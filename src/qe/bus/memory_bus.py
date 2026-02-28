from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from qe.bus.bus_metrics import get_bus_metrics
from qe.bus.protocol import validate_envelope
from qe.models.envelope import Envelope

if TYPE_CHECKING:
    from qe.bus.event_log import EventLog

log = logging.getLogger(__name__)

# ── Default bus configuration ──────────────────────────────────────────────
_DEFAULT_MAX_RETRIES = 2  # retries per handler (total attempts = 1 + retries)
_DEFAULT_RETRY_BASE_DELAY = 0.1  # seconds, doubles each retry
_DEFAULT_DEDUP_TTL = 300  # seconds to remember seen envelope IDs
_DEFAULT_DEDUP_MAX_SIZE = 50_000  # max entries in dedup cache
_DEFAULT_TOPIC_CONCURRENCY = 50  # max concurrent handlers per topic
_DLQ_TOPIC = "system.dlq"

_NON_RETRYABLE = (ValueError, TypeError, KeyError, AttributeError, NotImplementedError)


def is_retryable(exc: Exception) -> bool:
    return not isinstance(exc, _NON_RETRYABLE)


class DeadLetterEntry:
    """A failed envelope with error context for inspection / replay."""

    __slots__ = ("envelope", "handler_name", "error", "attempts", "failed_at")

    def __init__(
        self,
        envelope: Envelope,
        handler_name: str,
        error: str,
        attempts: int,
    ) -> None:
        self.envelope = envelope
        self.handler_name = handler_name
        self.error = error
        self.attempts = attempts
        self.failed_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_id": self.envelope.envelope_id,
            "topic": self.envelope.topic,
            "source_service_id": self.envelope.source_service_id,
            "handler_name": self.handler_name,
            "error": self.error,
            "attempts": self.attempts,
            "failed_at": self.failed_at,
            "payload": self.envelope.payload,
        }


class MemoryBus:
    def __init__(
        self,
        event_log: EventLog | None = None,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        dedup_ttl: float = _DEFAULT_DEDUP_TTL,
        topic_concurrency: int = _DEFAULT_TOPIC_CONCURRENCY,
    ) -> None:
        self._subscribers: dict[
            str, list[Callable[[Envelope], Awaitable[None]]]
        ] = defaultdict(list)
        self._event_log = event_log

        # ── Retry configuration ──
        self._max_retries = max_retries

        # ── Idempotency: dedup cache (envelope_id → timestamp) ──
        self._seen_ids: dict[str, float] = {}
        self._dedup_ttl = dedup_ttl
        self._dedup_max_size = _DEFAULT_DEDUP_MAX_SIZE

        # ── Backpressure: per-topic semaphores ──
        self._topic_concurrency = topic_concurrency
        self._topic_semaphores: dict[str, asyncio.Semaphore] = {}

        # ── Dead Letter Queue ──
        self._dlq: deque[DeadLetterEntry] = deque(maxlen=1000)

        # ── Publish listeners (synchronous observers of every published envelope) ──
        self._publish_listeners: list[Callable[[Envelope], None]] = []

    def set_event_log(self, event_log: EventLog) -> None:
        self._event_log = event_log

    def add_publish_listener(self, callback: Callable[[Envelope], None]) -> None:
        """Register a listener called synchronously on every published envelope."""
        self._publish_listeners.append(callback)

    def subscribe(self, topic: str, handler: Callable[[Envelope], Awaitable[None]]) -> None:
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[Envelope], Awaitable[None]]) -> None:
        self._subscribers[topic] = [h for h in self._subscribers[topic] if h is not handler]

    # ── Core publish ───────────────────────────────────────────────────────

    def publish(self, envelope: Envelope) -> list[asyncio.Task]:
        """Publish an envelope to all subscribers.

        Provides:
        - Idempotency: duplicate envelope_ids within TTL window are dropped
        - Backpressure: per-topic semaphore limits concurrent handler tasks
        - Retry: failed handlers are retried with exponential backoff
        - DLQ: permanently failed envelopes are quarantined

        Returns the list of created asyncio.Tasks so callers can optionally
        await them.
        """
        validate_envelope(envelope)

        # ── Idempotency check ──
        now = time.time()
        self._evict_stale_ids(now)
        if envelope.envelope_id in self._seen_ids:
            log.debug(
                "bus.dedup_dropped envelope_id=%s topic=%s",
                envelope.envelope_id,
                envelope.topic,
            )
            return []
        self._seen_ids[envelope.envelope_id] = now

        log.debug(
            "bus.publish topic=%s envelope_id=%s source=%s correlation_id=%s",
            envelope.topic,
            envelope.envelope_id,
            envelope.source_service_id,
            envelope.correlation_id,
        )

        # Record bus metrics
        get_bus_metrics().record_publish(envelope.topic)

        # Notify publish listeners (used by Supervisor for loop detection)
        for listener in self._publish_listeners:
            listener(envelope)

        # Fire-and-forget durable logging
        if self._event_log is not None:
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self._log_event(envelope))
            except RuntimeError:
                pass  # No event loop — skip durable logging

        handlers = self._subscribers.get(envelope.topic, [])
        if not handlers:
            return []

        try:
            asyncio.get_running_loop()
            running_loop = True
        except RuntimeError:
            running_loop = False

        tasks: list[asyncio.Task] = []
        for handler in handlers:
            if running_loop:
                task = asyncio.create_task(
                    self._guarded_call(handler, envelope)
                )
                task.add_done_callback(self._handle_task_exception)
                tasks.append(task)
            else:
                asyncio.run(self._guarded_call(handler, envelope))

        return tasks

    async def publish_and_wait(self, envelope: Envelope) -> list[Any]:
        """Publish an envelope and await all handler completions.

        Returns a list of results (None for handlers that succeed, or the
        exception object for handlers that failed).
        """
        tasks = self.publish(envelope)
        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return list(results)

    async def request(
        self,
        envelope: Envelope,
        reply_topic: str,
        timeout_seconds: float = 10.0,
    ) -> Envelope | None:
        """Publish an envelope and wait for a correlated reply.

        Subscribes a one-shot handler on ``reply_topic`` that matches
        ``correlation_id == envelope.envelope_id``. Returns the reply
        ``Envelope`` or ``None`` on timeout.
        """
        result_future: asyncio.Future[Envelope] = asyncio.get_running_loop().create_future()

        async def _one_shot(reply: Envelope) -> None:
            if reply.correlation_id == envelope.envelope_id and not result_future.done():
                result_future.set_result(reply)

        self.subscribe(reply_topic, _one_shot)
        try:
            self.publish(envelope)
            return await asyncio.wait_for(result_future, timeout=timeout_seconds)
        except TimeoutError:
            return None
        finally:
            self.unsubscribe(reply_topic, _one_shot)

    async def publish_and_wait_first(self, envelope: Envelope) -> Any:
        """Publish and return the result of the first handler to complete."""
        tasks = self.publish(envelope)
        if not tasks:
            return None
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        first = next(iter(done))
        return first.result()

    # ── Handler execution with backpressure + retry + DLQ ──────────────────

    async def _guarded_call(
        self,
        handler: Callable[[Envelope], Awaitable[None]],
        envelope: Envelope,
    ) -> None:
        """Execute handler with TTL check, backpressure, retry, and DLQ."""
        # ── Envelope TTL check ──
        if envelope.ttl_seconds is not None:
            from datetime import UTC, datetime, timedelta

            expiry = envelope.timestamp + timedelta(seconds=envelope.ttl_seconds)
            if datetime.now(UTC) > expiry:
                handler_name = getattr(handler, "__qualname__", repr(handler))
                log.warning(
                    "bus.envelope_expired envelope_id=%s topic=%s ttl=%d",
                    envelope.envelope_id,
                    envelope.topic,
                    envelope.ttl_seconds,
                )
                self._route_to_dlq(
                    envelope, handler_name, "envelope_expired", 0
                )
                return

        sem = self._get_semaphore(envelope.topic)
        async with sem:
            await self._retry_call(handler, envelope)

    async def _retry_call(
        self,
        handler: Callable[[Envelope], Awaitable[None]],
        envelope: Envelope,
    ) -> None:
        """Call handler with retry + exponential backoff. Route to DLQ on exhaustion."""
        last_error: Exception | None = None
        handler_name = getattr(handler, "__qualname__", repr(handler))
        total_attempts = 1 + self._max_retries

        for attempt in range(total_attempts):
            start = time.monotonic()
            try:
                result = handler(envelope)
                if inspect.isawaitable(result):
                    await result
                elapsed_ms = (time.monotonic() - start) * 1000
                log.debug(
                    "bus.handler_done handler=%s topic=%s envelope_id=%s "
                    "attempt=%d duration_ms=%.1f",
                    handler_name,
                    envelope.topic,
                    envelope.envelope_id,
                    attempt + 1,
                    elapsed_ms,
                )
                get_bus_metrics().record_handler_done(
                    envelope.topic, elapsed_ms
                )
                return  # success
            except Exception as exc:
                last_error = exc
                elapsed_ms = (time.monotonic() - start) * 1000
                if not is_retryable(exc):
                    get_bus_metrics().record_handler_error(envelope.topic)
                    log.warning(
                        "bus.non_retryable handler=%s error=%s",
                        handler_name,
                        str(exc),
                    )
                    break  # fall through to DLQ
                if attempt < self._max_retries:
                    jitter = random.uniform(0, _DEFAULT_RETRY_BASE_DELAY)
                    delay = _DEFAULT_RETRY_BASE_DELAY * (2**attempt) + jitter
                    log.warning(
                        "bus.handler_retry handler=%s envelope_id=%s "
                        "topic=%s attempt=%d/%d error=%s delay=%.2fs",
                        handler_name,
                        envelope.envelope_id,
                        envelope.topic,
                        attempt + 1,
                        total_attempts,
                        str(exc),
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    get_bus_metrics().record_handler_error(envelope.topic)
                    log.error(
                        "bus.handler_failed_permanently handler=%s "
                        "envelope_id=%s topic=%s attempts=%d error=%s "
                        "duration_ms=%.1f",
                        handler_name,
                        envelope.envelope_id,
                        envelope.topic,
                        total_attempts,
                        str(exc),
                        elapsed_ms,
                    )

        # All retries exhausted → route to DLQ
        if last_error is not None:
            self._route_to_dlq(
                envelope,
                handler_name,
                str(last_error),
                total_attempts,
            )

    def _route_to_dlq(
        self,
        envelope: Envelope,
        handler_name: str,
        error: str,
        attempts: int,
    ) -> None:
        """Quarantine a permanently failed envelope in the dead-letter queue."""
        entry = DeadLetterEntry(envelope, handler_name, error, attempts)
        self._dlq.append(entry)
        get_bus_metrics().record_dlq(envelope.topic)
        log.error(
            "bus.dlq_routed envelope_id=%s topic=%s handler=%s "
            "error=%s attempts=%d dlq_size=%d",
            envelope.envelope_id,
            envelope.topic,
            handler_name,
            error,
            attempts,
            len(self._dlq),
        )

        # Publish DLQ notification (without recursion — DLQ topic handlers
        # are fire-and-forget with no retry)
        try:
            asyncio.get_running_loop()
            dlq_handlers = self._subscribers.get(_DLQ_TOPIC, [])
            for dlq_handler in dlq_handlers:
                dlq_envelope = Envelope(
                    topic=_DLQ_TOPIC,
                    source_service_id="bus",
                    correlation_id=envelope.envelope_id,
                    payload=entry.to_dict(),
                )
                asyncio.create_task(self._safe_call_no_retry(dlq_handler, dlq_envelope))
        except RuntimeError:
            pass

    async def _safe_call_no_retry(
        self,
        handler: Callable[[Envelope], Awaitable[None]],
        envelope: Envelope,
    ) -> None:
        """Fire-and-forget handler call with no retry (for DLQ notifications)."""
        try:
            result = handler(envelope)
            if inspect.isawaitable(result):
                await result
        except Exception:
            log.exception(
                "DLQ handler %s failed for %s",
                getattr(handler, "__qualname__", "?"),
                envelope.envelope_id,
            )

    # ── DLQ inspection and replay ──────────────────────────────────────────

    def dlq_list(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent DLQ entries for inspection."""
        entries = list(self._dlq)[-limit:]
        return [e.to_dict() for e in reversed(entries)]

    def dlq_size(self) -> int:
        return len(self._dlq)

    async def dlq_replay(self, envelope_id: str) -> bool:
        """Re-publish a DLQ'd envelope back into the bus.

        Removes the entry from the DLQ and clears its dedup record so it
        can be re-processed.
        """
        for i, entry in enumerate(self._dlq):
            if entry.envelope.envelope_id == envelope_id:
                # Remove from DLQ
                del self._dlq[i]
                # Clear dedup so re-publish is accepted
                self._seen_ids.pop(envelope_id, None)
                log.info(
                    "bus.dlq_replay envelope_id=%s topic=%s",
                    envelope_id,
                    entry.envelope.topic,
                )
                self.publish(entry.envelope)
                return True
        return False

    async def dlq_purge(self) -> int:
        """Remove all entries from the DLQ. Returns count purged."""
        count = len(self._dlq)
        self._dlq.clear()
        log.info("bus.dlq_purged count=%d", count)
        return count

    # ── Idempotency helpers ────────────────────────────────────────────────

    def _evict_stale_ids(self, now: float) -> None:
        """Remove expired entries from the dedup cache."""
        cutoff = now - self._dedup_ttl
        stale = [eid for eid, ts in self._seen_ids.items() if ts < cutoff]
        for eid in stale:
            del self._seen_ids[eid]

        # If still oversized, evict oldest entries to cap memory use.
        overflow = len(self._seen_ids) - self._dedup_max_size
        if overflow > 0:
            oldest = sorted(self._seen_ids.items(), key=lambda item: item[1])[:overflow]
            for eid, _ in oldest:
                del self._seen_ids[eid]

    # ── Backpressure helpers ───────────────────────────────────────────────

    def _get_semaphore(self, topic: str) -> asyncio.Semaphore:
        """Get or create a per-topic concurrency semaphore."""
        if topic not in self._topic_semaphores:
            self._topic_semaphores[topic] = asyncio.Semaphore(
                self._topic_concurrency
            )
        return self._topic_semaphores[topic]

    # ── Event logging ──────────────────────────────────────────────────────

    async def _log_event(self, envelope: Envelope) -> None:
        try:
            await self._event_log.append(envelope)
        except Exception:
            log.exception("Failed to log event %s", envelope.envelope_id)

    def _handle_task_exception(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            pass

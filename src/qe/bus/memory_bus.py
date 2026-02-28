from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from qe.bus.protocol import validate_envelope
from qe.models.envelope import Envelope

if TYPE_CHECKING:
    from qe.bus.event_log import EventLog

log = logging.getLogger(__name__)


class MemoryBus:
    def __init__(self, event_log: EventLog | None = None) -> None:
        self._subscribers: dict[
            str, list[Callable[[Envelope], Awaitable[None]]]
        ] = defaultdict(list)
        self._event_log = event_log

    def set_event_log(self, event_log: EventLog) -> None:
        self._event_log = event_log

    def subscribe(self, topic: str, handler: Callable[[Envelope], Awaitable[None]]) -> None:
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[Envelope], Awaitable[None]]) -> None:
        self._subscribers[topic] = [h for h in self._subscribers[topic] if h is not handler]

    def publish(self, envelope: Envelope) -> list[asyncio.Task]:
        """Publish an envelope to all subscribers.

        Returns the list of created asyncio.Tasks so callers can optionally
        await them.
        """
        validate_envelope(envelope)

        log.debug(
            "bus.publish topic=%s envelope_id=%s source=%s correlation_id=%s",
            envelope.topic,
            envelope.envelope_id,
            envelope.source_service_id,
            envelope.correlation_id,
        )

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
                task = asyncio.create_task(self._safe_call(handler, envelope))
                task.add_done_callback(self._handle_task_exception)
                tasks.append(task)
            else:
                asyncio.run(self._safe_call(handler, envelope))

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

    async def _log_event(self, envelope: Envelope) -> None:
        try:
            await self._event_log.append(envelope)
        except Exception:
            log.exception("Failed to log event %s", envelope.envelope_id)

    async def _safe_call(
        self,
        handler: Callable[[Envelope], Awaitable[None]],
        envelope: Envelope,
    ) -> None:
        start = time.monotonic()
        try:
            await handler(envelope)
        except Exception:
            log.exception(
                "Handler %s failed for envelope %s on topic %s",
                handler.__qualname__,
                envelope.envelope_id,
                envelope.topic,
            )
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            log.debug(
                "bus.handler_done handler=%s topic=%s envelope_id=%s duration_ms=%.1f",
                handler.__qualname__,
                envelope.topic,
                envelope.envelope_id,
                elapsed_ms,
            )

    def _handle_task_exception(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            pass

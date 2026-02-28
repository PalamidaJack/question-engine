from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

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

    def publish(self, envelope: Envelope) -> None:
        validate_envelope(envelope)

        # Fire-and-forget durable logging
        if self._event_log is not None:
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self._log_event(envelope))
            except RuntimeError:
                pass  # No event loop — skip durable logging

        handlers = self._subscribers.get(envelope.topic, [])
        if not handlers:
            return

        try:
            asyncio.get_running_loop()
            running_loop = True
        except RuntimeError:
            running_loop = False

        for handler in handlers:
            if running_loop:
                task = asyncio.create_task(self._safe_call(handler, envelope))
                task.add_done_callback(self._handle_task_exception)
            else:
                asyncio.run(self._safe_call(handler, envelope))

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
        try:
            await handler(envelope)
        except Exception:
            log.exception(
                "Handler %s failed for envelope %s on topic %s",
                handler.__qualname__,
                envelope.envelope_id,
                envelope.topic,
            )

    def _handle_task_exception(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            pass

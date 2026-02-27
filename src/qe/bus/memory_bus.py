import asyncio
import logging
from collections import defaultdict
from typing import Callable, Awaitable

from qe.models.envelope import Envelope
from qe.bus.protocol import validate_envelope


log = logging.getLogger(__name__)


class MemoryBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[Envelope], Awaitable[None]]]] = (
            defaultdict(list)
        )

    def subscribe(
        self,
        topic: str,
        handler: Callable[[Envelope], Awaitable[None]],
    ) -> None:
        self._subscribers[topic].append(handler)

    def unsubscribe(
        self,
        topic: str,
        handler: Callable[[Envelope], Awaitable[None]],
    ) -> None:
        self._subscribers[topic] = [h for h in self._subscribers[topic] if h is not handler]

    def publish(self, envelope: Envelope) -> None:
        """
        Fire-and-forget publish. Validates envelope, creates tasks for all handlers.
        """
        validate_envelope(envelope)  # raises ValueError on unknown topic

        for handler in self._subscribers.get(envelope.topic, []):
            # Wrap in a task so exceptions in one handler don't block others
            task = asyncio.create_task(self._safe_call(handler, envelope))
            task.add_done_callback(self._handle_task_exception)

    async def _safe_call(
        self,
        handler: Callable[[Envelope], Awaitable[None]],
        envelope: Envelope,
    ) -> None:
        try:
            await handler(envelope)
        except Exception:
            log.exception(
                f"Handler {handler.__qualname__} failed for envelope "
                f"{envelope.envelope_id} on topic {envelope.topic}"
            )

    def _handle_task_exception(self, task: asyncio.Task) -> None:
        # Suppress CancelledError and already-logged exceptions
        if not task.cancelled() and task.exception():
            pass  # already logged in _safe_call

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable

from qe.bus.protocol import validate_envelope
from qe.models.envelope import Envelope

log = logging.getLogger(__name__)


class MemoryBus:
    def __init__(self) -> None:
        self._subscribers: dict[
            str, list[Callable[[Envelope], Awaitable[None]]]
        ] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[Envelope], Awaitable[None]]) -> None:
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable[[Envelope], Awaitable[None]]) -> None:
        self._subscribers[topic] = [h for h in self._subscribers[topic] if h is not handler]

    def publish(self, envelope: Envelope) -> None:
        validate_envelope(envelope)
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

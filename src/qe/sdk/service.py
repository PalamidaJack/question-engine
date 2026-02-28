"""SDK base class for building custom services."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

log = logging.getLogger(__name__)

# Registry for topic handlers
_HANDLER_REGISTRY: dict[str, list[tuple[str, str]]] = {}


def handles(topic: str) -> Callable:
    """Decorator to register a method as a topic handler."""

    def decorator(
        func: Callable[..., Coroutine],
    ) -> Callable[..., Coroutine]:
        func._handles_topic = topic  # type: ignore[attr-defined]
        return func

    return decorator


class ServiceBase:
    """Base class for SDK-built services.

    Provides automatic bus subscription, heartbeat management,
    tool access, and structured logging.
    """

    genome: str = ""

    def __init__(
        self,
        bus: Any = None,
        substrate: Any = None,
        tool_registry: Any = None,
    ) -> None:
        self.bus = bus
        self.substrate = substrate
        self.tool_registry = tool_registry
        self._handlers: dict[str, Callable] = {}
        self._running = False
        self._discover_handlers()

    def _discover_handlers(self) -> None:
        """Find methods decorated with @handles."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if callable(attr) and hasattr(attr, "_handles_topic"):
                topic = attr._handles_topic
                self._handlers[topic] = attr
                log.debug(
                    "sdk.handler_registered topic=%s method=%s",
                    topic,
                    attr_name,
                )

    async def start(self) -> None:
        """Start the service and subscribe to topics."""
        self._running = True
        if self.bus:
            for topic, handler in self._handlers.items():
                self.bus.subscribe(topic, handler)
        log.info(
            "sdk.service_started class=%s topics=%s",
            self.__class__.__name__,
            list(self._handlers.keys()),
        )

    async def stop(self) -> None:
        """Stop the service."""
        self._running = False
        log.info(
            "sdk.service_stopped class=%s",
            self.__class__.__name__,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    def get_subscribed_topics(self) -> list[str]:
        """Return list of topics this service handles."""
        return list(self._handlers.keys())

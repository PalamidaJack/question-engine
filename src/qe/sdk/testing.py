"""SDK testing framework for services and tools."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

log = logging.getLogger(__name__)


class MockBus:
    """Mock bus for testing services."""

    def __init__(self) -> None:
        self._published: list[dict[str, Any]] = []
        self._subscriptions: dict[str, list] = {}

    def subscribe(self, topic: str, handler: Any) -> None:
        self._subscriptions.setdefault(topic, []).append(handler)

    def publish(self, envelope: Any) -> list:
        self._published.append(envelope)
        return []

    @property
    def published(self) -> list[dict[str, Any]]:
        return self._published

    def published_to(self, topic: str) -> list:
        """Get envelopes published to a specific topic."""
        return [
            e for e in self._published
            if getattr(e, "topic", None) == topic
        ]


class ServiceTestHarness:
    """Test harness for running services in isolation.

    Provides a mock bus and substrate so services can be
    tested without a live engine.
    """

    def __init__(self, genome_path: str = "") -> None:
        self.genome_path = genome_path
        self.bus = MockBus()
        self.substrate = MagicMock()
        self.substrate.db_path = ":memory:"
        self._service = None

    async def start_service(self, service_class: type) -> Any:
        """Instantiate and start a service."""
        self._service = service_class(
            bus=self.bus, substrate=self.substrate
        )
        await self._service.start()
        return self._service

    async def send(
        self, topic: str, payload: dict[str, Any]
    ) -> Any:
        """Send a test envelope to topic handlers."""
        handlers = self.bus._subscriptions.get(topic, [])
        results = []
        for handler in handlers:
            envelope = MagicMock()
            envelope.topic = topic
            envelope.payload = payload
            for k, v in payload.items():
                setattr(envelope, k, v)
            result = await handler(envelope)
            results.append(result)
        return TestResult(
            published=self.bus.published,
            handler_results=results,
        )

    async def stop(self) -> None:
        """Stop the service."""
        if self._service:
            await self._service.stop()


class TestResult:
    """Result from a test harness send."""

    def __init__(
        self,
        published: list[Any],
        handler_results: list[Any],
    ) -> None:
        self.published = published
        self.handler_results = handler_results

    def published_to(self, topic: str) -> bool:
        """Check if any envelope was published to this topic."""
        return any(
            getattr(e, "topic", None) == topic
            for e in self.published
        )

    @property
    def first_result(self) -> Any:
        """Get the first handler result."""
        return self.handler_results[0] if self.handler_results else None

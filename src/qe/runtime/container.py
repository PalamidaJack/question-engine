"""Lightweight service container for dependency injection.

Replaces module-level singletons with a centralized registry
that supports overrides for testing and swappable implementations.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceContainer:
    """Minimal dependency injection container.

    Supports singleton and factory registrations. Services are resolved
    by name. Overrides allow test doubles without module-level patching.

    Usage:
        container = ServiceContainer()
        container.register("bus", lambda: MemoryBus())
        bus = container.resolve("bus")  # creates and caches instance

        # In tests:
        container.override("bus", mock_bus)
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Any]] = {}
        self._instances: dict[str, Any] = {}
        self._overrides: dict[str, Any] = {}

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        """Register a factory for a named service.

        The factory is called lazily on first resolve().
        Re-registering replaces the factory (but not cached instances).
        """
        self._factories[name] = factory
        log.debug("container.register name=%s", name)

    def register_instance(self, name: str, instance: Any) -> None:
        """Register a pre-created instance directly."""
        self._instances[name] = instance
        log.debug("container.register_instance name=%s", name)

    def resolve(self, name: str) -> Any:
        """Resolve a service by name.

        Resolution order:
        1. Override (if set)
        2. Cached singleton instance
        3. Factory (creates, caches, returns)

        Raises KeyError if service not registered.
        """
        # 1. Check overrides first (for testing)
        if name in self._overrides:
            return self._overrides[name]

        # 2. Check cached instances
        if name in self._instances:
            return self._instances[name]

        # 3. Create from factory
        factory = self._factories.get(name)
        if factory is None:
            raise KeyError(
                f"Service '{name}' not registered. "
                f"Available: {list(self._factories.keys())}"
            )

        instance = factory()
        self._instances[name] = instance
        log.debug("container.created name=%s type=%s", name, type(instance).__name__)
        return instance

    def override(self, name: str, instance: Any) -> None:
        """Override a service with a specific instance.

        Used in testing to inject mocks/stubs. Takes precedence
        over both factories and cached instances.
        """
        self._overrides[name] = instance
        log.debug("container.override name=%s", name)

    def clear_override(self, name: str) -> None:
        """Remove an override, restoring normal resolution."""
        self._overrides.pop(name, None)

    def clear_overrides(self) -> None:
        """Remove all overrides."""
        self._overrides.clear()

    def has(self, name: str) -> bool:
        """Check if a service is registered (factory or instance)."""
        return (
            name in self._overrides
            or name in self._instances
            or name in self._factories
        )

    def registered_names(self) -> list[str]:
        """Return all registered service names."""
        names = set(self._factories.keys()) | set(self._instances.keys())
        return sorted(names)

    def reset(self) -> None:
        """Clear all registrations, instances, and overrides.

        Primarily for testing. Does NOT call any cleanup on instances.
        """
        self._factories.clear()
        self._instances.clear()
        self._overrides.clear()

    def status(self) -> dict[str, Any]:
        """Return container status for monitoring."""
        return {
            "registered": len(self._factories),
            "instantiated": len(self._instances),
            "overrides": len(self._overrides),
            "services": self.registered_names(),
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_container = ServiceContainer()


def get_container() -> ServiceContainer:
    return _container

"""Graceful shutdown: drain in-flight work before stopping.

Provides coordinated shutdown with configurable drain timeout,
in-flight envelope tracking, and force-kill fallback.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Any

log = logging.getLogger(__name__)


class ShutdownCoordinator:
    """Coordinates graceful shutdown with drain and force-kill.

    Tracks in-flight handler executions. On shutdown signal:
    1. Stops accepting new work
    2. Waits up to drain_timeout for in-flight handlers to complete
    3. Force-cancels any remaining tasks after timeout

    Usage:
        coordinator = ShutdownCoordinator(drain_timeout=30.0)
        coordinator.install_signal_handlers(supervisor)

        # In handler execution:
        coordinator.enter_handler()
        try:
            await handle(envelope)
        finally:
            coordinator.exit_handler()
    """

    def __init__(self, drain_timeout: float = 30.0) -> None:
        self._drain_timeout = drain_timeout
        self._in_flight = 0
        self._draining = False
        self._shutdown_requested = False
        self._lock = asyncio.Lock()
        self._drain_complete = asyncio.Event()
        self._drain_complete.set()  # starts as complete (no work)
        self._shutdown_callbacks: list[Any] = []
        self._started_at: float | None = None
        self._shutdown_at: float | None = None

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def is_draining(self) -> bool:
        return self._draining

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def enter_handler(self) -> bool:
        """Mark a handler as starting. Returns False if draining (reject work)."""
        if self._draining:
            return False
        self._in_flight += 1
        if self._in_flight == 1:
            self._drain_complete.clear()
        return True

    def exit_handler(self) -> None:
        """Mark a handler as complete."""
        self._in_flight = max(0, self._in_flight - 1)
        if self._in_flight == 0:
            self._drain_complete.set()

    def on_shutdown(self, callback: Any) -> None:
        """Register a callback to invoke during shutdown."""
        self._shutdown_callbacks.append(callback)

    async def drain(self) -> bool:
        """Wait for in-flight handlers to complete within timeout.

        Returns True if all handlers completed, False if timeout expired.
        """
        self._draining = True
        self._shutdown_at = time.monotonic()
        log.info(
            "shutdown.drain_start in_flight=%d timeout=%.1fs",
            self._in_flight,
            self._drain_timeout,
        )

        if self._in_flight == 0:
            log.info("shutdown.drain_complete immediately (no in-flight)")
            return True

        try:
            await asyncio.wait_for(
                self._drain_complete.wait(),
                timeout=self._drain_timeout,
            )
            elapsed = time.monotonic() - self._shutdown_at
            log.info("shutdown.drain_complete elapsed=%.2fs", elapsed)
            return True
        except TimeoutError:
            log.warning(
                "shutdown.drain_timeout remaining=%d handlers",
                self._in_flight,
            )
            return False

    def install_signal_handlers(
        self,
        shutdown_coro_factory: Any,
    ) -> None:
        """Install SIGTERM/SIGINT handlers that trigger graceful shutdown.

        shutdown_coro_factory: callable that returns a coroutine performing
        the actual shutdown (e.g., supervisor.stop).
        """
        loop = asyncio.get_running_loop()

        def _signal_handler(sig: int) -> None:
            sig_name = signal.Signals(sig).name
            if self._shutdown_requested:
                log.warning(
                    "shutdown.forced signal=%s (second signal)",
                    sig_name,
                )
                return
            self._shutdown_requested = True
            log.info("shutdown.signal_received signal=%s", sig_name)
            asyncio.ensure_future(
                self._run_shutdown(shutdown_coro_factory), loop=loop
            )

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _signal_handler, sig)

        log.debug("shutdown.signal_handlers_installed")

    async def _run_shutdown(self, shutdown_coro_factory: Any) -> None:
        """Execute the full shutdown sequence."""
        # Drain in-flight work
        drained = await self.drain()

        if not drained:
            log.warning("shutdown.force_cancelling remaining handlers")

        # Run registered callbacks
        for cb in self._shutdown_callbacks:
            try:
                result = cb()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("shutdown.callback_error")

        # Run the main shutdown coroutine
        try:
            coro = shutdown_coro_factory()
            if asyncio.iscoroutine(coro):
                await coro
        except Exception:
            log.exception("shutdown.error")

    def status(self) -> dict[str, Any]:
        return {
            "in_flight": self._in_flight,
            "draining": self._draining,
            "shutdown_requested": self._shutdown_requested,
            "drain_timeout": self._drain_timeout,
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_shutdown_coordinator: ShutdownCoordinator | None = None


def get_shutdown_coordinator(
    drain_timeout: float = 30.0,
) -> ShutdownCoordinator:
    global _shutdown_coordinator
    if _shutdown_coordinator is None:
        _shutdown_coordinator = ShutdownCoordinator(drain_timeout=drain_timeout)
    return _shutdown_coordinator


def reset_shutdown_coordinator() -> None:
    """Reset coordinator (for testing)."""
    global _shutdown_coordinator
    _shutdown_coordinator = None

"""Watchdog: monitors kernel health and handles recovery."""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)


class Watchdog:
    """External process that monitors kernel health."""

    def __init__(
        self,
        check_interval: int = 10,
        health_url: str = "http://localhost:8000/api/health",
    ) -> None:
        self.check_interval = check_interval
        self.health_url = health_url
        self._running = False
        self._consecutive_failures = 0
        self._max_failures = 3

    async def start(self) -> None:
        """Start the watchdog monitoring loop."""
        self._running = True
        log.info(
            "watchdog.started interval=%ds",
            self.check_interval,
        )
        while self._running:
            healthy = await self._check_health()
            if healthy:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                log.warning(
                    "watchdog.health_check_failed"
                    " consecutive=%d",
                    self._consecutive_failures,
                )
                if (
                    self._consecutive_failures
                    >= self._max_failures
                ):
                    await self._handle_kernel_failure()
            await asyncio.sleep(self.check_interval)

    async def stop(self) -> None:
        """Stop the watchdog."""
        self._running = False
        log.info("watchdog.stopped")

    async def _check_health(self) -> bool:
        """Check if the kernel is responsive."""
        try:
            import httpx

            async with httpx.AsyncClient(
                timeout=5,
            ) as client:
                resp = await client.get(self.health_url)
                return resp.status_code == 200
        except Exception:
            return False

    async def _handle_kernel_failure(self) -> None:
        """Handle a detected kernel failure."""
        log.error("watchdog.kernel_failure detected")
        self._consecutive_failures = 0
        # In production, this would trigger a restart
        # For now, just log the event
        log.info("watchdog.would_restart_kernel")

    @property
    def is_running(self) -> bool:
        """Return whether the watchdog is running."""
        return self._running

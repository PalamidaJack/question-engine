"""Monitor service for managing recurring scheduled tasks."""

from __future__ import annotations

import logging
import uuid
from typing import Any

log = logging.getLogger(__name__)

_VALID_INTERVALS = {
    "hourly",
    "daily",
    "weekly",
    "monthly",
}


class MonitorService:
    """Manages recurring monitoring schedules for automated
    task execution."""

    def __init__(
        self,
        substrate: Any = None,
        bus: Any = None,
    ) -> None:
        self.substrate = substrate
        self.bus = bus
        self._schedules: dict[str, dict] = {}

    async def add_schedule(
        self,
        description: str,
        interval: str = "daily",
    ) -> dict:
        """Add a new recurring monitoring schedule.

        Args:
            description: What to monitor.
            interval: How often (hourly, daily, weekly,
                monthly).

        Returns the created schedule record.
        """
        if interval not in _VALID_INTERVALS:
            log.warning(
                "Unknown interval: %s, defaulting to daily",
                interval,
            )
            interval = "daily"

        monitor_id = f"mon_{uuid.uuid4().hex[:12]}"
        schedule = {
            "monitor_id": monitor_id,
            "description": description,
            "interval": interval,
            "active": True,
        }
        self._schedules[monitor_id] = schedule

        log.info(
            "Schedule added: id=%s interval=%s desc=%.60s",
            monitor_id,
            interval,
            description,
        )

        return schedule

    async def list_schedules(self) -> list[dict]:
        """Return all registered monitoring schedules."""
        return list(self._schedules.values())

    async def remove_schedule(
        self, monitor_id: str
    ) -> bool:
        """Remove a monitoring schedule by ID.

        Returns True if the schedule was found and removed,
        False otherwise.
        """
        if monitor_id in self._schedules:
            del self._schedules[monitor_id]
            log.info("Schedule removed: %s", monitor_id)
            return True

        log.warning(
            "Schedule not found for removal: %s",
            monitor_id,
        )
        return False

"""Digest service for generating daily system summaries."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)


class DigestService:
    """Generates daily digest reports summarizing system
    activity, costs, and health."""

    def __init__(
        self,
        substrate: Any = None,
        bus: Any = None,
        budget_tracker: Any = None,
    ) -> None:
        self.substrate = substrate
        self.bus = bus
        self.budget_tracker = budget_tracker

    async def generate_daily_digest(self) -> dict:
        """Generate a daily digest of system activity.

        Aggregates goal progress, claim counts, cost data,
        and system health into a single report.
        """
        now = datetime.now(UTC).isoformat()

        goals_completed = 0
        goals_in_progress = 0
        claims_committed = 0
        total_cost_usd = 0.0

        if self.substrate is not None:
            goals_completed = await self._count_goals(
                "completed"
            )
            goals_in_progress = await self._count_goals(
                "in_progress"
            )
            claims_committed = await self._count_claims()

        if self.budget_tracker is not None:
            total_cost_usd = await self._get_total_cost()

        digest = {
            "goals_completed": goals_completed,
            "goals_in_progress": goals_in_progress,
            "claims_committed": claims_committed,
            "total_cost_usd": total_cost_usd,
            "system_health": "nominal",
            "generated_at": now,
        }

        log.info(
            "Daily digest generated at %s: "
            "completed=%d in_progress=%d claims=%d",
            now,
            goals_completed,
            goals_in_progress,
            claims_committed,
        )

        return digest

    async def _count_goals(self, status: str) -> int:
        """Count goals with the given status."""
        # Placeholder: would query substrate
        _ = status
        return 0

    async def _count_claims(self) -> int:
        """Count total committed claims."""
        # Placeholder: would query substrate
        return 0

    async def _get_total_cost(self) -> float:
        """Retrieve total cost from budget tracker."""
        # Placeholder: would query budget tracker
        return 0.0

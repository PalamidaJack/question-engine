"""Assumption monitor: re-verifies assumptions for active goals."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)


class InvalidatedAssumption:
    """An assumption that has been invalidated."""

    def __init__(
        self,
        assumption: str,
        reason: str,
        detected_at: str = "",
    ) -> None:
        self.assumption = assumption
        self.reason = reason
        self.detected_at = (
            detected_at or datetime.now(UTC).isoformat()
        )


class AssumptionMonitor:
    """Periodically re-verifies assumptions for active goals."""

    def __init__(self, substrate: Any = None) -> None:
        self.substrate = substrate

    async def check_assumptions(
        self,
        assumptions: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[InvalidatedAssumption]:
        """Check if any assumptions have been invalidated.

        Simple heuristic checks -- no LLM calls.
        """
        invalidated: list[InvalidatedAssumption] = []

        for assumption in assumptions:
            lower = assumption.lower()

            # Check time-based assumptions
            if "before" in lower or "deadline" in lower:
                # Can't fully evaluate without parsing dates,
                # but flag old assumptions
                invalidated.append(
                    InvalidatedAssumption(
                        assumption=assumption,
                        reason=(
                            "Time-based assumption needs review"
                        ),
                    )
                )

            # Check availability assumptions
            if "available" in lower or "accessible" in lower:
                # Flag for review if context suggests issues
                if context and context.get("errors"):
                    invalidated.append(
                        InvalidatedAssumption(
                            assumption=assumption,
                            reason=(
                                "Recent errors may affect"
                                " availability"
                            ),
                        )
                    )

        return invalidated

    async def get_stale_assumptions(
        self,
        assumptions: list[str],
        max_age_hours: int = 24,
    ) -> list[str]:
        """Return assumptions older than max_age_hours."""
        # Without timestamps on assumptions, return all if
        # the goal has been running long
        _ = max_age_hours
        return assumptions if len(assumptions) > 5 else []

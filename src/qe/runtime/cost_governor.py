"""Cost governance: per-goal caps, preflight estimation, auto-downgrade.

Works alongside BudgetTracker (global monthly spend) to enforce
fine-grained cost controls per goal and per LLM call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Rough cost estimates per 1M tokens (input) by model prefix
# Used for preflight estimation when exact pricing unavailable
_COST_PER_M_INPUT: dict[str, float] = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "o1": 15.00,
    "claude-haiku": 0.25,
    "claude-sonnet": 3.00,
    "claude-opus": 15.00,
    "gemini/gemini-2.0-flash": 0.10,
    "gemini/gemini-2.5-pro": 1.25,
    "groq/": 0.05,
    "ollama/": 0.0,
    "together_ai/": 0.20,
}

_COST_PER_M_OUTPUT: dict[str, float] = {
    "gpt-4o-mini": 0.60,
    "gpt-4o": 10.00,
    "o1": 60.00,
    "claude-haiku": 1.25,
    "claude-sonnet": 15.00,
    "claude-opus": 75.00,
    "gemini/gemini-2.0-flash": 0.40,
    "gemini/gemini-2.5-pro": 10.00,
    "groq/": 0.05,
    "ollama/": 0.0,
    "together_ai/": 0.20,
}


@dataclass
class CostEstimate:
    """Preflight cost estimate for an LLM call."""

    model: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    within_budget: bool
    remaining_goal_budget_usd: float | None = None


@dataclass
class GoalBudget:
    """Tracks spend for a single goal."""

    goal_id: str
    cap_usd: float  # max spend for this goal
    spent_usd: float = 0.0
    call_count: int = 0

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.cap_usd


class CostGovernor:
    """Enforces per-goal cost caps and provides preflight cost estimates."""

    def __init__(
        self,
        default_goal_cap_usd: float = 5.0,
        budget_tracker: Any | None = None,
    ) -> None:
        self.default_goal_cap_usd = default_goal_cap_usd
        self._budget_tracker = budget_tracker
        self._goal_budgets: dict[str, GoalBudget] = {}

    def register_goal(
        self, goal_id: str, cap_usd: float | None = None
    ) -> GoalBudget:
        """Register a goal with a cost cap."""
        budget = GoalBudget(
            goal_id=goal_id,
            cap_usd=cap_usd if cap_usd is not None else self.default_goal_cap_usd,
        )
        self._goal_budgets[goal_id] = budget
        log.debug(
            "cost_governor.register goal_id=%s cap_usd=%.2f",
            goal_id,
            budget.cap_usd,
        )
        return budget

    def get_goal_budget(self, goal_id: str) -> GoalBudget | None:
        return self._goal_budgets.get(goal_id)

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 500,
        goal_id: str | None = None,
    ) -> CostEstimate:
        """Estimate cost for an LLM call before making it.

        Args:
            model: Model name (e.g., "gpt-4o-mini").
            input_tokens: Estimated input token count.
            output_tokens: Estimated output token count (default 500).
            goal_id: Optional goal to check against goal budget.
        """
        input_rate = self._lookup_rate(_COST_PER_M_INPUT, model)
        output_rate = self._lookup_rate(_COST_PER_M_OUTPUT, model)

        est_cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

        within_budget = True
        remaining_goal = None

        # Check goal budget
        if goal_id:
            gb = self._goal_budgets.get(goal_id)
            if gb:
                remaining_goal = gb.remaining_usd
                if est_cost > gb.remaining_usd:
                    within_budget = False

        # Check global budget
        if self._budget_tracker and self._budget_tracker.remaining_pct() <= 0:
            within_budget = False

        return CostEstimate(
            model=model,
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=round(est_cost, 6),
            within_budget=within_budget,
            remaining_goal_budget_usd=remaining_goal,
        )

    def record_call(
        self,
        goal_id: str,
        cost_usd: float,
    ) -> bool:
        """Record actual cost after an LLM call. Returns True if within cap."""
        gb = self._goal_budgets.get(goal_id)
        if not gb:
            return True

        gb.spent_usd += cost_usd
        gb.call_count += 1

        if gb.exhausted:
            log.warning(
                "cost_governor.goal_budget_exhausted goal_id=%s "
                "spent=%.4f cap=%.2f calls=%d",
                goal_id,
                gb.spent_usd,
                gb.cap_usd,
                gb.call_count,
            )
            return False

        log.debug(
            "cost_governor.recorded goal_id=%s cost=%.6f "
            "spent=%.4f remaining=%.4f",
            goal_id,
            cost_usd,
            gb.spent_usd,
            gb.remaining_usd,
        )
        return True

    def suggest_downgrade(self, model: str) -> str | None:
        """Suggest a cheaper model when budget is tight.

        Returns a cheaper model name, or None if already cheapest.
        """
        downgrades: dict[str, str] = {
            "o1-preview": "gpt-4o",
            "gpt-4o": "gpt-4o-mini",
            "claude-sonnet-4-20250514": "claude-haiku-4-5-20251001",
            "gemini/gemini-2.5-pro-preview-06-05": "gemini/gemini-2.0-flash",
        }
        return downgrades.get(model)

    def goal_summary(self, goal_id: str) -> dict[str, Any] | None:
        """Return cost summary for a goal."""
        gb = self._goal_budgets.get(goal_id)
        if not gb:
            return None
        return {
            "goal_id": gb.goal_id,
            "cap_usd": gb.cap_usd,
            "spent_usd": round(gb.spent_usd, 6),
            "remaining_usd": round(gb.remaining_usd, 6),
            "call_count": gb.call_count,
            "exhausted": gb.exhausted,
        }

    @staticmethod
    def _lookup_rate(table: dict[str, float], model: str) -> float:
        """Look up cost rate, matching by prefix."""
        for prefix, rate in table.items():
            if model.startswith(prefix):
                return rate
        return 1.0  # fallback: assume $1/M tokens

"""Budget tracking for LLM spend via litellm cost callbacks."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

log = logging.getLogger(__name__)


class BudgetTracker:
    """Tracks cumulative LLM spend and enforces monthly budget limits."""

    def __init__(self, monthly_limit_usd: float = 50.0, alert_at_pct: float = 0.80) -> None:
        self.monthly_limit_usd = monthly_limit_usd
        self.alert_at_pct = alert_at_pct
        self._month_key: str = self._current_month_key()
        self._spend: dict[str, float] = {}  # model -> cumulative USD
        self._total_spend: float = 0.0
        self._alerted = False

    def record_cost(self, model: str, cost_usd: float) -> None:
        """Record a completed LLM call cost."""
        current_key = self._current_month_key()
        if current_key != self._month_key:
            self._reset(current_key)

        self._spend[model] = self._spend.get(model, 0.0) + cost_usd
        self._total_spend += cost_usd

        pct = self.remaining_pct()
        if pct < (1.0 - self.alert_at_pct) and not self._alerted:
            log.warning(
                "Budget alert: %.1f%% remaining ($%.4f / $%.2f)",
                pct * 100,
                self._total_spend,
                self.monthly_limit_usd,
            )
            self._alerted = True

        if pct <= 0:
            log.error(
                "Budget EXHAUSTED: $%.4f / $%.2f — switching to local-only",
                self._total_spend,
                self.monthly_limit_usd,
            )

    def remaining_pct(self) -> float:
        """Return fraction of budget remaining (0.0 to 1.0)."""
        current_key = self._current_month_key()
        if current_key != self._month_key:
            self._reset(current_key)

        if self.monthly_limit_usd <= 0:
            return 0.0
        return max(0.0, 1.0 - (self._total_spend / self.monthly_limit_usd))

    def total_spend(self) -> float:
        """Return total spend this month."""
        return self._total_spend

    def spend_by_model(self) -> dict[str, float]:
        """Return spend breakdown by model."""
        return dict(self._spend)

    def _reset(self, month_key: str) -> None:
        """Reset counters for a new month."""
        log.info("Budget reset for new month: %s", month_key)
        self._month_key = month_key
        self._spend.clear()
        self._total_spend = 0.0
        self._alerted = False

    @staticmethod
    def _current_month_key() -> str:
        now = datetime.now(UTC)
        return f"{now.year}-{now.month:02d}"

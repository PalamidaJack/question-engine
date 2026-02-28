"""Budget tracking for LLM spend via litellm cost callbacks."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

import aiosqlite

log = logging.getLogger(__name__)


class BudgetTracker:
    """Tracks cumulative LLM spend and enforces monthly budget limits.

    Supports optional SQLite persistence so spend data survives restarts.
    """

    def __init__(
        self,
        monthly_limit_usd: float = 50.0,
        alert_at_pct: float = 0.80,
        db_path: str | None = None,
    ) -> None:
        self.monthly_limit_usd = monthly_limit_usd
        self.alert_at_pct = alert_at_pct
        self._month_key: str = self._current_month_key()
        self._spend: dict[str, float] = {}  # model -> cumulative USD
        self._total_spend: float = 0.0
        self._alerted = False
        self._db_path = db_path
        self._save_lock = asyncio.Lock()
        self._pending_records: list[dict] = []

    def record_cost(
        self,
        model: str,
        cost_usd: float,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        service_id: str = "",
        envelope_id: str = "",
    ) -> None:
        """Record a completed LLM call cost."""
        current_key = self._current_month_key()
        if current_key != self._month_key:
            self._reset(current_key)

        self._spend[model] = self._spend.get(model, 0.0) + cost_usd
        self._total_spend += cost_usd

        log.debug(
            "budget.record model=%s cost_usd=%.6f tokens_in=%d tokens_out=%d "
            "service_id=%s total_spend=%.4f",
            model,
            cost_usd,
            tokens_in,
            tokens_out,
            service_id,
            self._total_spend,
        )

        # Queue for async persistence
        if self._db_path:
            self._pending_records.append({
                "month_key": self._month_key,
                "model": model,
                "cost_usd": cost_usd,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "service_id": service_id,
                "envelope_id": envelope_id,
                "created_at": datetime.now(UTC).isoformat(),
            })

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

    async def save(self) -> None:
        """Flush pending cost records to SQLite."""
        if not self._db_path or not self._pending_records:
            return

        async with self._save_lock:
            records = self._pending_records[:]
            self._pending_records.clear()

            async with aiosqlite.connect(self._db_path) as db:
                for rec in records:
                    await db.execute(
                        """
                        INSERT INTO budget_records
                            (month_key, model, cost_usd, tokens_in, tokens_out,
                             service_id, envelope_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rec["month_key"],
                            rec["model"],
                            rec["cost_usd"],
                            rec["tokens_in"],
                            rec["tokens_out"],
                            rec["service_id"],
                            rec["envelope_id"],
                            rec["created_at"],
                        ),
                    )
                # Update monthly summary
                await db.execute(
                    """
                    INSERT INTO budget_monthly (month_key, total_spend_usd, spend_by_model)
                    VALUES (?, ?, ?)
                    ON CONFLICT(month_key) DO UPDATE SET
                        total_spend_usd = ?,
                        spend_by_model = ?
                    """,
                    (
                        self._month_key,
                        self._total_spend,
                        json.dumps(self._spend),
                        self._total_spend,
                        json.dumps(self._spend),
                    ),
                )
                await db.commit()

        log.debug(
            "budget.saved records=%d total_spend=%.4f",
            len(records),
            self._total_spend,
        )

    async def load(self) -> None:
        """Load budget state from SQLite on startup."""
        if not self._db_path:
            return

        current_key = self._current_month_key()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    "SELECT total_spend_usd, spend_by_model"
                    " FROM budget_monthly WHERE month_key = ?",
                    (current_key,),
                )
                row = await cursor.fetchone()
        except Exception:
            log.debug("budget.load no budget tables yet — starting fresh")
            return

        if row:
            self._month_key = current_key
            self._total_spend = row[0]
            self._spend = json.loads(row[1])
            log.info(
                "budget.loaded month=%s total_spend=%.4f models=%d",
                current_key,
                self._total_spend,
                len(self._spend),
            )
        else:
            self._reset(current_key)

    def update_limits(
        self,
        monthly_limit_usd: float | None = None,
        alert_at_pct: float | None = None,
    ) -> None:
        """Update budget limits at runtime without restart."""
        if monthly_limit_usd is not None:
            self.monthly_limit_usd = monthly_limit_usd
        if alert_at_pct is not None:
            self.alert_at_pct = alert_at_pct
            self._alerted = False  # re-arm alert at new threshold

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

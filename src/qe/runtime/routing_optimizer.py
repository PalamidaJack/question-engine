"""Routing optimizer: data-driven model selection
with Thompson sampling."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import aiosqlite

log = logging.getLogger(__name__)


class ModelScore:
    """Score for a model on a specific task type."""

    def __init__(
        self,
        model: str,
        success_rate: float,
        avg_latency_ms: float,
        avg_cost_usd: float,
        sample_count: int,
    ) -> None:
        self.model = model
        self.success_rate = success_rate
        self.avg_latency_ms = avg_latency_ms
        self.avg_cost_usd = avg_cost_usd
        self.sample_count = sample_count
        self.composite_score = 0.0

    def compute_composite(
        self,
        budget_remaining: float = 1000.0,
        latency_budget_ms: int | None = None,
    ) -> float:
        """Compute composite score."""
        score = self.success_rate * 100
        # Penalize expensive models when budget is low
        if (
            budget_remaining < 10.0
            and self.avg_cost_usd > 0
        ):
            cost_ratio = self.avg_cost_usd / budget_remaining
            score -= cost_ratio * 20
        # Penalize slow models if latency budget given
        if (
            latency_budget_ms
            and self.avg_latency_ms > latency_budget_ms
        ):
            score -= 10
        self.composite_score = max(score, 0)
        return self.composite_score


class RoutingOptimizer:
    """Data-driven model selection with exploration."""

    def __init__(
        self,
        db_path: str | None = None,
        exploration_rate: float = 0.1,
    ) -> None:
        self._db_path = db_path
        self._exploration_rate = exploration_rate
        self._initialized = False
        # In-memory stats:
        # {(model, task_type): {"successes": int, ...}}
        self._stats: dict[
            tuple[str, str], dict[str, Any]
        ] = defaultdict(
            lambda: {
                "successes": 0,
                "total": 0,
                "total_latency": 0,
                "total_cost": 0.0,
            }
        )

    async def _ensure_table(self) -> None:
        if self._initialized or not self._db_path:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS routing_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    latency_ms INTEGER,
                    cost_usd REAL,
                    quality_score REAL,
                    created_at TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_routing
                ON routing_records(model, task_type)
            """)
            await db.commit()
        self._initialized = True

    async def record_outcome(
        self,
        model: str,
        task_type: str,
        success: bool,
        latency_ms: int = 0,
        cost_usd: float = 0.0,
        quality_score: float = 0.0,
    ) -> None:
        """Record a model outcome."""
        key = (model, task_type)
        stats = self._stats[key]
        stats["total"] += 1
        if success:
            stats["successes"] += 1
        stats["total_latency"] += latency_ms
        stats["total_cost"] += cost_usd

        if self._db_path:
            await self._ensure_table()
            async with aiosqlite.connect(
                self._db_path
            ) as db:
                await db.execute(
                    "INSERT INTO routing_records "
                    "(model, task_type, success, "
                    "latency_ms, cost_usd, "
                    "quality_score, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        model,
                        task_type,
                        success,
                        latency_ms,
                        cost_usd,
                        quality_score,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                await db.commit()

    def get_model_scores(
        self,
        task_type: str,
        available_models: list[str],
    ) -> list[ModelScore]:
        """Score each available model for a task type."""
        scores = []
        for model in available_models:
            key = (model, task_type)
            stats = self._stats.get(key)
            if stats and stats["total"] > 0:
                total = stats["total"]
                score = ModelScore(
                    model=model,
                    success_rate=(
                        stats["successes"] / total
                    ),
                    avg_latency_ms=(
                        stats["total_latency"] / total
                    ),
                    avg_cost_usd=(
                        stats["total_cost"] / total
                    ),
                    sample_count=total,
                )
            else:
                score = ModelScore(
                    model=model,
                    success_rate=0.5,  # Optimistic prior
                    avg_latency_ms=0,
                    avg_cost_usd=0,
                    sample_count=0,
                )
            scores.append(score)
        return scores

    def select_model(
        self,
        task_type: str,
        available_models: list[str],
        budget_remaining: float = 1000.0,
        latency_budget_ms: int | None = None,
    ) -> str:
        """Select best model using scoring +
        exploration."""
        if not available_models:
            raise ValueError("No models available")

        # Exploration: sometimes pick a random model
        if random.random() < self._exploration_rate:
            return random.choice(available_models)

        # Exploitation: pick the highest-scoring model
        scores = self.get_model_scores(
            task_type, available_models
        )
        for s in scores:
            s.compute_composite(
                budget_remaining, latency_budget_ms
            )
        scores.sort(
            key=lambda s: s.composite_score,
            reverse=True,
        )
        return scores[0].model

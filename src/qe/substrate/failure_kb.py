"""Failure knowledge base: stores and retrieves failure patterns and recovery strategies."""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import UTC, datetime

import aiosqlite

log = logging.getLogger(__name__)


class FailureKnowledgeBase:
    """Stores and retrieves failure patterns and recovery strategies."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._initialized = False

    async def _ensure_table(self) -> None:
        """Create the failure_records table if it doesn't exist."""
        if self._initialized:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS failure_records (
                    failure_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    model_used TEXT,
                    failure_class TEXT NOT NULL,
                    error_summary TEXT,
                    context_fingerprint TEXT,
                    recovery_strategy TEXT,
                    recovery_succeeded BOOLEAN,
                    created_at TIMESTAMP,
                    goal_id TEXT,
                    subtask_id TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_failure_lookup
                ON failure_records(failure_class, task_type)
            """)
            await db.commit()
        self._initialized = True

    async def record(
        self,
        task_type: str,
        failure_class: str,
        error_summary: str,
        recovery_strategy: str,
        success: bool,
        *,
        model_used: str = "",
        goal_id: str = "",
        subtask_id: str = "",
        context: str = "",
    ) -> str:
        """Record a failure and the outcome of a recovery attempt."""
        await self._ensure_table()
        failure_id = f"fail_{uuid.uuid4().hex[:12]}"
        fingerprint = hashlib.md5(  # noqa: S324
            f"{task_type}:{failure_class}:{error_summary[:200]}".encode()
        ).hexdigest()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO failure_records
                    (failure_id, task_type, model_used, failure_class,
                     error_summary, context_fingerprint, recovery_strategy,
                     recovery_succeeded, created_at, goal_id, subtask_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    failure_id,
                    task_type,
                    model_used,
                    failure_class,
                    error_summary,
                    fingerprint,
                    recovery_strategy,
                    success,
                    datetime.now(UTC).isoformat(),
                    goal_id,
                    subtask_id,
                ),
            )
            await db.commit()

        log.debug(
            "failure_kb.record id=%s class=%s strategy=%s success=%s",
            failure_id,
            failure_class,
            recovery_strategy,
            success,
        )
        return failure_id

    async def lookup(
        self,
        failure_class: str,
        task_type: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Find the most effective recovery strategies for this failure type.

        Returns strategies ranked by success rate.
        """
        await self._ensure_table()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT recovery_strategy,
                       COUNT(*) as attempts,
                       SUM(CASE WHEN recovery_succeeded THEN 1 ELSE 0 END) as successes
                FROM failure_records
                WHERE failure_class = ? AND task_type = ?
                GROUP BY recovery_strategy
                ORDER BY (CAST(successes AS REAL) / COUNT(*)) DESC
                LIMIT ?
                """,
                (failure_class, task_type, top_k),
            )
            rows = await cursor.fetchall()

        return [
            {
                "strategy": row[0],
                "attempts": row[1],
                "successes": row[2],
                "success_rate": row[2] / row[1] if row[1] > 0 else 0,
            }
            for row in rows
        ]

    async def get_avoidance_rules(self, task_type: str) -> list[dict]:
        """Get patterns the planner should avoid based on known failures."""
        await self._ensure_table()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT failure_class, error_summary,
                       COUNT(*) as occurrences
                FROM failure_records
                WHERE task_type = ? AND NOT recovery_succeeded
                GROUP BY failure_class, error_summary
                HAVING COUNT(*) >= 2
                ORDER BY occurrences DESC
                LIMIT 10
                """,
                (task_type,),
            )
            rows = await cursor.fetchall()

        return [
            {
                "failure_class": row[0],
                "pattern": row[1],
                "occurrences": row[2],
            }
            for row in rows
        ]

    async def count(self) -> int:
        """Total failure records."""
        await self._ensure_table()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM failure_records")
            row = await cursor.fetchone()
        return row[0]

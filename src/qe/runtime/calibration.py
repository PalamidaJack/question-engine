"""Confidence calibration: tracks relationship
between reported and actual accuracy."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime

import aiosqlite

log = logging.getLogger(__name__)


class CalibrationTracker:
    """Tracks calibration between reported confidence
    and actual accuracy for each (model, task_type) pair."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        self._initialized = False
        # In-memory buckets:
        # {(model, task_type): {bucket: [correct_bools]}}
        self._buckets: dict[
            tuple[str, str], dict[int, list[bool]]
        ] = defaultdict(lambda: defaultdict(list))

    async def _ensure_table(self) -> None:
        if self._initialized or not self._db_path:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS calibration_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    reported_confidence REAL NOT NULL,
                    actual_correct BOOLEAN NOT NULL,
                    created_at TIMESTAMP,
                    goal_id TEXT,
                    subtask_id TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_calibration
                ON calibration_records(model, task_type)
            """)
            await db.commit()
        self._initialized = True

    def _bucket_for(self, confidence: float) -> int:
        """Map confidence to a bucket (0-9 for 0.0-1.0)."""
        return min(int(confidence * 10), 9)

    async def record(
        self,
        model: str,
        task_type: str,
        reported_confidence: float,
        actual_correct: bool,
        *,
        goal_id: str = "",
        subtask_id: str = "",
    ) -> None:
        """Record a calibration data point."""
        bucket = self._bucket_for(reported_confidence)
        key = (model, task_type)
        self._buckets[key][bucket].append(actual_correct)

        if self._db_path:
            await self._ensure_table()
            async with aiosqlite.connect(
                self._db_path
            ) as db:
                await db.execute(
                    "INSERT INTO calibration_records "
                    "(model, task_type, "
                    "reported_confidence, "
                    "actual_correct, created_at, "
                    "goal_id, subtask_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        model,
                        task_type,
                        reported_confidence,
                        actual_correct,
                        datetime.now(UTC).isoformat(),
                        goal_id,
                        subtask_id,
                    ),
                )
                await db.commit()

    def calibrated_confidence(
        self,
        model: str,
        task_type: str,
        raw_confidence: float,
    ) -> float:
        """Adjust raw confidence based on historical
        calibration."""
        key = (model, task_type)
        bucket = self._bucket_for(raw_confidence)
        data = self._buckets.get(key, {}).get(bucket, [])
        if len(data) < 5:
            return raw_confidence  # Not enough data
        return sum(data) / len(data)

    def get_calibration_curve(
        self, model: str, task_type: str
    ) -> list[tuple[float, float]]:
        """Return (reported_midpoint, actual_accuracy)
        pairs."""
        key = (model, task_type)
        buckets = self._buckets.get(key, {})
        curve = []
        for bucket_id in sorted(buckets.keys()):
            data = buckets[bucket_id]
            if data:
                midpoint = (bucket_id + 0.5) / 10
                accuracy = sum(data) / len(data)
                curve.append((midpoint, accuracy))
        return curve

    async def count(self) -> int:
        """Total calibration records."""
        total = 0
        for buckets in self._buckets.values():
            for data in buckets.values():
                total += len(data)
        return total

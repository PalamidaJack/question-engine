"""Persistent dead-letter queue backed by SQLite.

Replaces the in-memory deque DLQ so entries survive process restarts.
Provides the same API surface as the MemoryBus DLQ methods.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import aiosqlite

from qe.models.envelope import Envelope

log = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS dead_letter_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id TEXT NOT NULL UNIQUE,
    topic TEXT NOT NULL,
    source_service_id TEXT NOT NULL,
    handler_name TEXT NOT NULL,
    error TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    payload TEXT NOT NULL,
    envelope_json TEXT NOT NULL,
    failed_at REAL NOT NULL,
    replayed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_dlq_envelope_id ON dead_letter_queue(envelope_id);
CREATE INDEX IF NOT EXISTS idx_dlq_topic ON dead_letter_queue(topic);
CREATE INDEX IF NOT EXISTS idx_dlq_failed_at ON dead_letter_queue(failed_at);
"""


class PersistentDLQ:
    """SQLite-backed dead-letter queue.

    Entries persist across restarts. Supports append, list,
    get, replay, and purge operations.
    """

    def __init__(self, db_path: str = "data/dlq.db") -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        """Create the DLQ table if it doesn't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_CREATE_TABLE)
            await db.commit()
        log.debug("persistent_dlq.initialized db=%s", self._db_path)

    async def append(
        self,
        envelope: Envelope,
        handler_name: str,
        error: str,
        attempts: int,
    ) -> None:
        """Add a failed envelope to the persistent DLQ."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO dead_letter_queue
                    (envelope_id, topic, source_service_id, handler_name,
                     error, attempts, payload, envelope_json, failed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.envelope_id,
                    envelope.topic,
                    envelope.source_service_id,
                    handler_name,
                    error,
                    attempts,
                    json.dumps(envelope.payload),
                    envelope.model_dump_json(),
                    time.time(),
                ),
            )
            await db.commit()
        log.debug(
            "persistent_dlq.append envelope_id=%s topic=%s",
            envelope.envelope_id,
            envelope.topic,
        )

    async def list_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        """List DLQ entries, most recent first."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT envelope_id, topic, source_service_id,
                       handler_name, error, attempts, payload, failed_at
                FROM dead_letter_queue
                WHERE replayed = 0
                ORDER BY failed_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()

        return [
            {
                "envelope_id": row[0],
                "topic": row[1],
                "source_service_id": row[2],
                "handler_name": row[3],
                "error": row[4],
                "attempts": row[5],
                "payload": json.loads(row[6]),
                "failed_at": row[7],
            }
            for row in rows
        ]

    async def get_entry(self, envelope_id: str) -> dict[str, Any] | None:
        """Get a specific DLQ entry by envelope ID."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT envelope_id, topic, source_service_id,
                       handler_name, error, attempts, payload,
                       envelope_json, failed_at
                FROM dead_letter_queue
                WHERE envelope_id = ? AND replayed = 0
                """,
                (envelope_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            return None

        return {
            "envelope_id": row[0],
            "topic": row[1],
            "source_service_id": row[2],
            "handler_name": row[3],
            "error": row[4],
            "attempts": row[5],
            "payload": json.loads(row[6]),
            "envelope_json": row[7],
            "failed_at": row[8],
        }

    async def remove(self, envelope_id: str) -> bool:
        """Mark a DLQ entry as replayed (soft delete). Returns True if found."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                UPDATE dead_letter_queue
                SET replayed = 1
                WHERE envelope_id = ? AND replayed = 0
                """,
                (envelope_id,),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def get_envelope(self, envelope_id: str) -> Envelope | None:
        """Retrieve the full Envelope object for replay."""
        entry = await self.get_entry(envelope_id)
        if entry is None:
            return None
        return Envelope.model_validate_json(entry["envelope_json"])

    async def size(self) -> int:
        """Count active (non-replayed) DLQ entries."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM dead_letter_queue WHERE replayed = 0"
            )
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def purge(self) -> int:
        """Remove all active DLQ entries. Returns count purged."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM dead_letter_queue WHERE replayed = 0"
            )
            await db.commit()
            count = cursor.rowcount
        log.info("persistent_dlq.purged count=%d", count)
        return count

    async def stats(self) -> dict[str, Any]:
        """Return DLQ statistics."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE replayed = 0) as active,
                    COUNT(*) FILTER (WHERE replayed = 1) as replayed,
                    COUNT(*) as total
                FROM dead_letter_queue
                """
            )
            row = await cursor.fetchone()

        if row:
            return {
                "active": row[0],
                "replayed": row[1],
                "total": row[2],
                "db_path": self._db_path,
            }
        return {"active": 0, "replayed": 0, "total": 0, "db_path": self._db_path}

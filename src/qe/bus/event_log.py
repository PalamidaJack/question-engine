"""Durable event log: SQLite-backed append-only log with replay support."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import aiosqlite

from qe.models.envelope import Envelope

log = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    envelope_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    source_service_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    logged_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_log_topic ON event_log(topic);
CREATE INDEX IF NOT EXISTS idx_event_log_timestamp ON event_log(timestamp);
"""


class EventLog:
    """Append-only SQLite event log for bus durability."""

    def __init__(self, db_path: str = "data/event_log.db") -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_CREATE_TABLE)
            await db.commit()

    async def append(self, envelope: Envelope) -> None:
        """Append an envelope to the durable log."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO event_log
                    (envelope_id, topic, source_service_id,
                     payload, timestamp, logged_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.envelope_id,
                    envelope.topic,
                    envelope.source_service_id,
                    json.dumps(envelope.payload),
                    envelope.timestamp.isoformat(),
                    datetime.now(UTC).isoformat(),
                ),
            )
            await db.commit()

    async def replay(
        self,
        since: datetime | None = None,
        topic: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Replay events from the log, optionally filtered by time and topic."""
        query = (
            "SELECT envelope_id, topic, source_service_id, "
            "payload, timestamp FROM event_log WHERE 1=1"
        )
        params: list = []

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        if topic:
            query += " AND topic = ?"
            params.append(topic)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        return [
            {
                "envelope_id": row[0],
                "topic": row[1],
                "source_service_id": row[2],
                "payload": json.loads(row[3]),
                "timestamp": row[4],
            }
            for row in rows
        ]

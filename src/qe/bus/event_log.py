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
    schema_version TEXT NOT NULL DEFAULT '1.0',
    topic TEXT NOT NULL,
    source_service_id TEXT NOT NULL,
    correlation_id TEXT,
    causation_id TEXT,
    payload TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    logged_at TEXT NOT NULL,
    ttl_seconds INTEGER
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
            # Backfill columns for existing databases created before
            # lineage + envelope metadata was persisted.
            cursor = await db.execute("PRAGMA table_info(event_log)")
            columns = {row[1] for row in await cursor.fetchall()}
            if "schema_version" not in columns:
                await db.execute(
                    "ALTER TABLE event_log "
                    "ADD COLUMN schema_version TEXT NOT NULL DEFAULT '1.0'"
                )
            if "correlation_id" not in columns:
                await db.execute(
                    "ALTER TABLE event_log ADD COLUMN correlation_id TEXT"
                )
            if "causation_id" not in columns:
                await db.execute(
                    "ALTER TABLE event_log ADD COLUMN causation_id TEXT"
                )
            if "ttl_seconds" not in columns:
                await db.execute(
                    "ALTER TABLE event_log ADD COLUMN ttl_seconds INTEGER"
                )
            await db.commit()

    async def append(self, envelope: Envelope) -> None:
        """Append an envelope to the durable log."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO event_log
                    (envelope_id, schema_version, topic, source_service_id,
                     correlation_id, causation_id, payload, timestamp, logged_at,
                     ttl_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.envelope_id,
                    envelope.schema_version,
                    envelope.topic,
                    envelope.source_service_id,
                    envelope.correlation_id,
                    envelope.causation_id,
                    json.dumps(envelope.payload),
                    envelope.timestamp.isoformat(),
                    datetime.now(UTC).isoformat(),
                    envelope.ttl_seconds,
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
            "SELECT envelope_id, schema_version, topic, source_service_id, "
            "correlation_id, causation_id, payload, timestamp, ttl_seconds "
            "FROM event_log WHERE 1=1"
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
                "schema_version": row[1],
                "topic": row[2],
                "source_service_id": row[3],
                "correlation_id": row[4],
                "causation_id": row[5],
                "payload": json.loads(row[6]),
                "timestamp": row[7],
                "ttl_seconds": row[8],
            }
            for row in rows
        ]

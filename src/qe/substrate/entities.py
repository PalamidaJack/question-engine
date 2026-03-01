"""Entity resolution: normalize names and manage canonical ↔ alias mappings."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime

import aiosqlite

log = logging.getLogger(__name__)

_SUFFIX_RE = re.compile(
    r"\s*\b(inc|corp|ltd|llc|co|plc|gmbh|sa|ag|pty|nv)\b\.?\s*$",
    re.IGNORECASE,
)


def normalize_entity(name: str) -> str:
    """Deterministic normalization: lowercase, strip suffixes, collapse whitespace."""
    name = name.strip().lower()
    name = _SUFFIX_RE.sub("", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


class EntityResolver:
    """SQLite-backed canonical name + alias lookup."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._alias_index_ready = False

    async def _ensure_alias_index(self, db: aiosqlite.Connection) -> None:
        """Create the entity_aliases index table if it doesn't exist."""
        if self._alias_index_ready:
            return
        await db.execute("""
            CREATE TABLE IF NOT EXISTS entity_aliases (
                alias TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL
            )
        """)
        await db.commit()
        self._alias_index_ready = True

    async def resolve(self, raw_name: str) -> str:
        """Return the canonical name for *raw_name*, or its normalized form."""
        normalized = normalize_entity(raw_name)

        async with aiosqlite.connect(self._db_path) as db:
            # Direct canonical match
            cursor = await db.execute(
                "SELECT canonical_name FROM entities WHERE canonical_name = ?",
                (normalized,),
            )
            row = await cursor.fetchone()
            if row:
                return row[0]

            # Fast indexed alias lookup
            await self._ensure_alias_index(db)
            cursor = await db.execute(
                "SELECT canonical_name FROM entity_aliases WHERE alias = ?",
                (normalized,),
            )
            row = await cursor.fetchone()
            if row:
                return row[0]

            # Fallback: scan JSON aliases (for entries not yet in index)
            cursor = await db.execute(
                "SELECT canonical_name, aliases FROM entities"
            )
            rows = await cursor.fetchall()
            for canonical, aliases_json in rows:
                aliases: list[str] = json.loads(aliases_json)
                if normalized in aliases:
                    # Back-fill the index for future lookups
                    try:
                        await db.execute(
                            "INSERT OR IGNORE INTO entity_aliases"
                            " (alias, canonical_name) VALUES (?, ?)",
                            (normalized, canonical),
                        )
                        await db.commit()
                    except Exception:
                        pass
                    return canonical

        return normalized

    async def add_alias(self, canonical_name: str, alias: str) -> None:
        """Add an alias for an existing or new entity."""
        normalized_alias = normalize_entity(alias)
        normalized_canonical = normalize_entity(canonical_name)
        now = datetime.now(UTC).isoformat()

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT aliases FROM entities WHERE canonical_name = ?",
                (normalized_canonical,),
            )
            row = await cursor.fetchone()

            if row:
                aliases: list[str] = json.loads(row[0])
                if normalized_alias not in aliases:
                    aliases.append(normalized_alias)
                    await db.execute(
                        "UPDATE entities SET aliases = ?, updated_at = ? WHERE canonical_name = ?",
                        (json.dumps(aliases), now, normalized_canonical),
                    )
                # Keep alias index in sync
                await self._ensure_alias_index(db)
                await db.execute(
                    "INSERT OR IGNORE INTO entity_aliases"
                    " (alias, canonical_name) VALUES (?, ?)",
                    (normalized_alias, normalized_canonical),
                )
            else:
                await db.execute(
                    "INSERT INTO entities "
                    "(canonical_name, aliases, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (normalized_canonical, json.dumps([normalized_alias]), now, now),
                )
            await db.commit()

    async def ensure_entity(self, raw_name: str) -> str:
        """Resolve and ensure the entity exists in the table. Returns canonical name."""
        canonical = await self.resolve(raw_name)
        now = datetime.now(UTC).isoformat()

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO entities (canonical_name, aliases, created_at, updated_at)
                VALUES (?, '[]', ?, ?)
                ON CONFLICT(canonical_name) DO NOTHING
                """,
                (canonical, now, now),
            )
            await db.commit()

        return canonical

    async def list_entities(self, db_path: str | None = None) -> list[dict]:
        """List all entities with their aliases."""
        path = db_path or self._db_path
        async with aiosqlite.connect(path) as db:
            cursor = await db.execute(
                "SELECT canonical_name, aliases, created_at, updated_at FROM entities"
            )
            rows = await cursor.fetchall()

        return [
            {
                "canonical_name": row[0],
                "aliases": json.loads(row[1]),
                "created_at": row[2],
                "updated_at": row[3],
            }
            for row in rows
        ]

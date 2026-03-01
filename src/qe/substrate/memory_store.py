"""Three-layer persistent memory system."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

import aiosqlite
from pydantic import BaseModel

log = logging.getLogger(__name__)


class MemoryEntry(BaseModel):
    """A single memory entry."""

    memory_id: str = ""
    category: Literal[
        "preference", "context", "project", "entity", "pattern"
    ] = "context"
    key: str = ""
    value: str = ""
    confidence: float = 1.0
    source: Literal["user_explicit", "inferred", "system"] = "system"
    created_at: str = ""
    updated_at: str = ""
    expires_at: str | None = None
    superseded_by: str | None = None


class MemoryStore:
    """Three-layer persistent memory system backed by SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._initialized = False

    async def _ensure_tables(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    memory_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source TEXT NOT NULL,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    superseded_by TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_category
                ON memory_entries(category)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_key
                ON memory_entries(key)
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP
                )
            """)
            await db.commit()
        self._initialized = True

    async def set_memory(
        self,
        category: str,
        key: str,
        value: str,
        *,
        confidence: float = 1.0,
        source: str = "user_explicit",
    ) -> MemoryEntry:
        """Set or update a memory entry."""
        await self._ensure_tables()
        now = datetime.now(UTC).isoformat()
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"

        async with aiosqlite.connect(self._db_path) as db:
            # Acquire write lock before read-modify-write to prevent race conditions
            await db.execute("BEGIN IMMEDIATE")

            # Check for existing entry with same category+key
            cursor = await db.execute(
                "SELECT memory_id FROM memory_entries "
                "WHERE category = ? AND key = ? "
                "AND superseded_by IS NULL",
                (category, key),
            )
            existing = await cursor.fetchone()
            if existing:
                # Supersede old entry
                await db.execute(
                    "UPDATE memory_entries "
                    "SET superseded_by = ? "
                    "WHERE memory_id = ?",
                    (memory_id, existing[0]),
                )

            await db.execute(
                "INSERT INTO memory_entries"
                " (memory_id, category, key, value,"
                " confidence, source, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory_id, category, key, value,
                    confidence, source, now, now,
                ),
            )
            await db.commit()

        return MemoryEntry(
            memory_id=memory_id,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            source=source,
            created_at=now,
            updated_at=now,
        )

    async def set_preference(
        self, key: str, value: str
    ) -> MemoryEntry:
        return await self.set_memory(
            "preference", key, value, source="user_explicit"
        )

    async def get_preferences(self) -> list[MemoryEntry]:
        return await self.get_by_category("preference")

    async def set_project_context(
        self, project_id: str, key: str, value: str
    ) -> MemoryEntry:
        return await self.set_memory(
            "project",
            f"{project_id}:{key}",
            value,
            source="system",
        )

    async def get_project_context(
        self, project_id: str
    ) -> list[MemoryEntry]:
        await self._ensure_tables()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM memory_entries "
                "WHERE category = 'project' AND key LIKE ? "
                "AND superseded_by IS NULL",
                (f"{project_id}:%",),
            )
            rows = await cursor.fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def set_entity_memory(
        self,
        entity_id: str,
        key: str,
        value: str,
        confidence: float = 0.8,
    ) -> MemoryEntry:
        return await self.set_memory(
            "entity",
            f"{entity_id}:{key}",
            value,
            confidence=confidence,
            source="inferred",
        )

    async def get_entity_memories(
        self, entity_id: str
    ) -> list[MemoryEntry]:
        await self._ensure_tables()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM memory_entries "
                "WHERE category = 'entity' AND key LIKE ? "
                "AND superseded_by IS NULL",
                (f"{entity_id}:%",),
            )
            rows = await cursor.fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def get_by_category(
        self, category: str
    ) -> list[MemoryEntry]:
        await self._ensure_tables()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM memory_entries "
                "WHERE category = ? "
                "AND superseded_by IS NULL "
                "ORDER BY updated_at DESC",
                (category,),
            )
            rows = await cursor.fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def get_all_active(self) -> list[MemoryEntry]:
        await self._ensure_tables()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM memory_entries "
                "WHERE superseded_by IS NULL "
                "ORDER BY updated_at DESC"
            )
            rows = await cursor.fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def delete(self, memory_id: str) -> bool:
        await self._ensure_tables()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "DELETE FROM memory_entries "
                "WHERE memory_id = ?",
                (memory_id,),
            )
            await db.commit()
        return cursor.rowcount > 0

    async def count(self, active_only: bool = True) -> int:
        await self._ensure_tables()
        async with aiosqlite.connect(self._db_path) as db:
            if active_only:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM memory_entries "
                    "WHERE superseded_by IS NULL"
                )
            else:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM memory_entries"
                )
            row = await cursor.fetchone()
        return row[0]

    # -- Projects ------------------------------------------------

    async def create_project(
        self, name: str, description: str = ""
    ) -> dict[str, Any]:
        await self._ensure_tables()
        project_id = f"proj_{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO projects "
                "(project_id, name, description, "
                "active, created_at, last_accessed) "
                "VALUES (?, ?, ?, 1, ?, ?)",
                (project_id, name, description, now, now),
            )
            await db.commit()
        return {
            "project_id": project_id,
            "name": name,
            "description": description,
            "active": True,
        }

    async def list_projects(
        self, active_only: bool = True
    ) -> list[dict[str, Any]]:
        await self._ensure_tables()
        async with aiosqlite.connect(self._db_path) as db:
            if active_only:
                cursor = await db.execute(
                    "SELECT project_id, name, "
                    "description, active "
                    "FROM projects WHERE active = 1"
                )
            else:
                cursor = await db.execute(
                    "SELECT project_id, name, "
                    "description, active "
                    "FROM projects"
                )
            rows = await cursor.fetchall()
        return [
            {
                "project_id": r[0],
                "name": r[1],
                "description": r[2],
                "active": bool(r[3]),
            }
            for r in rows
        ]

    def _row_to_entry(self, row: tuple) -> MemoryEntry:
        return MemoryEntry(
            memory_id=row[0],
            category=row[1],
            key=row[2],
            value=row[3],
            confidence=row[4] or 1.0,
            source=row[5] or "system",
            created_at=row[6] or "",
            updated_at=row[7] or "",
            expires_at=row[8],
            superseded_by=row[9],
        )

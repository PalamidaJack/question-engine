"""Persistent conversation storage backed by SQLite."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

log = logging.getLogger(__name__)


class ChatStore:
    """SQLite-backed conversation and message persistence.

    Stores conversations with messages, supports search,
    model tracking, and multi-model comparison responses.
    """

    def __init__(
        self, db_path: str | Path = "data/chat_history.db",
    ) -> None:
        self._db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        Path(self._db_path).parent.mkdir(
            parents=True, exist_ok=True,
        )
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL,
                updated_at REAL,
                primary_model TEXT,
                is_multi_model INTEGER DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model_id TEXT,
                timestamp REAL,
                latency_ms INTEGER,
                token_count INTEGER,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
            );
            CREATE TABLE IF NOT EXISTS multi_responses (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                prompt_index INTEGER,
                model_id TEXT NOT NULL,
                content TEXT NOT NULL,
                latency_ms INTEGER,
                token_count INTEGER,
                was_preferred INTEGER DEFAULT 0,
                timestamp REAL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_ts
                ON messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_multi_conv
                ON multi_responses(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_conv_updated
                ON conversations(updated_at);
        """)
        await self._db.commit()
        log.info(
            "chat_store.initialized db=%s", self._db_path,
        )

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # -- Conversation CRUD ---------------------------------------

    async def create_conversation(
        self,
        title: str | None = None,
        model: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        conv_id = str(uuid.uuid4())
        now = time.time()
        await self._db.execute(
            "INSERT INTO conversations "
            "(id, title, created_at, updated_at, "
            "primary_model, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                conv_id,
                title or "New Conversation",
                now,
                now,
                model,
                json.dumps(metadata or {}),
            ),
        )
        await self._db.commit()
        return conv_id

    async def get_conversation(
        self, conv_id: str,
    ) -> dict | None:
        cursor = await self._db.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conv_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._conv_row_to_dict(row, cursor.description)

    async def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        model: str | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM conversations"
        params: list = []
        if model:
            query += " WHERE primary_model = ?"
            params.append(model)
        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            self._conv_row_to_dict(r, cursor.description)
            for r in rows
        ]

    async def update_conversation(
        self, conv_id: str, **kwargs: Any,
    ) -> bool:
        allowed = {
            "title",
            "primary_model",
            "metadata",
            "is_multi_model",
        }
        updates = {
            k: v for k, v in kwargs.items() if k in allowed
        }
        if not updates:
            return False
        if "metadata" in updates and isinstance(
            updates["metadata"], dict,
        ):
            updates["metadata"] = json.dumps(
                updates["metadata"],
            )
        set_clause = ", ".join(
            f"{k} = ?" for k in updates
        )
        values = list(updates.values()) + [
            time.time(),
            conv_id,
        ]
        await self._db.execute(
            f"UPDATE conversations SET {set_clause}, "
            "updated_at = ? WHERE id = ?",
            values,
        )
        await self._db.commit()
        return True

    async def delete_conversation(
        self, conv_id: str,
    ) -> bool:
        await self._db.execute(
            "DELETE FROM messages "
            "WHERE conversation_id = ?",
            (conv_id,),
        )
        await self._db.execute(
            "DELETE FROM multi_responses "
            "WHERE conversation_id = ?",
            (conv_id,),
        )
        cursor = await self._db.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conv_id,),
        )
        await self._db.commit()
        return cursor.rowcount > 0

    # -- Messages ------------------------------------------------

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model_id: str | None = None,
        latency_ms: int | None = None,
        token_count: int | None = None,
        metadata: dict | None = None,
    ) -> str:
        msg_id = str(uuid.uuid4())
        now = time.time()
        await self._db.execute(
            "INSERT INTO messages "
            "(id, conversation_id, role, content, "
            "model_id, timestamp, latency_ms, "
            "token_count, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                msg_id,
                conversation_id,
                role,
                content,
                model_id,
                now,
                latency_ms,
                token_count,
                json.dumps(metadata or {}),
            ),
        )
        await self._db.execute(
            "UPDATE conversations SET updated_at = ?, "
            "message_count = message_count + 1 "
            "WHERE id = ?",
            (now, conversation_id),
        )
        # Auto-title from first user message
        if role == "user":
            cursor = await self._db.execute(
                "SELECT message_count "
                "FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            row = await cursor.fetchone()
            if row and row[0] <= 1:
                title = content[:80] + (
                    "..." if len(content) > 80 else ""
                )
                await self._db.execute(
                    "UPDATE conversations "
                    "SET title = ? WHERE id = ?",
                    (title, conversation_id),
                )
        await self._db.commit()
        return msg_id

    async def get_messages(
        self,
        conversation_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM messages "
            "WHERE conversation_id = ? "
            "ORDER BY timestamp ASC LIMIT ? OFFSET ?",
            (conversation_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [
            self._msg_row_to_dict(r, cursor.description)
            for r in rows
        ]

    # -- Multi-model responses -----------------------------------

    async def add_multi_response(
        self,
        conversation_id: str,
        prompt_index: int,
        model_id: str,
        content: str,
        latency_ms: int | None = None,
        token_count: int | None = None,
    ) -> str:
        resp_id = str(uuid.uuid4())
        now = time.time()
        await self._db.execute(
            "INSERT INTO multi_responses "
            "(id, conversation_id, prompt_index, "
            "model_id, content, latency_ms, "
            "token_count, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                resp_id,
                conversation_id,
                prompt_index,
                model_id,
                content,
                latency_ms,
                token_count,
                now,
            ),
        )
        await self._db.execute(
            "UPDATE conversations "
            "SET is_multi_model = 1, updated_at = ? "
            "WHERE id = ?",
            (now, conversation_id),
        )
        await self._db.commit()
        return resp_id

    async def set_preferred(
        self, response_id: str,
    ) -> bool:
        # Get conv_id and prompt_index to clear others
        cursor = await self._db.execute(
            "SELECT conversation_id, prompt_index "
            "FROM multi_responses WHERE id = ?",
            (response_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return False
        conv_id, prompt_idx = row
        # Clear existing preference for this prompt
        await self._db.execute(
            "UPDATE multi_responses "
            "SET was_preferred = 0 "
            "WHERE conversation_id = ? "
            "AND prompt_index = ?",
            (conv_id, prompt_idx),
        )
        # Set new preference
        await self._db.execute(
            "UPDATE multi_responses "
            "SET was_preferred = 1 WHERE id = ?",
            (response_id,),
        )
        await self._db.commit()
        return True

    async def get_multi_responses(
        self,
        conversation_id: str,
        prompt_index: int | None = None,
    ) -> list[dict]:
        query = (
            "SELECT * FROM multi_responses "
            "WHERE conversation_id = ?"
        )
        params: list = [conversation_id]
        if prompt_index is not None:
            query += " AND prompt_index = ?"
            params.append(prompt_index)
        query += " ORDER BY timestamp ASC"
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [
            self._multi_row_to_dict(r, cursor.description)
            for r in rows
        ]

    # -- Search --------------------------------------------------

    async def search(
        self, query: str, limit: int = 20,
    ) -> list[dict]:
        """Full-text search across messages.

        Returns matching conversations.
        """
        cursor = await self._db.execute(
            "SELECT DISTINCT c.* FROM conversations c "
            "JOIN messages m "
            "ON c.id = m.conversation_id "
            "WHERE m.content LIKE ? "
            "ORDER BY c.updated_at DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        return [
            self._conv_row_to_dict(r, cursor.description)
            for r in rows
        ]

    # -- Stats ---------------------------------------------------

    async def stats(self) -> dict:
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM conversations",
        )
        conv_count = (await cursor.fetchone())[0]
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM messages",
        )
        msg_count = (await cursor.fetchone())[0]
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM multi_responses "
            "WHERE was_preferred = 1",
        )
        pref_count = (await cursor.fetchone())[0]
        return {
            "conversations": conv_count,
            "messages": msg_count,
            "preferences_recorded": pref_count,
        }

    # -- Helpers -------------------------------------------------

    @staticmethod
    def _conv_row_to_dict(row, description) -> dict:
        d = {
            desc[0]: val
            for desc, val in zip(
                description, row, strict=False,
            )
        }
        if "metadata" in d and isinstance(
            d["metadata"], str,
        ):
            d["metadata"] = json.loads(d["metadata"])
        d["is_multi_model"] = bool(
            d.get("is_multi_model"),
        )
        return d

    @staticmethod
    def _msg_row_to_dict(row, description) -> dict:
        d = {
            desc[0]: val
            for desc, val in zip(
                description, row, strict=False,
            )
        }
        if "metadata" in d and isinstance(
            d["metadata"], str,
        ):
            d["metadata"] = json.loads(d["metadata"])
        return d

    @staticmethod
    def _multi_row_to_dict(row, description) -> dict:
        d = {
            desc[0]: val
            for desc, val in zip(
                description, row, strict=False,
            )
        }
        d["was_preferred"] = bool(
            d.get("was_preferred"),
        )
        return d

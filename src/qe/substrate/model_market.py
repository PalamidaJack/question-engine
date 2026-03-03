import logging
from datetime import datetime
from typing import Any

import aiosqlite

log = logging.getLogger(__name__)


class ModelMarketStore:
    def __init__(self, db_path: str = "data/belief_ledger.db"):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS free_model_inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                context_length INTEGER DEFAULT 32000,
                capabilities TEXT,
                is_available INTEGER DEFAULT 1,
                rate_limit_rpm INTEGER DEFAULT 20,
                rate_limit_rpd INTEGER DEFAULT 200,
                last_checked TEXT,
                last_error TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_latency_ms REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(provider, model_id)
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS provider_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL UNIQUE,
                api_base TEXT,
                requires_api_key INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1,
                rate_limit_default_rpm INTEGER DEFAULT 20,
                rate_limit_default_rpd INTEGER DEFAULT 200,
                notes TEXT,
                last_scraped TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS model_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                error_code TEXT,
                error_message TEXT,
                error_type TEXT,
                resolved INTEGER DEFAULT 0,
                resolution_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                resolved_at TEXT
            )
        """)

        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def add_or_update_model(
        self,
        provider: str,
        model_id: str,
        model_name: str,
        context_length: int = 32000,
        capabilities: str = "",
        rate_limit_rpm: int = 20,
        rate_limit_rpd: int = 200,
    ) -> None:
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT INTO free_model_inventory
            (provider, model_id, model_name, context_length,
             capabilities, rate_limit_rpm, rate_limit_rpd,
             last_checked, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, model_id) DO UPDATE SET
                model_name = excluded.model_name,
                context_length = excluded.context_length,
                capabilities = excluded.capabilities,
                rate_limit_rpm = excluded.rate_limit_rpm,
                rate_limit_rpd = excluded.rate_limit_rpd,
                last_checked = excluded.last_checked,
                updated_at = excluded.updated_at,
                is_available = 1
            """,
            (provider, model_id, model_name, context_length,
             capabilities, rate_limit_rpm, rate_limit_rpd, now, now),
        )
        await self._db.commit()

    async def get_available_models(self) -> list[dict[str, Any]]:
        cursor = await self._db.execute(
            """
            SELECT provider, model_id, model_name, context_length, capabilities,
                   rate_limit_rpm, rate_limit_rpd, avg_latency_ms
            FROM free_model_inventory
            WHERE is_available = 1
            ORDER BY provider, model_name
            """
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_model_by_id(self, provider: str, model_id: str) -> dict[str, Any] | None:
        cursor = await self._db.execute(
            """
            SELECT * FROM free_model_inventory
            WHERE provider = ? AND model_id = ?
            """,
            (provider, model_id),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def mark_model_unavailable(self, provider: str, model_id: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            UPDATE free_model_inventory
            SET is_available = 0, last_error = ?, last_checked = ?, updated_at = ?
            WHERE provider = ? AND model_id = ?
            """,
            (error, now, now, provider, model_id),
        )
        await self._db.commit()

    async def record_success(self, provider: str, model_id: str, latency_ms: float) -> None:
        await self._db.execute(
            """
            UPDATE free_model_inventory
            SET success_count = success_count + 1,
                avg_latency_ms = (avg_latency_ms * success_count + ?) / (success_count + 1),
                last_checked = ?
            WHERE provider = ? AND model_id = ?
            """,
            (latency_ms, datetime.utcnow().isoformat(), provider, model_id),
        )
        await self._db.commit()

    async def record_failure(self, provider: str, model_id: str, error: str) -> None:
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            UPDATE free_model_inventory
            SET failure_count = failure_count + 1,
                last_error = ?,
                last_checked = ?
            WHERE provider = ? AND model_id = ?
            """,
            (error, now, provider, model_id),
        )
        await self._db.commit()

    async def add_error_record(
        self,
        provider: str,
        model_id: str,
        error_code: str,
        error_message: str,
        error_type: str,
    ) -> None:
        await self._db.execute(
            """
            INSERT INTO model_errors (provider, model_id, error_code, error_message, error_type)
            VALUES (?, ?, ?, ?, ?)
            """,
            (provider, model_id, error_code, error_message, error_type),
        )
        await self._db.commit()

    async def get_unresolved_errors(self) -> list[dict[str, Any]]:
        cursor = await self._db.execute(
            """
            SELECT * FROM model_errors
            WHERE resolved = 0
            ORDER BY created_at DESC
            LIMIT 100
            """
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def resolve_error(self, error_id: int, resolution_notes: str) -> None:
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            UPDATE model_errors
            SET resolved = 1, resolved_at = ?, resolution_notes = ?
            WHERE id = ?
            """,
            (now, resolution_notes, error_id),
        )
        await self._db.commit()

    async def add_or_update_provider(
        self,
        provider: str,
        api_base: str,
        requires_api_key: bool = True,
        rate_limit_default_rpm: int = 20,
        rate_limit_default_rpd: int = 200,
        notes: str = "",
    ) -> None:
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT INTO provider_info
            (provider, api_base, requires_api_key,
             rate_limit_default_rpm, rate_limit_default_rpd,
             notes, last_scraped)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                api_base = excluded.api_base,
                requires_api_key = excluded.requires_api_key,
                rate_limit_default_rpm = excluded.rate_limit_default_rpm,
                rate_limit_default_rpd = excluded.rate_limit_default_rpd,
                notes = excluded.notes,
                last_scraped = excluded.last_scraped,
                is_active = 1
            """,
            (
                provider, api_base, 1 if requires_api_key else 0,
                rate_limit_default_rpm, rate_limit_default_rpd,
                notes, now,
            ),
        )
        await self._db.commit()

    async def get_active_providers(self) -> list[dict[str, Any]]:
        cursor = await self._db.execute(
            """
            SELECT * FROM provider_info WHERE is_active = 1
            """
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_stats(self) -> dict[str, Any]:
        cursor = await self._db.execute(
            "SELECT COUNT(*) as total, SUM(is_available) as available FROM free_model_inventory"
        )
        row = await cursor.fetchone()

        cursor2 = await self._db.execute(
            "SELECT COUNT(*) as unresolved FROM model_errors WHERE resolved = 0"
        )
        row2 = await cursor2.fetchone()

        return {
            "total_models": row["total"] if row else 0,
            "available_models": row["available"] if row else 0,
            "unresolved_errors": row2["unresolved"] if row2 else 0,
        }

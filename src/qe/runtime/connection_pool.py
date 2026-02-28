"""SQLite connection pooling for aiosqlite.

Replaces per-query aiosqlite.connect() with persistent pooled connections.
Reduces connection setup overhead and thread creation under load.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

log = logging.getLogger(__name__)


class ConnectionPool:
    """Async connection pool for a single SQLite database file.

    Maintains a pool of reusable aiosqlite connections. Connections
    are initialized with WAL mode and returned to the pool after use.

    Usage:
        pool = ConnectionPool("data/qe.db", max_size=5)
        await pool.initialize()

        async with pool.acquire() as db:
            await db.execute("SELECT ...")

        await pool.close()
    """

    def __init__(
        self,
        db_path: str,
        max_size: int = 5,
        wal_mode: bool = True,
    ) -> None:
        self._db_path = db_path
        self._max_size = max_size
        self._wal_mode = wal_mode
        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(
            maxsize=max_size
        )
        self._size = 0
        self._closed = False
        self._lock = asyncio.Lock()
        self._total_acquired = 0
        self._total_created = 0

    async def initialize(self) -> None:
        """Pre-create one connection to validate the database path."""
        conn = await self._create_connection()
        await self._pool.put(conn)

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new aiosqlite connection with optimal settings."""
        conn = await aiosqlite.connect(self._db_path)
        if self._wal_mode:
            await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        self._size += 1
        self._total_created += 1
        log.debug(
            "pool.connection_created db=%s pool_size=%d",
            self._db_path,
            self._size,
        )
        return conn

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[aiosqlite.Connection]:
        """Acquire a connection from the pool.

        Creates a new connection if the pool is empty and under max_size.
        Blocks if at capacity until a connection is returned.
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        conn: aiosqlite.Connection | None = None

        # Try to get from pool without blocking
        try:
            conn = self._pool.get_nowait()
        except asyncio.QueueEmpty:
            # Pool empty — create new if under limit
            async with self._lock:
                if self._size < self._max_size:
                    conn = await self._create_connection()

        if conn is None:
            # At capacity — block until one is returned
            conn = await self._pool.get()

        self._total_acquired += 1

        try:
            yield conn
        except Exception:
            # On error, try to return the connection; if broken, discard
            try:
                await self._pool.put(conn)
            except Exception:
                self._size -= 1
            raise
        else:
            # Return healthy connection to pool
            try:
                self._pool.put_nowait(conn)
            except asyncio.QueueFull:
                # Pool is full (shouldn't happen), close excess
                await conn.close()
                self._size -= 1

    async def close(self) -> None:
        """Close all connections in the pool."""
        self._closed = True
        closed = 0
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await conn.close()
                closed += 1
            except asyncio.QueueEmpty:
                break
        self._size = 0
        log.debug(
            "pool.closed db=%s connections_closed=%d",
            self._db_path,
            closed,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "db_path": self._db_path,
            "max_size": self._max_size,
            "current_size": self._size,
            "available": self._pool.qsize(),
            "in_use": self._size - self._pool.qsize(),
            "total_acquired": self._total_acquired,
            "total_created": self._total_created,
            "closed": self._closed,
        }


class PoolManager:
    """Manages connection pools for multiple database files.

    Provides a single pool per unique db_path. Thread-safe
    initialization via asyncio.Lock.
    """

    def __init__(self, default_max_size: int = 5) -> None:
        self._pools: dict[str, ConnectionPool] = {}
        self._default_max_size = default_max_size
        self._lock = asyncio.Lock()

    async def get_pool(
        self,
        db_path: str,
        max_size: int | None = None,
    ) -> ConnectionPool:
        """Get or create a connection pool for the given database path."""
        if db_path in self._pools:
            return self._pools[db_path]

        async with self._lock:
            # Double-check after acquiring lock
            if db_path in self._pools:
                return self._pools[db_path]

            pool = ConnectionPool(
                db_path,
                max_size=max_size or self._default_max_size,
            )
            await pool.initialize()
            self._pools[db_path] = pool
            log.debug("pool_manager.created db=%s", db_path)
            return pool

    @asynccontextmanager
    async def acquire(self, db_path: str) -> AsyncIterator[aiosqlite.Connection]:
        """Shortcut: get pool for db_path and acquire a connection."""
        pool = await self.get_pool(db_path)
        async with pool.acquire() as conn:
            yield conn

    async def close_all(self) -> None:
        """Close all managed pools."""
        for pool in self._pools.values():
            await pool.close()
        self._pools.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "pools": {
                path: pool.stats() for path, pool in self._pools.items()
            },
            "total_pools": len(self._pools),
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_pool_manager: PoolManager | None = None


def get_pool_manager() -> PoolManager:
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = PoolManager()
    return _pool_manager


def reset_pool_manager() -> None:
    """Reset pool manager (for testing)."""
    global _pool_manager
    _pool_manager = None

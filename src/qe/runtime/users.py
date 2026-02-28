"""Multi-user support: user context management and persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import aiosqlite

log = logging.getLogger(__name__)


@dataclass
class UserContext:
    """Represents a user's context and preferences."""

    user_id: str
    display_name: str = ""
    created_at: str = ""
    preferences: dict = field(default_factory=dict)
    active_project_id: str | None = None


class UserManager:
    """Manages user accounts, preferences, and active project context.

    Supports both persistent (SQLite via aiosqlite) and in-memory modes.
    When no db_path is provided, all data is kept in an in-memory dict.
    """

    def __init__(self, db_path: str = "") -> None:
        self._db_path = db_path
        self._initialized = False
        # In-memory store, used when no db_path is given
        self._users: dict[str, UserContext] = {}

    @property
    def _use_db(self) -> bool:
        return bool(self._db_path)

    async def _ensure_table(self) -> None:
        """Create the users table if it does not exist."""
        if self._initialized or not self._use_db:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP NOT NULL,
                    preferences TEXT NOT NULL DEFAULT '{}',
                    active_project_id TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_project
                ON users(active_project_id)
            """)
            await db.commit()
        self._initialized = True

    async def create_user(
        self,
        user_id: str,
        display_name: str = "",
    ) -> UserContext:
        """Create a new user.

        Args:
            user_id: unique identifier for the user.
            display_name: optional human-readable name.

        Returns:
            The created UserContext.

        Raises:
            ValueError: if user_id already exists.
        """
        now = datetime.now(UTC).isoformat()

        if self._use_db:
            await self._ensure_table()
            async with aiosqlite.connect(self._db_path) as db:
                # Check for existing user
                cursor = await db.execute(
                    "SELECT user_id FROM users WHERE user_id = ?",
                    (user_id,),
                )
                if await cursor.fetchone():
                    raise ValueError(f"User {user_id} already exists")

                await db.execute(
                    "INSERT INTO users "
                    "(user_id, display_name, created_at, preferences, active_project_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (user_id, display_name, now, "{}", None),
                )
                await db.commit()
        else:
            if user_id in self._users:
                raise ValueError(f"User {user_id} already exists")

        ctx = UserContext(
            user_id=user_id,
            display_name=display_name,
            created_at=now,
            preferences={},
            active_project_id=None,
        )
        self._users[user_id] = ctx
        log.info("Created user %s (%s)", user_id, display_name)
        return ctx

    async def get_user(self, user_id: str) -> UserContext | None:
        """Retrieve a user by ID. Returns None if not found."""
        if self._use_db:
            await self._ensure_table()
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    "SELECT user_id, display_name, created_at, "
                    "preferences, active_project_id "
                    "FROM users WHERE user_id = ?",
                    (user_id,),
                )
                row = await cursor.fetchone()
            if row is None:
                return None
            return self._row_to_context(row)

        return self._users.get(user_id)

    async def list_users(self) -> list[UserContext]:
        """Return all registered users."""
        if self._use_db:
            await self._ensure_table()
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    "SELECT user_id, display_name, created_at, "
                    "preferences, active_project_id "
                    "FROM users ORDER BY created_at ASC"
                )
                rows = await cursor.fetchall()
            return [self._row_to_context(r) for r in rows]

        return list(self._users.values())

    async def update_preferences(self, user_id: str, prefs: dict) -> None:
        """Merge new preferences into the user's existing preferences.

        Args:
            user_id: the user to update.
            prefs: dict of preference key-value pairs to merge.

        Raises:
            ValueError: if user not found.
        """
        if self._use_db:
            await self._ensure_table()
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    "SELECT preferences FROM users WHERE user_id = ?",
                    (user_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    raise ValueError(f"User {user_id} not found")

                existing: dict = json.loads(row[0])
                existing.update(prefs)

                await db.execute(
                    "UPDATE users SET preferences = ? WHERE user_id = ?",
                    (json.dumps(existing), user_id),
                )
                await db.commit()

            # Update in-memory cache
            if user_id in self._users:
                self._users[user_id].preferences.update(prefs)
        else:
            user = self._users.get(user_id)
            if user is None:
                raise ValueError(f"User {user_id} not found")
            user.preferences.update(prefs)

        log.info("Updated preferences for user %s: %s", user_id, list(prefs.keys()))

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user. Returns True if the user existed.

        Args:
            user_id: the user to delete.

        Returns:
            True if the user was found and deleted, False otherwise.
        """
        if self._use_db:
            await self._ensure_table()
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM users WHERE user_id = ?",
                    (user_id,),
                )
                await db.commit()
                deleted = cursor.rowcount > 0

            self._users.pop(user_id, None)
            if deleted:
                log.info("Deleted user %s", user_id)
            return deleted

        if user_id in self._users:
            del self._users[user_id]
            log.info("Deleted user %s", user_id)
            return True
        return False

    async def set_active_project(self, user_id: str, project_id: str) -> None:
        """Set the active project for a user.

        Args:
            user_id: the user to update.
            project_id: the project to set as active.

        Raises:
            ValueError: if user not found.
        """
        if self._use_db:
            await self._ensure_table()
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    "SELECT user_id FROM users WHERE user_id = ?",
                    (user_id,),
                )
                if not await cursor.fetchone():
                    raise ValueError(f"User {user_id} not found")

                await db.execute(
                    "UPDATE users SET active_project_id = ? WHERE user_id = ?",
                    (project_id, user_id),
                )
                await db.commit()

            if user_id in self._users:
                self._users[user_id].active_project_id = project_id
        else:
            user = self._users.get(user_id)
            if user is None:
                raise ValueError(f"User {user_id} not found")
            user.active_project_id = project_id

        log.info("Set active project for user %s to %s", user_id, project_id)

    async def get_user_context(self, user_id: str) -> UserContext:
        """Get full user context for prompt enrichment.

        Unlike get_user(), this raises if the user is not found, ensuring
        callers always get a valid context.

        Args:
            user_id: the user to look up.

        Returns:
            Full UserContext with preferences and active project.

        Raises:
            ValueError: if user not found.
        """
        ctx = await self.get_user(user_id)
        if ctx is None:
            raise ValueError(f"User {user_id} not found")
        return ctx

    def _row_to_context(self, row: tuple) -> UserContext:
        """Convert a database row to a UserContext."""
        ctx = UserContext(
            user_id=row[0],
            display_name=row[1] or "",
            created_at=row[2] or "",
            preferences=json.loads(row[3]) if row[3] else {},
            active_project_id=row[4],
        )
        # Update in-memory cache
        self._users[ctx.user_id] = ctx
        return ctx

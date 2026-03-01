"""Episodic Memory — Tier 1 short-term recall with LRU eviction.

In-memory hot store (OrderedDict) + SQLite warm overflow.
Stores recent tool calls, LLM responses, observations, and synthesis results.
Supports semantic + recency-weighted search for recall.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any, Literal

import aiosqlite
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


EPISODE_TYPES = Literal[
    "tool_call",
    "llm_response",
    "user_input",
    "observation",
    "claim_committed",
    "question_generated",
    "synthesis",
]


class Episode(BaseModel):
    """A single episodic memory entry."""

    episode_id: str = Field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:12]}")
    inquiry_id: str = ""
    goal_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    episode_type: EPISODE_TYPES
    content: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""  # short text summary for search
    token_count: int = 0
    relevance_to_goal: float = 0.5  # precomputed on storage
    accessed_count: int = 0


class EpisodicMemory:
    """Tier 1 episodic memory: in-memory LRU with SQLite overflow.

    Hot store (in-memory): most recent entries, fast access.
    Warm store (SQLite): older entries, searched on demand.
    """

    def __init__(
        self,
        db_path: str | None = None,
        max_hot_entries: int = 500,
        max_warm_entries: int = 5000,
    ) -> None:
        self._hot: OrderedDict[str, Episode] = OrderedDict()
        self._max_hot = max_hot_entries
        self._max_warm = max_warm_entries
        self._db_path = db_path
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create the warm store table if using SQLite."""
        if self._db_path is None:
            self._initialized = True
            return

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    episode_id TEXT PRIMARY KEY,
                    inquiry_id TEXT NOT NULL DEFAULT '',
                    goal_id TEXT NOT NULL DEFAULT '',
                    timestamp TEXT NOT NULL,
                    episode_type TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '{}',
                    summary TEXT NOT NULL DEFAULT '',
                    token_count INTEGER NOT NULL DEFAULT 0,
                    relevance_to_goal REAL NOT NULL DEFAULT 0.5,
                    accessed_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_goal ON episodic_memory(goal_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_type ON episodic_memory(episode_type)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ep_timestamp ON episodic_memory(timestamp)"
            )
            await db.commit()
        self._initialized = True

    async def store(self, episode: Episode) -> None:
        """Store an episode. Evicts from hot to warm on overflow."""
        async with self._lock:
            # Add to hot store
            self._hot[episode.episode_id] = episode
            self._hot.move_to_end(episode.episode_id)

            # Evict oldest from hot to warm if over capacity
            evicted: list[Episode] = []
            while len(self._hot) > self._max_hot:
                _oldest_id, oldest = self._hot.popitem(last=False)
                evicted.append(oldest)

        # Spill outside the lock to avoid holding it during I/O
        for ep in evicted:
            await self._spill_to_warm(ep)

    async def recall(
        self,
        query: str,
        top_k: int = 10,
        time_window_hours: float | None = None,
        goal_id: str | None = None,
        episode_type: str | None = None,
    ) -> list[Episode]:
        """Search across hot and warm stores.

        Uses keyword matching + recency weighting.
        For semantic search, use recall_semantic() with an embedding function.
        """
        candidates: list[tuple[float, Episode]] = []
        now = datetime.now(UTC)
        query_words = set(query.lower().split())

        # Search hot store
        for ep in self._hot.values():
            if goal_id and ep.goal_id != goal_id:
                continue
            if episode_type and ep.episode_type != episode_type:
                continue
            if time_window_hours is not None:
                age_hours = (now - ep.timestamp).total_seconds() / 3600
                if age_hours > time_window_hours:
                    continue

            score = self._score_episode(ep, query_words, now)
            candidates.append((score, ep))

        # Search warm store
        if self._db_path is not None and self._initialized:
            warm_episodes = await self._search_warm(
                query, goal_id, episode_type, time_window_hours
            )
            for ep in warm_episodes:
                score = self._score_episode(ep, query_words, now)
                candidates.append((score, ep))

        # Sort by score descending, return top_k
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = [ep for _, ep in candidates[:top_k]]

        # Update access counts
        for ep in results:
            ep.accessed_count += 1
            if ep.episode_id in self._hot:
                self._hot.move_to_end(ep.episode_id)

        return results

    async def recall_for_goal(
        self,
        goal_id: str,
        top_k: int = 20,
    ) -> list[Episode]:
        """All episodes for a goal, ordered by relevance then recency."""
        episodes: list[Episode] = []

        # Hot store
        for ep in self._hot.values():
            if ep.goal_id == goal_id:
                episodes.append(ep)

        # Warm store
        if self._db_path is not None and self._initialized:
            warm = await self._search_warm("", goal_id=goal_id)
            episodes.extend(warm)

        # Sort by relevance (desc) then timestamp (desc)
        episodes.sort(
            key=lambda e: (e.relevance_to_goal, e.timestamp.timestamp()),
            reverse=True,
        )

        return episodes[:top_k]

    async def clear_goal(self, goal_id: str) -> int:
        """Remove all episodes for a completed goal. Returns count removed."""
        removed = 0

        # Hot store
        to_remove = [eid for eid, ep in self._hot.items() if ep.goal_id == goal_id]
        for eid in to_remove:
            del self._hot[eid]
            removed += 1

        # Warm store
        if self._db_path is not None and self._initialized:
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM episodic_memory WHERE goal_id = ?",
                    (goal_id,),
                )
                removed += cursor.rowcount
                await db.commit()

        return removed

    def hot_count(self) -> int:
        """Number of entries in hot store."""
        return len(self._hot)

    async def warm_count(self) -> int:
        """Number of entries in warm store."""
        if self._db_path is None or not self._initialized:
            return 0
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM episodic_memory")
            row = await cursor.fetchone()
        return row[0] if row else 0

    def status(self) -> dict[str, Any]:
        """Monitoring snapshot."""
        return {
            "hot_entries": len(self._hot),
            "max_hot": self._max_hot,
        }

    def get_latest(self, limit: int = 20) -> list[Episode]:
        """Return the most recent episodes from the hot store."""
        items = list(self._hot.values())
        items.reverse()
        return items[:limit]

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _score_episode(
        self,
        episode: Episode,
        query_words: set[str],
        now: datetime,
    ) -> float:
        """Score an episode by keyword match + recency + relevance.

        Score = 0.4 * keyword_overlap + 0.3 * recency + 0.3 * relevance_to_goal
        """
        # Keyword overlap
        ep_text = f"{episode.summary} {json.dumps(episode.content)}".lower()
        ep_words = set(ep_text.split())
        if query_words and ep_words:
            overlap = len(query_words & ep_words) / max(len(query_words), 1)
        else:
            overlap = 0.0

        # Recency (exponential decay, half-life 1 hour)
        age_hours = max(0, (now - episode.timestamp).total_seconds() / 3600)
        recency = 0.5 ** (age_hours / 1.0)  # half-life = 1 hour

        # Relevance
        relevance = episode.relevance_to_goal

        return 0.4 * overlap + 0.3 * recency + 0.3 * relevance

    async def _spill_to_warm(self, episode: Episode) -> None:
        """Move an episode from hot to warm (SQLite)."""
        if self._db_path is None or not self._initialized:
            return  # No warm store configured, just drop

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO episodic_memory (
                    episode_id, inquiry_id, goal_id, timestamp, episode_type,
                    content, summary, token_count, relevance_to_goal, accessed_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.inquiry_id,
                    episode.goal_id,
                    episode.timestamp.isoformat(),
                    episode.episode_type,
                    json.dumps(episode.content),
                    episode.summary,
                    episode.token_count,
                    episode.relevance_to_goal,
                    episode.accessed_count,
                ),
            )

            # Enforce warm store limit
            count_cursor = await db.execute("SELECT COUNT(*) FROM episodic_memory")
            count = (await count_cursor.fetchone())[0]
            if count > self._max_warm:
                excess = count - self._max_warm
                await db.execute(
                    """
                    DELETE FROM episodic_memory WHERE episode_id IN (
                        SELECT episode_id FROM episodic_memory
                        ORDER BY timestamp ASC LIMIT ?
                    )
                    """,
                    (excess,),
                )

            await db.commit()

    async def _search_warm(
        self,
        query: str,
        goal_id: str | None = None,
        episode_type: str | None = None,
        time_window_hours: float | None = None,
    ) -> list[Episode]:
        """Search the warm SQLite store."""
        if self._db_path is None or not self._initialized:
            return []

        sql = "SELECT * FROM episodic_memory WHERE 1=1"
        params: list[Any] = []

        if goal_id:
            sql += " AND goal_id = ?"
            params.append(goal_id)

        if episode_type:
            sql += " AND episode_type = ?"
            params.append(episode_type)

        if time_window_hours is not None:
            sql += " AND timestamp > ?"
            from datetime import timedelta
            cutoff_dt = datetime.now(UTC) - timedelta(hours=time_window_hours)
            params.append(cutoff_dt.isoformat())

        sql += " ORDER BY timestamp DESC LIMIT 100"

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()

        episodes = []
        for row in rows:
            episodes.append(Episode(
                episode_id=row[0],
                inquiry_id=row[1],
                goal_id=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                episode_type=row[4],
                content=json.loads(row[5]) if row[5] else {},
                summary=row[6],
                token_count=row[7],
                relevance_to_goal=row[8],
                accessed_count=row[9],
            ))
        return episodes

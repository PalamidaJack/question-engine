"""Goal-scoped working memory for in-flight subtask collaboration.

Provides a key-value store keyed by goal_id so that subtasks within the
same goal can share intermediate results without external persistence.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class _Entry:
    """A single working-memory entry with optional TTL."""

    value: Any
    source: str
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float | None = None
    last_accessed: float = field(default_factory=time.monotonic)

    def is_expired(self, now: float | None = None) -> bool:
        if self.ttl_seconds is None:
            return False
        now = now or time.monotonic()
        return (now - self.created_at) >= self.ttl_seconds


class WorkingMemory:
    """Goal-scoped key-value store for in-flight subtask collaboration."""

    def __init__(self, max_entries_per_goal: int = 500) -> None:
        self._stores: dict[str, dict[str, _Entry]] = {}
        self._max_entries = max_entries_per_goal

    def put(
        self,
        goal_id: str,
        key: str,
        value: Any,
        source: str = "",
        ttl_seconds: float | None = None,
    ) -> None:
        """Store a value with optional TTL and LRU eviction."""
        store = self._stores.setdefault(goal_id, {})
        now = time.monotonic()

        # Evict LRU entries if at capacity and this is a new key
        if key not in store and len(store) >= self._max_entries:
            self._evict_lru(store)

        store[key] = _Entry(
            value=value,
            source=source,
            created_at=now,
            ttl_seconds=ttl_seconds,
            last_accessed=now,
        )

    def get(self, goal_id: str, key: str, default: Any = None) -> Any:
        """Retrieve a value with lazy TTL expiry."""
        store = self._stores.get(goal_id)
        if store is None:
            return default
        entry = store.get(key)
        if entry is None:
            return default
        if entry.is_expired():
            del store[key]
            return default
        entry.last_accessed = time.monotonic()
        return entry.value

    def get_all(self, goal_id: str) -> dict[str, Any]:
        """Return all non-expired entries for a goal."""
        store = self._stores.get(goal_id)
        if store is None:
            return {}
        now = time.monotonic()
        result: dict[str, Any] = {}
        expired_keys: list[str] = []
        for key, entry in store.items():
            if entry.is_expired(now):
                expired_keys.append(key)
            else:
                result[key] = entry.value
        for key in expired_keys:
            del store[key]
        return result

    def store_subtask_result(
        self,
        goal_id: str,
        subtask_id: str,
        output: dict[str, Any],
    ) -> None:
        """Store subtask output under ``subtask:{subtask_id}`` + individual keys."""
        self.put(goal_id, f"subtask:{subtask_id}", output, source=subtask_id)
        for key, value in output.items():
            self.put(goal_id, f"{subtask_id}.{key}", value, source=subtask_id)

    def build_context_for_subtask(
        self,
        goal_id: str,
        subtask_id: str,
        depends_on: list[str],
    ) -> dict[str, Any]:
        """Build context dict from dependency results."""
        context: dict[str, Any] = {}
        for dep_id in depends_on:
            dep_output = self.get(goal_id, f"subtask:{dep_id}")
            if dep_output is not None:
                context[dep_id] = dep_output
        return context

    def clear_goal(self, goal_id: str) -> None:
        """Cleanup on goal completion/failure."""
        self._stores.pop(goal_id, None)

    def status(self) -> dict[str, Any]:
        """Monitoring dict."""
        return {
            "active_goals": len(self._stores),
            "total_entries": sum(len(s) for s in self._stores.values()),
            "max_entries_per_goal": self._max_entries,
        }

    def _evict_lru(self, store: dict[str, _Entry]) -> None:
        """Evict the least-recently-accessed entry."""
        if not store:
            return
        lru_key = min(store, key=lambda k: store[k].last_accessed)
        del store[lru_key]

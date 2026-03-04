"""LRU cache for tool results — reduces redundant sub-agent calls."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

log = logging.getLogger(__name__)


class ToolResultCache:
    """Thread-safe LRU cache keyed on SHA-256(tool_name + sorted params).

    TTL-aware entries are evicted on access. Max capacity with LRU eviction.
    """

    def __init__(
        self,
        max_entries: int = 500,
        default_ttl_seconds: int = 300,
    ) -> None:
        self._max_entries = max_entries
        self._default_ttl = default_ttl_seconds
        self._lock = threading.Lock()
        # key -> (value, expires_at)
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(tool_name: str, params: dict[str, Any]) -> str:
        raw = tool_name + json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, tool_name: str, params: dict[str, Any]) -> str | None:
        key = self._make_key(tool_name, params)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: str,
        ttl_seconds: int | None = None,
    ) -> None:
        key = self._make_key(tool_name, params)
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires_at = time.monotonic() + ttl
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (result, expires_at)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0


# ── Singleton ──────────────────────────────────────────────────────────────

_cache = ToolResultCache()


def get_tool_result_cache() -> ToolResultCache:
    return _cache

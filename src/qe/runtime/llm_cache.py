"""LLM response caching to avoid redundant API calls.

Caches identical (model, messages, response_model) tuples with TTL expiry.
Injected into BaseService._call_llm() so all services benefit automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached LLM response with expiry metadata."""

    key: str
    response: Any
    created_at: float
    ttl_seconds: float
    model: str
    hit_count: int = 0

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


@dataclass
class CacheStats:
    """Running statistics for cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    entries: int = 0
    total_saved_calls: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "entries": self.entries,
            "hit_rate": round(self.hit_rate, 4),
            "total_saved_calls": self.total_saved_calls,
        }


class LLMCache:
    """In-memory TTL cache for LLM responses.

    Thread-safe. Keyed on hash of (model, messages, response_model_name).
    """

    def __init__(
        self,
        max_entries: int = 1000,
        default_ttl_seconds: float = 300.0,
        enabled: bool = True,
    ) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries
        self._default_ttl = default_ttl_seconds
        self._enabled = enabled
        self._stats = CacheStats()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def get(self, key: str) -> Any | None:
        """Look up a cached response. Returns None on miss or expiry."""
        if not self._enabled:
            return None

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            if entry.expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.entries = len(self._cache)
                return None

            entry.hit_count += 1
            self._stats.hits += 1
            self._stats.total_saved_calls += 1
            log.debug(
                "llm_cache.hit key=%s model=%s hit_count=%d",
                key[:16],
                entry.model,
                entry.hit_count,
            )
            return entry.response

    def put(
        self,
        key: str,
        response: Any,
        model: str,
        ttl_seconds: float | None = None,
    ) -> None:
        """Store a response in the cache."""
        if not self._enabled:
            return

        with self._lock:
            # Evict expired entries if at capacity
            if len(self._cache) >= self._max_entries:
                self._evict_expired()

            # If still at capacity, evict oldest entry
            if len(self._cache) >= self._max_entries:
                oldest_key = min(
                    self._cache, key=lambda k: self._cache[k].created_at
                )
                del self._cache[oldest_key]
                self._stats.evictions += 1

            self._cache[key] = CacheEntry(
                key=key,
                response=response,
                created_at=time.time(),
                ttl_seconds=ttl_seconds or self._default_ttl,
                model=model,
            )
            self._stats.entries = len(self._cache)

    def _evict_expired(self) -> None:
        """Remove all expired entries. Must be called with lock held."""
        expired = [k for k, v in self._cache.items() if v.expired]
        for k in expired:
            del self._cache[k]
            self._stats.evictions += 1
        self._stats.entries = len(self._cache)

    def clear(self) -> int:
        """Clear all cache entries. Returns count of entries removed."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.entries = 0
            return count

    def stats(self) -> dict[str, Any]:
        """Return cache performance statistics."""
        with self._lock:
            self._stats.entries = len(self._cache)
            return self._stats.to_dict()

    @staticmethod
    def make_key(
        model: str,
        messages: list[dict],
        response_model_name: str,
    ) -> str:
        """Generate a deterministic cache key from LLM call parameters."""
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "schema": response_model_name,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


# ── Singleton ──────────────────────────────────────────────────────────────

_llm_cache = LLMCache()


def get_llm_cache() -> LLMCache:
    return _llm_cache

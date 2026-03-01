"""Engram Cache — Three-band cache inspired by DeepSeek Engram.

Replaces LLMCache with a three-band lookup:
  Band 1 (Exact):    SHA-256 match → skip LLM entirely (O(1))
  Band 2 (Template): Semantic similarity >= threshold → cached template for adaptation
  Band 3 (Full):     No cache help → full reasoning required

Tracks compute-to-memory ratio. Target: 75-80% full / 20-25% cached
(DeepSeek Engram U-shaped law).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)

BandType = Literal["exact", "template", "full"]


@dataclass
class CacheEntry:
    """A cached response with TTL metadata."""

    key: str
    response: Any
    created_at: float
    ttl_seconds: float
    model: str
    query_text: str = ""  # for template matching
    hit_count: int = 0

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


@dataclass
class BandStats:
    """Running statistics per band and overall."""

    exact_hits: int = 0
    template_hits: int = 0
    full_misses: int = 0
    total_saved_cost_usd: float = 0.0

    @property
    def total(self) -> int:
        return self.exact_hits + self.template_hits + self.full_misses

    def compute_ratio(self) -> dict[str, float]:
        """Compute the compute-to-memory ratio. Target: 75-80% full."""
        total = self.total or 1
        return {
            "exact": self.exact_hits / total,
            "template": self.template_hits / total,
            "full": self.full_misses / total,
            "cached_pct": (self.exact_hits + self.template_hits) / total,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_hits": self.exact_hits,
            "template_hits": self.template_hits,
            "full_misses": self.full_misses,
            "total": self.total,
            **self.compute_ratio(),
            "total_saved_cost_usd": round(self.total_saved_cost_usd, 4),
        }


class EngramCache:
    """Three-band cache following DeepSeek Engram principles.

    Band 1 (Exact): Identical to v1's SHA-256 cache but with longer TTL.
    Band 2 (Template): Semantic similarity lookup; returns cached response
        as a template for adaptation by a fast model.
    Band 3 (Full): Novel query, no cache help.
    """

    def __init__(
        self,
        max_exact_entries: int = 1000,
        exact_ttl: float = 600.0,
        template_ttl: float = 3600.0,
        template_similarity: float = 0.90,
        embed_fn=None,
        enabled: bool = True,
    ) -> None:
        self._exact_cache: dict[str, CacheEntry] = {}
        self._template_store: list[CacheEntry] = []  # for similarity search
        self._lock = threading.Lock()
        self._max_exact = max_exact_entries
        self._exact_ttl = exact_ttl
        self._template_ttl = template_ttl
        self._template_sim = template_similarity
        self._embed_fn = embed_fn  # async (text) -> list[float]
        self._enabled = enabled
        self._stats = BandStats()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    async def lookup(
        self,
        model: str,
        messages: list[dict],
        schema_name: str = "",
    ) -> tuple[BandType, Any | None]:
        """Three-band cache lookup.

        Returns (band, cached_response_or_none).
        - ("exact", response): Exact match found, skip LLM.
        - ("template", response): Template match, adapt with fast model.
        - ("full", None): No cache help, full reasoning needed.
        """
        if not self._enabled:
            return ("full", None)

        # Band 1: Exact match (SHA-256)
        exact_key = self.make_key(model, messages, schema_name)
        with self._lock:
            entry = self._exact_cache.get(exact_key)
            if entry is not None and not entry.expired:
                entry.hit_count += 1
                self._stats.exact_hits += 1
                log.debug("engram.exact_hit key=%s model=%s", exact_key[:16], model)
                return ("exact", entry.response)
            elif entry is not None and entry.expired:
                del self._exact_cache[exact_key]

        # Band 2: Template match (semantic similarity)
        query_text = self._extract_query_text(messages)
        if query_text and self._embed_fn is not None:
            template_result = await self._find_template(query_text)
            if template_result is not None:
                self._stats.template_hits += 1
                log.debug("engram.template_hit similarity=%.3f", template_result[1])
                return ("template", template_result[0])

        # Band 3: Full reasoning needed
        self._stats.full_misses += 1
        return ("full", None)

    def store(
        self,
        model: str,
        messages: list[dict],
        schema_name: str,
        response: Any,
        cost_usd: float = 0.0,
    ) -> None:
        """Store a response in both exact and template caches."""
        if not self._enabled:
            return

        exact_key = self.make_key(model, messages, schema_name)
        query_text = self._extract_query_text(messages)
        now = time.time()

        with self._lock:
            # Exact cache: evict if at capacity
            if len(self._exact_cache) >= self._max_exact:
                self._evict_expired_exact()
            if len(self._exact_cache) >= self._max_exact:
                oldest_key = min(
                    self._exact_cache, key=lambda k: self._exact_cache[k].created_at
                )
                del self._exact_cache[oldest_key]

            self._exact_cache[exact_key] = CacheEntry(
                key=exact_key,
                response=response,
                created_at=now,
                ttl_seconds=self._exact_ttl,
                model=model,
                query_text=query_text,
            )

        # Template cache: store with longer TTL for similarity matching
        if query_text:
            template_entry = CacheEntry(
                key=f"tmpl_{exact_key[:16]}",
                response=response,
                created_at=now,
                ttl_seconds=self._template_ttl,
                model=model,
                query_text=query_text,
            )
            with self._lock:
                self._template_store.append(template_entry)
                # Evict expired templates
                self._template_store = [
                    t for t in self._template_store if not t.expired
                ]

    def get_exact(self, key: str) -> Any | None:
        """Direct exact-match lookup (backward compatible with LLMCache.get)."""
        if not self._enabled:
            return None
        with self._lock:
            entry = self._exact_cache.get(key)
            if entry is None or entry.expired:
                if entry is not None:
                    del self._exact_cache[key]
                self._stats.full_misses += 1
                return None
            entry.hit_count += 1
            self._stats.exact_hits += 1
            return entry.response

    def put_exact(
        self,
        key: str,
        response: Any,
        model: str,
        ttl_seconds: float | None = None,
    ) -> None:
        """Direct exact-match store (backward compatible with LLMCache.put)."""
        if not self._enabled:
            return
        with self._lock:
            if len(self._exact_cache) >= self._max_exact:
                self._evict_expired_exact()
            if len(self._exact_cache) >= self._max_exact:
                oldest = min(
                    self._exact_cache, key=lambda k: self._exact_cache[k].created_at
                )
                del self._exact_cache[oldest]

            self._exact_cache[key] = CacheEntry(
                key=key,
                response=response,
                created_at=time.time(),
                ttl_seconds=ttl_seconds or self._exact_ttl,
                model=model,
            )

    def clear(self) -> int:
        """Clear all cache entries."""
        with self._lock:
            count = len(self._exact_cache) + len(self._template_store)
            self._exact_cache.clear()
            self._template_store.clear()
            return count

    def stats(self) -> dict[str, Any]:
        """Return cache performance statistics."""
        with self._lock:
            result = self._stats.to_dict()
            result["exact_entries"] = len(self._exact_cache)
            result["template_entries"] = len(self._template_store)
            return result

    @staticmethod
    def make_key(
        model: str,
        messages: list[dict],
        response_model_name: str,
    ) -> str:
        """Generate a deterministic cache key (backward compatible with LLMCache)."""
        payload = json.dumps(
            {"model": model, "messages": messages, "schema": response_model_name},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    async def _find_template(
        self,
        query_text: str,
    ) -> tuple[Any, float] | None:
        """Find a semantically similar cached response.

        Returns (response, similarity) if found above threshold, else None.
        """
        if not self._template_store or self._embed_fn is None:
            return None

        try:
            query_emb = await self._embed_fn(query_text)
        except Exception:
            log.warning("engram.embed_failed", exc_info=True)
            return None

        best_sim = 0.0
        best_response = None

        # Simple linear scan (template store is small)
        for entry in self._template_store:
            if entry.expired or not entry.query_text:
                continue
            try:
                entry_emb = await self._embed_fn(entry.query_text)
                sim = self._cosine_similarity(query_emb, entry_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_response = entry.response
            except Exception:
                continue

        if best_sim >= self._template_sim and best_response is not None:
            return (best_response, best_sim)

        return None

    @staticmethod
    def _extract_query_text(messages: list[dict]) -> str:
        """Extract the user query text from messages for template matching."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _evict_expired_exact(self) -> None:
        """Remove expired entries from exact cache. Must hold lock."""
        expired = [k for k, v in self._exact_cache.items() if v.expired]
        for k in expired:
            del self._exact_cache[k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ── Singleton ──────────────────────────────────────────────────────────────

_engram_cache: EngramCache | None = None


def get_engram_cache(**kwargs) -> EngramCache:
    """Get or create the global EngramCache singleton."""
    global _engram_cache
    if _engram_cache is None:
        _engram_cache = EngramCache(**kwargs)
    return _engram_cache

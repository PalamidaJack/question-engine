"""Tests for the ToolResultCache — LRU cache for tool results."""

import time

from qe.runtime.tool_result_cache import ToolResultCache


class TestToolResultCache:
    def test_hit_and_miss(self):
        cache = ToolResultCache(max_entries=10)
        cache.put("search", {"q": "hello"}, "result1")
        assert cache.get("search", {"q": "hello"}) == "result1"
        assert cache.get("search", {"q": "world"}) is None

    def test_expiry(self):
        cache = ToolResultCache(max_entries=10, default_ttl_seconds=0)
        cache.put("tool", {"a": 1}, "val", ttl_seconds=0)
        # Expired immediately
        time.sleep(0.01)
        assert cache.get("tool", {"a": 1}) is None

    def test_eviction_at_capacity(self):
        cache = ToolResultCache(max_entries=2)
        cache.put("t1", {}, "v1")
        cache.put("t2", {}, "v2")
        cache.put("t3", {}, "v3")  # should evict t1
        assert cache.get("t1", {}) is None
        assert cache.get("t2", {}) == "v2"
        assert cache.get("t3", {}) == "v3"

    def test_deterministic_keys(self):
        cache = ToolResultCache()
        # Same params in different order should produce same key
        key1 = cache._make_key("tool", {"a": 1, "b": 2})
        key2 = cache._make_key("tool", {"b": 2, "a": 1})
        assert key1 == key2

    def test_stats(self):
        cache = ToolResultCache()
        cache.put("t", {"x": 1}, "val")
        cache.get("t", {"x": 1})  # hit
        cache.get("t", {"x": 2})  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1

    def test_clear(self):
        cache = ToolResultCache()
        cache.put("t", {}, "val")
        cache.clear()
        assert cache.get("t", {}) is None
        stats = cache.stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0

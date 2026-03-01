"""Tests for EngramCache — Three-band cache inspired by DeepSeek Engram."""

import asyncio

import pytest

from qe.runtime.engram_cache import BandStats, EngramCache


@pytest.fixture
def cache():
    return EngramCache(max_exact_entries=10, exact_ttl=5.0, template_ttl=60.0)


# ── Band 1: Exact Match ──────────────────────────────────────────────────────


class TestExactBand:
    async def test_exact_hit(self, cache):
        messages = [{"role": "user", "content": "What is the P/E ratio of AAPL?"}]
        cache.store("gpt-4o", messages, "FinanceSchema", {"pe_ratio": 28.5})

        band, response = await cache.lookup("gpt-4o", messages, "FinanceSchema")
        assert band == "exact"
        assert response == {"pe_ratio": 28.5}

    async def test_exact_miss(self, cache):
        messages = [{"role": "user", "content": "Novel question never seen"}]
        band, response = await cache.lookup("gpt-4o", messages, "Schema")
        assert band == "full"
        assert response is None

    async def test_exact_expired(self, cache):
        cache = EngramCache(exact_ttl=0.01)  # Very short TTL
        messages = [{"role": "user", "content": "Test"}]
        cache.store("gpt-4o", messages, "S", "response")

        await asyncio.sleep(0.02)  # Wait for expiry

        band, response = await cache.lookup("gpt-4o", messages, "S")
        assert band == "full"
        assert response is None

    async def test_different_model_different_key(self, cache):
        messages = [{"role": "user", "content": "Same question"}]
        cache.store("gpt-4o", messages, "S", "response_a")

        band, response = await cache.lookup("claude-sonnet", messages, "S")
        assert band == "full"  # Different model = different key

    async def test_exact_eviction_at_capacity(self, cache):
        # Fill cache to capacity
        for i in range(12):
            messages = [{"role": "user", "content": f"Question {i}"}]
            cache.store("gpt-4o", messages, "S", f"response_{i}")

        stats = cache.stats()
        assert stats["exact_entries"] <= 10


# ── Band 2: Template Match ──────────────────────────────────────────────────


class TestTemplateBand:
    async def test_template_hit_with_embed_fn(self):
        async def mock_embed(text):
            # Simple mock: returns same vector for similar questions
            if "investment" in text.lower():
                return [1.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0]

        cache = EngramCache(embed_fn=mock_embed, template_similarity=0.9)

        # Store a response
        messages = [{"role": "user", "content": "What are good investment opportunities?"}]
        cache.store("gpt-4o", messages, "S", {"answer": "Tech and healthcare"})

        # Query with similar question (same embedding due to "investment")
        similar_messages = [{"role": "user", "content": "Find untapped investment ideas"}]
        band, response = await cache.lookup("gpt-4o", similar_messages, "S")

        assert band == "template"
        assert response == {"answer": "Tech and healthcare"}

    async def test_template_miss_dissimilar(self):
        async def mock_embed(text):
            if "investment" in text.lower():
                return [1.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0]  # orthogonal = 0 similarity

        cache = EngramCache(embed_fn=mock_embed, template_similarity=0.9)
        messages = [{"role": "user", "content": "What are good investment opportunities?"}]
        cache.store("gpt-4o", messages, "S", {"answer": "something"})

        # Query with unrelated question (different embedding)
        diff_messages = [{"role": "user", "content": "What is the weather today?"}]
        band, response = await cache.lookup("gpt-4o", diff_messages, "S")

        assert band == "full"
        assert response is None

    async def test_no_template_without_embed_fn(self, cache):
        """Without embed_fn, template band is skipped."""
        messages = [{"role": "user", "content": "Test question"}]
        cache.store("gpt-4o", messages, "S", "response")

        similar = [{"role": "user", "content": "Similar test question"}]
        band, response = await cache.lookup("gpt-4o", similar, "S")

        # Without embed_fn, can't do template matching
        assert band == "full"


# ── Band Stats ──────────────────────────────────────────────────────────────


class TestBandStats:
    def test_compute_ratio_empty(self):
        stats = BandStats()
        ratio = stats.compute_ratio()
        assert ratio["cached_pct"] == 0.0

    def test_compute_ratio_all_cached(self):
        stats = BandStats(exact_hits=10, template_hits=5, full_misses=0)
        ratio = stats.compute_ratio()
        assert ratio["cached_pct"] == pytest.approx(1.0)

    def test_compute_ratio_mixed(self):
        stats = BandStats(exact_hits=5, template_hits=5, full_misses=40)
        ratio = stats.compute_ratio()
        assert ratio["cached_pct"] == pytest.approx(0.2)
        assert ratio["full"] == pytest.approx(0.8)

    async def test_stats_tracking(self, cache):
        # Generate some hits and misses
        messages = [{"role": "user", "content": "Cached question"}]
        cache.store("gpt-4o", messages, "S", "response")

        await cache.lookup("gpt-4o", messages, "S")  # exact hit
        await cache.lookup("gpt-4o", messages, "S")  # exact hit
        await cache.lookup("gpt-4o", [{"role": "user", "content": "novel"}], "S")  # miss

        stats = cache.stats()
        assert stats["exact_hits"] == 2
        assert stats["full_misses"] == 1
        assert stats["total"] == 3


# ── Backward Compatibility ──────────────────────────────────────────────────


class TestBackwardCompat:
    def test_make_key_same_as_llm_cache(self, cache):
        """EngramCache.make_key should be compatible with LLMCache.make_key."""
        key = EngramCache.make_key(
            "gpt-4o",
            [{"role": "user", "content": "test"}],
            "MySchema",
        )
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest

    def test_get_exact_put_exact(self, cache):
        """Direct get/put for backward compat with LLMCache."""
        key = "test_key_123"
        cache.put_exact(key, "response", "gpt-4o")

        result = cache.get_exact(key)
        assert result == "response"

    def test_get_exact_miss(self, cache):
        result = cache.get_exact("nonexistent_key")
        assert result is None


# ── Clear and Disable ──────────────────────────────────────────────────────


class TestClearAndDisable:
    def test_clear(self, cache):
        messages = [{"role": "user", "content": "test"}]
        cache.store("gpt-4o", messages, "S", "response")

        count = cache.clear()
        assert count >= 1
        assert cache.stats()["exact_entries"] == 0

    async def test_disabled_cache(self):
        cache = EngramCache(enabled=False)
        messages = [{"role": "user", "content": "test"}]
        cache.store("gpt-4o", messages, "S", "response")

        band, response = await cache.lookup("gpt-4o", messages, "S")
        assert band == "full"
        assert response is None

    async def test_toggle_enabled(self, cache):
        messages = [{"role": "user", "content": "test"}]
        cache.store("gpt-4o", messages, "S", "response")

        cache.enabled = False
        band, _ = await cache.lookup("gpt-4o", messages, "S")
        assert band == "full"

        cache.enabled = True
        band, response = await cache.lookup("gpt-4o", messages, "S")
        assert band == "exact"
        assert response == "response"

"""Tests for the embedding store (Phase 2)."""

from pathlib import Path

import aiosqlite
import pytest

from qe.substrate.embeddings import (
    EmbeddingStore,
    _cosine_similarity,
    _pack_floats,
    _unpack_floats,
)

# ── Vector math helpers ──────────────────────────────────────────────────────


class TestVectorMath:
    def test_cosine_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_cosine_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_cosine_similar(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.8, 0.1]
        sim = _cosine_similarity(a, b)
        assert sim > 0.9

    def test_cosine_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == 0.0


class TestPackUnpack:
    def test_roundtrip(self):
        original = [1.0, 2.5, -3.7, 0.0, 100.0]
        packed = _pack_floats(original)
        unpacked = _unpack_floats(packed)
        for a, b in zip(original, unpacked, strict=True):
            assert abs(a - b) < 1e-5

    def test_empty(self):
        packed = _pack_floats([])
        assert _unpack_floats(packed) == []


# ── EmbeddingStore ───────────────────────────────────────────────────────────


@pytest.fixture
async def store(tmp_path):
    """Create an EmbeddingStore with initialized tables."""
    db_path = str(tmp_path / "embed_test.db")
    migration = (
        Path(__file__).parent.parent.parent
        / "src/qe/substrate/migrations/0005_embeddings.sql"
    )
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(migration.read_text())
        await db.commit()
    return EmbeddingStore(db_path, model="test-model")


class TestEmbeddingStore:
    @pytest.mark.asyncio
    async def test_store_and_count(self, store):
        await store.store(
            "item-1", "hello world", embedding=[0.1, 0.2, 0.3]
        )
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_store_multiple(self, store):
        await store.store("a", "text a", embedding=[1.0, 0.0])
        await store.store("b", "text b", embedding=[0.0, 1.0])
        await store.store("c", "text c", embedding=[0.7, 0.7])
        assert await store.count() == 3

    @pytest.mark.asyncio
    async def test_search_returns_ranked(self, store):
        # Store 3 vectors: one very similar to query, one less, one orthogonal
        await store.store("close", "very close", embedding=[1.0, 0.1])
        await store.store("medium", "medium match", embedding=[0.7, 0.7])
        await store.store("far", "far away", embedding=[0.0, 1.0])

        results = await store.search(
            "query",
            top_k=10,
            min_similarity=0.0,
            query_embedding=[1.0, 0.0],
        )
        assert len(results) == 3
        # Closest should be first
        assert results[0].id == "close"
        assert results[0].similarity > results[1].similarity
        assert results[1].similarity > results[2].similarity

    @pytest.mark.asyncio
    async def test_search_min_similarity(self, store):
        await store.store("a", "text a", embedding=[1.0, 0.0])
        await store.store("b", "text b", embedding=[0.0, 1.0])

        results = await store.search(
            "query",
            min_similarity=0.9,
            query_embedding=[1.0, 0.0],
        )
        assert len(results) == 1
        assert results[0].id == "a"

    @pytest.mark.asyncio
    async def test_search_top_k(self, store):
        for i in range(10):
            await store.store(
                f"item-{i}",
                f"text {i}",
                embedding=[1.0 - i * 0.05, i * 0.05],
            )
        results = await store.search(
            "query",
            top_k=3,
            min_similarity=0.0,
            query_embedding=[1.0, 0.0],
        )
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_empty_store(self, store):
        results = await store.search(
            "query", query_embedding=[1.0, 0.0]
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, store):
        await store.store(
            "meta-1",
            "text",
            metadata={"source": "test", "type": "claim"},
            embedding=[0.5, 0.5],
        )
        results = await store.search(
            "query",
            min_similarity=0.0,
            query_embedding=[0.5, 0.5],
        )
        assert results[0].metadata["source"] == "test"

    @pytest.mark.asyncio
    async def test_upsert(self, store):
        await store.store("id-1", "original", embedding=[1.0, 0.0])
        await store.store("id-1", "updated", embedding=[0.0, 1.0])
        assert await store.count() == 1
        results = await store.search(
            "q", min_similarity=0.0, query_embedding=[0.0, 1.0]
        )
        assert results[0].text == "updated"

    @pytest.mark.asyncio
    async def test_delete(self, store):
        await store.store("del-1", "text", embedding=[1.0, 0.0])
        assert await store.count() == 1
        deleted = await store.delete("del-1")
        assert deleted is True
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        deleted = await store.delete("nope")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_find_contradictions(self, store):
        await store.store(
            "claim-1",
            "the sky is blue",
            embedding=[0.9, 0.1, 0.0],
        )
        await store.store(
            "claim-2",
            "the sky is red",
            embedding=[0.85, 0.15, 0.0],
        )
        await store.store(
            "claim-3",
            "water is wet",
            embedding=[0.0, 0.0, 1.0],
        )

        contradictions = await store.find_contradictions(
            "the sky is green",
            threshold=0.8,
            claim_embedding=[0.88, 0.12, 0.0],
        )
        # claim-1 and claim-2 are similar to query, claim-3 is not
        ids = [c.id for c in contradictions]
        assert "claim-1" in ids or "claim-2" in ids
        assert "claim-3" not in ids

    @pytest.mark.asyncio
    async def test_find_contradictions_excludes_exact_match(self, store):
        await store.store("c1", "exact text", embedding=[1.0, 0.0])
        contradictions = await store.find_contradictions(
            "exact text",
            threshold=0.0,
            claim_embedding=[1.0, 0.0],
        )
        assert len(contradictions) == 0

"""Tests for HNSW vector indexing and executor concurrency."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import aiosqlite
import pytest

from qe.substrate.embeddings import _HNSW_AVAILABLE, EmbeddingStore

# ── HNSW Indexing Tests ──────────────────────────────────────────────────


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "hnsw_test.db")
    migration = (
        Path(__file__).parent.parent.parent
        / "src/qe/substrate/migrations/0005_embeddings.sql"
    )
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(migration.read_text())
        await db.commit()
    return EmbeddingStore(db_path, model="test-model")


@pytest.mark.skipif(not _HNSW_AVAILABLE, reason="hnswlib not installed")
class TestHNSWIndexing:
    """Verify HNSW index is used and produces correct results."""

    @pytest.mark.asyncio
    async def test_hnsw_index_built_lazily(self, store):
        """Index should not exist until first search."""
        assert store._hnsw_index is None
        assert store._hnsw_dirty is True

        await store.store("a", "text a", embedding=[1.0, 0.0, 0.0])
        await store.store("b", "text b", embedding=[0.0, 1.0, 0.0])

        # Index still not built (store() adds incrementally but
        # _ensure_hnsw only runs on search)
        # After stores to a dirty index, it's still dirty
        results = await store.search(
            "q", top_k=10, min_similarity=0.0,
            query_embedding=[1.0, 0.0, 0.0],
        )
        # Now the index should be built
        assert store._hnsw_index is not None
        assert store._hnsw_dirty is False
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_hnsw_search_ranked_correctly(self, store):
        """HNSW search should return results in similarity order."""
        await store.store("close", "close", embedding=[0.9, 0.1, 0.0])
        await store.store("medium", "medium", embedding=[0.5, 0.5, 0.0])
        await store.store("far", "far", embedding=[0.0, 0.0, 1.0])

        results = await store.search(
            "q", top_k=10, min_similarity=0.0,
            query_embedding=[1.0, 0.0, 0.0],
        )
        assert results[0].id == "close"
        assert results[0].similarity > results[1].similarity

    @pytest.mark.asyncio
    async def test_hnsw_respects_min_similarity(self, store):
        await store.store("a", "text a", embedding=[1.0, 0.0])
        await store.store("b", "text b", embedding=[0.0, 1.0])

        results = await store.search(
            "q", min_similarity=0.9,
            query_embedding=[1.0, 0.0],
        )
        assert len(results) == 1
        assert results[0].id == "a"

    @pytest.mark.asyncio
    async def test_hnsw_rebuild_after_delete(self, store):
        """Delete should mark index dirty, next search rebuilds."""
        await store.store("a", "text a", embedding=[1.0, 0.0])
        await store.store("b", "text b", embedding=[0.0, 1.0])

        # First search builds the index
        await store.search("q", query_embedding=[1.0, 0.0])
        assert store._hnsw_dirty is False

        # Delete marks dirty
        await store.delete("b")
        assert store._hnsw_dirty is True

        # Next search rebuilds with only 1 entry
        results = await store.search(
            "q", top_k=10, min_similarity=0.0,
            query_embedding=[1.0, 0.0],
        )
        assert len(results) == 1
        assert results[0].id == "a"
        assert store._hnsw_dirty is False

    @pytest.mark.asyncio
    async def test_hnsw_incremental_add(self, store):
        """Items added after index build appear in search."""
        await store.store("a", "text a", embedding=[1.0, 0.0, 0.0])

        # Build index
        await store.search("q", query_embedding=[1.0, 0.0, 0.0])
        assert store._hnsw_index is not None

        # Add another item (incremental, no rebuild)
        await store.store("b", "text b", embedding=[0.0, 1.0, 0.0])
        assert store._hnsw_dirty is False  # no rebuild needed

        results = await store.search(
            "q", top_k=10, min_similarity=0.0,
            query_embedding=[0.0, 1.0, 0.0],
        )
        ids = [r.id for r in results]
        assert "b" in ids

    @pytest.mark.asyncio
    async def test_hnsw_empty_store_search(self, store):
        results = await store.search(
            "q", query_embedding=[1.0, 0.0],
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_hnsw_upsert_marks_dirty(self, store):
        """Upserting an existing id should mark index dirty."""
        await store.store("a", "original", embedding=[1.0, 0.0])

        # Build index
        await store.search("q", query_embedding=[1.0, 0.0])
        assert store._hnsw_dirty is False

        # Upsert same id with different embedding
        await store.store("a", "updated", embedding=[0.0, 1.0])
        assert store._hnsw_dirty is True

        # Search should reflect the updated embedding after rebuild
        results = await store.search(
            "q", top_k=10, min_similarity=0.0,
            query_embedding=[0.0, 1.0],
        )
        assert results[0].id == "a"
        assert results[0].text == "updated"


# ── Executor Concurrency Tests ───────────────────────────────────────────


class TestExecutorConcurrency:
    """ExecutorService processes multiple tasks concurrently."""

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        from qe.bus.memory_bus import MemoryBus
        from qe.models.envelope import Envelope
        from qe.services.executor import ExecutorService

        bus = MemoryBus()
        executor = ExecutorService(
            bus=bus, substrate=None, model="gpt-4o-mini",
            max_concurrency=3,
        )

        # Track concurrency level
        active = 0
        peak_active = 0

        async def instrumented_run_task(envelope):
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            await asyncio.sleep(0.05)  # simulate work
            active -= 1

        executor._run_task = instrumented_run_task

        await executor.start()

        # Fire 5 tasks concurrently
        tasks = []
        for i in range(5):
            env = Envelope(
                topic="tasks.dispatched",
                source_service_id="dispatcher",
                correlation_id=f"goal_{i}",
                payload={
                    "goal_id": f"goal_{i}",
                    "subtask_id": f"sub_{i}",
                    "task_type": "research",
                    "description": f"Task {i}",
                    "model_tier": "balanced",
                    "depends_on": [],
                    "contract": {},
                },
            )
            tasks.append(executor._handle_dispatched(env))

        await asyncio.gather(*tasks)

        # With max_concurrency=3, peak should be <= 3
        assert peak_active <= 3
        # But should actually run some concurrently (peak > 1)
        assert peak_active > 1

    @pytest.mark.asyncio
    async def test_semaphore_default(self):
        from qe.services.executor import ExecutorService

        executor = ExecutorService(
            bus=MagicMock(), substrate=None, model="gpt-4o-mini",
        )
        # Default is 5
        assert executor._semaphore._value == 5

    @pytest.mark.asyncio
    async def test_semaphore_custom(self):
        from qe.services.executor import ExecutorService

        executor = ExecutorService(
            bus=MagicMock(), substrate=None, model="gpt-4o-mini",
            max_concurrency=10,
        )
        assert executor._semaphore._value == 10

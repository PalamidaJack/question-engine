"""Tests for EpisodicMemory — Tier 1 short-term recall."""

from datetime import UTC, datetime, timedelta

import pytest

from qe.runtime.episodic_memory import Episode, EpisodicMemory


@pytest.fixture
def memory():
    """In-memory only episodic memory (no SQLite warm store)."""
    return EpisodicMemory(db_path=None, max_hot_entries=5)


@pytest.fixture
async def memory_with_db(tmp_path):
    """Episodic memory with SQLite warm store."""
    db_path = str(tmp_path / "episodic.db")
    mem = EpisodicMemory(db_path=db_path, max_hot_entries=3, max_warm_entries=10)
    await mem.initialize()
    return mem


def make_episode(**overrides) -> Episode:
    defaults = {
        "goal_id": "goal_1",
        "episode_type": "tool_call",
        "content": {"tool": "web_search", "result": "some data"},
        "summary": "Web search for market data",
        "relevance_to_goal": 0.7,
    }
    defaults.update(overrides)
    return Episode(**defaults)


# ── Basic Operations ──────────────────────────────────────────────────────


class TestBasicOperations:
    async def test_store_and_count(self, memory):
        ep = make_episode()
        await memory.store(ep)
        assert memory.hot_count() == 1

    async def test_store_multiple(self, memory):
        for i in range(3):
            await memory.store(make_episode(summary=f"Episode {i}"))
        assert memory.hot_count() == 3

    async def test_recall_by_goal(self, memory):
        await memory.store(make_episode(goal_id="goal_1"))
        await memory.store(make_episode(goal_id="goal_2"))

        results = await memory.recall_for_goal("goal_1")
        assert len(results) == 1
        assert results[0].goal_id == "goal_1"

    async def test_recall_by_query(self, memory):
        await memory.store(make_episode(summary="Healthcare sector analysis"))
        await memory.store(make_episode(summary="Energy transition data"))

        results = await memory.recall("healthcare", top_k=5)
        assert len(results) >= 1
        assert "healthcare" in results[0].summary.lower() or "Healthcare" in results[0].summary

    async def test_clear_goal(self, memory):
        await memory.store(make_episode(goal_id="goal_1"))
        await memory.store(make_episode(goal_id="goal_2"))

        removed = await memory.clear_goal("goal_1")
        assert removed == 1
        assert memory.hot_count() == 1


# ── LRU Eviction ──────────────────────────────────────────────────────


class TestLRUEviction:
    async def test_hot_eviction_without_db(self, memory):
        """When no DB, overflow entries are simply dropped."""
        for i in range(10):
            await memory.store(make_episode(summary=f"Episode {i}"))

        # Max hot is 5, so only 5 should remain
        assert memory.hot_count() == 5

    async def test_hot_to_warm_spill(self, memory_with_db):
        """Overflow entries spill to warm SQLite store."""
        for i in range(5):
            await memory_with_db.store(make_episode(summary=f"Episode {i}"))

        # Max hot is 3, so 2 should have spilled to warm
        assert memory_with_db.hot_count() == 3
        warm = await memory_with_db.warm_count()
        assert warm == 2

    async def test_warm_overflow_eviction(self, memory_with_db):
        """Warm store enforces its own max_entries."""
        # Store more than max_warm (10) entries
        for i in range(15):
            await memory_with_db.store(make_episode(summary=f"Episode {i}"))

        warm = await memory_with_db.warm_count()
        assert warm <= 10


# ── Search and Scoring ──────────────────────────────────────────────────────


class TestSearchScoring:
    async def test_recency_weighting(self, memory):
        """More recent episodes should score higher for same query."""
        old = make_episode(
            summary="Investment opportunity analysis",
            timestamp=datetime.now(UTC) - timedelta(hours=5),
        )
        recent = make_episode(
            summary="Investment opportunity analysis",
            timestamp=datetime.now(UTC),
        )
        await memory.store(old)
        await memory.store(recent)

        results = await memory.recall("investment opportunity", top_k=2)
        # Recent should come first due to recency weighting
        assert results[0].timestamp >= results[1].timestamp

    async def test_relevance_weighting(self, memory):
        await memory.store(make_episode(summary="Market data", relevance_to_goal=0.9))
        await memory.store(make_episode(summary="Market data", relevance_to_goal=0.1))

        results = await memory.recall("market", top_k=2)
        assert results[0].relevance_to_goal >= results[1].relevance_to_goal

    async def test_filter_by_episode_type(self, memory):
        await memory.store(make_episode(episode_type="tool_call"))
        await memory.store(make_episode(episode_type="llm_response"))

        results = await memory.recall("", episode_type="tool_call")
        assert all(r.episode_type == "tool_call" for r in results)

    async def test_time_window_filter(self, memory):
        old = make_episode(timestamp=datetime.now(UTC) - timedelta(hours=48))
        recent = make_episode(timestamp=datetime.now(UTC))
        await memory.store(old)
        await memory.store(recent)

        results = await memory.recall("", time_window_hours=24)
        assert len(results) == 1

    async def test_access_count_incremented(self, memory):
        ep = make_episode()
        await memory.store(ep)

        await memory.recall("web search")
        await memory.recall("web search")

        results = await memory.recall_for_goal("goal_1")
        assert results[0].accessed_count >= 2


# ── Cross-store Search ──────────────────────────────────────────────────────


class TestCrossStoreSearch:
    async def test_recall_searches_warm(self, memory_with_db):
        """Recall should search both hot and warm stores."""
        for i in range(5):
            await memory_with_db.store(
                make_episode(summary=f"Episode {i}", goal_id="goal_1")
            )

        # Some are in warm now
        results = await memory_with_db.recall_for_goal("goal_1")
        assert len(results) == 5

    async def test_clear_goal_clears_both(self, memory_with_db):
        for i in range(5):
            await memory_with_db.store(make_episode(goal_id="goal_1"))

        removed = await memory_with_db.clear_goal("goal_1")
        assert removed == 5
        assert memory_with_db.hot_count() == 0


# ── Status ──────────────────────────────────────────────────────────────


class TestEpisodicStatus:
    def test_status_empty(self, memory):
        status = memory.status()
        assert status["hot_entries"] == 0
        assert status["max_hot"] == 5

    async def test_status_with_data(self, memory):
        await memory.store(make_episode())
        status = memory.status()
        assert status["hot_entries"] == 1

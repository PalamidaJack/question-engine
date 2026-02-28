"""Tests for MAGMA multi-graph query system."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.substrate.magma import (
    GraphResult,
    MultiGraphQuery,
)


class TestMultiGraphQueryAvailability:
    def test_no_stores_returns_empty(self):
        mgq = MultiGraphQuery()
        assert mgq.available_graphs() == []

    def test_embeddings_provides_semantic(self):
        mgq = MultiGraphQuery(embeddings=MagicMock())
        assert "semantic" in mgq.available_graphs()

    def test_event_log_provides_temporal_and_causal(self):
        mgq = MultiGraphQuery(event_log=MagicMock())
        graphs = mgq.available_graphs()
        assert "temporal" in graphs
        assert "causal" in graphs

    def test_entity_resolver_provides_entity(self):
        mgq = MultiGraphQuery(entity_resolver=MagicMock())
        assert "entity" in mgq.available_graphs()


class TestMultiGraphQueryMerge:
    def test_merge_deduplicates_by_id(self):
        results = {
            "semantic": [
                GraphResult(graph="semantic", id="a", text="hello", score=0.9),
            ],
            "temporal": [
                GraphResult(graph="temporal", id="a", text="hello", score=0.5),
            ],
        }
        merged = MultiGraphQuery._merge(
            results,
            {"semantic": 0.5, "temporal": 0.5},
            top_k=10,
        )
        assert len(merged) == 1
        assert merged[0].id == "a"
        # 0.9*0.5 + 0.5*0.5 = 0.45 + 0.25 = 0.70
        assert merged[0].composite_score == pytest.approx(0.70)
        assert merged[0].graph_scores["semantic"] == 0.9
        assert merged[0].graph_scores["temporal"] == 0.5

    def test_merge_respects_top_k(self):
        results = {
            "semantic": [
                GraphResult(graph="semantic", id=f"item_{i}", text=f"t{i}", score=1.0 - i * 0.1)
                for i in range(10)
            ],
        }
        merged = MultiGraphQuery._merge(
            results, {"semantic": 1.0}, top_k=3
        )
        assert len(merged) == 3
        # Should be sorted by score descending
        assert merged[0].id == "item_0"
        assert merged[2].id == "item_2"

    def test_merge_empty_input(self):
        merged = MultiGraphQuery._merge({}, {"semantic": 1.0}, top_k=10)
        assert merged == []


class TestMultiGraphQueryIntegration:
    @pytest.fixture
    def mock_embeddings(self):
        store = AsyncMock()
        result = MagicMock()
        result.id = "emb_1"
        result.text = "Test claim"
        result.similarity = 0.85
        result.metadata = {"type": "claim"}
        store.search.return_value = [result]
        return store

    @pytest.fixture
    def mock_event_log(self):
        log = AsyncMock()
        log.replay.return_value = [
            {
                "envelope_id": "evt_1",
                "topic": "claims.committed",
                "source_service_id": "researcher",
                "payload": {"text": "Test claim about AI"},
                "timestamp": "2026-02-28T10:00:00+00:00",
            }
        ]
        return log

    @pytest.mark.asyncio
    async def test_semantic_only_query(self, mock_embeddings):
        mgq = MultiGraphQuery(embeddings=mock_embeddings)
        results = await mgq.multi_query(
            "test query",
            graphs=frozenset(["semantic"]),
        )
        assert len(results) == 1
        assert results[0].id == "emb_1"
        mock_embeddings.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multi_graph_parallel_query(self, mock_embeddings, mock_event_log):
        mgq = MultiGraphQuery(
            embeddings=mock_embeddings,
            event_log=mock_event_log,
        )
        results = await mgq.multi_query("AI test claim")
        # Should have results from both semantic and temporal
        assert len(results) >= 1
        mock_embeddings.search.assert_awaited_once()
        # Event log queried for both temporal and causal
        assert mock_event_log.replay.await_count >= 1

    @pytest.mark.asyncio
    async def test_no_stores_returns_empty(self):
        mgq = MultiGraphQuery()
        results = await mgq.multi_query("anything")
        assert results == []


class TestChainDepth:
    def test_no_children(self):
        depth = MultiGraphQuery._chain_depth("a", {})
        assert depth == 0

    def test_single_level(self):
        depth = MultiGraphQuery._chain_depth("a", {"a": ["b", "c"]})
        assert depth == 1

    def test_two_levels(self):
        children = {"a": ["b"], "b": ["c"]}
        depth = MultiGraphQuery._chain_depth("a", children)
        assert depth == 2

    def test_max_depth_capped(self):
        # Chain of 10 nodes but max_depth=3
        children = {f"n{i}": [f"n{i+1}"] for i in range(10)}
        depth = MultiGraphQuery._chain_depth("n0", children, max_depth=3)
        assert depth == 3

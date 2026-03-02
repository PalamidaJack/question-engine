"""Tests for CognitiveAgentPool."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.models.arena import ArenaResult
from qe.runtime.cognitive_agent_pool import AgentSlot, CognitiveAgentPool
from qe.runtime.competitive_arena import CompetitiveArena
from qe.runtime.strategy_models import StrategyConfig
from qe.services.inquiry.schemas import InquiryResult


def _make_mock_engine(
    goal_id: str = "g1",
    insights: list | None = None,
    cost: float = 0.01,
    findings: str = "found something",
) -> MagicMock:
    """Create a mock InquiryEngine that returns a canned result."""
    engine = MagicMock()
    result = InquiryResult(
        inquiry_id="inq_test",
        goal_id=goal_id,
        status="completed",
        iterations_completed=2,
        total_questions_generated=3,
        total_questions_answered=2,
        findings_summary=findings,
        insights=insights or [],
        total_cost_usd=cost,
        duration_seconds=1.0,
    )
    engine.run_inquiry = AsyncMock(return_value=result)
    return engine


class TestSpawnRetire:
    @pytest.mark.asyncio
    async def test_spawn_creates_agent(self):
        pool = CognitiveAgentPool(max_agents=3)
        agent = await pool.spawn_agent("economics", "balanced")
        assert agent.specialization == "economics"
        assert agent.status == "idle"
        assert pool.get_slot(agent.agent_id) is not None

    @pytest.mark.asyncio
    async def test_spawn_registers_in_underlying_pool(self):
        pool = CognitiveAgentPool(max_agents=3)
        agent = await pool.spawn_agent("tech")
        record = pool._pool.get(agent.agent_id)
        assert record is not None
        assert record.model_tier == "balanced"

    @pytest.mark.asyncio
    async def test_spawn_with_strategy(self):
        pool = CognitiveAgentPool(max_agents=3)
        strat = StrategyConfig(name="depth_first")
        agent = await pool.spawn_agent(strategy=strat)
        slot = pool.get_slot(agent.agent_id)
        assert slot is not None
        assert slot.strategy.name == "depth_first"

    @pytest.mark.asyncio
    async def test_spawn_at_capacity_raises(self):
        pool = CognitiveAgentPool(max_agents=1)
        await pool.spawn_agent()
        with pytest.raises(RuntimeError, match="capacity"):
            await pool.spawn_agent()

    @pytest.mark.asyncio
    async def test_retire_removes_agent(self):
        pool = CognitiveAgentPool(max_agents=3)
        agent = await pool.spawn_agent()
        assert await pool.retire_agent(agent.agent_id) is True
        assert pool.get_slot(agent.agent_id) is None
        assert pool._pool.get(agent.agent_id) is None

    @pytest.mark.asyncio
    async def test_retire_nonexistent(self):
        pool = CognitiveAgentPool()
        assert await pool.retire_agent("no_such_id") is False

    @pytest.mark.asyncio
    async def test_retire_cancels_active_task(self):
        pool = CognitiveAgentPool(max_agents=3)
        agent = await pool.spawn_agent()
        slot = pool.get_slot(agent.agent_id)

        # Create a long-running task
        async def _long_task():
            await asyncio.sleep(100)

        slot.current_task = asyncio.create_task(_long_task())
        assert await pool.retire_agent(agent.agent_id) is True
        assert slot.agent.status == "retired"


class TestParallelInquiry:
    @pytest.mark.asyncio
    async def test_fan_out_returns_results(self):
        engines = []

        def factory():
            e = _make_mock_engine(goal_id="g1")
            engines.append(e)
            return e

        pool = CognitiveAgentPool(max_agents=3, engine_factory=factory)
        await pool.spawn_agent("econ")
        await pool.spawn_agent("tech")

        results = await pool.run_parallel_inquiry("g1", "test goal")
        assert len(results) == 2
        assert all(r.goal_id == "g1" for r in results)

    @pytest.mark.asyncio
    async def test_fan_out_specific_agents(self):
        pool = CognitiveAgentPool(
            max_agents=3,
            engine_factory=lambda: _make_mock_engine(),
        )
        a1 = await pool.spawn_agent("econ")
        await pool.spawn_agent("tech")
        a3 = await pool.spawn_agent("science")

        results = await pool.run_parallel_inquiry(
            "g1", "test", agent_ids=[a1.agent_id, a3.agent_id]
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_pool_returns_empty(self):
        pool = CognitiveAgentPool()
        results = await pool.run_parallel_inquiry("g1", "test")
        assert results == []

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Semaphore limits concurrent executions."""
        concurrent = {"max": 0, "current": 0}

        async def _counting_run(*args, **kwargs):
            concurrent["current"] += 1
            concurrent["max"] = max(concurrent["max"], concurrent["current"])
            await asyncio.sleep(0.05)
            concurrent["current"] -= 1
            return InquiryResult(
                inquiry_id="inq",
                goal_id="g1",
                status="completed",
            )

        def factory():
            e = MagicMock()
            e.run_inquiry = _counting_run
            return e

        pool = CognitiveAgentPool(max_agents=5, engine_factory=factory)
        # Reduce semaphore to 2 to test limiting
        pool._semaphore = asyncio.Semaphore(2)
        for i in range(5):
            await pool.spawn_agent(f"spec_{i}")

        await pool.run_parallel_inquiry("g1", "test")
        assert concurrent["max"] <= 2


class TestMergeResults:
    @pytest.mark.asyncio
    async def test_merge_empty(self):
        pool = CognitiveAgentPool()
        merged = await pool.merge_results([])
        assert merged.status == "completed"
        assert merged.insights == []

    @pytest.mark.asyncio
    async def test_merge_insights_union(self):
        pool = CognitiveAgentPool()
        r1 = InquiryResult(
            inquiry_id="i1", goal_id="g1",
            insights=[{"insight_id": "a", "headline": "A"}],
            total_cost_usd=0.01,
            findings_summary="short",
        )
        r2 = InquiryResult(
            inquiry_id="i2", goal_id="g1",
            insights=[
                {"insight_id": "a", "headline": "A"},  # duplicate
                {"insight_id": "b", "headline": "B"},
            ],
            total_cost_usd=0.02,
            findings_summary="much longer findings",
        )
        merged = await pool.merge_results([r1, r2])
        assert len(merged.insights) == 2  # deduped
        assert merged.total_cost_usd == pytest.approx(0.03)
        assert merged.findings_summary == "much longer findings"

    @pytest.mark.asyncio
    async def test_merge_best_status(self):
        pool = CognitiveAgentPool()
        r1 = InquiryResult(inquiry_id="i1", goal_id="g1", status="failed")
        r2 = InquiryResult(inquiry_id="i2", goal_id="g1", status="completed")
        merged = await pool.merge_results([r1, r2])
        assert merged.status == "completed"


class TestPoolStatus:
    @pytest.mark.asyncio
    async def test_pool_status_reporting(self):
        pool = CognitiveAgentPool(max_agents=3)
        await pool.spawn_agent("econ")
        await pool.spawn_agent("tech")

        status = pool.pool_status()
        assert status["total_agents"] == 2
        assert status["active_agents"] == 0
        assert status["max_agents"] == 3
        assert len(status["agents"]) == 2

    @pytest.mark.asyncio
    async def test_agent_slot_access(self):
        pool = CognitiveAgentPool(max_agents=3)
        agent = await pool.spawn_agent("econ")
        slot = pool.get_slot(agent.agent_id)
        assert isinstance(slot, AgentSlot)
        assert slot.agent.agent_id == agent.agent_id

    @pytest.mark.asyncio
    async def test_get_slot_nonexistent(self):
        pool = CognitiveAgentPool()
        assert pool.get_slot("nope") is None


class TestCompetitiveInquiry:
    @pytest.mark.asyncio
    async def test_competitive_inquiry_with_arena(self):
        """When arena is set and >= 2 results, tournament runs."""
        arena = MagicMock(spec=CompetitiveArena)
        arena_result = ArenaResult(goal_id="g1", winner_id="agent_a")
        arena.run_tournament = AsyncMock(return_value=arena_result)

        pool = CognitiveAgentPool(
            max_agents=3,
            engine_factory=lambda: _make_mock_engine(),
            arena=arena,
        )
        await pool.spawn_agent("econ")
        await pool.spawn_agent("tech")

        result = await pool.run_competitive_inquiry("g1", "test goal")

        assert isinstance(result, ArenaResult)
        assert result.winner_id == "agent_a"
        arena.run_tournament.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_competitive_inquiry_without_arena(self):
        """Without arena, falls back to merge_results."""
        pool = CognitiveAgentPool(
            max_agents=3,
            engine_factory=lambda: _make_mock_engine(),
        )
        await pool.spawn_agent("econ")
        await pool.spawn_agent("tech")

        result = await pool.run_competitive_inquiry("g1", "test goal")

        assert isinstance(result, InquiryResult)
        assert result.goal_id == "g1"

    @pytest.mark.asyncio
    async def test_competitive_inquiry_single_result(self):
        """With < 2 results, falls back to merge even with arena."""
        arena = MagicMock(spec=CompetitiveArena)
        arena.run_tournament = AsyncMock()

        pool = CognitiveAgentPool(
            max_agents=3,
            engine_factory=lambda: _make_mock_engine(),
            arena=arena,
        )
        await pool.spawn_agent("econ")

        result = await pool.run_competitive_inquiry("g1", "test goal")

        assert isinstance(result, InquiryResult)
        arena.run_tournament.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_competitive_inquiry_empty_pool(self):
        """Empty pool returns empty merged result."""
        arena = MagicMock(spec=CompetitiveArena)
        pool = CognitiveAgentPool(arena=arena)

        result = await pool.run_competitive_inquiry("g1", "test goal")

        assert isinstance(result, InquiryResult)
        arena.run_tournament.assert_not_awaited()

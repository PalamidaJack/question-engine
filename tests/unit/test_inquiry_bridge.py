"""Tests for InquiryBridge — cross-loop glue connecting Inquiry, Knowledge, and Strategy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.models.envelope import Envelope
from qe.runtime.episodic_memory import EpisodicMemory
from qe.runtime.inquiry_bridge import InquiryBridge

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_bus():
    """Build a mock bus with subscribe/unsubscribe/publish."""
    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    bus.publish = MagicMock()
    return bus


def _make_bridge(**overrides) -> InquiryBridge:
    """Build an InquiryBridge with mocked dependencies."""
    episodic = AsyncMock(spec=EpisodicMemory)
    episodic.store = AsyncMock()

    defaults = {
        "bus": _make_bus(),
        "episodic_memory": episodic,
        "strategy_evolver": None,
        "knowledge_loop": None,
    }
    defaults.update(overrides)
    return InquiryBridge(**defaults)


def _make_evolver(current_strategy: str | None = "breadth_first"):
    """Build a mock strategy evolver."""
    evolver = MagicMock()
    evolver._current_strategy = current_strategy
    evolver.record_outcome = MagicMock()
    return evolver


def _make_knowledge_loop():
    """Build a mock knowledge loop."""
    kl = AsyncMock()
    kl.trigger_consolidation = AsyncMock()
    return kl


def _envelope(topic: str, payload: dict) -> Envelope:
    return Envelope(topic=topic, source_service_id="test", payload=payload)


# ── Init Tests ───────────────────────────────────────────────────────────


class TestInquiryBridgeInit:
    def test_stores_deps(self):
        bus = _make_bus()
        episodic = AsyncMock(spec=EpisodicMemory)
        evolver = _make_evolver()
        kl = _make_knowledge_loop()
        bridge = InquiryBridge(
            bus=bus, episodic_memory=episodic,
            strategy_evolver=evolver, knowledge_loop=kl,
        )
        assert bridge._bus is bus
        assert bridge._episodic is episodic
        assert bridge._evolver is evolver
        assert bridge._knowledge_loop is kl

    def test_works_without_strategy_evolver(self):
        bridge = _make_bridge(strategy_evolver=None)
        assert bridge._evolver is None

    def test_works_without_knowledge_loop(self):
        bridge = _make_bridge(knowledge_loop=None)
        assert bridge._knowledge_loop is None


# ── Lifecycle Tests ──────────────────────────────────────────────────────


class TestInquiryBridgeLifecycle:
    def test_start_subscribes_5_topics(self):
        bridge = _make_bridge()
        bridge.start()
        assert bridge._bus.subscribe.call_count == 5
        topics = {c.args[0] for c in bridge._bus.subscribe.call_args_list}
        assert topics == {
            "inquiry.started", "inquiry.completed",
            "inquiry.failed", "inquiry.insight_generated",
            "arena.tournament_completed",
        }

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self):
        bridge = _make_bridge()
        bridge.start()
        await bridge.stop()
        assert bridge._bus.unsubscribe.call_count == 5
        assert bridge._running is False

    def test_double_start_is_noop(self):
        bridge = _make_bridge()
        bridge.start()
        bridge.start()
        # Still only 5 subscribes
        assert bridge._bus.subscribe.call_count == 5


# ── inquiry.started ──────────────────────────────────────────────────────


class TestOnInquiryStarted:
    @pytest.mark.asyncio
    async def test_stores_observation_episode(self):
        bridge = _make_bridge()
        env = _envelope("inquiry.started", {
            "inquiry_id": "inq_1", "goal_id": "g_1", "goal": "Research AI",
        })
        await bridge._on_inquiry_started(env)
        bridge._episodic.store.assert_called_once()
        ep = bridge._episodic.store.call_args.args[0]
        assert ep.episode_type == "observation"
        assert bridge._episodes_stored == 1

    @pytest.mark.asyncio
    async def test_episode_has_correct_ids(self):
        bridge = _make_bridge()
        env = _envelope("inquiry.started", {
            "inquiry_id": "inq_42", "goal_id": "g_42", "goal": "test",
        })
        await bridge._on_inquiry_started(env)
        ep = bridge._episodic.store.call_args.args[0]
        assert ep.goal_id == "g_42"
        assert ep.inquiry_id == "inq_42"


# ── inquiry.completed ────────────────────────────────────────────────────


class TestOnInquiryCompleted:
    @pytest.mark.asyncio
    async def test_stores_synthesis_episode(self):
        bridge = _make_bridge()
        env = _envelope("inquiry.completed", {
            "inquiry_id": "inq_1", "goal_id": "g_1",
            "status": "completed", "iterations": 3, "insights": 2,
        })
        await bridge._on_inquiry_completed(env)
        bridge._episodic.store.assert_called_once()
        ep = bridge._episodic.store.call_args.args[0]
        assert ep.episode_type == "synthesis"
        assert "completed" in ep.summary
        assert bridge._episodes_stored == 1

    @pytest.mark.asyncio
    async def test_records_strategy_outcome_success(self):
        evolver = _make_evolver("depth_first")
        bridge = _make_bridge(strategy_evolver=evolver)
        env = _envelope("inquiry.completed", {
            "inquiry_id": "inq_1", "goal_id": "g_1",
            "status": "completed", "iterations": 3, "insights": 2,
        })
        await bridge._on_inquiry_completed(env)
        evolver.record_outcome.assert_called_once()
        outcome = evolver.record_outcome.call_args.args[0]
        assert outcome.strategy_name == "depth_first"
        assert outcome.success is True
        assert outcome.insights_count == 2
        assert bridge._outcomes_recorded == 1

    @pytest.mark.asyncio
    async def test_records_outcome_failure_when_zero_insights(self):
        evolver = _make_evolver("breadth_first")
        bridge = _make_bridge(strategy_evolver=evolver)
        env = _envelope("inquiry.completed", {
            "inquiry_id": "inq_1", "goal_id": "g_1",
            "status": "completed", "iterations": 3, "insights": 0,
        })
        await bridge._on_inquiry_completed(env)
        outcome = evolver.record_outcome.call_args.args[0]
        assert outcome.success is False

    @pytest.mark.asyncio
    async def test_triggers_knowledge_consolidation(self):
        kl = _make_knowledge_loop()
        bridge = _make_bridge(knowledge_loop=kl)
        env = _envelope("inquiry.completed", {
            "inquiry_id": "inq_1", "goal_id": "g_1",
            "status": "completed", "iterations": 1, "insights": 1,
        })
        await bridge._on_inquiry_completed(env)
        kl.trigger_consolidation.assert_awaited_once()
        assert bridge._consolidations_triggered == 1

    @pytest.mark.asyncio
    async def test_no_crash_when_evolver_none(self):
        bridge = _make_bridge(strategy_evolver=None)
        env = _envelope("inquiry.completed", {
            "inquiry_id": "inq_1", "goal_id": "g_1",
            "status": "completed", "iterations": 1, "insights": 1,
        })
        await bridge._on_inquiry_completed(env)
        assert bridge._outcomes_recorded == 0

    @pytest.mark.asyncio
    async def test_no_crash_when_knowledge_loop_none(self):
        bridge = _make_bridge(knowledge_loop=None)
        env = _envelope("inquiry.completed", {
            "inquiry_id": "inq_1", "goal_id": "g_1",
            "status": "completed", "iterations": 1, "insights": 1,
        })
        await bridge._on_inquiry_completed(env)
        assert bridge._consolidations_triggered == 0

    @pytest.mark.asyncio
    async def test_no_crash_when_no_current_strategy(self):
        evolver = _make_evolver(current_strategy=None)
        bridge = _make_bridge(strategy_evolver=evolver)
        env = _envelope("inquiry.completed", {
            "inquiry_id": "inq_1", "goal_id": "g_1",
            "status": "completed", "iterations": 1, "insights": 1,
        })
        await bridge._on_inquiry_completed(env)
        evolver.record_outcome.assert_not_called()
        assert bridge._outcomes_recorded == 0


# ── inquiry.failed ───────────────────────────────────────────────────────


class TestOnInquiryFailed:
    @pytest.mark.asyncio
    async def test_stores_failure_episode(self):
        bridge = _make_bridge()
        env = _envelope("inquiry.failed", {
            "inquiry_id": "inq_1", "goal_id": "g_1", "iteration": 2,
        })
        await bridge._on_inquiry_failed(env)
        bridge._episodic.store.assert_called_once()
        ep = bridge._episodic.store.call_args.args[0]
        assert ep.episode_type == "observation"
        assert "failed" in ep.summary

    @pytest.mark.asyncio
    async def test_records_negative_outcome(self):
        evolver = _make_evolver("breadth_first")
        bridge = _make_bridge(strategy_evolver=evolver)
        env = _envelope("inquiry.failed", {
            "inquiry_id": "inq_1", "goal_id": "g_1", "iteration": 2,
        })
        await bridge._on_inquiry_failed(env)
        outcome = evolver.record_outcome.call_args.args[0]
        assert outcome.success is False
        assert outcome.insights_count == 0


# ── inquiry.insight_generated ────────────────────────────────────────────


class TestOnInsightGenerated:
    @pytest.mark.asyncio
    async def test_stores_synthesis_episode(self):
        bridge = _make_bridge()
        env = _envelope("inquiry.insight_generated", {
            "inquiry_id": "inq_1", "insight_id": "ins_1",
            "headline": "AI adoption accelerating",
        })
        await bridge._on_insight_generated(env)
        bridge._episodic.store.assert_called_once()
        ep = bridge._episodic.store.call_args.args[0]
        assert ep.episode_type == "synthesis"

    @pytest.mark.asyncio
    async def test_summary_includes_headline(self):
        bridge = _make_bridge()
        env = _envelope("inquiry.insight_generated", {
            "inquiry_id": "inq_1", "insight_id": "ins_1",
            "headline": "Market is growing",
        })
        await bridge._on_insight_generated(env)
        ep = bridge._episodic.store.call_args.args[0]
        assert "Market is growing" in ep.summary


# ── Bus Events ───────────────────────────────────────────────────────────


# ── arena.tournament_completed ──────────────────────────────────────────


class TestOnArenaTournamentCompleted:
    @pytest.mark.asyncio
    async def test_arena_completed_stores_episode(self):
        bridge = _make_bridge()
        env = _envelope("arena.tournament_completed", {
            "arena_id": "arena_abc",
            "goal_id": "g_1",
            "winner_id": "agent_a",
            "sycophancy_detected": False,
            "match_count": 3,
        })
        await bridge._on_arena_tournament_completed(env)
        bridge._episodic.store.assert_called_once()
        ep = bridge._episodic.store.call_args.args[0]
        assert ep.episode_type == "synthesis"
        assert "winner=agent_a" in ep.summary
        assert bridge._episodes_stored == 1

    @pytest.mark.asyncio
    async def test_arena_completed_sycophancy_in_summary(self):
        bridge = _make_bridge()
        env = _envelope("arena.tournament_completed", {
            "arena_id": "arena_def",
            "goal_id": "g_2",
            "winner_id": "agent_b",
            "sycophancy_detected": True,
        })
        await bridge._on_arena_tournament_completed(env)
        ep = bridge._episodic.store.call_args.args[0]
        assert "sycophancy_detected=True" in ep.summary

    @pytest.mark.asyncio
    async def test_arena_completed_no_crash_on_store_failure(self):
        bridge = _make_bridge()
        bridge._episodic.store.side_effect = RuntimeError("DB error")
        env = _envelope("arena.tournament_completed", {
            "arena_id": "arena_xyz",
            "goal_id": "g_3",
            "winner_id": "agent_c",
            "sycophancy_detected": False,
        })
        # Should not raise
        await bridge._on_arena_tournament_completed(env)
        assert bridge._episodes_stored == 0


# ── Bus Events ───────────────────────────────────────────────────────────


class TestBridgeEvents:
    @pytest.mark.asyncio
    async def test_publishes_strategy_outcome_event(self):
        evolver = _make_evolver("breadth_first")
        bridge = _make_bridge(strategy_evolver=evolver)
        env = _envelope("inquiry.completed", {
            "inquiry_id": "inq_1", "goal_id": "g_1",
            "status": "completed", "iterations": 1, "insights": 2,
        })
        await bridge._on_inquiry_completed(env)
        publish_calls = bridge._bus.publish.call_args_list
        bridge_calls = [
            c for c in publish_calls
            if c.args[0].topic == "bridge.strategy_outcome_recorded"
        ]
        assert len(bridge_calls) == 1
        payload = bridge_calls[0].args[0].payload
        assert payload["strategy_name"] == "breadth_first"
        assert payload["success"] is True
        assert payload["insights_count"] == 2

    @pytest.mark.asyncio
    async def test_publishes_negative_outcome_event_on_failure(self):
        evolver = _make_evolver("depth_first")
        bridge = _make_bridge(strategy_evolver=evolver)
        env = _envelope("inquiry.failed", {
            "inquiry_id": "inq_1", "goal_id": "g_1", "iteration": 2,
        })
        await bridge._on_inquiry_failed(env)
        publish_calls = bridge._bus.publish.call_args_list
        bridge_calls = [
            c for c in publish_calls
            if c.args[0].topic == "bridge.strategy_outcome_recorded"
        ]
        assert len(bridge_calls) == 1
        assert bridge_calls[0].args[0].payload["success"] is False


# ── Status ───────────────────────────────────────────────────────────────


class TestBridgeStatus:
    def test_status_before_start(self):
        bridge = _make_bridge()
        s = bridge.status()
        assert s["running"] is False
        assert s["episodes_stored"] == 0
        assert s["outcomes_recorded"] == 0
        assert s["consolidations_triggered"] == 0

    def test_status_after_start(self):
        bridge = _make_bridge()
        bridge.start()
        assert bridge.status()["running"] is True


# ── Integration ──────────────────────────────────────────────────────────


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Full started → insight → completed lifecycle."""
        evolver = _make_evolver("hypothesis_driven")
        kl = _make_knowledge_loop()
        bridge = _make_bridge(strategy_evolver=evolver, knowledge_loop=kl)

        # inquiry.started
        await bridge._on_inquiry_started(_envelope("inquiry.started", {
            "inquiry_id": "inq_1", "goal_id": "g_1", "goal": "Analyze market",
        }))

        # inquiry.insight_generated
        await bridge._on_insight_generated(_envelope("inquiry.insight_generated", {
            "inquiry_id": "inq_1", "insight_id": "ins_1",
            "headline": "Market is growing",
        }))

        # inquiry.completed
        await bridge._on_inquiry_completed(_envelope("inquiry.completed", {
            "inquiry_id": "inq_1", "goal_id": "g_1",
            "status": "completed", "iterations": 3, "insights": 1,
        }))

        # Verify episodes: started(observation) + insight(synthesis) + completed(synthesis) = 3
        assert bridge._episodes_stored == 3
        assert bridge._episodic.store.call_count == 3

        # Verify strategy outcome recorded
        assert bridge._outcomes_recorded == 1
        evolver.record_outcome.assert_called_once()

        # Verify knowledge consolidation triggered
        assert bridge._consolidations_triggered == 1
        kl.trigger_consolidation.assert_awaited_once()

        # Verify bus event published
        publish_calls = bridge._bus.publish.call_args_list
        bridge_events = [
            c for c in publish_calls
            if c.args[0].topic == "bridge.strategy_outcome_recorded"
        ]
        assert len(bridge_events) == 1

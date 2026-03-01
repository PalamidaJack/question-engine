"""Tests for Phase C: GoalSynthesizer service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.envelope import Envelope
from qe.models.goal import (
    GoalDecomposition,
    GoalState,
    Subtask,
    SubtaskResult,
)
from qe.services.synthesizer.service import GoalResult, GoalSynthesizer, SynthesisInput

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_bus():
    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    bus.publish = MagicMock()
    return bus


def _make_goal_store(state: GoalState | None = None):
    store = AsyncMock()
    store.load_goal = AsyncMock(return_value=state)
    store.save_goal = AsyncMock()
    return store


def _make_completed_goal(goal_id: str = "g1") -> GoalState:
    subtasks = [
        Subtask(subtask_id="s1", description="Research topic", task_type="research"),
        Subtask(subtask_id="s2", description="Analyze data", task_type="analysis"),
    ]
    state = GoalState(
        goal_id=goal_id,
        description="Research quantum computing advances",
        status="completed",
        decomposition=GoalDecomposition(
            goal_id=goal_id,
            original_description="Research quantum computing advances",
            strategy="Two-phase: research then analyze",
            subtasks=subtasks,
        ),
        subtask_states={"s1": "completed", "s2": "completed"},
        subtask_results={
            "s1": SubtaskResult(
                subtask_id="s1", goal_id=goal_id, status="completed",
                output={"content": "Quantum computing has advanced in error correction"},
                model_used="gpt-4o", latency_ms=1500, cost_usd=0.01,
            ),
            "s2": SubtaskResult(
                subtask_id="s2", goal_id=goal_id, status="completed",
                output={"content": "Key insight: topological qubits show most promise"},
                model_used="gpt-4o", latency_ms=2000, cost_usd=0.02,
                tool_calls=[{"tool": "web_search"}],
            ),
        },
    )
    return state


def _mock_synthesis():
    return SynthesisInput(
        summary="Quantum computing has advanced significantly, especially in error correction.",
        key_findings=["Error correction improved", "Topological qubits promising"],
        confidence=0.85,
        recommendations=["Invest in topological qubit research"],
    )


# ── Model Validation ────────────────────────────────────────────────────────


class TestModels:
    def test_goal_result_defaults(self):
        result = GoalResult(goal_id="g1", summary="test")
        assert result.confidence == 0.5
        assert result.findings == []
        assert result.total_cost_usd == 0.0

    def test_goal_result_full(self):
        result = GoalResult(
            goal_id="g1", summary="Full result",
            findings=[{"finding": "A"}], confidence=0.9,
            recommendations=["Do X"], total_cost_usd=0.05,
            subtask_count=3,
        )
        assert result.subtask_count == 3
        assert result.confidence == 0.9

    def test_synthesis_input_validation(self):
        synth = SynthesisInput(
            summary="test", key_findings=["a", "b"],
            confidence=0.7, recommendations=["do X"],
        )
        assert synth.confidence == 0.7

    def test_synthesis_input_confidence_clamped(self):
        with pytest.raises(ValueError):
            SynthesisInput(summary="test", confidence=1.5)


# ── Start/Stop ───────────────────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_subscribes(self):
        bus = _make_bus()
        synth = GoalSynthesizer(bus=bus, goal_store=_make_goal_store())
        await synth.start()
        bus.subscribe.assert_called_once_with("goals.completed", synth._on_goal_completed)
        assert synth._running is True

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self):
        bus = _make_bus()
        synth = GoalSynthesizer(bus=bus, goal_store=_make_goal_store())
        await synth.start()
        await synth.stop()
        bus.unsubscribe.assert_called_once_with("goals.completed", synth._on_goal_completed)
        assert synth._running is False


# ── Subtask Collection ───────────────────────────────────────────────────────


class TestCollectSubtaskSummaries:
    def test_collects_completed_subtasks(self):
        state = _make_completed_goal()
        synth = GoalSynthesizer(bus=_make_bus(), goal_store=_make_goal_store())
        summaries = synth._collect_subtask_summaries(state)
        assert len(summaries) == 2
        assert summaries[0]["task_type"] == "research"
        assert summaries[1]["status"] == "completed"

    def test_handles_no_decomposition(self):
        state = GoalState(goal_id="g1")
        synth = GoalSynthesizer(bus=_make_bus(), goal_store=_make_goal_store())
        summaries = synth._collect_subtask_summaries(state)
        assert summaries == []


# ── Provenance ───────────────────────────────────────────────────────────────


class TestBuildProvenance:
    def test_includes_model_cost_latency(self):
        state = _make_completed_goal()
        synth = GoalSynthesizer(bus=_make_bus(), goal_store=_make_goal_store())
        prov = synth._build_provenance(state)
        assert len(prov) == 2
        assert prov[0]["model_used"] == "gpt-4o"
        assert prov[0]["cost_usd"] == 0.01
        assert prov[1]["tool_calls_count"] == 1


# ── LLM Synthesis ────────────────────────────────────────────────────────────


class TestSynthesis:
    @pytest.mark.asyncio
    async def test_synthesize_returns_goal_result(self):
        state = _make_completed_goal()
        store = _make_goal_store(state)
        bus = _make_bus()
        synth = GoalSynthesizer(bus=bus, goal_store=store, model="gpt-4o")

        mock_synthesis = _mock_synthesis()
        with patch("qe.services.synthesizer.service.instructor") as mock_instr:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_synthesis)
            mock_instr.from_litellm.return_value = mock_client

            result = await synth.synthesize("g1")

        assert isinstance(result, GoalResult)
        assert result.goal_id == "g1"
        assert result.confidence == 0.85
        assert len(result.findings) == 2
        assert result.total_cost_usd == pytest.approx(0.03)
        assert result.subtask_count == 2
        store.save_goal.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_synthesize_with_dialectic(self):
        state = _make_completed_goal()
        store = _make_goal_store(state)
        bus = _make_bus()

        dialectic = AsyncMock()
        report = MagicMock()
        report.revised_confidence = 0.72
        report.counterarguments = [MagicMock()]
        report.perspectives = [MagicMock(), MagicMock()]
        dialectic.full_dialectic = AsyncMock(return_value=report)

        synth = GoalSynthesizer(
            bus=bus, goal_store=store, dialectic_engine=dialectic,
        )

        mock_synthesis = _mock_synthesis()
        with patch("qe.services.synthesizer.service.instructor") as mock_instr:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_synthesis)
            mock_instr.from_litellm.return_value = mock_client

            result = await synth.synthesize("g1")

        assert result.confidence == 0.72
        assert result.dialectic_review is not None
        assert result.dialectic_review["revised_confidence"] == 0.72
        assert result.dialectic_review["counterarguments_count"] == 1

    @pytest.mark.asyncio
    async def test_result_stored_in_metadata(self):
        state = _make_completed_goal()
        store = _make_goal_store(state)
        synth = GoalSynthesizer(bus=_make_bus(), goal_store=store)

        with patch("qe.services.synthesizer.service.instructor") as mock_instr:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_synthesis())
            mock_instr.from_litellm.return_value = mock_client

            await synth.synthesize("g1")

        saved_state = store.save_goal.call_args[0][0]
        assert "goal_result" in saved_state.metadata

    @pytest.mark.asyncio
    async def test_goal_not_found_raises(self):
        store = _make_goal_store(None)
        synth = GoalSynthesizer(bus=_make_bus(), goal_store=store)
        with pytest.raises(ValueError, match="Goal not found"):
            await synth.synthesize("nonexistent")


# ── Event Handling ───────────────────────────────────────────────────────────


class TestOnGoalCompleted:
    @pytest.mark.asyncio
    async def test_publishes_synthesized_event(self):
        state = _make_completed_goal()
        store = _make_goal_store(state)
        bus = _make_bus()
        synth = GoalSynthesizer(bus=bus, goal_store=store)

        with patch("qe.services.synthesizer.service.instructor") as mock_instr:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_synthesis())
            mock_instr.from_litellm.return_value = mock_client

            envelope = Envelope(
                topic="goals.completed",
                source_service_id="dispatcher",
                payload={"goal_id": "g1", "subtask_count": 2},
            )
            await synth._on_goal_completed(envelope)

        # Find the goals.synthesized publish call
        calls = bus.publish.call_args_list
        synth_call = [c for c in calls if c[0][0].topic == "goals.synthesized"]
        assert len(synth_call) == 1
        payload = synth_call[0][0][0].payload
        assert payload["goal_id"] == "g1"
        assert payload["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_publishes_synthesis_failed_on_error(self):
        store = _make_goal_store(None)
        bus = _make_bus()
        synth = GoalSynthesizer(bus=bus, goal_store=store)

        envelope = Envelope(
            topic="goals.completed",
            source_service_id="dispatcher",
            payload={"goal_id": "g1", "subtask_count": 2},
        )
        await synth._on_goal_completed(envelope)

        calls = bus.publish.call_args_list
        fail_call = [c for c in calls if c[0][0].topic == "goals.synthesis_failed"]
        assert len(fail_call) == 1
        assert fail_call[0][0][0].payload["goal_id"] == "g1"


# ── Bus Schema Registration ─────────────────────────────────────────────────


class TestBusSchemas:
    def test_synthesized_topic_registered(self):
        from qe.bus.protocol import TOPICS
        assert "goals.synthesized" in TOPICS

    def test_synthesis_failed_topic_registered(self):
        from qe.bus.protocol import TOPICS
        assert "goals.synthesis_failed" in TOPICS

    def test_synthesized_schema_in_registry(self):
        from qe.bus.schemas import TOPIC_SCHEMAS
        assert "goals.synthesized" in TOPIC_SCHEMAS

    def test_synthesis_failed_schema_in_registry(self):
        from qe.bus.schemas import TOPIC_SCHEMAS
        assert "goals.synthesis_failed" in TOPIC_SCHEMAS

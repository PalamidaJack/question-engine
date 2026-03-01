"""Integration tests for the full goal orchestration pipeline.

Tests the end-to-end flow: plan → dispatch → execute → synthesize
using MemoryBus + GoalStore + Dispatcher + ExecutorService + GoalSynthesizer
with mocked LLM calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.goal import (
    ExecutionContract,
    GoalDecomposition,
    GoalState,
    Subtask,
    SubtaskResult,
)
from qe.services.dispatcher.service import Dispatcher
from qe.services.executor.service import ExecutorService
from qe.services.synthesizer.service import GoalSynthesizer, SynthesisInput

# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_litellm_response(content="LLM result"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.model_dump = MagicMock(return_value={"role": "assistant", "content": content})
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _mock_synthesis():
    return SynthesisInput(
        summary="Synthesized: quantum computing advances are significant",
        key_findings=["Error correction improved", "Topological qubits promising"],
        confidence=0.82,
        recommendations=["Monitor topological qubit research"],
    )


class FakeGoalStore:
    """Minimal in-memory goal store for integration tests."""

    def __init__(self):
        self._goals: dict[str, GoalState] = {}
        self._checkpoint_counter = 0

    async def save_goal(self, state: GoalState) -> None:
        self._goals[state.goal_id] = state

    async def load_goal(self, goal_id: str) -> GoalState | None:
        return self._goals.get(goal_id)

    async def save_checkpoint(self, goal_id: str, state: GoalState) -> str:
        self._checkpoint_counter += 1
        return f"ckpt_{self._checkpoint_counter}"

    async def list_active_goals(self) -> list[GoalState]:
        return [s for s in self._goals.values() if s.status in ("planning", "executing")]


# ── Integration Tests ────────────────────────────────────────────────────────


class TestGoalPipelineE2E:
    @pytest.mark.asyncio
    async def test_two_subtask_goal_plan_dispatch_execute_synthesize(self):
        """Full pipeline: plan → dispatch → execute → synthesize for 2 subtasks."""
        bus = MemoryBus()
        store = FakeGoalStore()

        dispatcher = Dispatcher(bus=bus, goal_store=store)
        executor = ExecutorService(
            bus=bus, substrate=None, model="test-model",
        )

        synthesized_events = []

        def _capture_synthesized(envelope):
            if envelope.topic == "goals.synthesized":
                synthesized_events.append(envelope)

        bus.subscribe("goals.synthesized", _capture_synthesized)

        synth = GoalSynthesizer(bus=bus, goal_store=store, model="test-model")
        await synth.start()

        # Create a planned goal
        subtasks = [
            Subtask(
                subtask_id="s1", description="Research quantum computing",
                task_type="research",
            ),
            Subtask(
                subtask_id="s2", description="Analyze findings",
                task_type="analysis", depends_on=["s1"],
            ),
        ]
        state = GoalState(
            goal_id="g1",
            description="Research quantum computing advances",
            status="executing",
            decomposition=GoalDecomposition(
                goal_id="g1",
                original_description="Research quantum computing advances",
                strategy="Research then analyze",
                subtasks=subtasks,
            ),
            subtask_states={"s1": "pending", "s2": "pending"},
        )

        # Mock LLM for executor and synthesizer
        mock_response = _mock_litellm_response(
            "Research findings: significant advances"
        )

        with patch("qe.services.executor.service.litellm") as mock_exec_llm, \
             patch("qe.services.executor.service.get_rate_limiter") as mock_rl, \
             patch("qe.services.synthesizer.service.instructor") as mock_instr:

            mock_rl.return_value.acquire = AsyncMock()
            mock_exec_llm.acompletion = AsyncMock(return_value=mock_response)
            mock_exec_llm.completion_cost = MagicMock(return_value=0.001)

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_synthesis())
            mock_instr.from_litellm.return_value = mock_client

            # Start executor
            await executor.start()

            # Submit goal to dispatcher (this dispatches s1)
            await dispatcher.submit_goal(state)

            # Allow event processing
            await asyncio.sleep(0.1)

            # s1 should have been dispatched and executed
            # Simulate the verification bypass — directly handle completion
            s1_result = SubtaskResult(
                subtask_id="s1", goal_id="g1", status="completed",
                output={"content": "Research findings about quantum error correction"},
                model_used="test-model", latency_ms=100,
            )
            await dispatcher.handle_subtask_completed("g1", s1_result)

            # s2 should now be dispatched
            await asyncio.sleep(0.1)

            s2_result = SubtaskResult(
                subtask_id="s2", goal_id="g1", status="completed",
                output={"content": "Analysis: topological qubits are most promising"},
                model_used="test-model", latency_ms=150,
            )
            await dispatcher.handle_subtask_completed("g1", s2_result)

            # Allow synthesis event to process
            await asyncio.sleep(0.1)

        await executor.stop()
        await synth.stop()

        # Verify goal completed and synthesized
        loaded = await store.load_goal("g1")
        assert loaded is not None
        assert loaded.status == "completed"
        assert "goal_result" in loaded.metadata

        goal_result = loaded.metadata["goal_result"]
        assert goal_result["confidence"] == 0.82
        assert len(goal_result["findings"]) == 2

    @pytest.mark.asyncio
    async def test_subtask_failure_retry_then_succeed(self):
        """Failed subtask gets retried by dispatcher and eventually succeeds."""
        bus = MemoryBus()
        store = FakeGoalStore()
        dispatcher = Dispatcher(bus=bus, goal_store=store)

        subtasks = [
            Subtask(
                subtask_id="s1", description="Research", task_type="research",
                contract=ExecutionContract(max_retries=2),
            ),
        ]
        state = GoalState(
            goal_id="g1", description="Test", status="executing",
            decomposition=GoalDecomposition(
                goal_id="g1", original_description="Test",
                strategy="test", subtasks=subtasks,
            ),
            subtask_states={"s1": "dispatched"},
        )
        dispatcher._active_goals["g1"] = state
        await store.save_goal(state)

        # First attempt fails
        failed_result = SubtaskResult(
            subtask_id="s1", goal_id="g1", status="failed",
            output={"error": "timeout"},
        )
        await dispatcher.handle_subtask_completed("g1", failed_result)

        # Should be reset for retry (pending then redispatched)
        assert state.subtask_states["s1"] in ("pending", "dispatched")
        assert state.metadata["retry_counts"]["s1"] == 1

        # Simulate redispatch + success
        state.subtask_states["s1"] = "dispatched"
        success_result = SubtaskResult(
            subtask_id="s1", goal_id="g1", status="completed",
            output={"content": "Success on retry"},
        )
        await dispatcher.handle_subtask_completed("g1", success_result)

        assert state.status == "completed"

    @pytest.mark.asyncio
    async def test_synthesis_with_dialectic_review(self):
        """Synthesis with dialectic engine revises confidence."""
        bus = MemoryBus()
        store = FakeGoalStore()

        dialectic = AsyncMock()
        report = MagicMock()
        report.revised_confidence = 0.65
        report.counterarguments = [MagicMock()]
        report.perspectives = []
        dialectic.full_dialectic = AsyncMock(return_value=report)

        synth = GoalSynthesizer(
            bus=bus, goal_store=store, dialectic_engine=dialectic,
        )

        # Create completed goal in store
        subtasks = [
            Subtask(subtask_id="s1", description="Research", task_type="research"),
        ]
        state = GoalState(
            goal_id="g1", description="Test", status="completed",
            decomposition=GoalDecomposition(
                goal_id="g1", original_description="Test",
                strategy="test", subtasks=subtasks,
            ),
            subtask_states={"s1": "completed"},
            subtask_results={
                "s1": SubtaskResult(
                    subtask_id="s1", goal_id="g1", status="completed",
                    output={"content": "Findings"}, cost_usd=0.01,
                ),
            },
        )
        await store.save_goal(state)

        with patch("qe.services.synthesizer.service.instructor") as mock_instr:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_synthesis())
            mock_instr.from_litellm.return_value = mock_client

            result = await synth.synthesize("g1")

        assert result.confidence == 0.65  # revised by dialectic
        assert result.dialectic_review is not None

    @pytest.mark.asyncio
    async def test_get_result_from_metadata(self):
        """After synthesis, goal_result is stored in metadata and retrievable."""
        bus = MemoryBus()
        store = FakeGoalStore()
        synth = GoalSynthesizer(bus=bus, goal_store=store)

        subtasks = [
            Subtask(subtask_id="s1", description="Research", task_type="research"),
        ]
        state = GoalState(
            goal_id="g1", description="Test", status="completed",
            decomposition=GoalDecomposition(
                goal_id="g1", original_description="Test",
                strategy="test", subtasks=subtasks,
            ),
            subtask_states={"s1": "completed"},
            subtask_results={
                "s1": SubtaskResult(
                    subtask_id="s1", goal_id="g1", status="completed",
                    output={"content": "Findings"}, cost_usd=0.01,
                ),
            },
        )
        await store.save_goal(state)

        with patch("qe.services.synthesizer.service.instructor") as mock_instr:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=_mock_synthesis())
            mock_instr.from_litellm.return_value = mock_client

            await synth.synthesize("g1")

        loaded = await store.load_goal("g1")
        assert loaded is not None
        result = loaded.metadata.get("goal_result")
        assert result is not None
        assert result["summary"].startswith("Synthesized:")
        assert result["subtask_count"] == 1

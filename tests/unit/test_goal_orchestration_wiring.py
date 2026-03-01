"""Tests for Phase D: Goal orchestration pipeline wiring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.bus.protocol import TOPICS
from qe.bus.schemas import (
    TOPIC_SCHEMAS,
    GoalCompletedPayload,
)
from qe.models.goal import (
    ExecutionContract,
    GoalDecomposition,
    GoalState,
    Subtask,
    SubtaskResult,
)
from qe.services.dispatcher.service import Dispatcher

# ── Topic/Schema Registration ───────────────────────────────────────────────


class TestTopicRegistration:
    def test_contract_violated_in_topics(self):
        assert "tasks.contract_violated" in TOPICS

    def test_synthesized_in_topics(self):
        assert "goals.synthesized" in TOPICS

    def test_synthesis_failed_in_topics(self):
        assert "goals.synthesis_failed" in TOPICS

    def test_all_new_topics_have_schemas(self):
        for topic in ("tasks.contract_violated", "goals.synthesized", "goals.synthesis_failed"):
            assert topic in TOPIC_SCHEMAS, f"{topic} not in TOPIC_SCHEMAS"

    def test_goal_completed_has_results_summary(self):
        payload = GoalCompletedPayload(goal_id="g1", subtask_count=2)
        assert payload.subtask_results_summary == {}

        payload = GoalCompletedPayload(
            goal_id="g1", subtask_count=2,
            subtask_results_summary={"s1": {"status": "completed", "content_preview": "test"}},
        )
        assert "s1" in payload.subtask_results_summary


# ── Dispatcher Retry ─────────────────────────────────────────────────────────


def _make_bus():
    bus = MagicMock()
    bus.publish = MagicMock()
    return bus


def _make_goal_store():
    store = AsyncMock()
    store.save_goal = AsyncMock()
    store.save_checkpoint = AsyncMock(return_value="ckpt_1")
    return store


def _make_dispatched_goal(goal_id="g1", num_subtasks=4, max_retries=2):
    subtasks = []
    for i in range(num_subtasks):
        subtasks.append(Subtask(
            subtask_id=f"s{i+1}",
            description=f"Subtask {i+1}",
            task_type="research",
            contract=ExecutionContract(max_retries=max_retries),
        ))

    state = GoalState(
        goal_id=goal_id,
        description="Test goal",
        status="executing",
        decomposition=GoalDecomposition(
            goal_id=goal_id,
            original_description="Test goal",
            strategy="test",
            subtasks=subtasks,
        ),
        subtask_states={f"s{i+1}": "pending" for i in range(num_subtasks)},
    )
    return state


class TestDispatcherRetry:
    @pytest.mark.asyncio
    async def test_failed_subtask_retried(self):
        """Failed subtask gets reset to pending and redispatched."""
        bus = _make_bus()
        store = _make_goal_store()
        dispatcher = Dispatcher(bus=bus, goal_store=store)

        state = _make_dispatched_goal(num_subtasks=2, max_retries=2)
        state.subtask_states["s1"] = "dispatched"
        dispatcher._active_goals["g1"] = state

        failed_result = SubtaskResult(
            subtask_id="s1", goal_id="g1", status="failed",
            output={"error": "timeout"},
        )

        await dispatcher.handle_subtask_completed("g1", failed_result)

        # Subtask should be reset for retry (pending then redispatched)
        assert state.subtask_states["s1"] in ("pending", "dispatched")
        assert state.metadata["retry_counts"]["s1"] == 1

    @pytest.mark.asyncio
    async def test_retries_exhausted_checks_threshold(self):
        """After exhausting retries, goal fails if >50% subtasks failed."""
        bus = _make_bus()
        store = _make_goal_store()
        dispatcher = Dispatcher(bus=bus, goal_store=store)

        state = _make_dispatched_goal(num_subtasks=2, max_retries=0)
        state.subtask_states["s1"] = "failed"
        state.subtask_results["s1"] = SubtaskResult(
            subtask_id="s1", goal_id="g1", status="failed",
            output={"error": "err"},
        )
        state.metadata["retry_counts"] = {"s1": 0}
        dispatcher._active_goals["g1"] = state

        failed_result = SubtaskResult(
            subtask_id="s2", goal_id="g1", status="failed",
            output={"error": "err2"},
        )

        await dispatcher.handle_subtask_completed("g1", failed_result)

        # >50% failed → goal should fail
        assert state.status == "failed"
        published = [c[0][0] for c in bus.publish.call_args_list]
        fail_events = [e for e in published if e.topic == "goals.failed"]
        assert len(fail_events) == 1

    @pytest.mark.asyncio
    async def test_completed_goal_has_results_summary(self):
        """goals.completed payload includes subtask_results_summary."""
        bus = _make_bus()
        store = _make_goal_store()
        dispatcher = Dispatcher(bus=bus, goal_store=store)

        state = _make_dispatched_goal(num_subtasks=1, max_retries=0)
        state.subtask_states["s1"] = "dispatched"
        dispatcher._active_goals["g1"] = state

        result = SubtaskResult(
            subtask_id="s1", goal_id="g1", status="completed",
            output={"content": "Research findings about quantum computing"},
        )
        await dispatcher.handle_subtask_completed("g1", result)

        published = [c[0][0] for c in bus.publish.call_args_list]
        completed = [e for e in published if e.topic == "goals.completed"]
        assert len(completed) == 1
        payload = completed[0].payload
        assert "subtask_results_summary" in payload
        assert "s1" in payload["subtask_results_summary"]

    @pytest.mark.asyncio
    async def test_is_dispatchable_deps_met(self):
        """Subtask is dispatchable when dependencies are completed."""
        store = _make_goal_store()
        dispatcher = Dispatcher(bus=_make_bus(), goal_store=store)

        subtasks = [
            Subtask(subtask_id="s1", description="A", task_type="research"),
            Subtask(subtask_id="s2", description="B", task_type="analysis", depends_on=["s1"]),
        ]
        state = GoalState(
            goal_id="g1", status="executing",
            decomposition=GoalDecomposition(
                goal_id="g1", original_description="test",
                strategy="test", subtasks=subtasks,
            ),
            subtask_states={"s1": "completed", "s2": "pending"},
        )

        assert dispatcher._is_dispatchable(state, "s2") is True
        assert dispatcher._is_dispatchable(state, "s1") is True

    @pytest.mark.asyncio
    async def test_is_dispatchable_deps_not_met(self):
        """Subtask is not dispatchable when dependencies are incomplete."""
        store = _make_goal_store()
        dispatcher = Dispatcher(bus=_make_bus(), goal_store=store)

        subtasks = [
            Subtask(subtask_id="s1", description="A", task_type="research"),
            Subtask(subtask_id="s2", description="B", task_type="analysis", depends_on=["s1"]),
        ]
        state = GoalState(
            goal_id="g1", status="executing",
            decomposition=GoalDecomposition(
                goal_id="g1", original_description="test",
                strategy="test", subtasks=subtasks,
            ),
            subtask_states={"s1": "pending", "s2": "pending"},
        )

        assert dispatcher._is_dispatchable(state, "s2") is False


# ── Feature Flag ─────────────────────────────────────────────────────────────


class TestFeatureFlag:
    def test_goal_orchestration_flag_defined(self):
        from qe.runtime.feature_flags import FeatureFlagStore
        store = FeatureFlagStore()
        store.define("goal_orchestration", enabled=False)
        assert store.is_enabled("goal_orchestration") is False
        store.enable("goal_orchestration")
        assert store.is_enabled("goal_orchestration") is True

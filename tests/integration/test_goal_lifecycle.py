"""Test 2F: End-to-end goal lifecycle with real MemoryBus, no LLM."""

import asyncio
from pathlib import Path

import aiosqlite
import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.envelope import Envelope
from qe.models.goal import (
    GoalDecomposition,
    GoalState,
    Subtask,
    SubtaskResult,
)
from qe.services.dispatcher.service import Dispatcher
from qe.substrate.goal_store import GoalStore


@pytest.fixture
async def goal_db(tmp_path):
    db_path = str(tmp_path / "lifecycle_test.db")
    migration = (
        Path(__file__).parent.parent.parent
        / "src/qe/substrate/migrations/0006_goals.sql"
    )
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(migration.read_text())
        await db.commit()
    return db_path


def _build_goal() -> GoalState:
    """Build a two-subtask goal: s1 (research) → s2 (synthesis)."""
    s1 = Subtask(subtask_id="s1", description="Research phase", task_type="research")
    s2 = Subtask(
        subtask_id="s2",
        description="Synthesis phase",
        task_type="synthesis",
        depends_on=["s1"],
    )
    decomp = GoalDecomposition(
        goal_id="g_lifecycle",
        original_description="Full lifecycle test",
        strategy="Research then synthesize",
        subtasks=[s1, s2],
    )
    return GoalState(
        goal_id="g_lifecycle",
        description="Full lifecycle test",
        status="executing",
        decomposition=decomp,
        subtask_states={"s1": "pending", "s2": "pending"},
    )


class TestGoalLifecycle:
    """Full lifecycle: submit → dispatch s1 → complete s1 → dispatch s2 →
    complete s2 → goals.completed event + persisted state."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, goal_db):
        bus = MemoryBus()
        store = GoalStore(goal_db)
        dispatcher = Dispatcher(bus=bus, goal_store=store)

        # Track events published to specific topics
        dispatched_events: list[Envelope] = []
        completed_events: list[Envelope] = []

        async def _on_dispatched(env: Envelope) -> None:
            dispatched_events.append(env)

        async def _on_completed(env: Envelope) -> None:
            completed_events.append(env)

        bus.subscribe("tasks.dispatched", _on_dispatched)
        bus.subscribe("goals.completed", _on_completed)

        # Wire dispatcher to receive task results (like app.py does)
        async def _on_task_result(env: Envelope) -> None:
            result = SubtaskResult.model_validate(env.payload)
            await dispatcher.handle_subtask_completed(result.goal_id, result)

        bus.subscribe("tasks.completed", _on_task_result)

        # 1. Submit goal
        state = _build_goal()
        await dispatcher.submit_goal(state)

        # Allow async handlers to run
        await asyncio.sleep(0.05)

        # s1 should be dispatched (no deps)
        assert state.subtask_states["s1"] == "dispatched"
        assert state.subtask_states["s2"] == "pending"
        assert len(dispatched_events) == 1
        assert dispatched_events[0].payload["subtask_id"] == "s1"

        # 2. Simulate s1 completion via bus (like executor would)
        bus.publish(
            Envelope(
                topic="tasks.completed",
                source_service_id="executor",
                correlation_id="g_lifecycle",
                payload={
                    "subtask_id": "s1",
                    "goal_id": "g_lifecycle",
                    "status": "completed",
                    "output": {"result": "Found 10 sources"},
                    "cost_usd": 0.005,
                    "latency_ms": 500,
                },
            )
        )
        await asyncio.sleep(0.1)

        # s2 should now be dispatched
        assert state.subtask_states["s1"] == "completed"
        assert state.subtask_states["s2"] == "dispatched"
        assert len(dispatched_events) == 2
        assert dispatched_events[1].payload["subtask_id"] == "s2"

        # 3. Simulate s2 completion
        bus.publish(
            Envelope(
                topic="tasks.completed",
                source_service_id="executor",
                correlation_id="g_lifecycle",
                payload={
                    "subtask_id": "s2",
                    "goal_id": "g_lifecycle",
                    "status": "completed",
                    "output": {"result": "Synthesized report"},
                    "cost_usd": 0.008,
                    "latency_ms": 800,
                },
            )
        )
        await asyncio.sleep(0.1)

        # 4. Verify goals.completed event was published
        assert len(completed_events) == 1
        assert completed_events[0].payload["goal_id"] == "g_lifecycle"

        # 5. Goal should be removed from active tracking
        assert dispatcher.get_goal_state("g_lifecycle") is None

        # 6. Verify persisted state in store
        persisted = await store.load_goal("g_lifecycle")
        assert persisted is not None
        assert persisted.status == "completed"
        assert persisted.completed_at is not None
        assert "s1" in persisted.subtask_results
        assert "s2" in persisted.subtask_results
        assert persisted.subtask_results["s1"].cost_usd == pytest.approx(0.005)
        assert persisted.subtask_results["s2"].cost_usd == pytest.approx(0.008)

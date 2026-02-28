"""Tests for Phase 3: Task Decomposition — Goal models, store, and dispatcher."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import aiosqlite
import pytest

from qe.models.goal import (
    ExecutionContract,
    GoalDecomposition,
    GoalState,
    Subtask,
    SubtaskResult,
)
from qe.services.dispatcher.service import Dispatcher
from qe.services.planner.service import PlannerService
from qe.substrate.goal_store import GoalStore

# ── Goal models ──────────────────────────────────────────────────────────────


class TestGoalModels:
    def test_subtask_defaults(self):
        s = Subtask(description="test", task_type="research")
        assert s.subtask_id.startswith("sub_")
        assert s.model_tier == "balanced"
        assert s.depends_on == []
        assert s.contract.max_retries == 3

    def test_goal_state_defaults(self):
        g = GoalState()
        assert g.goal_id.startswith("goal_")
        assert g.status == "planning"
        assert g.decomposition is None

    def test_execution_contract(self):
        c = ExecutionContract(
            preconditions=["data available"],
            postconditions=["result.length >= 3"],
            timeout_seconds=60,
        )
        assert c.timeout_seconds == 60
        assert len(c.preconditions) == 1

    def test_goal_decomposition(self):
        s1 = Subtask(
            subtask_id="s1",
            description="Research topic",
            task_type="research",
        )
        s2 = Subtask(
            subtask_id="s2",
            description="Analyze results",
            task_type="analysis",
            depends_on=["s1"],
        )
        d = GoalDecomposition(
            goal_id="g1",
            original_description="Test goal",
            strategy="Research then analyze",
            subtasks=[s1, s2],
        )
        assert len(d.subtasks) == 2
        assert d.subtasks[1].depends_on == ["s1"]


# ── GoalStore ────────────────────────────────────────────────────────────────


@pytest.fixture
async def goal_db(tmp_path):
    db_path = str(tmp_path / "goals_test.db")
    migration = (
        Path(__file__).parent.parent.parent
        / "src/qe/substrate/migrations/0006_goals.sql"
    )
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(migration.read_text())
        await db.commit()
    return db_path


def _make_state(goal_id="goal_test", **kwargs):
    defaults = {
        "goal_id": goal_id,
        "description": "Test goal",
        "status": "executing",
    }
    defaults.update(kwargs)
    return GoalState(**defaults)


class TestGoalStore:
    @pytest.mark.asyncio
    async def test_save_and_load(self, goal_db):
        store = GoalStore(goal_db)
        state = _make_state()
        await store.save_goal(state)

        loaded = await store.load_goal("goal_test")
        assert loaded is not None
        assert loaded.goal_id == "goal_test"
        assert loaded.status == "executing"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, goal_db):
        store = GoalStore(goal_db)
        assert await store.load_goal("nope") is None

    @pytest.mark.asyncio
    async def test_list_goals(self, goal_db):
        store = GoalStore(goal_db)
        await store.save_goal(_make_state("g1", status="executing"))
        await store.save_goal(_make_state("g2", status="completed"))
        await store.save_goal(_make_state("g3", status="executing"))

        all_goals = await store.list_goals()
        assert len(all_goals) == 3

        executing = await store.list_goals(status="executing")
        assert len(executing) == 2

    @pytest.mark.asyncio
    async def test_list_active_goals(self, goal_db):
        store = GoalStore(goal_db)
        await store.save_goal(_make_state("g1", status="executing"))
        await store.save_goal(_make_state("g2", status="completed"))
        await store.save_goal(_make_state("g3", status="planning"))

        active = await store.list_active_goals()
        assert len(active) == 2
        ids = {g.goal_id for g in active}
        assert ids == {"g1", "g3"}

    @pytest.mark.asyncio
    async def test_save_with_decomposition(self, goal_db):
        store = GoalStore(goal_db)
        decomp = GoalDecomposition(
            goal_id="g1",
            original_description="Test",
            strategy="Test strategy",
            subtasks=[
                Subtask(
                    subtask_id="s1",
                    description="Do thing",
                    task_type="research",
                )
            ],
        )
        state = _make_state("g1", decomposition=decomp)
        await store.save_goal(state)

        loaded = await store.load_goal("g1")
        assert loaded.decomposition is not None
        assert loaded.decomposition.strategy == "Test strategy"
        assert len(loaded.decomposition.subtasks) == 1

    @pytest.mark.asyncio
    async def test_checkpoint(self, goal_db):
        store = GoalStore(goal_db)
        state = _make_state(
            "g1",
            subtask_states={"s1": "completed", "s2": "pending"},
        )
        ckpt_id = await store.save_checkpoint("g1", state)
        assert ckpt_id.startswith("ckpt_")

        loaded = await store.load_checkpoint("g1", ckpt_id)
        assert loaded is not None
        assert loaded["subtask_states"]["s1"] == "completed"

    @pytest.mark.asyncio
    async def test_update_status(self, goal_db):
        store = GoalStore(goal_db)
        state = _make_state("g1", status="executing")
        await store.save_goal(state)

        state.status = "completed"
        state.completed_at = datetime.now(UTC)
        await store.save_goal(state)

        loaded = await store.load_goal("g1")
        assert loaded.status == "completed"
        assert loaded.completed_at is not None


# ── Dispatcher ───────────────────────────────────────────────────────────────


class TestDispatcher:
    @pytest.fixture
    async def dispatcher(self, goal_db):
        store = GoalStore(goal_db)
        bus = MagicMock()
        bus.publish = MagicMock(return_value=[])
        return Dispatcher(bus=bus, goal_store=store)

    def _make_ready_state(self):
        s1 = Subtask(
            subtask_id="s1",
            description="Step 1",
            task_type="research",
        )
        s2 = Subtask(
            subtask_id="s2",
            description="Step 2",
            task_type="analysis",
            depends_on=["s1"],
        )
        decomp = GoalDecomposition(
            goal_id="g1",
            original_description="Test",
            strategy="Two steps",
            subtasks=[s1, s2],
        )
        return GoalState(
            goal_id="g1",
            description="Test",
            status="executing",
            decomposition=decomp,
            subtask_states={"s1": "pending", "s2": "pending"},
        )

    @pytest.mark.asyncio
    async def test_submit_dispatches_ready(self, dispatcher):
        state = self._make_ready_state()
        await dispatcher.submit_goal(state)

        # s1 has no deps, should be dispatched; s2 depends on s1
        assert state.subtask_states["s1"] == "dispatched"
        assert state.subtask_states["s2"] == "pending"
        assert dispatcher.bus.publish.called

    @pytest.mark.asyncio
    async def test_subtask_completion_dispatches_dependents(self, dispatcher):
        state = self._make_ready_state()
        await dispatcher.submit_goal(state)

        result = SubtaskResult(
            subtask_id="s1",
            goal_id="g1",
            status="completed",
        )
        await dispatcher.handle_subtask_completed("g1", result)

        # s2 should now be dispatched since s1 is complete
        assert state.subtask_states["s2"] == "dispatched"

    @pytest.mark.asyncio
    async def test_goal_completes_when_all_done(self, dispatcher):
        state = self._make_ready_state()
        await dispatcher.submit_goal(state)

        # Complete s1
        await dispatcher.handle_subtask_completed(
            "g1",
            SubtaskResult(subtask_id="s1", goal_id="g1", status="completed"),
        )
        # Complete s2
        await dispatcher.handle_subtask_completed(
            "g1",
            SubtaskResult(subtask_id="s2", goal_id="g1", status="completed"),
        )

        # Goal should be completed and removed from active goals
        assert dispatcher.get_goal_state("g1") is None

    @pytest.mark.asyncio
    async def test_pause_and_resume(self, dispatcher):
        state = self._make_ready_state()
        await dispatcher.submit_goal(state)

        assert await dispatcher.pause_goal("g1")
        assert state.status == "paused"

        assert await dispatcher.resume_goal("g1")
        assert state.status == "executing"

    @pytest.mark.asyncio
    async def test_cancel(self, dispatcher):
        state = self._make_ready_state()
        await dispatcher.submit_goal(state)

        assert await dispatcher.cancel_goal("g1")
        assert dispatcher.get_goal_state("g1") is None

    @pytest.mark.asyncio
    async def test_orphan_result_ignored(self, dispatcher):
        # Completing a subtask for a non-existent goal shouldn't crash
        result = SubtaskResult(
            subtask_id="s1", goal_id="nonexistent", status="completed"
        )
        await dispatcher.handle_subtask_completed("nonexistent", result)


# ── Planner helpers ──────────────────────────────────────────────────────────


class TestPlannerHelpers:
    def test_get_ready_subtasks_no_deps(self):
        planner = PlannerService.__new__(PlannerService)
        s1 = Subtask(
            subtask_id="s1",
            description="Step 1",
            task_type="research",
        )
        decomp = GoalDecomposition(
            goal_id="g1",
            original_description="Test",
            strategy="One step",
            subtasks=[s1],
        )
        state = GoalState(
            goal_id="g1",
            decomposition=decomp,
            subtask_states={"s1": "pending"},
        )
        ready = planner.get_ready_subtasks(state)
        assert len(ready) == 1
        assert ready[0].subtask_id == "s1"

    def test_get_ready_subtasks_blocked(self):
        planner = PlannerService.__new__(PlannerService)
        s1 = Subtask(
            subtask_id="s1",
            description="Step 1",
            task_type="research",
        )
        s2 = Subtask(
            subtask_id="s2",
            description="Step 2",
            task_type="analysis",
            depends_on=["s1"],
        )
        decomp = GoalDecomposition(
            goal_id="g1",
            original_description="Test",
            strategy="Two steps",
            subtasks=[s1, s2],
        )
        state = GoalState(
            goal_id="g1",
            decomposition=decomp,
            subtask_states={"s1": "pending", "s2": "pending"},
        )
        ready = planner.get_ready_subtasks(state)
        assert len(ready) == 1
        assert ready[0].subtask_id == "s1"

    def test_get_ready_subtasks_after_completion(self):
        planner = PlannerService.__new__(PlannerService)
        s1 = Subtask(
            subtask_id="s1",
            description="Step 1",
            task_type="research",
        )
        s2 = Subtask(
            subtask_id="s2",
            description="Step 2",
            task_type="analysis",
            depends_on=["s1"],
        )
        decomp = GoalDecomposition(
            goal_id="g1",
            original_description="Test",
            strategy="Two steps",
            subtasks=[s1, s2],
        )
        state = GoalState(
            goal_id="g1",
            decomposition=decomp,
            subtask_states={"s1": "completed", "s2": "pending"},
        )
        ready = planner.get_ready_subtasks(state)
        assert len(ready) == 1
        assert ready[0].subtask_id == "s2"


# ── API endpoint tests ───────────────────────────────────────────────────────


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from qe.api.app import app

    return TestClient(app, raise_server_exceptions=False)


def test_goals_list_503_when_not_started(client):
    resp = client.get("/api/goals")
    assert resp.status_code == 503


def test_goals_submit_503_when_not_started(client):
    resp = client.post("/api/goals", json={"description": "test"})
    assert resp.status_code == 503


def test_goals_get_503_when_not_started(client):
    resp = client.get("/api/goals/g1")
    assert resp.status_code == 503

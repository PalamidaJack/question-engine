"""Tests for Phase 3: Task Decomposition — Goal models, store, and dispatcher."""

from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import aiosqlite
import pytest
from pydantic import ValidationError

from qe.models.goal import (
    ExecutionContract,
    GoalDecomposition,
    GoalState,
    Subtask,
    SubtaskResult,
)
from qe.services.dispatcher.service import Dispatcher
from qe.services.planner.schemas import DecompositionOutput, ProblemRepresentation, SubtaskPlan
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
    async def test_metadata_persistence(self, goal_db):
        store = GoalStore(goal_db)
        state = _make_state(
            "g_meta",
            metadata={
                "origin_user_id": "user-123",
                "origin_channel": "telegram",
            },
        )
        await store.save_goal(state)

        loaded = await store.load_goal("g_meta")
        assert loaded is not None
        assert loaded.metadata["origin_user_id"] == "user-123"
        assert loaded.metadata["origin_channel"] == "telegram"

    @pytest.mark.asyncio
    async def test_metadata_defaults_to_empty(self, goal_db):
        store = GoalStore(goal_db)
        state = _make_state("g_no_meta")
        await store.save_goal(state)

        loaded = await store.load_goal("g_no_meta")
        assert loaded is not None
        assert loaded.metadata == {}

    @pytest.mark.asyncio
    async def test_subtask_results_round_trip(self, goal_db):
        """Test 2A: subtask_results survive save → load cycle."""
        store = GoalStore(goal_db)
        result = SubtaskResult(
            subtask_id="s1",
            goal_id="g_rt",
            status="completed",
            output={"result": "found 42 sources"},
            model_used="gpt-4o",
            tokens_used={"prompt": 100, "completion": 50},
            latency_ms=1234,
            cost_usd=0.0042,
        )
        state = _make_state(
            "g_rt",
            subtask_results={"s1": result},
            subtask_states={"s1": "completed"},
        )
        await store.save_goal(state)

        loaded = await store.load_goal("g_rt")
        assert loaded is not None
        assert "s1" in loaded.subtask_results
        r = loaded.subtask_results["s1"]
        assert r.status == "completed"
        assert r.output == {"result": "found 42 sources"}
        assert r.model_used == "gpt-4o"
        assert r.latency_ms == 1234
        assert r.cost_usd == pytest.approx(0.0042)
        assert r.tokens_used == {"prompt": 100, "completion": 50}

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

    @pytest.mark.asyncio
    async def test_diamond_dependency_ordering(self, dispatcher):
        """Test 2C: Diamond DAG — D waits for both B and C to complete."""
        a = Subtask(subtask_id="a", description="Root", task_type="research")
        b = Subtask(
            subtask_id="b", description="Left", task_type="analysis", depends_on=["a"]
        )
        c = Subtask(
            subtask_id="c", description="Right", task_type="analysis", depends_on=["a"]
        )
        d = Subtask(
            subtask_id="d",
            description="Join",
            task_type="synthesis",
            depends_on=["b", "c"],
        )
        decomp = GoalDecomposition(
            goal_id="g_diamond",
            original_description="Diamond test",
            strategy="A -> B+C -> D",
            subtasks=[a, b, c, d],
        )
        state = GoalState(
            goal_id="g_diamond",
            description="Diamond test",
            status="executing",
            decomposition=decomp,
            subtask_states={"a": "pending", "b": "pending", "c": "pending", "d": "pending"},
        )
        await dispatcher.submit_goal(state)

        # Only A should be dispatched initially
        assert state.subtask_states["a"] == "dispatched"
        assert state.subtask_states["b"] == "pending"
        assert state.subtask_states["c"] == "pending"
        assert state.subtask_states["d"] == "pending"

        # Complete A → B and C become ready
        await dispatcher.handle_subtask_completed(
            "g_diamond",
            SubtaskResult(subtask_id="a", goal_id="g_diamond", status="completed"),
        )
        assert state.subtask_states["b"] == "dispatched"
        assert state.subtask_states["c"] == "dispatched"
        assert state.subtask_states["d"] == "pending"  # still blocked

        # Complete B only → D still blocked (C not done)
        await dispatcher.handle_subtask_completed(
            "g_diamond",
            SubtaskResult(subtask_id="b", goal_id="g_diamond", status="completed"),
        )
        assert state.subtask_states["d"] == "pending"

        # Complete C → D should now dispatch
        await dispatcher.handle_subtask_completed(
            "g_diamond",
            SubtaskResult(subtask_id="c", goal_id="g_diamond", status="completed"),
        )
        assert state.subtask_states["d"] == "dispatched"


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


# ── DAG Validation ──────────────────────────────────────────────────────────


def _topo_sort(subtasks: list[Subtask]) -> list[str]:
    """Kahn's algorithm. Raises ValueError on cycle."""
    id_set = {s.subtask_id for s in subtasks}
    in_degree: dict[str, int] = {s.subtask_id: 0 for s in subtasks}
    adj: dict[str, list[str]] = {s.subtask_id: [] for s in subtasks}
    for s in subtasks:
        for dep in s.depends_on:
            adj[dep].append(s.subtask_id)
            in_degree[s.subtask_id] += 1

    queue = deque(sid for sid, d in in_degree.items() if d == 0)
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in adj[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(id_set):
        raise ValueError("Cycle detected in DAG")
    return order


_DAG_CASES = [
    # (name, subtasks_spec): each spec is (id, [deps])
    ("single", [("s1", [])]),
    ("linear_2", [("s1", []), ("s2", ["s1"])]),
    ("linear_3", [("s1", []), ("s2", ["s1"]), ("s3", ["s2"])]),
    ("fan_out", [("s1", []), ("s2", ["s1"]), ("s3", ["s1"]), ("s4", ["s1"])]),
    ("fan_in", [("s1", []), ("s2", []), ("s3", ["s1", "s2"])]),
    ("diamond", [("a", []), ("b", ["a"]), ("c", ["a"]), ("d", ["b", "c"])]),
    ("wide_parallel", [("s1", []), ("s2", []), ("s3", []), ("s4", [])]),
    ("deep_chain", [(f"s{i}", [f"s{i-1}"] if i > 0 else []) for i in range(6)]),
    (
        "mixed",
        [("s1", []), ("s2", []), ("s3", ["s1"]), ("s4", ["s2", "s3"]), ("s5", ["s4"])],
    ),
    (
        "complex_diamond",
        [
            ("a", []),
            ("b", ["a"]),
            ("c", ["a"]),
            ("d", ["b"]),
            ("e", ["c"]),
            ("f", ["d", "e"]),
        ],
    ),
]


class TestDAGValidation:
    """Test 2B: Validate 10 different DAG structures."""

    @pytest.mark.parametrize("name,spec", _DAG_CASES, ids=[c[0] for c in _DAG_CASES])
    def test_dag_structure(self, name, spec):
        subtasks = [
            Subtask(
                subtask_id=sid,
                description=f"Task {sid}",
                task_type="research",
                depends_on=deps,
            )
            for sid, deps in spec
        ]
        id_set = {s.subtask_id for s in subtasks}

        # All dependency refs point to real subtask IDs
        for s in subtasks:
            for dep in s.depends_on:
                assert dep in id_set, f"{s.subtask_id} depends on unknown {dep}"

        # No self-dependencies
        for s in subtasks:
            assert s.subtask_id not in s.depends_on, f"{s.subtask_id} depends on itself"

        # At least one root node (no deps)
        roots = [s for s in subtasks if not s.depends_on]
        assert len(roots) >= 1, "DAG has no root node"

        # No cycles (topological sort succeeds)
        order = _topo_sort(subtasks)
        assert len(order) == len(subtasks)


# ── Crash Recovery ──────────────────────────────────────────────────────────


class TestCrashRecovery:
    """Test 2D: Crash recovery via dispatcher.reconcile()."""

    @pytest.mark.asyncio
    async def test_executing_goal_resumed(self, goal_db):
        """Goal in 'executing' state is resumed with correct subtask states."""
        store = GoalStore(goal_db)
        bus = MagicMock()
        bus.publish = MagicMock(return_value=[])

        s1 = Subtask(subtask_id="s1", description="Step 1", task_type="research")
        s2 = Subtask(
            subtask_id="s2",
            description="Step 2",
            task_type="analysis",
            depends_on=["s1"],
        )
        decomp = GoalDecomposition(
            goal_id="g_crash",
            original_description="Crash test",
            strategy="Two steps",
            subtasks=[s1, s2],
        )
        result = SubtaskResult(
            subtask_id="s1",
            goal_id="g_crash",
            status="completed",
            cost_usd=0.01,
        )
        state = _make_state(
            "g_crash",
            status="executing",
            decomposition=decomp,
            subtask_states={"s1": "completed", "s2": "pending"},
            subtask_results={"s1": result},
        )
        await store.save_goal(state)

        # Simulate restart: new dispatcher, no in-memory goals
        dispatcher = Dispatcher(bus=bus, goal_store=store)
        counts = await dispatcher.reconcile()

        assert counts["resumed"] == 1
        assert counts["failed_stale"] == 0

        # Goal should be in active goals now
        recovered = dispatcher.get_goal_state("g_crash")
        assert recovered is not None
        assert recovered.status == "executing"
        # s1 result should have survived via Fix 1A
        assert "s1" in recovered.subtask_results
        assert recovered.subtask_results["s1"].cost_usd == pytest.approx(0.01)
        # s2 should have been dispatched (deps met)
        assert recovered.subtask_states["s2"] == "dispatched"

    @pytest.mark.asyncio
    async def test_planning_goal_failed_as_stale(self, goal_db):
        """Goal in 'planning' state is failed as stale on reconcile."""
        store = GoalStore(goal_db)
        bus = MagicMock()
        bus.publish = MagicMock(return_value=[])

        state = _make_state("g_stale", status="planning")
        await store.save_goal(state)

        dispatcher = Dispatcher(bus=bus, goal_store=store)
        counts = await dispatcher.reconcile()

        assert counts["resumed"] == 0
        assert counts["failed_stale"] == 1

        # Goal should be marked failed in the store
        loaded = await store.load_goal("g_stale")
        assert loaded.status == "failed"


# ── ProblemRepresentation schema enforcement ────────────────────────────────


class TestProblemRepresentation:
    """Test 2E: DecompositionOutput requires a representation field."""

    def test_valid_decomposition_output(self):
        output = DecompositionOutput(
            representation=ProblemRepresentation(
                core_problem="Test problem",
                actual_need="Test need",
            ),
            strategy="Test strategy",
            subtasks=[
                SubtaskPlan(
                    description="Do research",
                    task_type="research",
                )
            ],
        )
        assert output.representation.core_problem == "Test problem"

    def test_missing_representation_raises(self):
        with pytest.raises(ValidationError, match="representation"):
            DecompositionOutput(
                strategy="No representation",
                subtasks=[
                    SubtaskPlan(
                        description="Do research",
                        task_type="research",
                    )
                ],
            )

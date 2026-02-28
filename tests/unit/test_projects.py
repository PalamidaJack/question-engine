"""Tests for the project management layer."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from qe.models.goal import GoalState, Project


class TestProjectModel:
    def test_project_defaults(self):
        p = Project(name="Test Project")
        assert p.name == "Test Project"
        assert p.status == "active"
        assert p.project_id.startswith("proj_")
        assert p.tags == []

    def test_goal_state_has_project_fields(self):
        g = GoalState(
            description="test goal",
            project_id="proj_abc",
            tags=["research", "urgent"],
        )
        assert g.project_id == "proj_abc"
        assert g.tags == ["research", "urgent"]
        assert g.started_at is None
        assert g.due_at is None

    def test_goal_state_with_timeline(self):
        now = datetime.now(UTC)
        g = GoalState(
            description="test",
            started_at=now,
            due_at=now,
        )
        assert g.started_at == now
        assert g.due_at == now


class TestGoalStoreProjects:
    @pytest.fixture
    async def store(self, tmp_path):
        import aiosqlite

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    description TEXT,
                    status TEXT,
                    decomposition TEXT,
                    subtask_states TEXT,
                    subtask_results TEXT,
                    created_at TEXT,
                    completed_at TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    goal_id TEXT,
                    subtask_states TEXT,
                    subtask_results TEXT,
                    created_at TEXT
                )
            """)
            await db.commit()

        from qe.substrate.goal_store import GoalStore

        return GoalStore(db_path)

    @pytest.mark.asyncio
    async def test_save_and_load_project(self, store):
        p = Project(name="My Project", description="Test desc", owner="alice")
        await store.save_project(p)

        loaded = await store.get_project(p.project_id)
        assert loaded is not None
        assert loaded.name == "My Project"
        assert loaded.owner == "alice"

    @pytest.mark.asyncio
    async def test_list_projects(self, store):
        p1 = Project(name="Active Project", status="active")
        p2 = Project(name="Archived Project", status="archived")
        await store.save_project(p1)
        await store.save_project(p2)

        all_projects = await store.list_projects()
        assert len(all_projects) == 2

        active = await store.list_projects(status="active")
        assert len(active) == 1
        assert active[0].name == "Active Project"

    @pytest.mark.asyncio
    async def test_goal_with_project_assignment(self, store):
        p = Project(name="Research")
        await store.save_project(p)

        g = GoalState(
            description="Analyze market data",
            project_id=p.project_id,
            tags=["research"],
        )
        await store.save_goal(g)

        goals = await store.get_project_goals(p.project_id)
        assert len(goals) == 1
        assert goals[0].goal_id == g.goal_id
        assert goals[0].project_id == p.project_id

    @pytest.mark.asyncio
    async def test_project_metrics(self, store):
        p = Project(name="Metrics Test")
        await store.save_project(p)

        # Completed goal
        g1 = GoalState(
            description="Goal 1",
            status="completed",
            project_id=p.project_id,
            completed_at=datetime.now(UTC),
        )
        await store.save_goal(g1)

        # In-progress goal
        g2 = GoalState(
            description="Goal 2",
            status="executing",
            project_id=p.project_id,
        )
        await store.save_goal(g2)

        metrics = await store.get_project_metrics(p.project_id)
        assert metrics["total_goals"] == 2
        assert metrics["completed"] == 1
        assert metrics["in_progress"] == 1
        assert metrics["completion_rate"] == 0.5


class TestProjectEndpoints:
    @pytest.fixture
    def client(self):
        from starlette.testclient import TestClient

        from qe.api.app import app

        return TestClient(app, raise_server_exceptions=False)

    def test_list_projects_returns_503_without_engine(self, client):
        res = client.get("/api/projects")
        assert res.status_code == 503

    def test_create_project_returns_503_without_engine(self, client):
        res = client.post("/api/projects", json={"name": "Test"})
        assert res.status_code == 503

    def test_health_live_returns_503_without_doctor(self, client):
        res = client.get("/api/health/live")
        assert res.status_code == 503

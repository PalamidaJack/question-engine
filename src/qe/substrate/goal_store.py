"""Goal and project persistence for crash recovery and PM tracking."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime

import aiosqlite

from qe.models.goal import GoalState, Project

log = logging.getLogger(__name__)


class GoalStore:
    """Persists goal state to SQLite for crash recovery."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def _ensure_pm_columns(self) -> None:
        """Add project management columns if they don't exist yet."""
        async with aiosqlite.connect(self._db_path) as db:
            # Check if project_id column exists
            cursor = await db.execute("PRAGMA table_info(goals)")
            columns = {row[1] for row in await cursor.fetchall()}
            if "project_id" not in columns:
                await db.execute(
                    "ALTER TABLE goals ADD COLUMN project_id TEXT"
                )
                await db.execute(
                    "ALTER TABLE goals ADD COLUMN started_at TEXT"
                )
                await db.execute(
                    "ALTER TABLE goals ADD COLUMN due_at TEXT"
                )
                await db.execute(
                    "ALTER TABLE goals ADD COLUMN tags TEXT DEFAULT '[]'"
                )
                await db.commit()
                log.info("goal_store: added PM columns to goals table")

            if "metadata" not in columns:
                await db.execute(
                    "ALTER TABLE goals ADD COLUMN metadata JSON DEFAULT '{}'"
                )
                await db.commit()
                log.info("goal_store: added metadata column to goals table")

            # Create projects table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    owner TEXT DEFAULT '',
                    status TEXT DEFAULT 'active',
                    tags TEXT DEFAULT '[]',
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            await db.commit()

    async def save_goal(self, state: GoalState) -> None:
        """Save or update a goal's state."""
        await self._ensure_pm_columns()
        decomp_json = (
            state.decomposition.model_dump_json()
            if state.decomposition
            else None
        )
        results_json = json.dumps(
            {k: v.model_dump() for k, v in state.subtask_results.items()},
            default=str,
        )

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO goals
                    (goal_id, description, status, decomposition,
                     subtask_states, subtask_results, created_at, completed_at,
                     project_id, started_at, due_at, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(goal_id) DO UPDATE SET
                    status = excluded.status,
                    decomposition = excluded.decomposition,
                    subtask_states = excluded.subtask_states,
                    subtask_results = excluded.subtask_results,
                    completed_at = excluded.completed_at,
                    project_id = excluded.project_id,
                    started_at = excluded.started_at,
                    due_at = excluded.due_at,
                    tags = excluded.tags,
                    metadata = excluded.metadata
                """,
                (
                    state.goal_id,
                    state.description,
                    state.status,
                    decomp_json,
                    json.dumps(state.subtask_states),
                    results_json,
                    state.created_at.isoformat(),
                    (
                        state.completed_at.isoformat()
                        if state.completed_at
                        else None
                    ),
                    state.project_id,
                    (
                        state.started_at.isoformat()
                        if state.started_at
                        else None
                    ),
                    (
                        state.due_at.isoformat()
                        if state.due_at
                        else None
                    ),
                    json.dumps(state.tags),
                    json.dumps(state.metadata),
                ),
            )
            await db.commit()

    async def load_goal(self, goal_id: str) -> GoalState | None:
        """Load a goal by ID."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM goals WHERE goal_id = ?",
                (goal_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            return None
        return self._row_to_state(row)

    async def list_goals(
        self, status: str | None = None
    ) -> list[GoalState]:
        """List all goals, optionally filtered by status."""
        if status:
            query = "SELECT * FROM goals WHERE status = ? ORDER BY created_at DESC"
            params: tuple = (status,)
        else:
            query = "SELECT * FROM goals ORDER BY created_at DESC"
            params = ()

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        return [self._row_to_state(row) for row in rows]

    async def list_active_goals(self) -> list[GoalState]:
        """List goals in planning or executing state."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM goals WHERE status IN ('planning', 'executing') "
                "ORDER BY created_at DESC"
            )
            rows = await cursor.fetchall()

        return [self._row_to_state(row) for row in rows]

    async def save_checkpoint(
        self, goal_id: str, goal_state: GoalState
    ) -> str:
        """Save a checkpoint for rollback. Returns checkpoint_id."""
        checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"
        results_json = json.dumps(
            {k: v.model_dump() for k, v in goal_state.subtask_results.items()},
            default=str,
        )

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO checkpoints
                    (checkpoint_id, goal_id, subtask_states,
                     subtask_results, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    goal_id,
                    json.dumps(goal_state.subtask_states),
                    results_json,
                    datetime.now(UTC).isoformat(),
                ),
            )
            await db.commit()

        return checkpoint_id

    async def load_checkpoint(
        self, goal_id: str, checkpoint_id: str
    ) -> dict | None:
        """Load a specific checkpoint."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ? AND goal_id = ?",
                (checkpoint_id, goal_id),
            )
            row = await cursor.fetchone()

        if row is None:
            return None
        return {
            "checkpoint_id": row[0],
            "goal_id": row[1],
            "subtask_states": json.loads(row[2]),
            "subtask_results": json.loads(row[3]),
            "created_at": row[4],
        }

    # ── Project CRUD ────────────────────────────────────────────────

    async def save_project(self, project: Project) -> None:
        """Create or update a project."""
        await self._ensure_pm_columns()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO projects
                    (project_id, name, description, owner, status,
                     tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    owner = excluded.owner,
                    status = excluded.status,
                    tags = excluded.tags,
                    updated_at = excluded.updated_at
                """,
                (
                    project.project_id,
                    project.name,
                    project.description,
                    project.owner,
                    project.status,
                    json.dumps(project.tags),
                    project.created_at.isoformat(),
                    project.updated_at.isoformat(),
                ),
            )
            await db.commit()

    async def get_project(self, project_id: str) -> Project | None:
        """Load a project by ID."""
        await self._ensure_pm_columns()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM projects WHERE project_id = ?",
                (project_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_project(row)

    async def list_projects(
        self, status: str | None = None
    ) -> list[Project]:
        """List all projects, optionally filtered by status."""
        await self._ensure_pm_columns()
        if status:
            query = "SELECT * FROM projects WHERE status = ? ORDER BY created_at DESC"
            params: tuple = (status,)
        else:
            query = "SELECT * FROM projects ORDER BY created_at DESC"
            params = ()

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        return [self._row_to_project(row) for row in rows]

    async def get_project_goals(self, project_id: str) -> list[GoalState]:
        """List all goals belonging to a project."""
        await self._ensure_pm_columns()
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM goals WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            )
            rows = await cursor.fetchall()
        return [self._row_to_state(row) for row in rows]

    async def get_project_metrics(self, project_id: str) -> dict:
        """Compute cycle time and cost metrics for a project."""
        goals = await self.get_project_goals(project_id)

        total_goals = len(goals)
        completed = [g for g in goals if g.status == "completed"]
        failed = [g for g in goals if g.status == "failed"]
        in_progress = [g for g in goals if g.status in ("planning", "executing")]

        # Cycle time: average time from created_at to completed_at
        cycle_times: list[float] = []
        total_cost = 0.0
        total_subtasks = 0
        for g in completed:
            if g.completed_at and g.created_at:
                delta = (g.completed_at - g.created_at).total_seconds()
                cycle_times.append(delta)
            for sr in g.subtask_results.values():
                total_cost += sr.cost_usd
                total_subtasks += 1

        avg_cycle_seconds = (
            sum(cycle_times) / len(cycle_times) if cycle_times else 0.0
        )

        return {
            "project_id": project_id,
            "total_goals": total_goals,
            "completed": len(completed),
            "failed": len(failed),
            "in_progress": len(in_progress),
            "avg_cycle_time_seconds": round(avg_cycle_seconds, 1),
            "total_cost_usd": round(total_cost, 4),
            "total_subtasks": total_subtasks,
            "completion_rate": (
                round(len(completed) / total_goals, 2)
                if total_goals > 0
                else 0.0
            ),
        }

    # ── Row Converters ───────────────────────────────────────────────

    def _row_to_state(self, row: tuple) -> GoalState:
        """Convert a database row to a GoalState."""
        from qe.models.goal import GoalDecomposition

        decomp = None
        if row[3]:
            decomp = GoalDecomposition.model_validate_json(row[3])

        # Handle both old (8-column) and new (12-column) schemas
        project_id = row[8] if len(row) > 8 else None
        started_at = (
            datetime.fromisoformat(row[9]) if len(row) > 9 and row[9] else None
        )
        due_at = (
            datetime.fromisoformat(row[10]) if len(row) > 10 and row[10] else None
        )
        tags = json.loads(row[11]) if len(row) > 11 and row[11] else []
        metadata = json.loads(row[12]) if len(row) > 12 and row[12] else {}

        return GoalState(
            goal_id=row[0],
            description=row[1],
            status=row[2],
            decomposition=decomp,
            subtask_states=json.loads(row[4]) if row[4] else {},
            subtask_results={},
            created_at=datetime.fromisoformat(row[6]),
            completed_at=(
                datetime.fromisoformat(row[7]) if row[7] else None
            ),
            project_id=project_id,
            started_at=started_at,
            due_at=due_at,
            tags=tags,
            metadata=metadata,
        )

    @staticmethod
    def _row_to_project(row: tuple) -> Project:
        """Convert a database row to a Project."""
        return Project(
            project_id=row[0],
            name=row[1],
            description=row[2] or "",
            owner=row[3] or "",
            status=row[4] or "active",
            tags=json.loads(row[5]) if row[5] else [],
            created_at=datetime.fromisoformat(row[6]) if row[6] else datetime.now(UTC),
            updated_at=datetime.fromisoformat(row[7]) if row[7] else datetime.now(UTC),
        )

"""Goal persistence for crash recovery."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime

import aiosqlite

from qe.models.goal import GoalState

log = logging.getLogger(__name__)


class GoalStore:
    """Persists goal state to SQLite for crash recovery."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def save_goal(self, state: GoalState) -> None:
        """Save or update a goal's state."""
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
                     subtask_states, subtask_results, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(goal_id) DO UPDATE SET
                    status = excluded.status,
                    decomposition = excluded.decomposition,
                    subtask_states = excluded.subtask_states,
                    subtask_results = excluded.subtask_results,
                    completed_at = excluded.completed_at
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

    def _row_to_state(self, row: tuple) -> GoalState:
        """Convert a database row to a GoalState."""
        from qe.models.goal import GoalDecomposition

        decomp = None
        if row[3]:
            decomp = GoalDecomposition.model_validate_json(row[3])

        return GoalState(
            goal_id=row[0],
            description=row[1],
            status=row[2],
            decomposition=decomp,
            subtask_states=json.loads(row[4]) if row[4] else {},
            subtask_results={},  # Not fully restored — subtask_results are complex
            created_at=datetime.fromisoformat(row[6]),
            completed_at=(
                datetime.fromisoformat(row[7]) if row[7] else None
            ),
        )

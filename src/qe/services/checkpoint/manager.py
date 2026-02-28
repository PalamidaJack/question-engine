"""CheckpointManager: graph-aware rollback over GoalStore checkpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

import aiosqlite

from qe.substrate.goal_store import GoalStore

log = logging.getLogger(__name__)


class CheckpointManager:
    """Wraps GoalStore checkpoint primitives with graph-aware rollback."""

    def __init__(self, goal_store: GoalStore) -> None:
        self._store = goal_store

    async def find_rollback_point(
        self, goal_id: str, failed_subtask_id: str
    ) -> str | None:
        """Walk checkpoints in reverse.

        Returns the latest checkpoint where *failed_subtask_id* was still pending/dispatched.
        """
        # Query checkpoint table directly (GoalState.checkpoints is not persisted)
        async with aiosqlite.connect(self._store._db_path) as db:
            cursor = await db.execute(
                "SELECT checkpoint_id, subtask_states FROM checkpoints "
                "WHERE goal_id = ? ORDER BY created_at DESC",
                (goal_id,),
            )
            rows = await cursor.fetchall()

        if not rows:
            return None

        for row in rows:
            ckpt_id = row[0]
            subtask_states: dict[str, str] = json.loads(row[1]) if row[1] else {}
            status = subtask_states.get(failed_subtask_id)
            if status in (None, "pending", "dispatched"):
                log.info(
                    "checkpoint.rollback_point goal=%s subtask=%s checkpoint=%s",
                    goal_id,
                    failed_subtask_id,
                    ckpt_id,
                )
                return ckpt_id

        return None

    async def rollback_to(
        self, goal_id: str, checkpoint_id: str
    ) -> dict[str, Any] | None:
        """Load checkpoint and restore subtask_states/subtask_results on the GoalState."""
        ckpt = await self._store.load_checkpoint(goal_id, checkpoint_id)
        if ckpt is None:
            log.warning(
                "checkpoint.rollback_missing goal=%s checkpoint=%s",
                goal_id,
                checkpoint_id,
            )
            return None

        state = await self._store.load_goal(goal_id)
        if state is None:
            return None

        state.subtask_states = ckpt["subtask_states"]

        # Restore subtask_results: keep only results for subtasks that were
        # completed at checkpoint time.
        from qe.models.goal import SubtaskResult

        ckpt_results = ckpt.get("subtask_results", {})
        restored: dict[str, SubtaskResult] = {}
        for sid, data in ckpt_results.items():
            if isinstance(data, dict):
                restored[sid] = SubtaskResult.model_validate(data)
            else:
                restored[sid] = data
        state.subtask_results = restored

        await self._store.save_goal(state)

        log.info(
            "checkpoint.rolled_back goal=%s checkpoint=%s",
            goal_id,
            checkpoint_id,
        )
        return {
            "goal_id": goal_id,
            "checkpoint_id": checkpoint_id,
            "subtask_states": state.subtask_states,
        }

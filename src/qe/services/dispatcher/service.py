"""Dispatcher: manages goal execution by tracking subtask dependency graphs."""

from __future__ import annotations

import logging
from typing import Any

from qe.models.envelope import Envelope
from qe.models.goal import GoalState, SubtaskResult
from qe.substrate.goal_store import GoalStore

log = logging.getLogger(__name__)


def _text_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two texts (0.0–1.0)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


class Dispatcher:
    """Deterministic orchestration engine for goal execution.

    Manages state, tracks dependencies, routes subtasks. Does not call LLMs.
    """

    def __init__(
        self,
        bus: Any,
        goal_store: GoalStore,
        drift_threshold: float = 0.05,
    ) -> None:
        self.bus = bus
        self.goal_store = goal_store
        self._drift_threshold = drift_threshold
        self._active_goals: dict[str, GoalState] = {}

    async def reconcile(self) -> dict[str, int]:
        """Load in-flight goals from store on startup and resume or fail them.

        Goals in 'executing' or 'planning' state are stale if the process
        restarted. Executing goals are resumed; planning goals are failed
        (their LLM decomposition was interrupted).

        Returns counts: {"resumed": N, "failed_stale": M}.
        """
        active_goals = await self.goal_store.list_active_goals()
        resumed = 0
        failed_stale = 0

        for state in active_goals:
            if state.goal_id in self._active_goals:
                continue  # already tracked in memory

            if state.status == "executing":
                # Resume: re-add to active goals and dispatch ready subtasks
                self._active_goals[state.goal_id] = state
                await self._dispatch_ready(state)
                resumed += 1
                log.info(
                    "reconcile.resumed goal_id=%s subtasks=%d",
                    state.goal_id,
                    len(state.subtask_states),
                )
            elif state.status == "planning":
                # Planning was interrupted mid-LLM — fail closed
                state.transition_to("failed")
                await self.goal_store.save_goal(state)
                failed_stale += 1
                log.warning(
                    "reconcile.failed_stale goal_id=%s "
                    "reason=planning_interrupted",
                    state.goal_id,
                )

        if resumed or failed_stale:
            log.info(
                "reconcile.done resumed=%d failed_stale=%d",
                resumed,
                failed_stale,
            )

        return {"resumed": resumed, "failed_stale": failed_stale}

    async def submit_goal(self, state: GoalState) -> None:
        """Accept a planned goal and begin execution."""
        self._active_goals[state.goal_id] = state
        await self.goal_store.save_goal(state)

        log.info(
            "dispatcher.goal_submitted goal_id=%s subtasks=%d",
            state.goal_id,
            len(state.subtask_states),
        )

        # Dispatch ready subtasks
        await self._dispatch_ready(state)

    async def handle_subtask_completed(
        self,
        goal_id: str,
        result: SubtaskResult,
    ) -> None:
        """Handle a completed subtask. Dispatch dependents if ready."""
        state = self._active_goals.get(goal_id)
        if not state:
            log.warning(
                "dispatcher.orphan_result goal_id=%s subtask_id=%s",
                goal_id,
                result.subtask_id,
            )
            return

        state.subtask_states[result.subtask_id] = result.status
        state.subtask_results[result.subtask_id] = result

        if result.status == "completed":
            self._check_drift(state, result)

        log.info(
            "dispatcher.subtask_done goal_id=%s subtask_id=%s status=%s",
            goal_id,
            result.subtask_id,
            result.status,
        )

        # Create checkpoint
        ckpt_id = await self.goal_store.save_checkpoint(goal_id, state)
        state.checkpoints.append(ckpt_id)

        # Check if all subtasks are complete
        if self._all_complete(state):
            state.transition_to("completed")
            await self.goal_store.save_goal(state)
            del self._active_goals[goal_id]

            log.info("dispatcher.goal_completed goal_id=%s", goal_id)

            self.bus.publish(
                Envelope(
                    topic="goals.completed",
                    source_service_id="dispatcher",
                    correlation_id=goal_id,
                    payload={
                        "goal_id": goal_id,
                        "subtask_count": len(state.subtask_states),
                    },
                )
            )
            return

        # Check for failures
        if result.status == "failed":
            failed_count = sum(
                1
                for s in state.subtask_states.values()
                if s == "failed"
            )
            if failed_count >= 3:
                state.transition_to("failed")
                await self.goal_store.save_goal(state)
                del self._active_goals[goal_id]

                log.error(
                    "dispatcher.goal_failed goal_id=%s failures=%d",
                    goal_id,
                    failed_count,
                )

                self.bus.publish(
                    Envelope(
                        topic="goals.failed",
                        source_service_id="dispatcher",
                        correlation_id=goal_id,
                        payload={
                            "goal_id": goal_id,
                            "reason": "too_many_failures",
                        },
                    )
                )
                return

        await self.goal_store.save_goal(state)
        await self._dispatch_ready(state)

    async def pause_goal(self, goal_id: str) -> bool:
        """Pause a running goal."""
        state = self._active_goals.get(goal_id)
        if not state:
            return False
        state.transition_to("paused")
        await self.goal_store.save_goal(state)
        log.info("dispatcher.goal_paused goal_id=%s", goal_id)
        return True

    async def resume_goal(self, goal_id: str) -> bool:
        """Resume a paused goal."""
        state = self._active_goals.get(goal_id)
        if not state or state.status != "paused":
            # Try loading from store
            state = await self.goal_store.load_goal(goal_id)
            if not state or state.status != "paused":
                return False
            self._active_goals[goal_id] = state

        state.transition_to("executing")
        await self.goal_store.save_goal(state)
        await self._dispatch_ready(state)
        log.info("dispatcher.goal_resumed goal_id=%s", goal_id)
        return True

    async def cancel_goal(self, goal_id: str) -> bool:
        """Cancel a goal."""
        state = self._active_goals.pop(goal_id, None)
        if state:
            state.transition_to("failed")
            await self.goal_store.save_goal(state)
            log.info("dispatcher.goal_cancelled goal_id=%s", goal_id)
            return True
        return False

    def get_goal_state(self, goal_id: str) -> GoalState | None:
        """Get the current state of a goal."""
        return self._active_goals.get(goal_id)

    def _check_drift(self, state: GoalState, result: SubtaskResult) -> None:
        """Publish a drift event if the subtask output diverges from the goal."""
        output_text = result.output.get("content", "")
        if not output_text:
            return

        sim = _text_similarity(state.description, output_text)
        if sim < self._drift_threshold:
            log.warning(
                "dispatcher.drift_detected goal_id=%s subtask_id=%s "
                "similarity=%.3f threshold=%.3f",
                state.goal_id,
                result.subtask_id,
                sim,
                self._drift_threshold,
            )
            self.bus.publish(
                Envelope(
                    topic="goals.drift_detected",
                    source_service_id="dispatcher",
                    correlation_id=state.goal_id,
                    payload={
                        "goal_id": state.goal_id,
                        "subtask_id": result.subtask_id,
                        "similarity": sim,
                        "threshold": self._drift_threshold,
                    },
                )
            )

    async def _dispatch_ready(self, state: GoalState) -> None:
        """Find and dispatch subtasks whose dependencies are met."""
        if not state.decomposition or state.status != "executing":
            return

        for subtask in state.decomposition.subtasks:
            if state.subtask_states.get(subtask.subtask_id) != "pending":
                continue

            deps_met = all(
                state.subtask_states.get(dep) == "completed"
                for dep in subtask.depends_on
            )
            if not deps_met:
                continue

            state.subtask_states[subtask.subtask_id] = "dispatched"
            log.info(
                "dispatcher.dispatch goal_id=%s subtask_id=%s type=%s",
                state.goal_id,
                subtask.subtask_id,
                subtask.task_type,
            )

            self.bus.publish(
                Envelope(
                    topic="tasks.dispatched",
                    source_service_id="dispatcher",
                    correlation_id=state.goal_id,
                    payload={
                        "goal_id": state.goal_id,
                        "subtask_id": subtask.subtask_id,
                        "description": subtask.description,
                        "task_type": subtask.task_type,
                        "model_tier": subtask.model_tier,
                        "depends_on": subtask.depends_on,
                        "contract": subtask.contract.model_dump(),
                    },
                )
            )

    def _all_complete(self, state: GoalState) -> bool:
        """Check if all subtasks have completed successfully."""
        return all(
            status == "completed"
            for status in state.subtask_states.values()
        )

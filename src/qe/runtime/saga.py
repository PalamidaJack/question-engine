"""Saga compensation: rollback partial goal progress on failure.

Implements the saga pattern where each completed subtask registers
a compensation action. On goal failure, compensations execute in
reverse order to undo partial work.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CompensationAction:
    """A single undo action for a completed subtask."""

    subtask_id: str
    action_type: str  # "retract_claims", "delete_output", "notify", "noop"
    parameters: dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    executed: bool = False
    error: str | None = None


@dataclass
class SagaState:
    """Tracks compensation actions for a goal's saga."""

    goal_id: str
    actions: list[CompensationAction] = field(default_factory=list)
    status: str = "active"  # "active", "compensating", "compensated", "failed"
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def register(
        self,
        subtask_id: str,
        action_type: str,
        parameters: dict[str, Any] | None = None,
    ) -> CompensationAction:
        """Register a compensation action for a completed subtask."""
        action = CompensationAction(
            subtask_id=subtask_id,
            action_type=action_type,
            parameters=parameters or {},
        )
        self.actions.append(action)
        log.debug(
            "saga.register goal_id=%s subtask=%s type=%s",
            self.goal_id,
            subtask_id,
            action_type,
        )
        return action

    @property
    def pending_compensations(self) -> list[CompensationAction]:
        """Return unexecuted compensation actions in reverse order."""
        return [a for a in reversed(self.actions) if not a.executed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "status": self.status,
            "action_count": len(self.actions),
            "executed_count": sum(1 for a in self.actions if a.executed),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class SagaCoordinator:
    """Coordinates saga execution and compensation for goals.

    Each goal gets a SagaState that tracks compensation actions.
    On failure, compensations run in reverse registration order.
    """

    def __init__(self) -> None:
        self._sagas: dict[str, SagaState] = {}

    def create_saga(self, goal_id: str) -> SagaState:
        """Create a new saga for a goal."""
        saga = SagaState(goal_id=goal_id)
        self._sagas[goal_id] = saga
        log.debug("saga.created goal_id=%s", goal_id)
        return saga

    def get_saga(self, goal_id: str) -> SagaState | None:
        return self._sagas.get(goal_id)

    def register_compensation(
        self,
        goal_id: str,
        subtask_id: str,
        action_type: str,
        parameters: dict[str, Any] | None = None,
    ) -> CompensationAction | None:
        """Register a compensation action for a subtask within a goal's saga."""
        saga = self._sagas.get(goal_id)
        if saga is None:
            saga = self.create_saga(goal_id)
        return saga.register(subtask_id, action_type, parameters)

    async def compensate(
        self,
        goal_id: str,
        executor: CompensationExecutor | None = None,
    ) -> CompensationResult:
        """Execute all compensation actions for a goal in reverse order.

        Returns a result summarizing what was compensated and any errors.
        """
        saga = self._sagas.get(goal_id)
        if saga is None:
            return CompensationResult(
                goal_id=goal_id,
                status="no_saga",
                actions_total=0,
                actions_executed=0,
            )

        saga.status = "compensating"
        executor = executor or NoopExecutor()
        errors: list[dict[str, Any]] = []
        executed = 0

        for action in saga.pending_compensations:
            try:
                await executor.execute(action)
                action.executed = True
                executed += 1
                log.info(
                    "saga.compensated goal_id=%s subtask=%s type=%s",
                    goal_id,
                    action.subtask_id,
                    action.action_type,
                )
            except Exception as exc:
                action.error = str(exc)
                errors.append({
                    "subtask_id": action.subtask_id,
                    "action_type": action.action_type,
                    "error": str(exc),
                })
                log.error(
                    "saga.compensation_failed goal_id=%s subtask=%s error=%s",
                    goal_id,
                    action.subtask_id,
                    exc,
                )

        saga.status = "compensated" if not errors else "failed"
        saga.completed_at = time.time()

        return CompensationResult(
            goal_id=goal_id,
            status=saga.status,
            actions_total=len(saga.actions),
            actions_executed=executed,
            errors=errors,
        )

    def remove_saga(self, goal_id: str) -> None:
        """Remove a completed saga from tracking."""
        self._sagas.pop(goal_id, None)

    def active_sagas(self) -> list[dict[str, Any]]:
        """Return summaries of all active sagas."""
        return [s.to_dict() for s in self._sagas.values()]


@dataclass
class CompensationResult:
    """Outcome of a compensation run."""

    goal_id: str
    status: str  # "compensated", "failed", "no_saga"
    actions_total: int
    actions_executed: int
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "status": self.status,
            "actions_total": self.actions_total,
            "actions_executed": self.actions_executed,
            "errors": self.errors,
        }


class CompensationExecutor:
    """Base class for executing compensation actions."""

    async def execute(self, action: CompensationAction) -> None:
        raise NotImplementedError


class NoopExecutor(CompensationExecutor):
    """Default executor that logs but takes no action.

    Real implementations would retract claims, delete outputs, etc.
    """

    async def execute(self, action: CompensationAction) -> None:
        log.debug(
            "saga.noop_execute subtask=%s type=%s params=%s",
            action.subtask_id,
            action.action_type,
            action.parameters,
        )


# ── Singleton ──────────────────────────────────────────────────────────────

_saga_coordinator = SagaCoordinator()


def get_saga_coordinator() -> SagaCoordinator:
    return _saga_coordinator

"""Goal and task decomposition models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExecutionContract(BaseModel):
    """Machine-checkable success criteria for a subtask."""

    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    timeout_seconds: int = 120
    max_retries: int = 3
    fallback_strategy: str | None = None


class Subtask(BaseModel):
    """A single unit of work within a goal decomposition."""

    subtask_id: str = Field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:12]}")
    description: str
    task_type: Literal[
        "research",
        "analysis",
        "fact_check",
        "synthesis",
        "code_execution",
        "web_search",
        "document_generation",
    ]
    depends_on: list[str] = Field(default_factory=list)
    model_tier: Literal["fast", "balanced", "powerful", "local"] = "balanced"
    tools_required: list[str] = Field(default_factory=list)
    contract: ExecutionContract = Field(default_factory=ExecutionContract)
    assigned_service_id: str | None = None
    assigned_model: str | None = None


class GoalDecomposition(BaseModel):
    """LLM-generated plan for achieving a goal."""

    goal_id: str
    original_description: str
    strategy: str = Field(description="Human-readable explanation of the approach")
    subtasks: list[Subtask]
    assumptions: list[str] = Field(default_factory=list)
    estimated_cost_usd: float = 0.0
    estimated_time_seconds: int = 0


class SubtaskResult(BaseModel):
    """Result of executing a subtask."""

    subtask_id: str
    goal_id: str
    status: Literal["completed", "failed", "recovered"]
    output: dict[str, Any] = Field(default_factory=dict)
    model_used: str = ""
    tokens_used: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
    cost_usd: float = 0.0
    tool_calls: list[dict] = Field(default_factory=list)
    verification_result: dict | None = None
    recovery_history: list[dict] = Field(default_factory=list)


class InvalidTransition(Exception):
    """Raised when attempting an illegal goal state transition."""


# Valid state transitions: from_status -> {allowed_to_statuses}
_GOAL_TRANSITIONS: dict[str, set[str]] = {
    "planning": {"executing", "failed", "paused"},
    "executing": {"completed", "failed", "paused"},
    "paused": {"executing", "failed"},
    "completed": set(),  # terminal
    "failed": set(),  # terminal
}


class GoalState(BaseModel):
    """Tracks the execution state of a goal."""

    goal_id: str = Field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:12]}")
    description: str = ""
    status: Literal["planning", "executing", "completed", "failed", "paused"] = (
        "planning"
    )
    decomposition: GoalDecomposition | None = None
    subtask_states: dict[str, str] = Field(default_factory=dict)
    subtask_results: dict[str, SubtaskResult] = Field(default_factory=dict)
    checkpoints: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    due_at: datetime | None = None
    project_id: str | None = None
    tags: list[str] = Field(default_factory=list)

    def transition_to(self, new_status: str) -> None:
        """Validated state transition. Raises InvalidTransition if illegal."""
        allowed = _GOAL_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise InvalidTransition(
                f"Cannot transition goal {self.goal_id} "
                f"from '{self.status}' to '{new_status}'. "
                f"Allowed: {allowed or 'none (terminal state)'}"
            )
        self.status = new_status  # type: ignore[assignment]
        if new_status == "executing" and self.started_at is None:
            self.started_at = datetime.now(UTC)
        if new_status in ("completed", "failed"):
            self.completed_at = datetime.now(UTC)


class Project(BaseModel):
    """A project groups related goals for tracking and reporting."""

    project_id: str = Field(default_factory=lambda: f"proj_{uuid.uuid4().hex[:12]}")
    name: str
    description: str = ""
    owner: str = ""
    status: Literal["active", "completed", "archived"] = "active"
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

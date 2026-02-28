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
    completed_at: datetime | None = None

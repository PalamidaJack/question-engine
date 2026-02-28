"""Typed Pydantic response models for API endpoints.

Enables FastAPI auto-OpenAPI schema generation and contract testing.
These models define the response shapes that consumers can rely on.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Health & Status ────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class ServiceInfo(BaseModel):
    service_id: str
    display_name: str
    status: str
    turn_count: int
    circuit_broken: bool


class BudgetInfo(BaseModel):
    total_spend: float
    remaining_pct: float
    limit_usd: float
    by_model: dict[str, float]


class StatusResponse(BaseModel):
    services: list[ServiceInfo]
    budget: BudgetInfo


# ── Claims ─────────────────────────────────────────────────────────────────


class ClaimResponse(BaseModel):
    claim_id: str
    subject_entity_id: str
    predicate: str
    object_value: str
    confidence: float
    created_at: str
    source_service_id: str = ""
    superseded_by: str | None = None
    tags: list[str] = Field(default_factory=list)


class ClaimsListResponse(BaseModel):
    claims: list[ClaimResponse]
    count: int


# ── Goals ──────────────────────────────────────────────────────────────────


class GoalSummary(BaseModel):
    goal_id: str
    description: str
    status: str
    subtask_count: int
    created_at: str
    completed_at: str | None = None


class GoalsListResponse(BaseModel):
    goals: list[GoalSummary]
    count: int


class GoalSubmitResponse(BaseModel):
    goal_id: str
    status: str
    subtask_count: int
    strategy: str = ""


# ── Projects ───────────────────────────────────────────────────────────────


class ProjectSummary(BaseModel):
    project_id: str
    name: str
    status: str
    goal_count: int = 0
    completed_goals: int = 0


class ProjectsListResponse(BaseModel):
    projects: list[ProjectSummary]
    count: int


# ── Events ─────────────────────────────────────────────────────────────────


class EventsListResponse(BaseModel):
    events: list[dict[str, Any]]
    count: int


# ── DLQ ────────────────────────────────────────────────────────────────────


class DLQEntry(BaseModel):
    envelope_id: str
    topic: str
    source_service_id: str = ""
    handler_name: str = ""
    error: str = ""
    attempts: int = 0
    failed_at: float = 0.0
    payload: dict[str, Any] = Field(default_factory=dict)


class DLQListResponse(BaseModel):
    entries: list[DLQEntry]
    count: int


# ── Generic ────────────────────────────────────────────────────────────────


class SubmitResponse(BaseModel):
    envelope_id: str
    status: str


class ErrorResponse(BaseModel):
    error: str
    code: str = ""


class SuccessResponse(BaseModel):
    status: str

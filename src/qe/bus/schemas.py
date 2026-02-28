"""Event payload schemas for typed bus communication.

Provides Pydantic models for critical topic payloads, enabling
validation on publish/subscribe and schema versioning.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ── Payload schemas per topic ──────────────────────────────────────────────


class ClaimProposedPayload(BaseModel):
    """Payload for claims.proposed topic."""

    claim_id: str = ""
    schema_version: str = "1.0"
    subject_entity_id: str
    predicate: str
    object_value: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_service_id: str = ""
    source_envelope_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimCommittedPayload(ClaimProposedPayload):
    """Payload for claims.committed topic."""

    created_at: str = ""


class GoalSubmittedPayload(BaseModel):
    """Payload for goals.submitted topic."""

    goal_id: str
    description: str


class GoalCompletedPayload(BaseModel):
    """Payload for goals.completed topic."""

    goal_id: str
    subtask_count: int = 0


class GoalFailedPayload(BaseModel):
    """Payload for goals.failed topic."""

    goal_id: str
    reason: str = ""


class TaskDispatchedPayload(BaseModel):
    """Payload for tasks.dispatched topic."""

    goal_id: str
    subtask_id: str
    description: str = ""
    task_type: str = ""
    model_tier: str = "balanced"
    depends_on: list[str] = Field(default_factory=list)
    contract: dict[str, Any] = Field(default_factory=dict)


class SystemErrorPayload(BaseModel):
    """Payload for system.error topic."""

    envelope_id: str = ""
    topic: str = ""
    error: str = ""


class SystemCircuitBreakPayload(BaseModel):
    """Payload for system.circuit_break topic."""

    service_id: str
    reason: str = ""
    count: int = 0
    window_seconds: int = 60


class HeartbeatPayload(BaseModel):
    """Payload for system.heartbeat topic."""

    turn_count: int = 0
    status: str = "alive"


class DLQPayload(BaseModel):
    """Payload for system.dlq topic."""

    envelope_id: str
    topic: str
    source_service_id: str = ""
    handler_name: str = ""
    error: str = ""
    attempts: int = 0
    failed_at: float = 0.0
    payload: dict[str, Any] = Field(default_factory=dict)


class GateDeniedPayload(BaseModel):
    """Payload for system.gate_denied topic."""

    reason: str = ""
    risk_score: float = 0.0
    patterns: list[str] = Field(default_factory=list)


# ── Schema Registry ────────────────────────────────────────────────────────

# Maps topic -> payload model for validation
TOPIC_SCHEMAS: dict[str, type[BaseModel]] = {
    "claims.proposed": ClaimProposedPayload,
    "claims.committed": ClaimCommittedPayload,
    "goals.submitted": GoalSubmittedPayload,
    "goals.completed": GoalCompletedPayload,
    "goals.failed": GoalFailedPayload,
    "tasks.dispatched": TaskDispatchedPayload,
    "system.error": SystemErrorPayload,
    "system.circuit_break": SystemCircuitBreakPayload,
    "system.heartbeat": HeartbeatPayload,
    "system.dlq": DLQPayload,
    "system.gate_denied": GateDeniedPayload,
}


def validate_payload(topic: str, payload: dict[str, Any]) -> BaseModel | None:
    """Validate a payload against its topic schema.

    Returns the validated model if schema exists, None if no schema
    registered for this topic. Raises ValidationError on bad data.
    """
    schema = TOPIC_SCHEMAS.get(topic)
    if schema is None:
        return None
    return schema.model_validate(payload)


def get_schema(topic: str) -> type[BaseModel] | None:
    """Look up the schema model for a topic."""
    return TOPIC_SCHEMAS.get(topic)

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    claim_id: str = Field(default_factory=lambda: f"clm_{uuid.uuid4().hex[:12]}")
    schema_version: str = "1.0"
    subject_entity_id: str
    predicate: str
    object_value: str
    confidence: float
    source_service_id: str
    source_envelope_ids: list[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None
    superseded_by: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class Prediction(BaseModel):
    prediction_id: str = Field(default_factory=lambda: f"prd_{uuid.uuid4().hex[:12]}")
    schema_version: str = "1.0"
    statement: str
    confidence: float
    resolution_criteria: str
    resolution_deadline: datetime | None = None
    source_service_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None
    resolution: Literal["confirmed", "denied", "unresolved"] = "unresolved"
    resolution_evidence_ids: list[str] = Field(default_factory=list)


class NullResult(BaseModel):
    null_result_id: str = Field(default_factory=lambda: f"nul_{uuid.uuid4().hex[:12]}")
    schema_version: str = "1.0"
    query: str
    search_scope: str
    source_service_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    significance: Literal["low", "medium", "high"] = "low"
    notes: str = ""

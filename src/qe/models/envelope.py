from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any
import uuid


class Envelope(BaseModel):
    envelope_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = "1.0"
    topic: str
    source_service_id: str
    correlation_id: str | None = None
    causation_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: dict[str, Any]
    ttl_seconds: int | None = None

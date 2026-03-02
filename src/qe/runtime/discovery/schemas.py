"""Data models for model discovery."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DiscoveredModel(BaseModel):
    """A model discovered from a provider API."""

    model_id: str  # e.g. "openrouter/google/gemini-2.0-flash-exp:free"
    provider: str  # e.g. "openrouter"
    base_model_name: str  # e.g. "gemini-2.0-flash-exp"
    is_free: bool = True
    context_length: int = 4096
    supports_tool_calling: bool = False
    supports_json_mode: bool = False
    supports_system_messages: bool = True
    supports_streaming: bool = True
    quality_tier: str = "fast"  # fast / balanced / powerful
    cost_per_m_input: float = 0.0
    cost_per_m_output: float = 0.0
    rate_limit_rpm: int = 60
    discovered_at: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)
    status: Literal["active", "degraded", "gone"] = "active"


class ModelHealthMetrics(BaseModel):
    """Rolling-window health metrics for a single model."""

    model_id: str
    total_calls: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_quality_score: float = 0.5
    last_error: str = ""
    last_error_at: datetime | None = None
    window_start: datetime = Field(default_factory=datetime.now)
    # Rolling buffer of recent latencies for p95 computation
    _latencies: list[float] = []

    model_config = {"arbitrary_types_allowed": True}


class TierAssignment(BaseModel):
    """Maps a system tier to a primary model + ordered fallbacks."""

    tier: str  # fast / balanced / powerful
    primary: str  # model_id
    fallbacks: list[str] = Field(default_factory=list)
    reason: str = ""
    assigned_at: datetime = Field(default_factory=datetime.now)
    auto_assigned: bool = True  # False if user manually pinned

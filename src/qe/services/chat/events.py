"""Typed chat progress events for SSE streaming.

Defines a small family of Pydantic models used by the SSE endpoint.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatProgressEvent(BaseModel):
    type: Literal["llm_start", "llm_complete", "tool_start", "tool_complete", "error", "complete"]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    iteration: int = 0


class LLMCompleteEvent(ChatProgressEvent):
    type: Literal["llm_complete"] = "llm_complete"
    model: str = ""
    call_tokens: dict = Field(default_factory=dict)
    call_cost_usd: float = 0.0
    has_tool_calls: bool = False


class ToolCompleteEvent(ChatProgressEvent):
    type: Literal["tool_complete"] = "tool_complete"
    tool_name: str = ""
    result_preview: str = ""
    duration_ms: float = 0.0


class ErrorEvent(ChatProgressEvent):
    type: Literal["error"] = "error"
    message: str = ""


class CompleteEvent(ChatProgressEvent):
    type: Literal["complete"] = "complete"
    summary: str = ""

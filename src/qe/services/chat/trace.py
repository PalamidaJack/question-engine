"""Structured execution traces for chat agent loops."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMCallTrace(BaseModel):
    """Trace of a single LLM call within the agent loop."""

    iteration: int = 0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    has_tool_calls: bool = False


class ToolCallTrace(BaseModel):
    """Trace of a single tool call."""

    iteration: int = 0
    tool_name: str = ""
    duration_ms: float = 0.0
    blocked: bool = False
    error: bool = False
    cache_hit: bool = False
    result_size: int = 0


class RoutingDecision(BaseModel):
    """Record of a model routing decision."""

    iteration: int = 0
    selected_model: str = ""
    task_hint: str | None = None
    reason: str = ""


class ChatTrace(BaseModel):
    """Full structured trace of a chat agent loop execution."""

    session_id: str = ""
    message_id: str = ""
    llm_calls: list[LLMCallTrace] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    routing_decisions: list[RoutingDecision] = Field(default_factory=list)
    total_iterations: int = 0
    total_tool_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: float = 0.0
    outcome: str = ""  # "success", "timeout", "error", "max_iterations"

    def summary(self) -> dict:
        """Human-readable summary of the trace."""
        return {
            "iterations": self.total_iterations,
            "tool_calls": self.total_tool_calls,
            "tokens": self.total_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
            "duration_ms": round(self.total_duration_ms, 1),
            "outcome": self.outcome,
            "models_used": list({lc.model for lc in self.llm_calls}),
            "tools_used": list({tc.tool_name for tc in self.tool_calls}),
        }

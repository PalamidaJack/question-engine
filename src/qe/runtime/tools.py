"""Tool registry with capability enforcement and lazy discovery."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


@dataclass
class _ToolStats:
    """Per-tool aggregate statistics."""

    name: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    last_error: str | None = None
    last_called_at: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.total_calls if self.total_calls else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_calls if self.total_calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "last_error": self.last_error,
            "last_called_at": self.last_called_at,
        }


class ToolMetrics:
    """Collects per-tool quality metrics (success rate, latency, errors)."""

    def __init__(self) -> None:
        self._stats: dict[str, _ToolStats] = {}

    def record_success(self, name: str, latency_ms: float) -> None:
        s = self._stats.setdefault(name, _ToolStats(name=name))
        s.total_calls += 1
        s.successes += 1
        s.total_latency_ms += latency_ms
        s.last_called_at = time.time()

    def record_failure(self, name: str, latency_ms: float, error: str) -> None:
        s = self._stats.setdefault(name, _ToolStats(name=name))
        s.total_calls += 1
        s.failures += 1
        s.total_latency_ms += latency_ms
        s.last_error = error
        s.last_called_at = time.time()

    def get_stats(self, name: str) -> dict[str, Any] | None:
        s = self._stats.get(name)
        return s.to_dict() if s else None

    def all_stats(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in sorted(self._stats.values(), key=lambda s: s.name)]

    def summary(self) -> dict[str, Any]:
        total = sum(s.total_calls for s in self._stats.values())
        errors = sum(s.failures for s in self._stats.values())
        return {
            "total_tool_calls": total,
            "total_errors": errors,
            "overall_success_rate": round((total - errors) / total, 4) if total else 1.0,
            "tools_tracked": len(self._stats),
        }


class ToolSpec(BaseModel):
    """Specification for a registered tool."""

    name: str
    description: str
    requires_capability: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 30
    ephemeral_output: bool = True
    category: str = ""
    cacheable: bool = False
    cache_ttl_seconds: int = 300
    preferred_tier: str = ""  # e.g. "fast", "balanced", "powerful"

    class Config:
        arbitrary_types_allowed = True


class ToolRegistry:
    """Plugin-based tool registry with capability enforcement."""

    def __init__(self, tool_metrics: ToolMetrics | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[..., Coroutine]] = {}
        self._tool_metrics = tool_metrics

    def register(
        self,
        spec: ToolSpec,
        handler: Callable[..., Coroutine],
    ) -> None:
        """Register a tool with its handler."""
        self._tools[spec.name] = spec
        self._handlers[spec.name] = handler
        log.debug("tool_registry.registered name=%s category=%s", spec.name, spec.category)

    def get_tool(self, name: str) -> ToolSpec | None:
        """Get a tool specification by name."""
        return self._tools.get(name)

    def get_handler(self, name: str) -> Callable[..., Coroutine] | None:
        """Get a tool handler by name."""
        return self._handlers.get(name)

    def get_tools_for_capabilities(
        self, capabilities: set[str]
    ) -> list[ToolSpec]:
        """Return tools allowed by the given capability set."""
        result = []
        for spec in self._tools.values():
            if spec.requires_capability is None:
                result.append(spec)
            elif spec.requires_capability in capabilities:
                result.append(spec)
        return result

    def get_tool_schemas(
        self,
        capabilities: set[str],
        mode: str = "discovery",
    ) -> list[dict]:
        """Return tool schemas for LLM function calling.

        mode="discovery": Returns only the meta-tool (search_available_tools)
        mode="direct": Returns all available tool schemas
        mode="relevant": Pre-filtered by capabilities
        """
        if mode == "discovery":
            return [{
                "type": "function",
                "function": {
                    "name": "search_available_tools",
                    "description": (
                        "Describe what you need to do, and this tool will "
                        "return the specific tools available for that task."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "What you need to accomplish",
                            }
                        },
                        "required": ["description"],
                    },
                },
            }]

        tools = self.get_tools_for_capabilities(capabilities)
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    def search_tools(
        self, description: str, capabilities: set[str]
    ) -> list[ToolSpec]:
        """Find tools matching a natural language description."""
        available = self.get_tools_for_capabilities(capabilities)
        lower = description.lower()

        # Simple keyword matching
        scored: list[tuple[int, ToolSpec]] = []
        for tool in available:
            score = 0
            for word in lower.split():
                if word in tool.name.lower():
                    score += 3
                if word in tool.description.lower():
                    score += 1
                if word in tool.category.lower():
                    score += 2
            if score > 0:
                scored.append((score, tool))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in scored]

    def list_all(self) -> list[ToolSpec]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_tier_for_tool(self, name: str) -> str | None:
        """Return the preferred model tier for a tool, or None."""
        spec = self._tools.get(name)
        return spec.preferred_tier if spec and spec.preferred_tier else None

    async def execute(
        self, name: str, params: dict[str, Any]
    ) -> Any:
        """Execute a tool by name with the given parameters."""
        handler = self._handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")

        start = time.monotonic()
        try:
            result = await handler(**params)
            if self._tool_metrics is not None:
                elapsed = (time.monotonic() - start) * 1000
                self._tool_metrics.record_success(name, elapsed)
            return result
        except Exception as exc:
            if self._tool_metrics is not None:
                elapsed = (time.monotonic() - start) * 1000
                self._tool_metrics.record_failure(name, elapsed, str(exc))
            raise

    # ── Progressive tool loading (#62) ───────────────────────────────

    def get_schemas_for_intent(
        self, intent: str, *, core_tools: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return tool schemas relevant to the detected intent.

        Starts with core tools, adds specialized ones on demand.
        Gated behind ``progressive_tool_loading`` feature flag.
        """
        core = set(core_tools or [
            "query_beliefs", "list_entities",
            "get_entity_details", "submit_observation",
        ])

        # Intent-to-tool mapping
        intent_tools: dict[str, list[str]] = {
            "research": [
                "deep_research", "swarm_research",
                "crystallize_insights",
            ],
            "analysis": [
                "reason_about", "crystallize_insights",
            ],
            "planning": ["plan_and_execute"],
            "knowledge": [
                "consolidate_knowledge", "query_beliefs",
            ],
            "delegation": ["delegate_to_agent"],
            "budget": ["get_budget_status"],
        }

        active_names = set(core)
        if intent in intent_tools:
            active_names.update(intent_tools[intent])

        schemas = []
        for name, spec in self._tools.items():
            if name in active_names:
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.input_schema,
                    },
                })
        return schemas

    def tool_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

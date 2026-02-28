"""Tool registry with capability enforcement and lazy discovery."""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


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

    class Config:
        arbitrary_types_allowed = True


class ToolRegistry:
    """Plugin-based tool registry with capability enforcement."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[..., Coroutine]] = {}

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

    async def execute(
        self, name: str, params: dict[str, Any]
    ) -> Any:
        """Execute a tool by name with the given parameters."""
        handler = self._handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")
        return await handler(**params)

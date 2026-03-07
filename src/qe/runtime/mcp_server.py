"""QE as MCP Server — expose cognitive tools via MCP protocol.

Allows Claude Desktop and other MCP-compatible agents to call QE's
cognitive tools via stdio/SSE transport.
Gated behind ``mcp_server`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class MCPToolDefinition:
    """An MCP tool definition exposed by QE."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPCallResult:
    """Result from an MCP tool call."""

    tool_name: str
    success: bool
    result: Any = None
    error: str = ""


class MCPServer:
    """MCP server exposing QE cognitive tools."""

    def __init__(
        self,
        tool_registry: Any = None,
        name: str = "question-engine",
        version: str = "1.0.0",
    ) -> None:
        self._tool_registry = tool_registry
        self._name = name
        self._version = version
        self._exposed_tools: dict[str, MCPToolDefinition] = {}

    def register_tool(self, tool: MCPToolDefinition) -> None:
        self._exposed_tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self._exposed_tools.values()
        ]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> MCPCallResult:
        """Execute an MCP tool call."""
        if tool_name not in self._exposed_tools:
            return MCPCallResult(
                tool_name=tool_name, success=False,
                error=f"Unknown tool: {tool_name}",
            )

        if self._tool_registry is None:
            return MCPCallResult(
                tool_name=tool_name, success=False,
                error="No tool registry configured",
            )

        try:
            result = await self._tool_registry.execute(
                tool_name, arguments,
            )
            return MCPCallResult(
                tool_name=tool_name, success=True,
                result=result,
            )
        except Exception as e:
            return MCPCallResult(
                tool_name=tool_name, success=False,
                error=str(e),
            )

    def server_info(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "version": self._version,
            "tools": len(self._exposed_tools),
            "protocol": "mcp",
        }

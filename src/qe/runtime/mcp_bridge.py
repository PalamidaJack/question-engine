"""MCP Bridge: connects to MCP servers via stdio JSON-RPC and registers tools."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    capability: str = "mcp"


class _MCPConnection:
    """Manages a single MCP server subprocess and JSON-RPC communication."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._tools: list[dict] = []

    async def start(self) -> list[dict]:
        """Start the subprocess, initialize, and discover tools."""
        import os

        env = {**os.environ, **self.config.env}
        self._process = await asyncio.create_subprocess_exec(
            self.config.command,
            *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        # JSON-RPC initialize
        init_resp = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "qe-mcp-bridge", "version": "1.0.0"},
        })
        if init_resp is None:
            raise ConnectionError(
                f"MCP server {self.config.name} failed to initialize"
            )
        # Send initialized notification
        await self._send_notification("notifications/initialized")
        # Discover tools
        tools_resp = await self._send_request("tools/list", {})
        if tools_resp and "tools" in tools_resp:
            self._tools = tools_resp["tools"]
        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool on the MCP server and return text content."""
        resp = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        if resp is None:
            return f"MCP tool '{name}' returned no response."
        content = resp.get("content", [])
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts) if parts else json.dumps(resp)

    def stop(self) -> None:
        """Terminate the subprocess."""
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
            except Exception:
                pass

    async def _send_request(
        self, method: str, params: dict,
    ) -> dict | None:
        """Send a JSON-RPC request and read the response."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            return None
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(request) + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()
        try:
            raw = await asyncio.wait_for(
                self._process.stdout.readline(), timeout=30.0,
            )
        except TimeoutError:
            return None
        if not raw:
            return None
        try:
            resp = json.loads(raw.decode())
            return resp.get("result")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    async def _send_notification(self, method: str) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return
        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        line = json.dumps(notification) + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()


class MCPBridge:
    """Connects to MCP servers and registers their tools in ToolRegistry."""

    def __init__(
        self,
        configs: list[MCPServerConfig],
        tool_registry: Any | None = None,
    ) -> None:
        self._configs = configs
        self._tool_registry = tool_registry
        self._connections: dict[str, _MCPConnection] = {}

    async def start(self) -> int:
        """Start all MCP servers and register discovered tools.

        Returns the number of tools registered.
        """
        total_registered = 0
        for config in self._configs:
            conn = _MCPConnection(config)
            try:
                tools = await conn.start()
                self._connections[config.name] = conn
                for tool in tools:
                    self._register_tool(config, conn, tool)
                    total_registered += 1
                log.info(
                    "mcp_bridge.connected server=%s tools=%d",
                    config.name, len(tools),
                )
            except Exception:
                log.warning(
                    "mcp_bridge.connect_failed server=%s",
                    config.name, exc_info=True,
                )
                conn.stop()
        return total_registered

    def stop(self) -> None:
        """Stop all MCP server connections."""
        for name, conn in self._connections.items():
            try:
                conn.stop()
                log.debug("mcp_bridge.stopped server=%s", name)
            except Exception:
                pass
        self._connections.clear()

    def _register_tool(
        self,
        config: MCPServerConfig,
        conn: _MCPConnection,
        tool_def: dict,
    ) -> None:
        """Register a single MCP tool in the ToolRegistry."""
        if not self._tool_registry:
            return
        from qe.runtime.tools import ToolSpec

        tool_name = tool_def.get("name", "")
        if not tool_name:
            return

        spec = ToolSpec(
            name=f"mcp_{config.name}_{tool_name}",
            description=tool_def.get("description", f"MCP tool: {tool_name}"),
            requires_capability=config.capability,
            input_schema=tool_def.get("inputSchema", {}),
            timeout_seconds=60,
            category=f"mcp:{config.name}",
        )

        async def _handler(
            _conn=conn, _name=tool_name, **kwargs,
        ) -> str:
            return await _conn.call_tool(_name, kwargs)

        self._tool_registry.register(spec, _handler)

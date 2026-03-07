"""Runtime context: live references to system state for capability awareness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MCPServerSummary:
    """Snapshot of a connected MCP server."""

    name: str
    tool_count: int
    tool_names: list[str] = field(default_factory=list)
    alive: bool = True


@dataclass
class RuntimeContext:
    """Lightweight container holding live references to runtime objects.

    Queries always reflect current state — no caching.
    """

    workspace_root: Path | None = None
    project_root: Path | None = None
    mcp_bridge: Any | None = None       # live MCPBridge reference
    peer_registry: Any | None = None    # live PeerRegistry reference

    def mcp_server_summaries(self) -> list[MCPServerSummary]:
        """Return summaries of all connected MCP servers."""
        bridge = self.mcp_bridge
        if bridge is None:
            return []
        connections: dict = getattr(bridge, "_connections", {})
        summaries: list[MCPServerSummary] = []
        for name, conn in connections.items():
            tools: list[dict] = getattr(conn, "_tools", [])
            tool_names = [
                t.get("name", "") for t in tools
            ]
            summaries.append(MCPServerSummary(
                name=name,
                tool_count=len(tools),
                tool_names=tool_names,
                alive=getattr(conn, "is_alive", False),
            ))
        return summaries

    def peer_count(self) -> int:
        """Total registered peers."""
        reg = self.peer_registry
        if reg is None:
            return 0
        peers = getattr(reg, "_peers", {})
        return len(peers)

    def healthy_peer_count(self) -> int:
        """Count of healthy peers."""
        reg = self.peer_registry
        if reg is None:
            return 0
        try:
            return len(reg.list_healthy())
        except Exception:
            return 0

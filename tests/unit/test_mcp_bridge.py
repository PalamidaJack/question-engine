"""Tests for the MCP Bridge."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.runtime.mcp_bridge import MCPBridge, MCPServerConfig, _MCPConnection


def _mock_process(tools: list[dict] | None = None):
    """Create a mock subprocess with stdin/stdout for JSON-RPC."""
    proc = MagicMock()
    proc.returncode = None
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()

    responses = []
    # Response to initialize
    responses.append(json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"protocolVersion": "2024-11-05", "capabilities": {}},
    }).encode() + b"\n")
    # Response to tools/list
    tool_list = tools or []
    responses.append(json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "result": {"tools": tool_list},
    }).encode() + b"\n")

    call_idx = {"i": 0}

    async def readline():
        idx = call_idx["i"]
        call_idx["i"] += 1
        if idx < len(responses):
            return responses[idx]
        return b""

    proc.stdout = MagicMock()
    proc.stdout.readline = readline
    proc.stderr = MagicMock()
    proc.terminate = MagicMock()
    return proc


class TestMCPBridge:
    @pytest.mark.asyncio
    async def test_registers_tools(self):
        """MCPBridge discovers tools from server and registers them."""
        tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
            {
                "name": "write_file",
                "description": "Write a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                },
            },
        ]
        proc = _mock_process(tools)
        registry = MagicMock()

        config = MCPServerConfig(name="filesystem", command="mcp-fs")
        bridge = MCPBridge(configs=[config], tool_registry=registry)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            count = await bridge.start()

        assert count == 2
        assert registry.register.call_count == 2
        # Verify tool names are prefixed
        registered_names = [
            call.args[0].name for call in registry.register.call_args_list
        ]
        assert "mcp_filesystem_read_file" in registered_names
        assert "mcp_filesystem_write_file" in registered_names

    @pytest.mark.asyncio
    async def test_calls_tool(self):
        """MCPBridge dispatches tool calls to the server."""
        tools = [{"name": "echo", "description": "Echo back", "inputSchema": {}}]
        proc = _mock_process(tools)

        # Add a response for tools/call
        call_response = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {
                "content": [{"type": "text", "text": "hello world"}],
            },
        }).encode() + b"\n"

        # Override readline to also return the call response
        original_readline = proc.stdout.readline
        call_count = {"i": 0}

        async def extended_readline():
            call_count["i"] += 1
            if call_count["i"] <= 2:
                return await original_readline()
            return call_response

        proc.stdout.readline = extended_readline

        config = MCPServerConfig(name="test", command="mcp-test")
        conn = _MCPConnection(config)

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await conn.start()
            result = await conn.call_tool("echo", {"text": "hello"})

        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_handles_connection_failure(self):
        """MCPBridge handles server connection failure gracefully."""
        config = MCPServerConfig(name="broken", command="nonexistent")
        registry = MagicMock()
        bridge = MCPBridge(configs=[config], tool_registry=registry)

        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=FileNotFoundError("not found")),
        ):
            count = await bridge.start()

        assert count == 0
        registry.register.assert_not_called()

    def test_stop_terminates_processes(self):
        """MCPBridge.stop() terminates all server processes."""
        config = MCPServerConfig(name="test", command="mcp-test")
        bridge = MCPBridge(configs=[config])

        # Manually inject a mock connection
        mock_conn = MagicMock()
        bridge._connections["test"] = mock_conn

        bridge.stop()

        mock_conn.stop.assert_called_once()
        assert len(bridge._connections) == 0

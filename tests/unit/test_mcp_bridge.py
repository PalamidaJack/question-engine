"""Tests for the MCP Bridge — reconnection, adaptive timeouts, health loop."""

from unittest.mock import MagicMock

import pytest

from qe.runtime.mcp_bridge import MCPServerConfig, _MCPConnection


class TestMCPConnectionLiveness:
    def test_is_alive_when_process_running(self):
        config = MCPServerConfig(name="test", command="echo")
        conn = _MCPConnection(config)
        conn._process = MagicMock()
        conn._process.returncode = None
        assert conn.is_alive is True

    def test_not_alive_when_process_exited(self):
        config = MCPServerConfig(name="test", command="echo")
        conn = _MCPConnection(config)
        conn._process = MagicMock()
        conn._process.returncode = 1
        assert conn.is_alive is False

    def test_not_alive_when_no_process(self):
        config = MCPServerConfig(name="test", command="echo")
        conn = _MCPConnection(config)
        assert conn.is_alive is False


class TestMCPReconnection:
    @pytest.mark.asyncio
    async def test_ensure_alive_when_already_alive(self):
        config = MCPServerConfig(name="test", command="echo")
        conn = _MCPConnection(config)
        conn._process = MagicMock()
        conn._process.returncode = None
        assert await conn.ensure_alive() is True
        assert conn._restart_count == 0

    @pytest.mark.asyncio
    async def test_max_restarts_exceeded(self):
        config = MCPServerConfig(name="test", command="echo")
        conn = _MCPConnection(config)
        conn._restart_count = 3
        conn._process = MagicMock()
        conn._process.returncode = 1
        assert await conn.ensure_alive() is False


class TestAdaptiveTimeouts:
    def test_default_timeout_with_no_history(self):
        config = MCPServerConfig(name="test", command="echo")
        conn = _MCPConnection(config)
        assert conn.get_adaptive_timeout("some_tool") == 30.0

    def test_adaptive_timeout_from_latency_history(self):
        config = MCPServerConfig(name="test", command="echo")
        conn = _MCPConnection(config)
        for _ in range(10):
            conn.record_tool_latency("fast_tool", 2.0)
        timeout = conn.get_adaptive_timeout("fast_tool")
        assert timeout == 10.0  # p95=2.0, ×3=6.0, floor=10.0

    def test_adaptive_timeout_respects_ceiling(self):
        config = MCPServerConfig(name="test", command="echo")
        conn = _MCPConnection(config)
        for _ in range(10):
            conn.record_tool_latency("slow_tool", 50.0)
        timeout = conn.get_adaptive_timeout("slow_tool")
        assert timeout == 120.0  # p95=50, ×3=150, ceiling=120

    def test_record_latency_caps_at_50_samples(self):
        config = MCPServerConfig(name="test", command="echo")
        conn = _MCPConnection(config)
        for i in range(60):
            conn.record_tool_latency("tool", float(i))
        assert len(conn._tool_latencies["tool"]) == 50

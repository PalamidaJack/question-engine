"""Tests for Phase A: Tool infrastructure wiring."""

from __future__ import annotations

import pytest

from qe.runtime.service import BaseService
from qe.runtime.tool_bootstrap import create_default_gate, create_default_registry
from qe.runtime.tool_gate import GateDecision, SecurityPolicy
from qe.runtime.workspace import WorkspaceManager


class TestCreateDefaultRegistry:
    def test_registers_six_tools(self):
        registry = create_default_registry()
        assert len(registry.list_all()) == 6

    def test_tool_names(self):
        registry = create_default_registry()
        names = {t.name for t in registry.list_all()}
        assert names == {
            "web_search", "web_fetch", "file_read",
            "file_write", "code_execute", "browser_navigate",
        }

    def test_tool_schemas_for_web_search(self):
        registry = create_default_registry()
        schemas = registry.get_tool_schemas({"web_search"}, mode="direct")
        names = [s["function"]["name"] for s in schemas]
        assert "web_search" in names
        assert "web_fetch" in names


class TestCreateDefaultGate:
    def test_blocks_localhost(self):
        gate = create_default_gate(policies=[
            SecurityPolicy(
                name="default",
                max_calls_per_goal=50,
                blocked_domains=["localhost", "127.0.0.1"],
            ),
        ])
        result = gate.validate(
            "web_search",
            {"query": "localhost test"},
            capabilities={"web_search"},
            goal_id="g1",
        )
        assert result.decision == GateDecision.DENY

    def test_allows_normal_domains(self):
        gate = create_default_gate(policies=[
            SecurityPolicy(
                name="default",
                max_calls_per_goal=50,
                blocked_domains=["localhost", "127.0.0.1"],
            ),
        ])
        result = gate.validate(
            "web_search",
            {"query": "quantum computing"},
            capabilities={"web_search"},
            goal_id="g1",
        )
        assert result.decision == GateDecision.ALLOW

    def test_denies_missing_capability(self):
        gate = create_default_gate()
        result = gate.validate(
            "web_search",
            {"query": "test"},
            capabilities=set(),
            goal_id="g1",
        )
        assert result.decision == GateDecision.DENY

    def test_rate_limit_exceeded(self):
        gate = create_default_gate(policies=[
            SecurityPolicy(name="default", max_calls_per_goal=2),
        ])
        caps = {"web_search"}
        gate.validate("web_search", {}, caps, goal_id="g1")
        gate.validate("web_search", {}, caps, goal_id="g1")
        result = gate.validate("web_search", {}, caps, goal_id="g1")
        assert result.decision == GateDecision.DENY
        assert "Rate limit" in result.reason


class TestBaseServiceToolRefs:
    def setup_method(self):
        BaseService._shared_tool_registry = None
        BaseService._shared_tool_gate = None

    def teardown_method(self):
        BaseService._shared_tool_registry = None
        BaseService._shared_tool_gate = None

    def test_set_tool_registry(self):
        registry = create_default_registry()
        BaseService.set_tool_registry(registry)
        assert BaseService._shared_tool_registry is registry

    def test_set_tool_gate(self):
        gate = create_default_gate()
        BaseService.set_tool_gate(gate)
        assert BaseService._shared_tool_gate is gate


class TestWorkspaceSandbox:
    def test_sandbox_path_escaping(self, tmp_path):
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        with pytest.raises(ValueError, match="escapes workspace"):
            mgr.sandbox_path("goal_1", "../../etc/passwd")

    def test_sandbox_path_valid(self, tmp_path):
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        path = mgr.sandbox_path("goal_1", "output.txt")
        assert "goal_1" in str(path)
        assert path.name == "output.txt"

"""Tests for the ToolGate — security policy enforcement for tool invocations."""

from qe.runtime.tool_gate import GateDecision, GateResult, SecurityPolicy, ToolGate

# ── Capability checks ─────────────────────────────────────────────────────


class TestCapabilityChecks:
    def test_allow_when_capability_present(self):
        gate = ToolGate()
        result = gate.validate("web_search", {}, {"web_search"})
        assert result.decision == GateDecision.ALLOW

    def test_deny_web_search_without_capability(self):
        gate = ToolGate()
        result = gate.validate("web_search", {}, set())
        assert result.decision == GateDecision.DENY
        assert "web_search" in result.reason
        assert result.policy_name == "capability_check"

    def test_deny_file_read_without_capability(self):
        gate = ToolGate()
        result = gate.validate("file_read", {"path": "data.txt"}, set())
        assert result.decision == GateDecision.DENY
        assert "file_read" in result.reason

    def test_deny_file_write_without_capability(self):
        gate = ToolGate()
        result = gate.validate("file_write", {"path": "out.txt"}, set())
        assert result.decision == GateDecision.DENY
        assert "file_write" in result.reason

    def test_deny_code_execute_without_capability(self):
        gate = ToolGate()
        result = gate.validate("code_execute", {}, set())
        assert result.decision == GateDecision.DENY
        assert "code_execute" in result.reason

    def test_deny_browser_navigate_without_capability(self):
        gate = ToolGate()
        result = gate.validate("browser_navigate", {}, set())
        assert result.decision == GateDecision.DENY
        assert "browser_control" in result.reason

    def test_web_fetch_requires_web_search_capability(self):
        gate = ToolGate()
        result = gate.validate("web_fetch", {"url": "https://example.com"}, set())
        assert result.decision == GateDecision.DENY
        assert "web_search" in result.reason

    def test_unknown_tool_passes_without_capability(self):
        gate = ToolGate()
        result = gate.validate("custom_tool", {}, set())
        assert result.decision == GateDecision.ALLOW

    def test_empty_capabilities_blocks_known_tools(self):
        gate = ToolGate()
        for tool in ("web_search", "web_fetch", "file_read", "file_write",
                      "code_execute", "browser_navigate"):
            result = gate.validate(tool, {}, set())
            assert result.decision == GateDecision.DENY, f"{tool} should be denied"


# ── Domain blocking ───────────────────────────────────────────────────────


class TestDomainBlocking:
    def test_web_fetch_blocked_domain(self):
        policy = SecurityPolicy(name="block_evil", blocked_domains=["evil.com"])
        gate = ToolGate(policies=[policy])
        result = gate.validate(
            "web_fetch", {"url": "https://evil.com/page"}, {"web_search"},
        )
        assert result.decision == GateDecision.DENY
        assert "evil.com" in result.reason
        assert result.policy_name == "block_evil"

    def test_web_search_blocked_domain_in_query(self):
        policy = SecurityPolicy(name="block_evil", blocked_domains=["evil.com"])
        gate = ToolGate(policies=[policy])
        result = gate.validate(
            "web_search", {"query": "site:evil.com how to hack"}, {"web_search"},
        )
        assert result.decision == GateDecision.DENY

    def test_allowed_domain_passes(self):
        policy = SecurityPolicy(name="block_evil", blocked_domains=["evil.com"])
        gate = ToolGate(policies=[policy])
        result = gate.validate(
            "web_fetch", {"url": "https://good.com/page"}, {"web_search"},
        )
        assert result.decision == GateDecision.ALLOW


# ── Rate limiting ─────────────────────────────────────────────────────────


class TestRateLimiting:
    def test_within_limit_allows(self):
        policy = SecurityPolicy(name="rate_limit", max_calls_per_goal=3)
        gate = ToolGate(policies=[policy])
        for _ in range(3):
            result = gate.validate("custom_tool", {}, set(), goal_id="g1")
            assert result.decision == GateDecision.ALLOW

    def test_exceeding_limit_denies(self):
        policy = SecurityPolicy(name="rate_limit", max_calls_per_goal=3)
        gate = ToolGate(policies=[policy])
        for _ in range(3):
            gate.validate("custom_tool", {}, set(), goal_id="g1")
        result = gate.validate("custom_tool", {}, set(), goal_id="g1")
        assert result.decision == GateDecision.DENY
        assert "Rate limit" in result.reason

    def test_different_goals_counted_separately(self):
        policy = SecurityPolicy(name="rate_limit", max_calls_per_goal=1)
        gate = ToolGate(policies=[policy])
        r1 = gate.validate("custom_tool", {}, set(), goal_id="g1")
        r2 = gate.validate("custom_tool", {}, set(), goal_id="g2")
        assert r1.decision == GateDecision.ALLOW
        assert r2.decision == GateDecision.ALLOW

    def test_call_counting_increments(self):
        policy = SecurityPolicy(name="rate_limit", max_calls_per_goal=10)
        gate = ToolGate(policies=[policy])
        gate.validate("custom_tool", {}, set(), goal_id="g1")
        gate.validate("custom_tool", {}, set(), goal_id="g1")
        assert gate._call_counts["g1"]["custom_tool"] == 2

    def test_reset_counts_clears(self):
        policy = SecurityPolicy(name="rate_limit", max_calls_per_goal=10)
        gate = ToolGate(policies=[policy])
        gate.validate("custom_tool", {}, set(), goal_id="g1")
        gate.reset_counts("g1")
        assert "g1" not in gate._call_counts


# ── Path sandboxing ───────────────────────────────────────────────────────


class TestPathSandboxing:
    def test_path_traversal_denied_file_read(self):
        gate = ToolGate()
        result = gate.validate("file_read", {"path": "../../etc/passwd"}, {"file_read"})
        assert result.decision == GateDecision.DENY
        assert result.policy_name == "sandbox_check"

    def test_absolute_path_denied_file_write(self):
        gate = ToolGate()
        result = gate.validate("file_write", {"path": "/etc/passwd"}, {"file_write"})
        assert result.decision == GateDecision.DENY

    def test_relative_path_allowed(self):
        gate = ToolGate()
        result = gate.validate("file_read", {"path": "data/notes.txt"}, {"file_read"})
        assert result.decision == GateDecision.ALLOW


# ── Human-in-the-loop escalation ──────────────────────────────────────────


class TestHumanInTheLoop:
    def test_require_hil_escalates(self):
        policy = SecurityPolicy(name="web_search", require_hil=True)
        gate = ToolGate(policies=[policy])
        result = gate.validate("web_search", {}, {"web_search"})
        assert result.decision == GateDecision.ESCALATE


# ── GateResult model ──────────────────────────────────────────────────────


class TestGateResult:
    def test_gate_result_fields(self):
        result = GateResult(
            decision=GateDecision.DENY, reason="test reason", policy_name="test_policy",
        )
        assert result.decision == GateDecision.DENY
        assert result.reason == "test reason"
        assert result.policy_name == "test_policy"

    def test_gate_result_defaults(self):
        result = GateResult(decision=GateDecision.ALLOW)
        assert result.reason == ""
        assert result.policy_name == ""

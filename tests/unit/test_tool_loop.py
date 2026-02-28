"""Tests for the agentic tool-calling loop in BaseService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.genome import Blueprint, CapabilityDeclaration, ModelPreference
from qe.runtime.service import BaseService
from qe.runtime.tool_bootstrap import create_default_registry
from qe.runtime.tool_gate import GateDecision, GateResult, ToolGate
from qe.runtime.tools import ToolRegistry, ToolSpec

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_tool_call(tool_call_id: str, name: str, arguments: str) -> MagicMock:
    """Create a mock tool_call matching litellm's structure."""
    tc = MagicMock()
    tc.id = tool_call_id
    tc.function.name = name
    tc.function.arguments = arguments
    tc.model_dump.return_value = {
        "id": tool_call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    return tc


def _make_response(content: str | None = None, tool_calls: list | None = None) -> MagicMock:
    """Create a mock litellm ModelResponse."""
    resp = MagicMock()
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
    }
    resp.choices = [MagicMock(message=msg)]
    return resp


def _make_service(
    capabilities: CapabilityDeclaration | None = None,
) -> BaseService:
    """Create a minimal BaseService subclass for testing."""
    bp = Blueprint(
        service_id="test-tool-svc",
        display_name="Test Tool Service",
        version="1.0",
        system_prompt="You are a test agent.",
        model_preference=ModelPreference(tier="balanced"),
        capabilities=capabilities or CapabilityDeclaration(web_search=True),
    )

    class _TestService(BaseService):
        async def handle_response(self, envelope, response):
            pass

        def get_response_schema(self, topic):
            raise NotImplementedError

    return _TestService(bp, MagicMock(), None)


@pytest.fixture(autouse=True)
def _reset_shared_state():
    """Reset BaseService class-level state between tests."""
    BaseService._shared_tool_registry = None
    BaseService._shared_tool_gate = None
    BaseService._shared_budget = None
    yield
    BaseService._shared_tool_registry = None
    BaseService._shared_tool_gate = None
    BaseService._shared_budget = None


def _setup_registry_with_test_tool() -> ToolRegistry:
    """Create a registry with a single test tool."""
    registry = ToolRegistry()
    spec = ToolSpec(
        name="web_search",
        description="Search the web",
        requires_capability="web_search",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    async def _handler(query: str) -> dict:
        return {"results": [f"result for: {query}"]}

    registry.register(spec, _handler)
    return registry


# ── CapabilityDeclaration tests ──────────────────────────────────────────────


class TestCapabilitySet:
    def test_to_capability_set_empty(self):
        caps = CapabilityDeclaration()
        assert caps.to_capability_set() == set()

    def test_to_capability_set_some(self):
        caps = CapabilityDeclaration(
            web_search=True, file_read=True, code_execute=True
        )
        assert caps.to_capability_set() == {"web_search", "file_read", "code_execute"}

    def test_to_capability_set_all(self):
        caps = CapabilityDeclaration(
            web_search=True,
            file_read=True,
            file_write=True,
            code_execute=True,
            browser_control=True,
        )
        assert caps.to_capability_set() == {
            "web_search",
            "file_read",
            "file_write",
            "code_execute",
            "browser_control",
        }


# ── Bootstrap tests ─────────────────────────────────────────────────────────


class TestBootstrap:
    def test_create_default_registry_registers_all_tools(self):
        registry = create_default_registry()
        tools = registry.list_all()
        names = {t.name for t in tools}
        assert names == {
            "web_search",
            "web_fetch",
            "file_read",
            "file_write",
            "code_execute",
            "browser_navigate",
        }


# ── _call_llm_with_tools tests ──────────────────────────────────────────────


class TestCallLlmWithTools:
    @pytest.mark.asyncio
    async def test_bypasses_instructor(self):
        svc = _make_service()
        mock_resp = _make_response(content="hello")
        with patch("qe.runtime.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            result = await svc._call_llm_with_tools(
                "gpt-4o-mini",
                [{"role": "user", "content": "hi"}],
                [{"type": "function", "function": {"name": "test"}}],
            )
        mock_litellm.acompletion.assert_awaited_once()
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == "auto"
        assert result is mock_resp

    @pytest.mark.asyncio
    async def test_records_budget(self):
        svc = _make_service()
        mock_budget = MagicMock()
        BaseService._shared_budget = mock_budget
        mock_resp = _make_response(content="hello")
        with patch("qe.runtime.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.completion_cost.return_value = 0.001
            await svc._call_llm_with_tools(
                "gpt-4o-mini",
                [{"role": "user", "content": "hi"}],
                [],
            )
        mock_budget.record_cost.assert_called_once()


# ── _run_tool_loop tests ────────────────────────────────────────────────────


class TestRunToolLoop:
    @pytest.mark.asyncio
    async def test_raises_without_registry(self):
        svc = _make_service()
        with pytest.raises(RuntimeError, match="ToolRegistry not configured"):
            await svc._run_tool_loop(
                "gpt-4o-mini", [{"role": "user", "content": "hi"}]
            )

    @pytest.mark.asyncio
    async def test_text_response_no_tools(self):
        """LLM returns text on first call — loop exits immediately."""
        registry = _setup_registry_with_test_tool()
        BaseService.set_tool_registry(registry)
        BaseService.set_tool_gate(ToolGate())
        svc = _make_service()

        mock_resp = _make_response(content="Here is your answer.")
        with patch.object(svc, "_call_llm_with_tools", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_resp
            text, audit = await svc._run_tool_loop(
                "gpt-4o-mini", [{"role": "user", "content": "hi"}]
            )
        assert text == "Here is your answer."
        assert audit == []
        mock_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_one_tool_call_then_text(self):
        """LLM calls a tool, gets result, then returns text."""
        registry = _setup_registry_with_test_tool()
        BaseService.set_tool_registry(registry)
        BaseService.set_tool_gate(ToolGate())
        svc = _make_service()

        tc = _make_tool_call("call_1", "web_search", '{"query": "test"}')
        resp_with_tool = _make_response(tool_calls=[tc])
        resp_final = _make_response(content="Based on search results...")

        with patch.object(svc, "_call_llm_with_tools", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [resp_with_tool, resp_final]
            text, audit = await svc._run_tool_loop(
                "gpt-4o-mini", [{"role": "user", "content": "search for test"}]
            )

        assert text == "Based on search results..."
        assert len(audit) == 1
        assert audit[0]["tool_name"] == "web_search"
        assert audit[0]["result"] is not None
        assert audit[0]["error"] is None
        assert mock_call.await_count == 2

    @pytest.mark.asyncio
    async def test_gate_deny_skips_execution(self):
        """ToolGate DENY prevents tool execution."""
        # Register tool without capability requirement (so it passes registry filter)
        registry = ToolRegistry()
        spec = ToolSpec(
            name="web_search",
            description="Search the web",
            requires_capability=None,  # no registry-level filtering
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )

        async def _handler(query: str) -> dict:
            return {"results": [f"result for: {query}"]}

        registry.register(spec, _handler)
        BaseService.set_tool_registry(registry)

        # ToolGate will deny web_search without "web_search" capability
        gate = ToolGate()
        BaseService.set_tool_gate(gate)

        svc = _make_service(capabilities=CapabilityDeclaration())

        tc = _make_tool_call("call_1", "web_search", '{"query": "test"}')
        resp_with_tool = _make_response(tool_calls=[tc])
        resp_final = _make_response(content="I cannot search.")

        with patch.object(svc, "_call_llm_with_tools", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [resp_with_tool, resp_final]
            text, audit = await svc._run_tool_loop(
                "gpt-4o-mini",
                [{"role": "user", "content": "search"}],
                capabilities=set(),  # empty → gate denies web_search
            )

        assert text == "I cannot search."
        assert len(audit) == 1
        assert audit[0]["gate_decision"] == "deny"
        assert audit[0]["error"] is not None
        assert audit[0]["result"] is None

    @pytest.mark.asyncio
    async def test_gate_escalate_skips_with_warning(self):
        """ToolGate ESCALATE skips tool and sends warning to LLM."""
        registry = _setup_registry_with_test_tool()
        BaseService.set_tool_registry(registry)

        # Gate that always escalates
        gate = MagicMock(spec=ToolGate)
        gate.validate.return_value = GateResult(
            decision=GateDecision.ESCALATE,
            reason="Requires human approval",
            policy_name="test_policy",
        )
        BaseService.set_tool_gate(gate)
        svc = _make_service()

        tc = _make_tool_call("call_1", "web_search", '{"query": "test"}')
        resp_with_tool = _make_response(tool_calls=[tc])
        resp_final = _make_response(content="Skipped tool call.")

        with patch.object(svc, "_call_llm_with_tools", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [resp_with_tool, resp_final]
            text, audit = await svc._run_tool_loop(
                "gpt-4o-mini", [{"role": "user", "content": "search"}]
            )

        assert text == "Skipped tool call."
        assert len(audit) == 1
        assert audit[0]["gate_decision"] == "escalate"
        assert "human approval" in audit[0]["error"]

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        """Loop stops at max_iterations and produces a final summary."""
        registry = _setup_registry_with_test_tool()
        BaseService.set_tool_registry(registry)
        BaseService.set_tool_gate(ToolGate())
        svc = _make_service()

        tc = _make_tool_call("call_1", "web_search", '{"query": "test"}')
        resp_with_tool = _make_response(tool_calls=[tc])
        resp_final = _make_response(content="Max iterations summary.")

        with (
            patch.object(svc, "_call_llm_with_tools", new_callable=AsyncMock) as mock_call,
            patch("qe.runtime.service.litellm") as mock_litellm,
        ):
            # Always return tool calls to force max iterations
            mock_call.return_value = resp_with_tool
            mock_litellm.acompletion = AsyncMock(return_value=resp_final)

            text, audit = await svc._run_tool_loop(
                "gpt-4o-mini",
                [{"role": "user", "content": "search"}],
                max_iterations=3,
            )

        assert text == "Max iterations summary."
        assert len(audit) == 3  # one tool call per iteration
        assert mock_call.await_count == 3

    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """Tool handler exception is caught and sent back to LLM."""
        registry = ToolRegistry()
        spec = ToolSpec(
            name="failing_tool",
            description="A tool that fails",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        )

        async def _handler(x: str) -> dict:
            raise RuntimeError("Tool crashed")

        registry.register(spec, _handler)
        BaseService.set_tool_registry(registry)
        BaseService.set_tool_gate(ToolGate())
        svc = _make_service(capabilities=CapabilityDeclaration())

        tc = _make_tool_call("call_1", "failing_tool", '{"x": "test"}')
        resp_with_tool = _make_response(tool_calls=[tc])
        resp_final = _make_response(content="Tool failed, moving on.")

        with patch.object(svc, "_call_llm_with_tools", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = [resp_with_tool, resp_final]
            text, audit = await svc._run_tool_loop(
                "gpt-4o-mini",
                [{"role": "user", "content": "do thing"}],
                capabilities=set(),
            )

        assert text == "Tool failed, moving on."
        assert len(audit) == 1
        assert "Tool crashed" in audit[0]["error"]
        assert audit[0]["result"] is None

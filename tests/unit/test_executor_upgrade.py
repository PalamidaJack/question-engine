"""Tests for Phase B: ExecutorService upgrade with tool loop, contract validation, and retry."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.errors import ExecutorContractError, ExecutorToolError
from qe.models.envelope import Envelope
from qe.models.goal import SubtaskResult
from qe.services.executor.service import _TOOL_TASK_TYPES, ExecutorService

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_bus():
    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    bus.publish = MagicMock()
    return bus


def _make_registry():
    registry = MagicMock()
    registry.get_tool_schemas = MagicMock(return_value=[{
        "type": "function",
        "function": {"name": "web_search", "description": "Search", "parameters": {}},
    }])
    registry.execute = AsyncMock(return_value="search result: quantum computing advances")
    return registry


def _make_gate(decision="allow"):
    gate = MagicMock()
    result = MagicMock()
    result.decision = decision
    result.reason = "denied" if decision != "allow" else ""
    gate.validate = MagicMock(return_value=result)
    return gate


def _mock_litellm_response(content="LLM response", tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    tc_data = None
    if tool_calls:
        tc_data = [{
            "id": "tc1", "type": "function",
            "function": {"name": "web_search", "arguments": '{"query": "test"}'},
        }]
    msg.model_dump = MagicMock(return_value={
        "role": "assistant", "content": content, "tool_calls": tc_data,
    })
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_tool_call(name="web_search", args='{"query": "test"}', tc_id="tc1"):
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    tc.id = tc_id
    return tc


# ── Error Classes ────────────────────────────────────────────────────────────


class TestErrorClasses:
    def test_executor_contract_error(self):
        err = ExecutorContractError("precondition failed")
        assert err.code == "QE_EXECUTOR_CONTRACT_VIOLATION"
        assert err.is_retryable is False

    def test_executor_tool_error(self):
        err = ExecutorToolError("tool failed")
        assert err.code == "QE_EXECUTOR_TOOL_FAILED"
        assert err.is_retryable is True
        assert err.retry_delay_ms == 2000


# ── Constructor ──────────────────────────────────────────────────────────────


class TestConstructor:
    def test_accepts_tool_params(self):
        bus = _make_bus()
        registry = _make_registry()
        gate = _make_gate()
        workspace = MagicMock()
        executor = ExecutorService(
            bus=bus, substrate=None,
            tool_registry=registry, tool_gate=gate,
            workspace_manager=workspace,
        )
        assert executor.tool_registry is registry
        assert executor.tool_gate is gate
        assert executor.workspace_manager is workspace


# ── Tool Loop ────────────────────────────────────────────────────────────────


class TestToolLoop:
    @pytest.mark.asyncio
    async def test_tool_loop_calls_and_returns(self):
        """LLM returns tool_calls then final answer; verify tool_registry.execute called."""
        bus = _make_bus()
        registry = _make_registry()
        gate = _make_gate()

        tc = _make_tool_call()
        # First call: LLM returns tool_calls
        resp_with_tools = _mock_litellm_response(content=None, tool_calls=[tc])
        # Second call: LLM returns final answer
        resp_final = _mock_litellm_response(content="Final answer with findings")

        executor = ExecutorService(
            bus=bus, substrate=None,
            tool_registry=registry, tool_gate=gate,
        )

        with patch("qe.services.executor.service.litellm") as mock_litellm, \
             patch("qe.services.executor.service.get_rate_limiter") as mock_rl:
            mock_rl.return_value.acquire = AsyncMock()
            mock_litellm.acompletion = AsyncMock(side_effect=[resp_with_tools, resp_final])
            mock_litellm.completion_cost = MagicMock(return_value=0.001)

            result = await executor._run_tool_loop(
                "web_search", "search for quantum computing", None, goal_id="g1",
            )

        assert result["content"] == "Final answer with findings"
        assert len(result["tool_calls"]) == 1
        registry.execute.assert_awaited_once()
        gate.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_loop_gate_deny(self):
        """Gate denies tool call; error in audit log."""
        bus = _make_bus()
        registry = _make_registry()
        gate = _make_gate(decision="deny")

        tc = _make_tool_call()
        resp_with_tools = _mock_litellm_response(content=None, tool_calls=[tc])
        resp_final = _mock_litellm_response(content="Answer without tool")

        executor = ExecutorService(
            bus=bus, substrate=None,
            tool_registry=registry, tool_gate=gate,
        )

        with patch("qe.services.executor.service.litellm") as mock_litellm, \
             patch("qe.services.executor.service.get_rate_limiter") as mock_rl:
            mock_rl.return_value.acquire = AsyncMock()
            mock_litellm.acompletion = AsyncMock(side_effect=[resp_with_tools, resp_final])

            result = await executor._run_tool_loop(
                "web_search", "test", None, goal_id="g1",
            )

        assert len(result["tool_calls"]) == 1
        assert "error" in result["tool_calls"][0]
        registry.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_execute_task_routes_to_tool_loop(self):
        """web_search task type routes to tool loop when registry is present."""
        bus = _make_bus()
        registry = _make_registry()
        gate = _make_gate()

        executor = ExecutorService(
            bus=bus, substrate=None,
            tool_registry=registry, tool_gate=gate,
        )

        with patch.object(executor, "_run_tool_loop", new_callable=AsyncMock) as mock_loop:
            mock_loop.return_value = {
                "content": "Tool result", "task_type": "web_search",
                "tool_calls": [], "cost_usd": 0.0,
            }
            result = await executor._execute_task(
                "web_search", "search test", {"goal_id": "g1", "subtask_id": "s1"},
            )

        assert result["content"] == "Tool result"
        mock_loop.assert_awaited_once()


# ── Contract Validation ──────────────────────────────────────────────────────


class TestContractValidation:
    def test_precondition_non_empty_passes(self):
        executor = ExecutorService(bus=_make_bus(), substrate=None)
        passed, reason = executor._validate_preconditions(
            {"preconditions": ["non_empty"]},
            {"dep1": "some data"},
        )
        assert passed is True

    def test_precondition_non_empty_fails(self):
        executor = ExecutorService(bus=_make_bus(), substrate=None)
        passed, reason = executor._validate_preconditions(
            {"preconditions": ["non_empty"]},
            None,
        )
        assert passed is False
        assert "non_empty" in reason

    def test_precondition_data_available_passes(self):
        executor = ExecutorService(bus=_make_bus(), substrate=None)
        passed, reason = executor._validate_preconditions(
            {"preconditions": ["data_available"]},
            {"dep1": "result"},
        )
        assert passed is True

    def test_precondition_data_available_fails(self):
        executor = ExecutorService(bus=_make_bus(), substrate=None)
        passed, reason = executor._validate_preconditions(
            {"preconditions": ["data_available"]},
            {},
        )
        assert passed is False

    def test_postcondition_non_empty_passes(self):
        executor = ExecutorService(bus=_make_bus(), substrate=None)
        passed, reason = executor._validate_postconditions(
            {"postconditions": ["non_empty"]},
            {"content": "some output"},
        )
        assert passed is True

    def test_postcondition_non_empty_fails(self):
        executor = ExecutorService(bus=_make_bus(), substrate=None)
        passed, reason = executor._validate_postconditions(
            {"postconditions": ["non_empty"]},
            {"content": ""},
        )
        assert passed is False

    def test_postcondition_min_length_passes(self):
        executor = ExecutorService(bus=_make_bus(), substrate=None)
        passed, reason = executor._validate_postconditions(
            {"postconditions": ["min_length:5"]},
            {"content": "hello world"},
        )
        assert passed is True

    def test_postcondition_min_length_fails(self):
        executor = ExecutorService(bus=_make_bus(), substrate=None)
        passed, reason = executor._validate_postconditions(
            {"postconditions": ["min_length:100"]},
            {"content": "short"},
        )
        assert passed is False
        assert "min_length" in reason

    @pytest.mark.asyncio
    async def test_precondition_violation_publishes_event(self):
        bus = _make_bus()
        executor = ExecutorService(bus=bus, substrate=None)

        with pytest.raises(ExecutorContractError):
            await executor._execute_task(
                "analysis", "test",
                {"goal_id": "g1", "subtask_id": "s1",
                 "contract": {"preconditions": ["non_empty"]}},
                dependency_context=None,
            )

        bus.publish.assert_called_once()
        envelope = bus.publish.call_args[0][0]
        assert envelope.topic == "tasks.contract_violated"
        assert envelope.payload["violation_type"] == "precondition"


# ── Retry Logic ──────────────────────────────────────────────────────────────


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        """Mock LLM to fail 2x then succeed; verify 3 attempts + recovery_history."""
        bus = _make_bus()
        executor = ExecutorService(bus=bus, substrate=None)

        call_count = 0

        async def _mock_execute(task_type, desc, payload, dependency_context=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate_limit exceeded")  # noqa: TRY002
            return {"content": "success", "task_type": task_type}

        with patch.object(executor, "_execute_task", side_effect=_mock_execute):
            envelope = Envelope(
                topic="tasks.dispatched",
                source_service_id="test",
                payload={
                    "goal_id": "g1", "subtask_id": "s1",
                    "task_type": "analysis", "description": "test",
                    "contract": {"max_retries": 3},
                },
            )
            await executor._run_task(envelope)

        assert call_count == 3
        published = bus.publish.call_args[0][0]
        assert published.topic == "tasks.completed"
        result = SubtaskResult.model_validate(published.payload)
        assert result.status == "completed"
        assert len(result.recovery_history) == 2  # 2 failures recorded

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable(self):
        """Non-retryable errors should not be retried."""
        bus = _make_bus()
        executor = ExecutorService(bus=bus, substrate=None)

        async def _mock_execute(task_type, desc, payload, dependency_context=None):
            raise ExecutorContractError("contract violation")

        with patch.object(executor, "_execute_task", side_effect=_mock_execute):
            envelope = Envelope(
                topic="tasks.dispatched",
                source_service_id="test",
                payload={
                    "goal_id": "g1", "subtask_id": "s1",
                    "task_type": "analysis", "description": "test",
                    "contract": {"max_retries": 3},
                },
            )
            await executor._run_task(envelope)

        published = bus.publish.call_args[0][0]
        assert published.topic == "tasks.failed"
        result = SubtaskResult.model_validate(published.payload)
        assert len(result.recovery_history) == 1  # Only 1 attempt

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        """Exhausting all retries publishes tasks.failed."""
        bus = _make_bus()
        executor = ExecutorService(bus=bus, substrate=None)

        async def _mock_execute(task_type, desc, payload, dependency_context=None):
            raise Exception("rate_limit error")  # noqa: TRY002

        with patch.object(executor, "_execute_task", side_effect=_mock_execute):
            envelope = Envelope(
                topic="tasks.dispatched",
                source_service_id="test",
                payload={
                    "goal_id": "g1", "subtask_id": "s1",
                    "task_type": "analysis", "description": "test",
                    "contract": {"max_retries": 2},
                },
            )
            await executor._run_task(envelope)

        published = bus.publish.call_args[0][0]
        assert published.topic == "tasks.failed"
        result = SubtaskResult.model_validate(published.payload)
        assert len(result.recovery_history) == 3  # 1 initial + 2 retries


# ── Task Type Mapping ────────────────────────────────────────────────────────


class TestTaskTypeMapping:
    def test_web_search_needs_tools(self):
        assert "web_search" in _TOOL_TASK_TYPES

    def test_code_execution_needs_tools(self):
        assert "code_execution" in _TOOL_TASK_TYPES

    def test_research_can_use_tools(self):
        assert "research" in _TOOL_TASK_TYPES

    def test_analysis_no_tools(self):
        assert "analysis" not in _TOOL_TASK_TYPES


# ── Bus Schema ───────────────────────────────────────────────────────────────


class TestBusSchema:
    def test_contract_violated_topic_registered(self):
        from qe.bus.protocol import TOPICS
        assert "tasks.contract_violated" in TOPICS

    def test_contract_violated_schema(self):
        from qe.bus.schemas import TaskContractViolatedPayload
        payload = TaskContractViolatedPayload(
            goal_id="g1", subtask_id="s1",
            violation_type="precondition",
            condition="non_empty", reason="empty",
        )
        assert payload.goal_id == "g1"
        assert payload.violation_type == "precondition"

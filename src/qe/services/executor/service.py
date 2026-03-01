"""ExecutorService: subscribes to tasks.dispatched and runs subtasks."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import litellm

from qe.errors import ExecutorContractError, ExecutorToolError, classify_error
from qe.models.envelope import Envelope
from qe.models.goal import SubtaskResult
from qe.runtime.budget import BudgetTracker
from qe.runtime.rate_limiter import get_rate_limiter

log = logging.getLogger(__name__)

# Task types handled via a direct LLM call (no tools needed)
_LLM_TASK_TYPES = frozenset(
    {"research", "analysis", "fact_check", "synthesis", "document_generation"}
)

# Task types that require tool access
_TOOL_TASK_TYPES: dict[str, set[str]] = {
    "web_search": {"web_search"},
    "research": {"web_search"},
    "code_execution": {"code_execute"},
}

# System prompt fragments keyed by task_type
_TASK_PROMPTS: dict[str, str] = {
    "research": (
        "You are a research assistant. Investigate the topic thoroughly "
        "and return structured findings."
    ),
    "analysis": (
        "You are an analytical assistant. Analyse the provided data or "
        "topic and return key insights."
    ),
    "fact_check": (
        "You are a fact-checking assistant. Verify the claims in the "
        "description and return a verdict with evidence."
    ),
    "synthesis": (
        "You are a synthesis assistant. Combine the provided information "
        "into a coherent summary."
    ),
    "document_generation": (
        "You are a document generation assistant. Produce the requested "
        "document based on the description."
    ),
    "web_search": (
        "You are a research assistant with web search capabilities. "
        "Use the available tools to find information."
    ),
    "code_execution": (
        "You are a coding assistant with code execution capabilities. "
        "Use the available tools to accomplish the task."
    ),
}


class ExecutorService:
    """Executes dispatched subtasks via LLM calls or tool loops.

    Not a BaseService subclass — subscribes to bus topics directly,
    similar to how Dispatcher works.
    """

    def __init__(
        self,
        bus: Any,
        substrate: Any,
        budget_tracker: BudgetTracker | None = None,
        model: str = "gpt-4o-mini",
        max_concurrency: int = 5,
        agent_id: str = "executor_default",
        tool_registry: Any | None = None,
        tool_gate: Any | None = None,
        workspace_manager: Any | None = None,
    ) -> None:
        self.bus = bus
        self.substrate = substrate
        self.budget_tracker = budget_tracker
        self.model = model
        self.agent_id = agent_id
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.tool_registry = tool_registry
        self.tool_gate = tool_gate
        self.workspace_manager = workspace_manager

    async def start(self) -> None:
        self.bus.subscribe("tasks.dispatched", self._handle_dispatched)
        log.info(
            "executor.started model=%s concurrency=%d tools=%s",
            self.model,
            self._semaphore._value,
            self.tool_registry is not None,
        )

    async def stop(self) -> None:
        self.bus.unsubscribe("tasks.dispatched", self._handle_dispatched)
        log.info("executor.stopped")

    async def _handle_dispatched(self, envelope: Envelope) -> None:
        # Agent routing filter: skip if assigned to a different agent
        assigned = envelope.payload.get("assigned_agent_id")
        if assigned is not None and assigned != self.agent_id:
            return
        async with self._semaphore:
            await self._run_task(envelope)

    async def _run_task(self, envelope: Envelope) -> None:
        payload = envelope.payload
        goal_id = payload["goal_id"]
        subtask_id = payload["subtask_id"]
        task_type = payload["task_type"]
        description = payload["description"]
        dependency_context = payload.get("dependency_context")
        contract = payload.get("contract", {})
        max_retries = contract.get("max_retries", 3)

        start = time.monotonic()
        recovery_history: list[dict[str, Any]] = []

        for attempt in range(max_retries + 1):
            try:
                output = await self._execute_task(
                    task_type, description, payload,
                    dependency_context=dependency_context,
                )
                output["_agent_id"] = self.agent_id
                latency_ms = int((time.monotonic() - start) * 1000)
                result = SubtaskResult(
                    subtask_id=subtask_id,
                    goal_id=goal_id,
                    status="completed",
                    output=output,
                    model_used=self.model,
                    latency_ms=latency_ms,
                    cost_usd=output.get("cost_usd", 0.0),
                    tool_calls=output.get("tool_calls", []),
                    recovery_history=recovery_history,
                )
                topic = "tasks.completed"
                log.info(
                    "executor.completed goal_id=%s subtask_id=%s type=%s ms=%d attempt=%d",
                    goal_id, subtask_id, task_type, latency_ms, attempt + 1,
                )
                break
            except Exception as exc:
                classified = classify_error(exc)
                recovery_history.append({
                    "attempt": attempt + 1,
                    "error": str(exc),
                    "error_code": classified.code,
                    "is_retryable": classified.is_retryable,
                })

                if classified.is_retryable and attempt < max_retries:
                    delay_s = classified.retry_delay_ms / 1000
                    log.warning(
                        "executor.retry goal_id=%s subtask_id=%s attempt=%d/%d "
                        "delay=%.1fs error=%s",
                        goal_id, subtask_id, attempt + 1, max_retries,
                        delay_s, str(exc),
                    )
                    await asyncio.sleep(delay_s)
                    continue

                latency_ms = int((time.monotonic() - start) * 1000)
                result = SubtaskResult(
                    subtask_id=subtask_id,
                    goal_id=goal_id,
                    status="failed",
                    output={"error": str(exc)},
                    model_used=self.model,
                    latency_ms=latency_ms,
                    recovery_history=recovery_history,
                )
                topic = "tasks.failed"
                log.error(
                    "executor.failed goal_id=%s subtask_id=%s type=%s error=%s attempts=%d",
                    goal_id, subtask_id, task_type, str(exc), attempt + 1,
                )
                break

        self.bus.publish(
            Envelope(
                topic=topic,
                source_service_id="executor",
                correlation_id=goal_id,
                payload=result.model_dump(mode="json"),
            )
        )

    def _validate_preconditions(
        self,
        contract: dict[str, Any],
        dependency_context: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        """Check contract preconditions against dependency_context."""
        for condition in contract.get("preconditions", []):
            if condition == "non_empty":
                if not dependency_context:
                    return False, "Precondition 'non_empty': dependency_context is empty"
            elif condition == "data_available":
                if not dependency_context or not any(dependency_context.values()):
                    return False, "Precondition 'data_available': no dependency results present"
        return True, ""

    def _validate_postconditions(
        self,
        contract: dict[str, Any],
        output: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check contract postconditions against output."""
        content = output.get("content", "")
        for condition in contract.get("postconditions", []):
            if condition == "non_empty":
                if not content:
                    return False, "Postcondition 'non_empty': output content is empty"
            elif condition.startswith("min_length:"):
                try:
                    min_len = int(condition.split(":")[1])
                except (IndexError, ValueError):
                    continue
                if len(content) < min_len:
                    return (
                        False,
                        f"Postcondition '{condition}': content length {len(content)} < {min_len}",
                    )
        return True, ""

    async def _run_tool_loop(
        self,
        task_type: str,
        description: str,
        dependency_context: dict[str, Any] | None,
        goal_id: str = "",
    ) -> dict[str, Any]:
        """Run an agentic tool loop for tasks that need tool access."""
        capabilities = _TOOL_TASK_TYPES.get(task_type, set())
        schemas = self.tool_registry.get_tool_schemas(capabilities, mode="direct")
        if not schemas:
            raise ExecutorToolError("No tool schemas available for capabilities")

        system_prompt = _TASK_PROMPTS.get(task_type, _TASK_PROMPTS["research"])
        user_content = description
        if dependency_context:
            context_lines = ["\n\n--- Prior subtask results ---"]
            for dep_id, content in dependency_context.items():
                context_lines.append(f"[{dep_id}]: {content}")
            user_content += "\n".join(context_lines)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        tool_calls_audit: list[dict[str, Any]] = []
        total_cost = 0.0

        for iteration in range(10):
            limiter = get_rate_limiter()
            await limiter.acquire(self.model)

            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                tools=schemas,
                tool_choice="auto",
            )

            if self.budget_tracker is not None:
                try:
                    cost = litellm.completion_cost(completion_response=response)
                except Exception:
                    cost = 0.0
                self.budget_tracker.record_cost(self.model, cost, service_id="executor")
                total_cost += cost

            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                return {
                    "content": msg.content or "",
                    "task_type": task_type,
                    "tool_calls": tool_calls_audit,
                    "cost_usd": total_cost,
                }

            # Append assistant message with tool_calls
            messages.append(msg.model_dump())

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                # Validate via gate
                if self.tool_gate is not None:
                    gate_result = self.tool_gate.validate(
                        fn_name, fn_args, capabilities, goal_id=goal_id,
                    )
                    if gate_result.decision != "allow":
                        error_msg = f"Tool gate denied: {gate_result.reason}"
                        tool_calls_audit.append({
                            "tool": fn_name, "params": fn_args,
                            "error": error_msg, "iteration": iteration,
                        })
                        messages.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "content": error_msg,
                        })
                        continue

                # Execute tool
                try:
                    tool_result = await self.tool_registry.execute(fn_name, fn_args)
                    result_str = str(tool_result) if tool_result is not None else ""
                except Exception as e:
                    result_str = f"Tool execution error: {e}"

                tool_calls_audit.append({
                    "tool": fn_name, "params": fn_args,
                    "result_preview": result_str[:200], "iteration": iteration,
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": result_str,
                })

        # Max iterations reached — get final summary
        messages.append({
            "role": "user",
            "content": "Summarize your findings based on the tool results above.",
        })
        limiter = get_rate_limiter()
        await limiter.acquire(self.model)
        response = await litellm.acompletion(model=self.model, messages=messages)
        final_content = response.choices[0].message.content or ""

        return {
            "content": final_content,
            "task_type": task_type,
            "tool_calls": tool_calls_audit,
            "cost_usd": total_cost,
        }

    async def _try_deterministic(
        self,
        task_type: str,
        description: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Attempt to resolve a task without an LLM call.

        Returns a result dict on success, or ``None`` to fall through to LLM.
        """
        if self.substrate is None:
            return None

        if task_type == "fact_check":
            return await self._fact_check_from_ledger(description)

        return None

    async def _fact_check_from_ledger(
        self, description: str
    ) -> dict[str, Any] | None:
        """Return a compiled verdict from the belief ledger if high-confidence claims exist."""
        try:
            claims = await self.substrate.search_full_text(description, limit=10)
            high_conf = [c for c in claims if c.confidence >= 0.8]
            if not high_conf:
                return None

            lines = []
            for c in high_conf:
                lines.append(
                    f"- {c.subject_entity_id} {c.predicate} {c.object_value} "
                    f"(confidence: {c.confidence:.0%})"
                )
            content = (
                f"Based on {len(high_conf)} existing verified claim(s):\n"
                + "\n".join(lines)
            )
            log.info(
                "executor.bypass task_type=fact_check claims=%d",
                len(high_conf),
            )
            return {
                "content": content,
                "task_type": "fact_check",
                "bypassed_llm": True,
            }
        except Exception:
            log.debug("executor.bypass_failed task_type=fact_check", exc_info=True)
            return None

    async def _execute_task(
        self,
        task_type: str,
        description: str,
        payload: dict[str, Any],
        dependency_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Dispatch to LLM call based on task_type.

        Tries a deterministic shortcut first; falls through to LLM on miss.
        Validates contract pre/postconditions when present.
        """
        contract = payload.get("contract", {})
        goal_id = payload.get("goal_id", "")
        subtask_id = payload.get("subtask_id", "")

        # Fast path: try to resolve without an LLM call
        deterministic = await self._try_deterministic(task_type, description, payload)
        if deterministic is not None:
            return deterministic

        # Validate preconditions
        if contract.get("preconditions"):
            passed, reason = self._validate_preconditions(contract, dependency_context)
            if not passed:
                self.bus.publish(Envelope(
                    topic="tasks.contract_violated",
                    source_service_id="executor",
                    correlation_id=goal_id,
                    payload={
                        "goal_id": goal_id,
                        "subtask_id": subtask_id,
                        "violation_type": "precondition",
                        "condition": str(contract["preconditions"]),
                        "reason": reason,
                    },
                ))
                raise ExecutorContractError(reason)

        # Tool loop path
        if self.tool_registry is not None and task_type in _TOOL_TASK_TYPES:
            output = await self._run_tool_loop(
                task_type, description, dependency_context, goal_id=goal_id,
            )
        else:
            # Standard single-LLM-call path
            output = await self._single_llm_call(
                task_type, description, dependency_context,
            )

        # Validate postconditions
        if contract.get("postconditions"):
            passed, reason = self._validate_postconditions(contract, output)
            if not passed:
                self.bus.publish(Envelope(
                    topic="tasks.contract_violated",
                    source_service_id="executor",
                    correlation_id=goal_id,
                    payload={
                        "goal_id": goal_id,
                        "subtask_id": subtask_id,
                        "violation_type": "postcondition",
                        "condition": str(contract["postconditions"]),
                        "reason": reason,
                    },
                ))
                raise ExecutorContractError(reason)

        return output

    async def _single_llm_call(
        self,
        task_type: str,
        description: str,
        dependency_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a single LLM call without tools."""
        user_content = description
        if dependency_context:
            context_lines = ["\n\n--- Prior subtask results ---"]
            for dep_id, content in dependency_context.items():
                context_lines.append(f"[{dep_id}]: {content}")
            user_content += "\n".join(context_lines)

        system_prompt = _TASK_PROMPTS.get(task_type, _TASK_PROMPTS["research"])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        limiter = get_rate_limiter()
        await limiter.acquire(self.model)

        start = time.monotonic()
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
        )
        latency_ms = (time.monotonic() - start) * 1000

        content = response.choices[0].message.content or ""
        cost = 0.0

        if self.budget_tracker is not None:
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = 0.0
            self.budget_tracker.record_cost(
                self.model, cost, service_id="executor"
            )
            log.debug(
                "executor.llm model=%s latency_ms=%.1f cost_usd=%.6f",
                self.model,
                latency_ms,
                cost,
            )

        return {"content": content, "task_type": task_type, "cost_usd": cost}

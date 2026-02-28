import asyncio
import inspect
import json
import logging
import time
from collections import Counter
from typing import Any

import instructor
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

from qe.models.envelope import Envelope
from qe.models.genome import Blueprint
from qe.runtime.budget import BudgetTracker
from qe.runtime.context_manager import ContextManager
from qe.runtime.llm_cache import get_llm_cache
from qe.runtime.logging_config import (
    ctx_correlation_id,
    ctx_envelope_id,
    ctx_service_id,
)
from qe.runtime.rate_limiter import get_rate_limiter
from qe.runtime.router import AutoRouter
from qe.runtime.sanitizer import InputSanitizer, SanitizeResult
from qe.runtime.tool_gate import GateDecision, ToolGate
from qe.runtime.tools import ToolRegistry

load_dotenv()

log = logging.getLogger(__name__)

# Shared sanitizer instance for all services
_sanitizer = InputSanitizer()


class BaseService:
    # Shared instances across all services in this process
    _shared_budget: BudgetTracker | None = None
    _shared_tool_registry: ToolRegistry | None = None
    _shared_tool_gate: ToolGate | None = None

    @classmethod
    def set_budget_tracker(cls, tracker: BudgetTracker) -> None:
        cls._shared_budget = tracker

    @classmethod
    def set_tool_registry(cls, registry: ToolRegistry) -> None:
        cls._shared_tool_registry = registry

    @classmethod
    def set_tool_gate(cls, gate: ToolGate) -> None:
        cls._shared_tool_gate = gate

    def __init__(self, blueprint: Blueprint, bus: Any, substrate: Any) -> None:
        self.blueprint = blueprint
        self.bus = bus
        self.substrate = substrate
        self.context_manager = ContextManager(blueprint)
        self.router = AutoRouter(blueprint.model_preference, self._shared_budget)
        self._turn_count = 0
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None

    async def _maybe_await(self, result: Any) -> Any:
        if inspect.isawaitable(result):
            return await result
        return result

    async def start(self) -> None:
        self._running = True
        for topic in self.blueprint.capabilities.bus_topics_subscribe:
            await self._maybe_await(self.bus.subscribe(topic, self._handle_envelope))
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        for topic in self.blueprint.capabilities.bus_topics_subscribe:
            await self._maybe_await(self.bus.unsubscribe(topic, self._handle_envelope))

    async def _handle_envelope(self, envelope: Envelope) -> None:
        assert envelope.topic in self.blueprint.capabilities.bus_topics_subscribe

        # Set correlation context for structured logging
        ctx_envelope_id.set(envelope.envelope_id)
        ctx_correlation_id.set(envelope.correlation_id or "")
        ctx_service_id.set(self.blueprint.service_id)

        start = time.monotonic()
        log.debug(
            "service.handle_start service_id=%s topic=%s envelope_id=%s",
            self.blueprint.service_id,
            envelope.topic,
            envelope.envelope_id,
        )

        messages = self.context_manager.build_messages(envelope, self._turn_count)
        model = self.router.select(envelope)
        schema = self.get_response_schema(envelope.topic)

        # ── Tripwire guardrails: run safety checks parallel to LLM call ──
        llm_task = asyncio.create_task(
            self._call_llm(model, messages, schema)
        )
        guard_task = asyncio.create_task(
            self._run_guardrails(envelope)
        )

        # Wait for guardrails first (they're fast); cancel LLM if tripped
        guard_result = await guard_task
        if guard_result is not None and guard_result.risk_score >= _sanitizer.threshold:
            llm_task.cancel()
            try:
                await llm_task
            except asyncio.CancelledError:
                pass
            log.warning(
                "guardrail.tripwire service_id=%s envelope_id=%s "
                "risk=%.2f patterns=%s — LLM call cancelled",
                self.blueprint.service_id,
                envelope.envelope_id,
                guard_result.risk_score,
                guard_result.matches,
            )
            self.bus.publish(
                Envelope(
                    topic="system.gate_denied",
                    source_service_id=self.blueprint.service_id,
                    correlation_id=envelope.envelope_id,
                    payload={
                        "reason": "guardrail_tripwire",
                        "risk_score": guard_result.risk_score,
                        "patterns": guard_result.matches,
                    },
                )
            )
            return

        response = await llm_task

        self._turn_count += 1
        if self._turn_count % self.blueprint.reinforcement_interval_turns == 0:
            self.context_manager.reinforce()

        await self.handle_response(envelope, response)

        elapsed_ms = (time.monotonic() - start) * 1000
        log.debug(
            "service.handle_done service_id=%s topic=%s envelope_id=%s duration_ms=%.1f",
            self.blueprint.service_id,
            envelope.topic,
            envelope.envelope_id,
            elapsed_ms,
        )

    async def handle_response(self, envelope: Envelope, response: Any) -> None:
        raise NotImplementedError

    def get_response_schema(self, topic: str) -> type[BaseModel]:
        raise NotImplementedError

    async def _call_llm(self, model: str, messages: list[dict], schema: type[BaseModel]) -> Any:
        # ── Cache lookup ──
        cache = get_llm_cache()
        cache_key = cache.make_key(model, messages, schema.__name__)
        cached = cache.get(cache_key)
        if cached is not None:
            log.debug(
                "llm.cache_hit model=%s service_id=%s schema=%s",
                model,
                self.blueprint.service_id,
                schema.__name__,
            )
            return cached

        # ── Rate limiting ──
        limiter = get_rate_limiter()
        await limiter.acquire(model)

        start = time.monotonic()
        client = instructor.from_litellm(litellm.acompletion)
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=schema,
        )
        latency_ms = (time.monotonic() - start) * 1000

        # ── Store in cache ──
        cache.put(cache_key, response, model)

        # Record cost if budget tracking is active
        if self._shared_budget is not None:
            cost = litellm.completion_cost(
                model=model,
                messages=messages,
                completion="",
            )
            self._shared_budget.record_cost(
                model,
                cost,
                service_id=self.blueprint.service_id,
            )
            log.debug(
                "llm.call model=%s service_id=%s latency_ms=%.1f cost_usd=%.6f",
                model,
                self.blueprint.service_id,
                latency_ms,
                cost,
            )
        else:
            log.debug(
                "llm.call model=%s service_id=%s latency_ms=%.1f",
                model,
                self.blueprint.service_id,
                latency_ms,
            )
        return response

    # ── Tool-calling support ─────────────────────────────────────────

    async def _call_llm_with_tools(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
    ) -> Any:
        """Call LLM with tool definitions. Returns raw litellm ModelResponse.

        Unlike _call_llm() which uses instructor for structured output,
        this calls litellm directly for function-calling. No caching
        (tool conversations are stateful).
        """
        limiter = get_rate_limiter()
        await limiter.acquire(model)

        start = time.monotonic()
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        latency_ms = (time.monotonic() - start) * 1000

        if self._shared_budget is not None:
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:
                cost = 0.0
            self._shared_budget.record_cost(
                model, cost, service_id=self.blueprint.service_id
            )
            log.debug(
                "llm.tool_call model=%s service_id=%s latency_ms=%.1f cost_usd=%.6f",
                model,
                self.blueprint.service_id,
                latency_ms,
                cost,
            )
        else:
            log.debug(
                "llm.tool_call model=%s service_id=%s latency_ms=%.1f",
                model,
                self.blueprint.service_id,
                latency_ms,
            )
        return response

    async def _run_tool_loop(
        self,
        model: str,
        messages: list[dict],
        *,
        capabilities: set[str] | None = None,
        goal_id: str = "",
        max_iterations: int = 10,
        tool_mode: str = "direct",
    ) -> tuple[str, list[dict]]:
        """Agentic tool-calling loop.

        Repeatedly calls the LLM with available tools until the LLM
        returns a text response (no tool_calls) or max_iterations is hit.

        Returns:
            (final_text_content, audit_log) where audit_log records each
            tool call and its result.
        """
        if self._shared_tool_registry is None:
            raise RuntimeError(
                "ToolRegistry not configured. "
                "Call BaseService.set_tool_registry() first."
            )

        if capabilities is None:
            capabilities = self.blueprint.capabilities.to_capability_set()

        tool_schemas = self._shared_tool_registry.get_tool_schemas(
            capabilities, mode=tool_mode
        )

        if not tool_schemas:
            log.warning(
                "tool_loop.no_tools service_id=%s capabilities=%s",
                self.blueprint.service_id,
                capabilities,
            )
            response = await self._call_llm_with_tools(model, messages, [])
            content = response.choices[0].message.content or ""
            return content, []

        audit_log: list[dict] = []

        for iteration in range(max_iterations):
            response = await self._call_llm_with_tools(
                model, messages, tool_schemas
            )
            choice = response.choices[0]
            assistant_message = choice.message

            if not assistant_message.tool_calls:
                content = assistant_message.content or ""
                log.debug(
                    "tool_loop.complete service_id=%s iterations=%d",
                    self.blueprint.service_id,
                    iteration + 1,
                )
                return content, audit_log

            # Append assistant message (with tool_calls) to conversation
            messages.append(assistant_message.model_dump())

            for tool_call in assistant_message.tool_calls:
                func = tool_call.function
                tool_name = func.name
                try:
                    tool_params = json.loads(func.arguments)
                except json.JSONDecodeError:
                    tool_params = {}

                call_record: dict[str, Any] = {
                    "iteration": iteration,
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_name,
                    "params": tool_params,
                    "gate_decision": None,
                    "result": None,
                    "error": None,
                }

                # ToolGate validation
                if self._shared_tool_gate is not None:
                    gate_result = self._shared_tool_gate.validate(
                        tool_name, tool_params, capabilities, goal_id
                    )
                    call_record["gate_decision"] = gate_result.decision.value

                    if gate_result.decision == GateDecision.DENY:
                        log.warning(
                            "tool_loop.gate_deny tool=%s reason=%s",
                            tool_name,
                            gate_result.reason,
                        )
                        error_content = (
                            f"Tool '{tool_name}' denied by security "
                            f"policy: {gate_result.reason}"
                        )
                        call_record["error"] = error_content
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_content,
                        })
                        audit_log.append(call_record)
                        continue

                    if gate_result.decision == GateDecision.ESCALATE:
                        log.warning(
                            "tool_loop.gate_escalate tool=%s reason=%s",
                            tool_name,
                            gate_result.reason,
                        )
                        error_content = (
                            f"Tool '{tool_name}' requires human approval "
                            f"(escalated): {gate_result.reason}. "
                            f"Skipping this call."
                        )
                        call_record["error"] = error_content
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_content,
                        })
                        audit_log.append(call_record)
                        continue

                # Execute via ToolRegistry
                try:
                    result = await self._shared_tool_registry.execute(
                        tool_name, tool_params
                    )
                    result_str = (
                        json.dumps(result)
                        if not isinstance(result, str)
                        else result
                    )
                    call_record["result"] = result_str
                except Exception as exc:
                    log.exception(
                        "tool_loop.execute_error tool=%s", tool_name
                    )
                    result_str = f"Error executing {tool_name}: {exc}"
                    call_record["error"] = result_str

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })
                audit_log.append(call_record)

        # Max iterations reached — final tool-free call for summary
        log.warning(
            "tool_loop.max_iterations service_id=%s max=%d",
            self.blueprint.service_id,
            max_iterations,
        )
        final_response = await litellm.acompletion(
            model=model, messages=messages
        )
        content = final_response.choices[0].message.content or ""
        return content, audit_log

    async def _run_guardrails(self, envelope: Envelope) -> SanitizeResult | None:
        """Run input sanitization on envelope payload text fields.

        Returns a SanitizeResult if risky content detected, None otherwise.
        Designed to run concurrently with the LLM call for tripwire
        cancellation — see _handle_envelope.
        """
        # Extract text fields from payload
        text_parts: list[str] = []
        for _key, value in envelope.payload.items():
            if isinstance(value, str):
                text_parts.append(value)

        if not text_parts:
            return None

        combined = " ".join(text_parts)
        result = _sanitizer.sanitize(combined)
        if result.risk_score > 0:
            log.debug(
                "guardrail.check service_id=%s risk=%.2f patterns=%s",
                self.blueprint.service_id,
                result.risk_score,
                result.matches,
            )
        return result

    async def _call_llm_consistent(
        self,
        model: str,
        messages: list[dict],
        schema: type[BaseModel],
        *,
        n_samples: int = 3,
    ) -> Any:
        """Self-consistency voting: run N parallel LLM calls, majority-vote.

        For structured outputs (Pydantic models), votes by comparing the
        JSON-serialized output. Returns the most common response.
        Falls back to single call if n_samples <= 1.
        """
        if n_samples <= 1:
            return await self._call_llm(model, messages, schema)

        tasks = [
            asyncio.create_task(self._call_llm(model, messages, schema))
            for _ in range(n_samples)
        ]

        responses: list[Any] = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                responses.append(result)
            except Exception:
                log.debug("consistency.sample_failed service_id=%s", self.blueprint.service_id)

        if not responses:
            raise RuntimeError("All self-consistency samples failed")

        if len(responses) == 1:
            return responses[0]

        # Vote by JSON representation
        serialized = []
        for r in responses:
            try:
                s = r.model_dump_json(exclude_none=True)
            except AttributeError:
                s = str(r)
            serialized.append(s)

        vote_counts = Counter(serialized)
        winner_json = vote_counts.most_common(1)[0][0]

        # Return the response object that matches the winner
        for r, s in zip(responses, serialized, strict=True):
            if s == winner_json:
                log.debug(
                    "consistency.voted service_id=%s samples=%d agreement=%d/%d",
                    self.blueprint.service_id,
                    n_samples,
                    vote_counts[winner_json],
                    len(responses),
                )
                return r

        return responses[0]

    async def reconfigure(self, new_config: dict[str, Any]) -> None:
        self.config = new_config

    async def _heartbeat_loop(self) -> None:
        while self._running:
            self.bus.publish(
                Envelope(
                    topic="system.heartbeat",
                    source_service_id=self.blueprint.service_id,
                    payload={"turn_count": self._turn_count, "status": "alive"},
                )
            )
            await asyncio.sleep(30)

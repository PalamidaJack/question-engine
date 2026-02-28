"""ExecutorService: subscribes to tasks.dispatched and runs subtasks."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import litellm

from qe.models.envelope import Envelope
from qe.models.goal import SubtaskResult
from qe.runtime.budget import BudgetTracker
from qe.runtime.rate_limiter import get_rate_limiter

log = logging.getLogger(__name__)

# Task types handled via a direct LLM call (no tools needed)
_LLM_TASK_TYPES = frozenset(
    {"research", "analysis", "fact_check", "synthesis", "document_generation"}
)

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
    ) -> None:
        self.bus = bus
        self.substrate = substrate
        self.budget_tracker = budget_tracker
        self.model = model
        self.agent_id = agent_id
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def start(self) -> None:
        self.bus.subscribe("tasks.dispatched", self._handle_dispatched)
        log.info(
            "executor.started model=%s concurrency=%d",
            self.model,
            self._semaphore._value,
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

        start = time.monotonic()
        try:
            output = await self._execute_task(
                task_type, description, payload, dependency_context=dependency_context
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
            )
            topic = "tasks.completed"
            log.info(
                "executor.completed goal_id=%s subtask_id=%s type=%s ms=%d",
                goal_id,
                subtask_id,
                task_type,
                latency_ms,
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            result = SubtaskResult(
                subtask_id=subtask_id,
                goal_id=goal_id,
                status="failed",
                output={"error": str(exc)},
                model_used=self.model,
                latency_ms=latency_ms,
            )
            topic = "tasks.failed"
            log.error(
                "executor.failed goal_id=%s subtask_id=%s type=%s error=%s",
                goal_id,
                subtask_id,
                task_type,
                str(exc),
            )

        self.bus.publish(
            Envelope(
                topic=topic,
                source_service_id="executor",
                correlation_id=goal_id,
                payload=result.model_dump(mode="json"),
            )
        )

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
        """
        # Fast path: try to resolve without an LLM call
        deterministic = await self._try_deterministic(task_type, description, payload)
        if deterministic is not None:
            return deterministic

        # Append dependency context to the user prompt if available
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

        return {"content": content, "task_type": task_type}

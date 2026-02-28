"""Recovery orchestrator: classifies failures and executes recovery strategies."""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import Any

from qe.models.envelope import Envelope
from qe.substrate.failure_kb import FailureKnowledgeBase

log = logging.getLogger(__name__)


class FailureClass(StrEnum):
    TRANSIENT = "transient"
    CAPABILITY = "capability"
    APPROACH = "approach"
    SPECIFICATION = "specification"
    UNRECOVERABLE = "unrecoverable"


# Keywords that indicate transient failures
_TRANSIENT_KEYWORDS = [
    "timeout", "rate_limit", "rate limit", "429", "503", "504",
    "connection", "network", "temporary", "retry",
]

# Keywords that indicate capability failures
_CAPABILITY_KEYWORDS = [
    "invalid json", "parse error", "schema", "format",
    "too long", "context length", "token limit",
]

# Model tier escalation order
_TIER_ESCALATION = {
    "local": "fast",
    "fast": "balanced",
    "balanced": "powerful",
    "powerful": None,
}


class RecoveryOrchestrator:
    """Classifies failures and executes graduated recovery strategies."""

    def __init__(
        self,
        failure_kb: FailureKnowledgeBase,
        bus: Any = None,
    ) -> None:
        self.failure_kb = failure_kb
        self.bus = bus

    def classify(self, error_summary: str, retry_count: int = 0) -> FailureClass:
        """Classify a failure based on error content."""
        lower = error_summary.lower()

        for keyword in _TRANSIENT_KEYWORDS:
            if keyword in lower:
                return FailureClass.TRANSIENT

        for keyword in _CAPABILITY_KEYWORDS:
            if keyword in lower:
                return FailureClass.CAPABILITY

        if retry_count >= 3:
            return FailureClass.UNRECOVERABLE

        return FailureClass.APPROACH

    def suggest_strategy(
        self,
        failure_class: FailureClass,
        current_tier: str = "balanced",
        retry_count: int = 0,
    ) -> str:
        """Suggest a recovery strategy for the given failure class."""
        if failure_class == FailureClass.TRANSIENT:
            return "retry_with_backoff"

        if failure_class == FailureClass.CAPABILITY:
            next_tier = _TIER_ESCALATION.get(current_tier)
            if next_tier:
                return f"escalate_to_{next_tier}"
            return "escalate_to_hil"

        if failure_class == FailureClass.APPROACH:
            if retry_count == 0:
                return "retry_with_simplified_prompt"
            if retry_count == 1:
                next_tier = _TIER_ESCALATION.get(current_tier)
                if next_tier:
                    return f"escalate_to_{next_tier}"
            return "escalate_to_hil"

        if failure_class == FailureClass.SPECIFICATION:
            return "replan_subtask"

        return "escalate_to_hil"

    async def attempt_recovery(
        self,
        task_type: str,
        error_summary: str,
        current_tier: str = "balanced",
        retry_count: int = 0,
        goal_id: str = "",
        subtask_id: str = "",
    ) -> dict[str, Any]:
        """Classify failure, consult KB, and suggest recovery.

        Returns a dict with failure_class, strategy, and whether to proceed.
        """
        failure_class = self.classify(error_summary, retry_count)

        # Consult failure KB for known strategies
        kb_strategies = await self.failure_kb.lookup(failure_class, task_type)

        # Use KB strategy if available and successful
        if kb_strategies and kb_strategies[0]["success_rate"] > 0.5:
            strategy = kb_strategies[0]["strategy"]
        else:
            strategy = self.suggest_strategy(
                failure_class, current_tier, retry_count
            )

        log.info(
            "recovery.plan class=%s strategy=%s goal=%s subtask=%s retry=%d",
            failure_class,
            strategy,
            goal_id,
            subtask_id,
            retry_count,
        )

        return {
            "failure_class": failure_class,
            "strategy": strategy,
            "should_retry": strategy != "escalate_to_hil",
            "escalate_tier": (
                strategy.removeprefix("escalate_to_")
                if strategy.startswith("escalate_to_") and strategy != "escalate_to_hil"
                else None
            ),
        }

    # ── Recovery Execution ────────────────────────────────────────────

    async def execute_strategy(
        self,
        strategy_info: dict[str, Any],
        original_dispatch_payload: dict[str, Any],
    ) -> Envelope | None:
        """Execute a recovery strategy and return an Envelope to publish.

        Returns a ``tasks.dispatched`` envelope for retries/escalations,
        an ``hil.approval_required`` envelope for HIL escalation,
        or ``None`` if no action can be taken.
        """
        strategy = strategy_info["strategy"]

        if strategy == "retry_with_backoff":
            retry = strategy_info.get("retry_count", 0)
            delay = min(2 ** retry, 30)
            await asyncio.sleep(delay)
            return self._build_redispatch(original_dispatch_payload)

        if strategy == "retry_with_simplified_prompt":
            new_desc = self._simplify_description(
                original_dispatch_payload.get("description", "")
            )
            return self._build_redispatch(
                original_dispatch_payload, description=new_desc
            )

        if strategy.startswith("escalate_to_") and strategy != "escalate_to_hil":
            new_tier = strategy.removeprefix("escalate_to_")
            return self._build_redispatch(
                original_dispatch_payload, model_tier=new_tier
            )

        if strategy in ("escalate_to_hil", "replan_subtask"):
            return self._build_hil_envelope(original_dispatch_payload)

        return None

    def _build_redispatch(
        self,
        payload: dict[str, Any],
        **overrides: Any,
    ) -> Envelope:
        """Create a new ``tasks.dispatched`` envelope preserving dispatch context."""
        new_payload = {
            "goal_id": payload.get("goal_id"),
            "subtask_id": payload.get("subtask_id"),
            "task_type": payload.get("task_type"),
            "description": payload.get("description"),
            "contract": payload.get("contract"),
            "model_tier": payload.get("model_tier", "balanced"),
            "dependency_context": payload.get("dependency_context"),
            "assigned_agent_id": payload.get("assigned_agent_id"),
        }
        new_payload.update(overrides)

        return Envelope(
            topic="tasks.dispatched",
            source_service_id="recovery",
            correlation_id=payload.get("goal_id"),
            payload=new_payload,
        )

    def _build_hil_envelope(self, payload: dict[str, Any]) -> Envelope:
        """Create an ``hil.approval_required`` envelope with context."""
        return Envelope(
            topic="hil.approval_required",
            source_service_id="recovery",
            correlation_id=payload.get("goal_id"),
            payload={
                "goal_id": payload.get("goal_id"),
                "subtask_id": payload.get("subtask_id"),
                "task_type": payload.get("task_type"),
                "description": payload.get("description"),
                "reason": "Recovery exhausted — human intervention required",
            },
        )

    @staticmethod
    def _simplify_description(desc: str) -> str:
        """Truncate and prepend 'Simplified: ' for retry-with-simplified-prompt."""
        truncated = desc[:500] if len(desc) > 500 else desc
        return f"Simplified: {truncated}"

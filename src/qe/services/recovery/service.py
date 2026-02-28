"""Recovery orchestrator: classifies failures and executes recovery strategies."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

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

"""VerificationGate: sits between Executor output and Dispatcher acceptance.

Subscribes to ``tasks.dispatched``, ``tasks.completed``, and ``tasks.failed``.
Runs verification on completed results and routes failures to recovery.
"""

from __future__ import annotations

import logging
from typing import Any

from qe.models.envelope import Envelope
from qe.models.goal import SubtaskResult
from qe.services.recovery.service import RecoveryOrchestrator
from qe.services.verification.service import CheckResult, VerificationService
from qe.substrate.failure_kb import FailureKnowledgeBase

log = logging.getLogger(__name__)


class VerificationGate:
    """Verification + recovery gate wired into the bus pipeline.

    Follows the same start()/stop() subscriber pattern as ExecutorService.
    """

    def __init__(
        self,
        bus: Any,
        verification_service: VerificationService,
        recovery_orchestrator: RecoveryOrchestrator,
        failure_kb: FailureKnowledgeBase,
        max_recovery_attempts: int = 3,
    ) -> None:
        self.bus = bus
        self._verification = verification_service
        self._recovery = recovery_orchestrator
        self._failure_kb = failure_kb
        self._max_recovery_attempts = max_recovery_attempts

        # Cached dispatch payloads keyed by (goal_id, subtask_id)
        self._dispatch_contexts: dict[tuple[str, str], dict[str, Any]] = {}
        # Recovery attempt counters keyed by (goal_id, subtask_id)
        self._retry_counts: dict[tuple[str, str], int] = {}

    async def start(self) -> None:
        self.bus.subscribe("tasks.dispatched", self._cache_dispatch_context)
        self.bus.subscribe("tasks.completed", self._on_task_completed)
        self.bus.subscribe("tasks.failed", self._on_task_failed)
        log.info("verification_gate.started")

    async def stop(self) -> None:
        self.bus.unsubscribe("tasks.dispatched", self._cache_dispatch_context)
        self.bus.unsubscribe("tasks.completed", self._on_task_completed)
        self.bus.unsubscribe("tasks.failed", self._on_task_failed)
        log.info("verification_gate.stopped")

    # ── Bus Handlers ──────────────────────────────────────────────────

    async def _cache_dispatch_context(self, envelope: Envelope) -> None:
        """Cache dispatch payload for later use by verification/recovery."""
        payload = envelope.payload
        key = (payload.get("goal_id", ""), payload.get("subtask_id", ""))
        self._dispatch_contexts[key] = payload
        log.debug("verification_gate.cached_context key=%s", key)

    async def _on_task_completed(self, envelope: Envelope) -> None:
        """Run verification pipeline on a completed subtask result."""
        result = SubtaskResult.model_validate(envelope.payload)
        key = (result.goal_id, result.subtask_id)
        dispatch_ctx = self._dispatch_contexts.get(key, {})

        # Extract contract and model_tier from cached dispatch context
        contract = dispatch_ctx.get("contract")
        if isinstance(contract, dict):
            pass  # already a dict
        elif hasattr(contract, "model_dump"):
            contract = contract.model_dump()
        model_tier = dispatch_ctx.get("model_tier", "balanced")

        report = await self._verification.verify(
            subtask_id=result.subtask_id,
            goal_id=result.goal_id,
            output=result.output,
            contract=contract,
            model_tier=model_tier,
        )

        # Attach verification report to the result
        result.verification_result = report.model_dump(mode="json")

        if report.overall in (CheckResult.PASS, CheckResult.WARN):
            # Clear retry count on success
            self._retry_counts.pop(key, None)
            self.bus.publish(
                Envelope(
                    topic="tasks.verified",
                    source_service_id="verification_gate",
                    correlation_id=result.goal_id,
                    payload=result.model_dump(mode="json"),
                )
            )
            log.info(
                "verification_gate.verified subtask=%s goal=%s result=%s",
                result.subtask_id,
                result.goal_id,
                report.overall,
            )
        else:
            # Verification failed
            self.bus.publish(
                Envelope(
                    topic="tasks.verification_failed",
                    source_service_id="verification_gate",
                    correlation_id=result.goal_id,
                    payload=result.model_dump(mode="json"),
                )
            )
            error_summary = "; ".join(report.details) or "Verification failed"
            await self._attempt_recovery(
                key, error_summary, dispatch_ctx, result
            )

    async def _on_task_failed(self, envelope: Envelope) -> None:
        """Route failed task to recovery."""
        result = SubtaskResult.model_validate(envelope.payload)
        key = (result.goal_id, result.subtask_id)
        dispatch_ctx = self._dispatch_contexts.get(key, {})

        error_summary = result.output.get("error", "Unknown error")
        await self._attempt_recovery(key, error_summary, dispatch_ctx, result)

    # ── Recovery ──────────────────────────────────────────────────────

    async def _attempt_recovery(
        self,
        key: tuple[str, str],
        error_summary: str,
        dispatch_ctx: dict[str, Any],
        result: SubtaskResult,
    ) -> None:
        """Attempt recovery for a failed/unverified subtask."""
        goal_id, subtask_id = key
        retry_count = self._retry_counts.get(key, 0)
        self._retry_counts[key] = retry_count + 1

        task_type = dispatch_ctx.get("task_type", "research")
        current_tier = dispatch_ctx.get("model_tier", "balanced")

        if retry_count >= self._max_recovery_attempts:
            log.warning(
                "verification_gate.retries_exhausted subtask=%s goal=%s retries=%d",
                subtask_id,
                goal_id,
                retry_count,
            )
            await self._failure_kb.record(
                task_type=task_type,
                failure_class="unrecoverable",
                error_summary=error_summary,
                recovery_strategy="escalate_to_hil",
                success=False,
                goal_id=goal_id,
                subtask_id=subtask_id,
            )
            hil_envelope = self._recovery._build_hil_envelope(dispatch_ctx)
            self.bus.publish(hil_envelope)
            return

        # Attempt recovery
        strategy_info = await self._recovery.attempt_recovery(
            task_type=task_type,
            error_summary=error_summary,
            current_tier=current_tier,
            retry_count=retry_count,
            goal_id=goal_id,
            subtask_id=subtask_id,
        )
        strategy_info["retry_count"] = retry_count

        recovery_envelope = await self._recovery.execute_strategy(
            strategy_info, dispatch_ctx
        )

        if recovery_envelope is not None:
            if recovery_envelope.topic == "tasks.dispatched":
                log.info(
                    "verification_gate.redispatch subtask=%s goal=%s strategy=%s",
                    subtask_id,
                    goal_id,
                    strategy_info["strategy"],
                )
            self.bus.publish(recovery_envelope)
            await self._failure_kb.record(
                task_type=task_type,
                failure_class=str(strategy_info["failure_class"]),
                error_summary=error_summary,
                recovery_strategy=strategy_info["strategy"],
                success=False,  # not yet known; updated on success path
                goal_id=goal_id,
                subtask_id=subtask_id,
            )
        else:
            # No action possible — escalate to HIL
            hil_envelope = self._recovery._build_hil_envelope(dispatch_ctx)
            self.bus.publish(hil_envelope)

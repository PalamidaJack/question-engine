"""GoalSynthesizer: aggregates subtask results into a coherent deliverable."""

from __future__ import annotations

import logging
import time
from typing import Any

import instructor
import litellm
from pydantic import BaseModel, Field

from qe.models.envelope import Envelope
from qe.models.goal import GoalState

log = logging.getLogger(__name__)


# ── Models ───────────────────────────────────────────────────────────────────


class SynthesisInput(BaseModel):
    """LLM response model for structured synthesis."""

    summary: str
    key_findings: list[str] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    recommendations: list[str] = Field(default_factory=list)


class GoalResult(BaseModel):
    """Final synthesized result for a completed goal."""

    goal_id: str
    summary: str
    findings: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    subtask_count: int = 0
    dialectic_review: dict[str, Any] | None = None


# ── Service ──────────────────────────────────────────────────────────────────


class GoalSynthesizer:
    """Aggregates subtask results into a coherent deliverable when a goal completes."""

    def __init__(
        self,
        bus: Any,
        goal_store: Any,
        dialectic_engine: Any | None = None,
        model: str = "gpt-4o",
        budget_tracker: Any | None = None,
    ) -> None:
        self.bus = bus
        self.goal_store = goal_store
        self._dialectic = dialectic_engine
        self._model = model
        self._budget_tracker = budget_tracker
        self._running = False

    async def start(self) -> None:
        """Subscribe to goals.completed and begin synthesis on completion."""
        self.bus.subscribe("goals.completed", self._on_goal_completed)
        self._running = True
        log.info("synthesizer.started model=%s", self._model)

    async def stop(self) -> None:
        """Unsubscribe from goals.completed."""
        self.bus.unsubscribe("goals.completed", self._on_goal_completed)
        self._running = False
        log.info("synthesizer.stopped")

    async def _on_goal_completed(self, envelope: Envelope) -> None:
        """Handle goals.completed event — synthesize results."""
        goal_id = envelope.payload.get("goal_id", "")
        if not goal_id:
            return
        try:
            result = await self.synthesize(goal_id)
            self.bus.publish(Envelope(
                topic="goals.synthesized",
                source_service_id="synthesizer",
                correlation_id=goal_id,
                payload={
                    "goal_id": goal_id,
                    "summary": result.summary[:200],
                    "confidence": result.confidence,
                    "findings_count": len(result.findings),
                    "total_cost_usd": result.total_cost_usd,
                },
            ))
            log.info(
                "synthesizer.completed goal_id=%s confidence=%.2f findings=%d",
                goal_id, result.confidence, len(result.findings),
            )
        except Exception as exc:
            self.bus.publish(Envelope(
                topic="goals.synthesis_failed",
                source_service_id="synthesizer",
                correlation_id=goal_id,
                payload={
                    "goal_id": goal_id,
                    "reason": str(exc),
                },
            ))
            log.error("synthesizer.failed goal_id=%s error=%s", goal_id, exc)

    async def synthesize(self, goal_id: str) -> GoalResult:
        """Synthesize all subtask results into a GoalResult."""
        state = await self.goal_store.load_goal(goal_id)
        if state is None:
            raise ValueError(f"Goal not found: {goal_id}")

        # Collect subtask summaries and provenance
        summaries = self._collect_subtask_summaries(state)
        provenance = self._build_provenance(state)

        # LLM synthesis
        start = time.monotonic()
        synthesis = await self._llm_synthesize(state.description, summaries)
        synth_latency_ms = int((time.monotonic() - start) * 1000)

        # Optional dialectic review
        dialectic_review = None
        confidence = synthesis.confidence
        if self._dialectic is not None:
            try:
                evidence = "\n".join(synthesis.key_findings)
                report = await self._dialectic.full_dialectic(
                    goal_id=goal_id,
                    conclusion=synthesis.summary,
                    evidence=evidence,
                )
                dialectic_review = {
                    "revised_confidence": report.revised_confidence,
                    "counterarguments_count": len(report.counterarguments),
                    "perspectives_count": len(report.perspectives),
                }
                confidence = report.revised_confidence
            except Exception:
                log.debug("synthesizer.dialectic_failed goal_id=%s", goal_id, exc_info=True)

        # Compute totals
        total_cost = sum(r.cost_usd for r in state.subtask_results.values())
        total_latency = sum(r.latency_ms for r in state.subtask_results.values())
        total_latency += synth_latency_ms

        # Build final result
        goal_result = GoalResult(
            goal_id=goal_id,
            summary=synthesis.summary,
            findings=[{"finding": f} for f in synthesis.key_findings],
            confidence=confidence,
            provenance=provenance,
            recommendations=synthesis.recommendations,
            total_cost_usd=total_cost,
            total_latency_ms=total_latency,
            subtask_count=len(state.subtask_results),
            dialectic_review=dialectic_review,
        )

        # Store in goal metadata
        state.metadata["goal_result"] = goal_result.model_dump(mode="json")
        await self.goal_store.save_goal(state)

        return goal_result

    def _collect_subtask_summaries(self, state: GoalState) -> list[dict[str, Any]]:
        """Collect summaries from all subtask results."""
        summaries: list[dict[str, Any]] = []
        if state.decomposition is None:
            return summaries

        for subtask in state.decomposition.subtasks:
            result = state.subtask_results.get(subtask.subtask_id)
            if result is None:
                continue
            summaries.append({
                "subtask_id": subtask.subtask_id,
                "description": subtask.description,
                "task_type": subtask.task_type,
                "status": result.status,
                "content": result.output.get("content", "")[:500],
            })
        return summaries

    def _build_provenance(self, state: GoalState) -> list[dict[str, Any]]:
        """Build provenance chain from subtask results."""
        provenance: list[dict[str, Any]] = []
        for sid, result in state.subtask_results.items():
            provenance.append({
                "subtask_id": sid,
                "model_used": result.model_used,
                "cost_usd": result.cost_usd,
                "latency_ms": result.latency_ms,
                "status": result.status,
                "tool_calls_count": len(result.tool_calls),
            })
        return provenance

    async def _llm_synthesize(
        self,
        goal_description: str,
        summaries: list[dict[str, Any]],
    ) -> SynthesisInput:
        """Call LLM to synthesize subtask results into a coherent deliverable."""
        summary_text = "\n".join(
            f"[{s['subtask_id']}] ({s['task_type']}, {s['status']}): {s['content']}"
            for s in summaries
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a synthesis assistant. Given a goal and the results "
                    "of multiple subtasks, produce a coherent summary with key "
                    "findings, a confidence score, and actionable recommendations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal: {goal_description}\n\n"
                    f"Subtask Results:\n{summary_text}\n\n"
                    "Synthesize these results into a coherent deliverable."
                ),
            },
        ]

        client = instructor.from_litellm(litellm.acompletion)
        result = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_model=SynthesisInput,
        )

        if self._budget_tracker is not None:
            try:
                # instructor wraps litellm, so cost tracking is approximate
                self._budget_tracker.record_cost(
                    self._model, 0.001, service_id="synthesizer",
                )
            except Exception:
                pass

        return result

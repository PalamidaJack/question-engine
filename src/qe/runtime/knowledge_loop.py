"""KnowledgeLoop — Background consolidation service (minutes-hours timescale).

Periodically consolidates short-term episodic findings into long-term
semantic beliefs, detects contradictions, manages hypothesis lifecycles
across inquiries, and retires failing procedural patterns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import instructor
import litellm
from pydantic import BaseModel, Field

from qe.runtime.episodic_memory import Episode, EpisodicMemory
from qe.runtime.feature_flags import get_flag_store
from qe.runtime.procedural_memory import ProceduralMemory
from qe.substrate.bayesian_belief import BayesianBeliefStore, EvidenceRecord

log = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────


class ConsolidationResult(BaseModel):
    """Summary of a single consolidation cycle."""

    episodes_scanned: int = 0
    patterns_detected: int = 0
    beliefs_promoted: int = 0
    contradictions_found: int = 0
    hypotheses_reviewed: int = 0
    hypotheses_confirmed: int = 0
    hypotheses_falsified: int = 0
    templates_retired: int = 0
    cycle_duration_s: float = 0.0


class ExtractedClaim(BaseModel):
    """LLM-extracted claim from episodic pattern."""

    subject_entity_id: str
    predicate: str
    object_value: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


# ── KnowledgeLoop ─────────────────────────────────────────────────────────


class KnowledgeLoop:
    """Background consolidation service: episodic → semantic → procedural.

    The middle of the three nested loops, operating on the
    minutes-hours timescale.
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        belief_store: BayesianBeliefStore,
        procedural_memory: ProceduralMemory,
        bus: Any = None,
        model: str = "gpt-4o-mini",
        consolidation_interval: float = 300.0,
        episode_lookback_hours: float = 1.0,
        promotion_confidence: float = 0.7,
        retirement_threshold: float = 0.2,
        min_evidence_count: int = 3,
    ) -> None:
        self._episodic = episodic_memory
        self._beliefs = belief_store
        self._procedural = procedural_memory
        self._bus = bus
        self._model = model
        self._consolidation_interval = consolidation_interval
        self._episode_lookback_hours = episode_lookback_hours
        self._promotion_confidence = promotion_confidence
        self._retirement_threshold = retirement_threshold
        self._min_evidence_count = min_evidence_count

        self._running = False
        self._loop_task: asyncio.Task[Any] | None = None
        self._cycles_total = 0
        self._beliefs_promoted_total = 0
        self._contradictions_total = 0
        self._last_cycle_at: datetime | None = None
        self._last_cycle_result: ConsolidationResult | None = None
        self._retired_template_ids: set[str] = set()
        self._retired_sequence_ids: set[str] = set()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background consolidation loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._consolidation_loop())
        log.info("knowledge_loop.started")

    async def stop(self) -> None:
        """Stop the background consolidation loop."""
        self._running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass
        self._loop_task = None
        log.info("knowledge_loop.stopped")

    def status(self) -> dict[str, Any]:
        """Return monitoring snapshot."""
        return {
            "running": self._running,
            "cycles_total": self._cycles_total,
            "beliefs_promoted_total": self._beliefs_promoted_total,
            "contradictions_total": self._contradictions_total,
            "last_cycle_at": (
                self._last_cycle_at.isoformat() if self._last_cycle_at else None
            ),
            "last_cycle_result": (
                self._last_cycle_result.model_dump()
                if self._last_cycle_result
                else None
            ),
        }

    async def trigger_consolidation(self) -> None:
        """Run consolidation immediately (event-driven, called by InquiryBridge)."""
        if not self._running:
            return
        try:
            await self._consolidate()
        except Exception:
            log.exception("knowledge_loop.triggered_consolidation_error")

    # ── Main loop ─────────────────────────────────────────────────────

    async def _consolidation_loop(self) -> None:
        """Periodic loop: sleep, then consolidate."""
        while self._running:
            try:
                await asyncio.sleep(self._consolidation_interval)
                await self._consolidate()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("knowledge_loop.consolidation_error")

    async def _consolidate(self) -> None:
        """Single consolidation cycle with four phases."""
        t0 = time.monotonic()
        result = ConsolidationResult()

        # Check feature flag
        if not get_flag_store().is_enabled("knowledge_consolidation"):
            return

        # Phase 1 — Episode Scan
        episodes = await self._episodic.recall(
            query="",
            top_k=200,
            time_window_hours=self._episode_lookback_hours,
        )
        result.episodes_scanned = len(episodes)

        # Group by episode_type, extract entity-predicate patterns
        # from synthesis and claim_committed episodes
        groups = self._group_episodes(episodes)

        # Phase 2 — Pattern Detection & Belief Promotion
        for _type_key, episode_group in groups.items():
            if len(episode_group) < self._min_evidence_count:
                continue

            result.patterns_detected += 1

            claim = await self._extract_claim_from_episodes(episode_group)
            if claim is None:
                continue
            if claim.confidence < self._promotion_confidence:
                continue

            # Promote to belief store
            try:
                from qe.models.claim import Claim

                belief_claim = Claim(
                    subject_entity_id=claim.subject_entity_id,
                    predicate=claim.predicate,
                    object_value=claim.object_value,
                    confidence=claim.confidence,
                    source_service_id="knowledge_loop",
                    source_envelope_ids=[],
                )
                evidence = EvidenceRecord(
                    source="knowledge_loop_consolidation",
                    supports=True,
                    strength=claim.confidence,
                    metadata={"episode_count": len(episode_group)},
                )
                await self._beliefs.update_belief(belief_claim, evidence)
                result.beliefs_promoted += 1

                self._publish("knowledge.belief_promoted", {
                    "subject_entity_id": claim.subject_entity_id,
                    "predicate": claim.predicate,
                    "object_value": claim.object_value,
                    "confidence": claim.confidence,
                    "evidence_count": len(episode_group),
                })
            except Exception:
                log.warning("knowledge_loop.belief_promotion_failed", exc_info=True)

        # Phase 3 — Hypothesis Review
        try:
            hypotheses = await self._beliefs.get_active_hypotheses()
            result.hypotheses_reviewed = len(hypotheses)

            for hyp in hypotheses:
                if hyp.current_probability >= 0.95:
                    result.hypotheses_confirmed += 1
                    self._publish("knowledge.hypothesis_updated", {
                        "hypothesis_id": hyp.hypothesis_id,
                        "old_status": "active",
                        "new_status": "confirmed",
                        "probability": hyp.current_probability,
                    })
                elif hyp.current_probability <= 0.05:
                    result.hypotheses_falsified += 1
                    self._publish("knowledge.hypothesis_updated", {
                        "hypothesis_id": hyp.hypothesis_id,
                        "old_status": "active",
                        "new_status": "falsified",
                        "probability": hyp.current_probability,
                    })
        except Exception:
            log.warning("knowledge_loop.hypothesis_review_failed", exc_info=True)

        # Phase 4 — Procedural Retirement
        try:
            templates = await self._procedural.get_best_templates(
                domain="general", top_k=50,
            )
            sequences = await self._procedural.get_best_sequences(
                domain="general", top_k=50,
            )

            for tmpl in templates:
                total = tmpl.success_count + tmpl.failure_count
                if total >= 10 and tmpl.success_rate < self._retirement_threshold:
                    if tmpl.template_id not in self._retired_template_ids:
                        self._retired_template_ids.add(tmpl.template_id)
                        result.templates_retired += 1

            for seq in sequences:
                total = seq.success_count + seq.failure_count
                if total >= 10 and seq.success_rate < self._retirement_threshold:
                    if seq.sequence_id not in self._retired_sequence_ids:
                        self._retired_sequence_ids.add(seq.sequence_id)
                        result.templates_retired += 1
        except Exception:
            log.warning("knowledge_loop.procedural_retirement_failed", exc_info=True)

        # Finalize
        result.cycle_duration_s = time.monotonic() - t0
        self._cycles_total += 1
        self._beliefs_promoted_total += result.beliefs_promoted
        self._contradictions_total += result.contradictions_found
        self._last_cycle_at = datetime.now(UTC)
        self._last_cycle_result = result

        self._publish("knowledge.consolidation_completed", {
            "episodes_scanned": result.episodes_scanned,
            "patterns_detected": result.patterns_detected,
            "beliefs_promoted": result.beliefs_promoted,
            "contradictions_found": result.contradictions_found,
            "hypotheses_reviewed": result.hypotheses_reviewed,
        })

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _group_episodes(
        episodes: list[Episode],
    ) -> dict[str, list[Episode]]:
        """Group episodes by type, filtering to consolidation-relevant types."""
        groups: dict[str, list[Episode]] = defaultdict(list)
        relevant_types = {"synthesis", "claim_committed"}
        for ep in episodes:
            if ep.episode_type in relevant_types:
                groups[ep.episode_type].append(ep)
        return dict(groups)

    async def _extract_claim_from_episodes(
        self,
        episodes: list[Episode],
    ) -> ExtractedClaim | None:
        """Use LLM to extract a structured claim from episodic observations."""
        summaries = []
        for ep in episodes[:10]:  # cap at 10 to stay within context
            text = ep.summary or str(ep.content)
            summaries.append(f"- [{ep.episode_type}] {text}")

        observations_text = "\n".join(summaries)

        try:
            client = instructor.from_litellm(litellm.acompletion)
            result = await client.chat.completions.create(
                model=self._model,
                response_model=ExtractedClaim,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract a single factual claim from these episodic "
                            "observations. Return subject, predicate, object_value, "
                            "and your confidence (0-1)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Observations:\n{observations_text}",
                    },
                ],
            )
            if result.confidence < 0.3:
                return None
            return result
        except Exception:
            log.warning("knowledge_loop.llm_extraction_failed", exc_info=True)
            return None

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish a bus event if bus is available."""
        if self._bus is None:
            return
        try:
            from qe.models.envelope import Envelope

            self._bus.publish(
                Envelope(
                    topic=topic,
                    source_service_id="knowledge_loop",
                    payload=payload,
                )
            )
        except Exception:
            log.debug("knowledge_loop.publish_failed topic=%s", topic)

"""Innovation Scout service: poll loop, HIL integration, apply/reject."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from qe.config import ScoutConfig
from qe.models.envelope import Envelope
from qe.models.scout import ScoutFeedbackRecord
from qe.runtime.feature_flags import get_flag_store
from qe.services.scout.analyzer import ScoutAnalyzer
from qe.services.scout.codegen import ScoutCodeGenerator
from qe.services.scout.pipeline import ScoutPipeline
from qe.services.scout.sandbox import ScoutSandbox
from qe.services.scout.sources import SourceManager
from qe.substrate.scout_store import ScoutStore

log = logging.getLogger(__name__)


class InnovationScoutService:
    """Self-improving meta-agent that scouts for improvements."""

    def __init__(
        self,
        bus,
        scout_store: ScoutStore,
        config: ScoutConfig | None = None,
        model: str = "gpt-4o-mini",
        balanced_model: str = "gpt-4o",
    ) -> None:
        self._bus = bus
        self._store = scout_store
        self._config = config or ScoutConfig()
        self._model = model
        self._balanced_model = balanced_model
        self._running = False
        self._poll_task: asyncio.Task | None = None
        self._cycles_completed = 0
        self._last_cycle_at: datetime | None = None

        # Build pipeline components
        self._source_manager = SourceManager(
            model=model,
            search_topics=self._config.search_topics,
        )
        self._analyzer = ScoutAnalyzer(
            model=model,
            min_composite_score=self._config.min_composite_score,
        )
        self._codegen = ScoutCodeGenerator(model=balanced_model)
        self._sandbox = ScoutSandbox()

        self._pipeline = ScoutPipeline(
            source_manager=self._source_manager,
            analyzer=self._analyzer,
            codegen=self._codegen,
            sandbox=self._sandbox,
            scout_store=self._store,
            bus=bus,
            max_findings_per_cycle=self._config.max_findings_per_cycle,
            max_proposals_per_cycle=self._config.max_proposals_per_cycle,
            hil_timeout_seconds=self._config.hil_timeout_seconds,
        )

    async def start(self) -> None:
        """Start the scout service."""
        self._running = True

        # Subscribe to HIL events
        await self._maybe_subscribe("hil.approved", self._on_hil_approved)
        await self._maybe_subscribe("hil.rejected", self._on_hil_rejected)

        # Start poll loop
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info("scout.started interval=%ds", self._config.poll_interval_seconds)

    async def stop(self) -> None:
        """Stop the scout service."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._maybe_unsubscribe("hil.approved", self._on_hil_approved)
        await self._maybe_unsubscribe("hil.rejected", self._on_hil_rejected)
        log.info("scout.stopped cycles=%d", self._cycles_completed)

    async def _poll_loop(self) -> None:
        """Background loop: run cycles at configured interval."""
        while self._running:
            try:
                await asyncio.sleep(self._config.poll_interval_seconds)
                if not self._running:
                    break
                await self._run_cycle()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("scout.poll_loop_error")

    async def _run_cycle(self) -> None:
        """Execute one scouting cycle if conditions are met."""
        if not get_flag_store().is_enabled("innovation_scout"):
            return

        # Check pending count
        pending = await self._store.count_proposals(status="pending_review")
        if pending >= self._config.max_pending_proposals:
            log.info(
                "scout.cycle_skipped reason=too_many_pending count=%d",
                pending,
            )
            return

        # Apply learning from past feedback
        await self._apply_learning()

        # Run the pipeline
        summary = await self._pipeline.run_cycle()
        self._cycles_completed += 1
        self._last_cycle_at = datetime.now(UTC)

        log.info(
            "scout.cycle_done cycle=%d findings=%d proposals=%d",
            self._cycles_completed,
            summary.get("findings_count", 0),
            summary.get("proposals_count", 0),
        )

    async def _apply_learning(self) -> None:
        """Apply feedback-driven adjustments before a cycle."""
        rejected_patterns = await self._store.get_rejected_patterns()
        self._source_manager.set_rejected_patterns(
            categories=rejected_patterns.get("rejected_categories", []),
            sources=rejected_patterns.get("rejected_sources", []),
        )

        # Dynamic threshold adjustment
        stats = await self._store.get_feedback_stats()
        total = stats.get("total", 0)
        if total >= 10:
            rate = stats.get("approval_rate", 0.5)
            if rate < 0.2:
                self._analyzer.update_threshold(
                    min(self._config.min_composite_score + 0.1, 0.9)
                )
            elif rate > 0.7:
                self._analyzer.update_threshold(
                    max(self._config.min_composite_score - 0.05, 0.2)
                )

    async def _on_hil_approved(self, envelope: Envelope) -> None:
        """Handle approved proposal: merge branch, update status, record feedback."""
        proposal_id = envelope.correlation_id
        if not proposal_id:
            return

        proposal = await self._store.get_proposal(proposal_id)
        if proposal is None or proposal.status != "pending_review":
            return

        # Merge branch to main
        merged = await self._sandbox.merge_branch(
            proposal.branch_name, proposal.idea.title,
        )

        if merged:
            # Clean up worktree and branch
            await self._sandbox.cleanup_worktree(
                proposal.worktree_path, proposal.branch_name, delete_branch=True,
            )

            now = datetime.now(UTC)
            await self._store.update_proposal_status(
                proposal_id, "applied", decided_at=now, applied_at=now,
            )

            # Record feedback for learning
            feedback = ScoutFeedbackRecord(
                proposal_id=proposal_id,
                decision="approved",
                feedback=envelope.payload.get("reason", ""),
                category=proposal.idea.category,
                source_type=proposal.idea.source_url[:50] if proposal.idea.source_url else "",
            )
            await self._store.save_feedback(feedback)

            self._publish("scout.proposal_applied", {
                "proposal_id": proposal_id,
                "title": proposal.idea.title,
                "branch_name": proposal.branch_name,
                "decision": "applied",
            })
            self._publish("scout.learning_recorded", {
                "record_id": feedback.record_id,
                "proposal_id": proposal_id,
                "decision": "approved",
                "category": proposal.idea.category,
            })
        else:
            log.error("scout.merge_failed proposal=%s", proposal_id)

    async def _on_hil_rejected(self, envelope: Envelope) -> None:
        """Handle rejected proposal: clean up, record feedback for learning."""
        proposal_id = envelope.correlation_id
        if not proposal_id:
            return

        proposal = await self._store.get_proposal(proposal_id)
        if proposal is None or proposal.status != "pending_review":
            return

        # Clean up worktree and branch
        await self._sandbox.cleanup_worktree(
            proposal.worktree_path, proposal.branch_name, delete_branch=True,
        )

        reason = envelope.payload.get("reason", "")
        now = datetime.now(UTC)
        await self._store.update_proposal_status(
            proposal_id, "rejected",
            reviewer_feedback=reason,
            decided_at=now,
        )

        # Record feedback for learning
        feedback = ScoutFeedbackRecord(
            proposal_id=proposal_id,
            decision="rejected",
            feedback=reason,
            category=proposal.idea.category,
            source_type=proposal.idea.source_url[:50] if proposal.idea.source_url else "",
        )
        await self._store.save_feedback(feedback)

        self._publish("scout.learning_recorded", {
            "record_id": feedback.record_id,
            "proposal_id": proposal_id,
            "decision": "rejected",
            "category": proposal.idea.category,
        })

    def status(self) -> dict:
        """Return service status."""
        return {
            "running": self._running,
            "cycles_completed": self._cycles_completed,
            "last_cycle_at": (
                self._last_cycle_at.isoformat() if self._last_cycle_at else None
            ),
            "poll_interval_seconds": self._config.poll_interval_seconds,
            "min_composite_score": self._analyzer._min_composite_score,
        }

    def _publish(self, topic: str, payload: dict) -> None:
        if self._bus:
            self._bus.publish(
                Envelope(
                    topic=topic,
                    source_service_id="innovation_scout",
                    payload=payload,
                )
            )

    async def _maybe_subscribe(self, topic: str, handler) -> None:
        result = self._bus.subscribe(topic, handler)
        if asyncio.iscoroutine(result):
            await result

    async def _maybe_unsubscribe(self, topic: str, handler) -> None:
        result = self._bus.unsubscribe(topic, handler)
        if asyncio.iscoroutine(result):
            await result

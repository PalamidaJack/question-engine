"""Six-phase scouting pipeline orchestration."""

from __future__ import annotations

import logging
import time
import uuid

from qe.models.envelope import Envelope
from qe.models.scout import ImprovementIdea, ImprovementProposal, ScoutFinding
from qe.services.scout.analyzer import ScoutAnalyzer
from qe.services.scout.codegen import ScoutCodeGenerator
from qe.services.scout.sandbox import ScoutSandbox
from qe.services.scout.sources import SourceManager
from qe.substrate.scout_store import ScoutStore

log = logging.getLogger(__name__)


class ScoutPipeline:
    """Orchestrate the 6-phase scouting cycle."""

    def __init__(
        self,
        source_manager: SourceManager,
        analyzer: ScoutAnalyzer,
        codegen: ScoutCodeGenerator,
        sandbox: ScoutSandbox,
        scout_store: ScoutStore,
        bus=None,
        max_findings_per_cycle: int = 20,
        max_proposals_per_cycle: int = 3,
        hil_timeout_seconds: int = 86400,
    ) -> None:
        self._sources = source_manager
        self._analyzer = analyzer
        self._codegen = codegen
        self._sandbox = sandbox
        self._store = scout_store
        self._bus = bus
        self._max_findings = max_findings_per_cycle
        self._max_proposals = max_proposals_per_cycle
        self._hil_timeout = hil_timeout_seconds

    async def run_cycle(self) -> dict:
        """Run one full scouting cycle. Returns cycle summary."""
        cycle_id = f"cyc_{uuid.uuid4().hex[:8]}"
        start = time.monotonic()

        self._publish("scout.cycle_started", {"cycle_id": cycle_id})

        # Phase 1: Source Discovery
        log.info("scout.phase1_source_discovery cycle=%s", cycle_id)
        queries = await self._sources.generate_queries()
        seen_urls: set[str] = set()
        findings = await self._sources.search(
            queries, seen_urls=seen_urls,
        )
        findings = findings[:self._max_findings]

        # Dedup against store
        new_findings: list[ScoutFinding] = []
        for f in findings:
            if not await self._store.has_finding_url(f.url):
                new_findings.append(f)
                await self._store.save_finding(f)
                self._publish("scout.finding_discovered", {
                    "finding_id": f.finding_id,
                    "url": f.url,
                    "title": f.title,
                    "source_type": f.source_type,
                    "relevance_score": f.relevance_score,
                })

        # Phase 2: Content Extraction
        log.info("scout.phase2_content_extraction count=%d", len(new_findings))
        enriched = await self._sources.fetch_content(new_findings)

        # Phase 3: Relevance & Feasibility Analysis
        log.info("scout.phase3_analysis count=%d", len(enriched))
        ideas = await self._analyzer.analyze(enriched)
        ideas.sort(key=lambda i: i.composite_score, reverse=True)

        for idea in ideas:
            self._publish("scout.idea_analyzed", {
                "idea_id": idea.idea_id,
                "finding_id": idea.finding_id,
                "title": idea.title,
                "category": idea.category,
                "composite_score": idea.composite_score,
            })

        # Take top N ideas for proposal generation
        top_ideas = ideas[:self._max_proposals]

        # Phases 4-6: Code Gen + Sandbox + Submit for each idea
        proposals: list[ImprovementProposal] = []
        for idea in top_ideas:
            try:
                proposal = await self._process_idea(idea)
                if proposal:
                    proposals.append(proposal)
            except Exception:
                log.exception("scout.idea_processing_failed idea=%s", idea.idea_id)

        duration = time.monotonic() - start
        summary = {
            "cycle_id": cycle_id,
            "findings_count": len(new_findings),
            "ideas_count": len(ideas),
            "proposals_count": len(proposals),
            "duration_s": round(duration, 2),
        }

        self._publish("scout.cycle_completed", summary)
        log.info("scout.cycle_completed %s", summary)
        return summary

    async def _process_idea(
        self,
        idea: ImprovementIdea,
    ) -> ImprovementProposal | None:
        """Phase 4-6: Code gen → sandbox test → submit for review."""
        proposal = ImprovementProposal(idea=idea, status="draft")

        # Phase 4: Code Generation
        log.info("scout.phase4_codegen idea=%s", idea.idea_id)
        file_contents = await self._codegen.read_affected_files(idea.affected_files)
        changes, impact, risk, rollback = await self._codegen.generate(
            idea, file_contents,
        )
        if not changes:
            log.info("scout.no_changes_generated idea=%s", idea.idea_id)
            return None

        proposal.changes = changes
        proposal.impact_assessment = impact
        proposal.risk_assessment = risk
        proposal.rollback_plan = rollback

        # Phase 5: Sandbox Testing
        log.info("scout.phase5_sandbox proposal=%s", proposal.proposal_id)
        proposal.status = "testing"
        try:
            slug = idea.title.replace(" ", "-").lower()[:30]
            worktree_path, branch_name = await self._sandbox.create_worktree(
                proposal.proposal_id, slug,
            )
            proposal.worktree_path = worktree_path
            proposal.branch_name = branch_name

            # Apply changes to worktree
            await self._sandbox.apply_changes(
                worktree_path, changes, file_contents,
            )

            # Run tests
            test_result = await self._sandbox.run_tests(worktree_path)
            proposal.test_result = test_result

            self._publish("scout.proposal_tested", {
                "proposal_id": proposal.proposal_id,
                "passed": test_result.passed,
                "total_tests": test_result.total_tests,
                "passed_tests": test_result.passed_tests,
                "failed_tests": test_result.failed_tests,
                "duration_s": test_result.duration_s,
            })

            if test_result.passed:
                proposal.status = "test_passed"
                # Capture diffs
                diff_changes = await self._sandbox.capture_diffs(worktree_path)
                if diff_changes:
                    proposal.changes = diff_changes
            else:
                proposal.status = "test_failed"
                # Clean up failed worktrees
                await self._sandbox.cleanup_worktree(
                    worktree_path, branch_name, delete_branch=True,
                )
                await self._store.save_proposal(proposal)
                return None

        except Exception:
            log.exception("scout.sandbox_failed proposal=%s", proposal.proposal_id)
            proposal.status = "test_failed"
            await self._store.save_proposal(proposal)
            return None

        # Phase 6: Package & Submit for review
        log.info("scout.phase6_submit proposal=%s", proposal.proposal_id)
        proposal.status = "pending_review"

        # Publish HIL approval request
        hil_envelope = Envelope(
            topic="hil.approval_required",
            source_service_id="innovation_scout",
            correlation_id=proposal.proposal_id,
            payload={
                "reason": "scout_proposal",
                "proposal_summary": (
                    f"[{idea.category}] {idea.title}: {idea.description[:200]}"
                ),
                "proposal_id": proposal.proposal_id,
                "timeout_seconds": self._hil_timeout,
            },
        )
        proposal.hil_envelope_id = hil_envelope.envelope_id

        if self._bus:
            self._bus.publish(hil_envelope)

        self._publish("scout.proposal_created", {
            "proposal_id": proposal.proposal_id,
            "title": idea.title,
            "category": idea.category,
            "branch_name": proposal.branch_name,
            "status": proposal.status,
        })

        await self._store.save_proposal(proposal)
        return proposal

    def _publish(self, topic: str, payload: dict) -> None:
        """Publish a bus event if bus is available."""
        if self._bus:
            self._bus.publish(
                Envelope(
                    topic=topic,
                    source_service_id="innovation_scout",
                    payload=payload,
                )
            )

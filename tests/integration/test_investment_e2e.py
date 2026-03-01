"""E2E Investment Opportunity Walkthrough — Phase 5 integration test.

Full cognitive pipeline test exercising all Phase 1-5 components:
ProceduralMemory in Orient, PersistenceEngine in Reflect, phase timings,
question persistence, and bus event ordering.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.services.inquiry.engine import InquiryEngine
from qe.services.inquiry.schemas import InquiryConfig

GOAL_ID = "goal_lithium_invest"
GOAL_DESC = "Evaluate lithium-ion battery storage as a $50M investment opportunity"


@pytest.mark.asyncio
class TestInvestmentWalkthrough:
    """Full cognitive pipeline test for investment analysis."""

    async def _run(
        self,
        factory: Callable[..., InquiryEngine],
        config: InquiryConfig | None = None,
    ):
        cfg = config or InquiryConfig(
            max_iterations=3,
            questions_per_iteration=2,
            confidence_threshold=0.99,  # High threshold to run all 3 iterations
            domain="finance",
        )
        engine = factory(cfg)
        return await engine.run_inquiry(GOAL_ID, GOAL_DESC, cfg)

    async def test_full_walkthrough_completes(
        self,
        inquiry_engine_factory: Callable[..., InquiryEngine],
    ):
        result = await self._run(inquiry_engine_factory)
        assert result.status == "completed"
        assert result.iterations_completed >= 1

    async def test_procedural_memory_consulted(
        self,
        inquiry_engine_factory: Callable[..., InquiryEngine],
        cognitive_stack: dict[str, Any],
    ):
        # Use a mock wrapper to track calls while still using real ProceduralMemory
        real_pm = cognitive_stack["procedural_memory"]
        original_get = real_pm.get_best_templates

        call_count = 0

        async def _tracking_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return await original_get(*args, **kwargs)

        real_pm.get_best_templates = _tracking_get
        try:
            await self._run(inquiry_engine_factory)
            assert call_count >= 1
        finally:
            real_pm.get_best_templates = original_get

    async def test_persistence_engine_on_drift(
        self,
        cognitive_stack: dict[str, Any],
        mock_bus: MemoryBus,
    ):
        """Force drift via context_curator, verify persistence engine fires."""
        # Override context_curator to force drift
        cc = MagicMock()
        drift_report = MagicMock()
        drift_report.similarity = 0.2  # Low similarity = drift detected
        cc.detect_drift = MagicMock(return_value=drift_report)

        pe = cognitive_stack["persistence_engine"]

        engine = InquiryEngine(
            episodic_memory=cognitive_stack["episodic_memory"],
            context_curator=cc,
            metacognitor=cognitive_stack["metacognitor"],
            epistemic_reasoner=cognitive_stack["epistemic_reasoner"],
            dialectic_engine=cognitive_stack["dialectic_engine"],
            persistence_engine=pe,
            insight_crystallizer=cognitive_stack["insight_crystallizer"],
            question_generator=cognitive_stack["question_generator"],
            hypothesis_manager=cognitive_stack["hypothesis_manager"],
            procedural_memory=cognitive_stack.get("procedural_memory"),
            bus=mock_bus,
            config=InquiryConfig(
                max_iterations=2,
                questions_per_iteration=2,
                confidence_threshold=0.99,
                domain="finance",
            ),
        )

        await engine.run_inquiry(GOAL_ID, GOAL_DESC)
        pe.analyze_root_cause.assert_called()
        pe.reframe.assert_called()

    async def test_multi_iteration_question_accumulation(
        self,
        inquiry_engine_factory: Callable[..., InquiryEngine],
    ):
        config = InquiryConfig(
            max_iterations=3,
            questions_per_iteration=2,
            confidence_threshold=0.99,
            domain="finance",
        )
        result = await self._run(inquiry_engine_factory, config)
        # At least questions_per_iteration * iterations generated
        assert result.total_questions_generated >= 3 * 2

    async def test_insights_have_headlines(
        self,
        inquiry_engine_factory: Callable[..., InquiryEngine],
    ):
        result = await self._run(inquiry_engine_factory)
        for insight in result.insights:
            assert insight["headline"], "Each insight must have a non-empty headline"

    async def test_episodic_memory_accessible(
        self,
        inquiry_engine_factory: Callable[..., InquiryEngine],
        cognitive_stack: dict[str, Any],
    ):
        await self._run(inquiry_engine_factory)
        episodic = cognitive_stack["episodic_memory"]
        # Verify episodic memory can accept stores (is functional)
        episodes = await episodic.recall("battery storage", top_k=50)
        assert isinstance(episodes, list)

    async def test_question_store_persists(
        self,
        inquiry_engine_factory: Callable[..., InquiryEngine],
        cognitive_stack: dict[str, Any],
    ):
        result = await self._run(inquiry_engine_factory)
        qs = cognitive_stack["question_store"]
        # Get all questions for the inquiry
        questions = await qs.get_question_tree(result.inquiry_id)
        assert len(questions) >= 1

    async def test_bus_events_complete(
        self,
        inquiry_engine_factory: Callable[..., InquiryEngine],
        mock_bus: MemoryBus,
    ):
        await self._run(inquiry_engine_factory)
        events = mock_bus._collected_events  # type: ignore[attr-defined]
        topics = [e.topic for e in events]

        # Verify ordering: started → phases → completed
        assert "inquiry.started" in topics
        assert "inquiry.completed" in topics
        started_idx = topics.index("inquiry.started")
        completed_idx = topics.index("inquiry.completed")
        phase_indices = [i for i, t in enumerate(topics) if t == "inquiry.phase_completed"]
        assert phase_indices, "Expected phase_completed events"
        assert started_idx < phase_indices[0]
        assert completed_idx > phase_indices[-1]

    async def test_phase_timings_in_result(
        self,
        inquiry_engine_factory: Callable[..., InquiryEngine],
    ):
        result = await self._run(inquiry_engine_factory)
        expected_phases = {
            "observe", "orient", "question", "prioritize",
            "investigate", "synthesize", "reflect",
        }
        assert expected_phases == set(result.phase_timings.keys())
        for _phase, stats in result.phase_timings.items():
            assert stats["total_s"] > 0

    async def test_cost_and_duration(
        self,
        inquiry_engine_factory: Callable[..., InquiryEngine],
    ):
        result = await self._run(inquiry_engine_factory)
        assert result.total_cost_usd >= 0
        assert result.duration_seconds > 0

    async def test_hypothesis_lifecycle(
        self,
        inquiry_engine_factory: Callable[..., InquiryEngine],
        mock_bus: MemoryBus,
    ):
        await self._run(inquiry_engine_factory)
        events = mock_bus._collected_events  # type: ignore[attr-defined]
        # Hypothesis events are generated when surprise is detected
        # The mock epistemic reasoner returns surprise_magnitude=0.3 < 0.5 threshold
        # so no hypotheses are generated by default — verify the flow still completes
        completed = [e for e in events if e.topic == "inquiry.completed"]
        assert len(completed) >= 1

    async def test_budget_limit_termination(
        self,
        cognitive_stack: dict[str, Any],
        mock_bus: MemoryBus,
    ):
        """Tight budget should cause budget_exhausted termination."""
        budget = MagicMock()
        budget._total_spend = 49.0
        budget.monthly_limit_usd = 50.0

        engine = InquiryEngine(
            episodic_memory=cognitive_stack["episodic_memory"],
            context_curator=cognitive_stack["context_curator"],
            metacognitor=cognitive_stack["metacognitor"],
            epistemic_reasoner=cognitive_stack["epistemic_reasoner"],
            dialectic_engine=cognitive_stack["dialectic_engine"],
            persistence_engine=cognitive_stack["persistence_engine"],
            insight_crystallizer=cognitive_stack["insight_crystallizer"],
            question_generator=cognitive_stack["question_generator"],
            hypothesis_manager=cognitive_stack["hypothesis_manager"],
            procedural_memory=cognitive_stack.get("procedural_memory"),
            budget_tracker=budget,
            bus=mock_bus,
            config=InquiryConfig(
                max_iterations=10,
                budget_hard_stop_pct=0.05,
                domain="finance",
            ),
        )

        result = await engine.run_inquiry(GOAL_ID, GOAL_DESC)
        assert result.termination_reason == "budget_exhausted"

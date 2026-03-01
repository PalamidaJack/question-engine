"""Tests for InquiryEngine — 7-phase loop, termination conditions, phase behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.cognition import (
    ApproachAssessment,
    DialecticReport,
    ReframingResult,
    RootCauseAnalysis,
    SurpriseDetection,
    UncertaintyAssessment,
)
from qe.services.inquiry.engine import InquiryEngine
from qe.services.inquiry.schemas import (
    InquiryConfig,
    Question,
)
from qe.substrate.bayesian_belief import Hypothesis


def _make_engine(**overrides):
    """Create an InquiryEngine with mocked components."""
    defaults = {
        "episodic_memory": MagicMock(),
        "context_curator": MagicMock(),
        "metacognitor": MagicMock(),
        "epistemic_reasoner": MagicMock(),
        "dialectic_engine": MagicMock(),
        "persistence_engine": MagicMock(),
        "insight_crystallizer": MagicMock(),
        "question_generator": MagicMock(),
        "hypothesis_manager": MagicMock(),
        "tool_registry": None,
        "budget_tracker": None,
        "bus": MagicMock(),
        "config": InquiryConfig(max_iterations=3, confidence_threshold=0.8),
    }
    defaults.update(overrides)

    # Set up common async returns (only for components not overridden)
    if "episodic_memory" not in overrides:
        em = defaults["episodic_memory"]
        em.recall_for_goal = AsyncMock(return_value=[])

    if "metacognitor" not in overrides:
        mc = defaults["metacognitor"]
        mc.suggest_next_approach = AsyncMock(
            return_value=ApproachAssessment(
                recommended_approach="Use web search",
                reasoning="Good starting point",
            )
        )

    if "question_generator" not in overrides:
        qg = defaults["question_generator"]
        qg.generate = AsyncMock(return_value=[
            Question(
                text="Generated Q1",
                expected_info_gain=0.8,
                relevance_to_goal=0.9,
                novelty_score=0.7,
            ),
        ])
        qg.prioritize = AsyncMock(side_effect=lambda goal, qs: sorted(
            qs, key=lambda q: q.expected_info_gain, reverse=True
        ))

    if "hypothesis_manager" not in overrides:
        hm = defaults["hypothesis_manager"]
        hm.get_active_hypotheses = AsyncMock(return_value=[])
        hm.generate_hypotheses = AsyncMock(return_value=[])

    if "epistemic_reasoner" not in overrides:
        ep = defaults["epistemic_reasoner"]
        ep.get_epistemic_state = MagicMock(return_value=None)
        ep.assess_uncertainty = AsyncMock(
            return_value=UncertaintyAssessment(finding_summary="test")
        )
        ep.detect_surprise = AsyncMock(return_value=None)
        ep.get_blind_spot_warning = MagicMock(return_value="")

    if "dialectic_engine" not in overrides:
        de = defaults["dialectic_engine"]
        de.full_dialectic = AsyncMock(return_value=DialecticReport(
            original_conclusion="test", revised_confidence=0.6
        ))

    if "insight_crystallizer" not in overrides:
        ic = defaults["insight_crystallizer"]
        ic.crystallize = AsyncMock(return_value=None)

    if "context_curator" not in overrides:
        cc = defaults["context_curator"]
        cc.detect_drift = MagicMock(return_value=None)

    return InquiryEngine(**defaults)


# ── Core flow tests ─────────────────────────────────────────────────────────


class TestInquiryEngineBasicFlow:
    @pytest.mark.asyncio
    async def test_basic_flow_completes(self):
        engine = _make_engine()
        result = await engine.run_inquiry("g_1", "Test goal")

        assert result.inquiry_id.startswith("inq_")
        assert result.goal_id == "g_1"
        assert result.status == "completed"
        assert result.iterations_completed > 0

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        # Generate 3 questions per iteration so there are always pending ones
        qg = MagicMock()
        qg.generate = AsyncMock(return_value=[
            Question(text="Q1"), Question(text="Q2"), Question(text="Q3"),
        ])
        qg.prioritize = AsyncMock(side_effect=lambda g, qs: qs)

        # Dialectic returns low confidence to avoid early termination
        de = MagicMock()
        de.full_dialectic = AsyncMock(return_value=DialecticReport(
            original_conclusion="test", revised_confidence=0.1,
        ))

        engine = _make_engine(
            question_generator=qg,
            dialectic_engine=de,
            config=InquiryConfig(max_iterations=2, confidence_threshold=0.99),
        )
        result = await engine.run_inquiry("g_1", "Test goal")

        assert result.iterations_completed <= 2
        assert result.termination_reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_budget_exhausted(self):
        budget = MagicMock()
        budget.remaining_pct.return_value = 0.02  # 2% remaining

        engine = _make_engine(
            budget_tracker=budget,
            config=InquiryConfig(max_iterations=10, budget_hard_stop_pct=0.05),
        )
        result = await engine.run_inquiry("g_1", "Test")

        assert result.termination_reason == "budget_exhausted"

    @pytest.mark.asyncio
    async def test_confidence_met(self):
        engine = _make_engine(
            config=InquiryConfig(max_iterations=10, confidence_threshold=0.5)
        )
        # Dialectic sets confidence to 0.6 which exceeds 0.5 threshold
        result = await engine.run_inquiry("g_1", "Test")

        assert result.termination_reason == "confidence_met"

    @pytest.mark.asyncio
    async def test_no_questions_terminates(self):
        qg = MagicMock()
        qg.generate = AsyncMock(return_value=[])
        qg.prioritize = AsyncMock(return_value=[])

        engine = _make_engine(question_generator=qg)
        result = await engine.run_inquiry("g_1", "Test")

        assert result.termination_reason == "all_questions_answered"


# ── Phase-specific tests ───────────────────────────────────────────────────


class TestPhaseObserve:
    @pytest.mark.asyncio
    async def test_observe_gathers_episodes(self):
        em = MagicMock()
        em.recall_for_goal = AsyncMock(return_value=["episode1", "episode2"])

        engine = _make_engine(episodic_memory=em)
        await engine.run_inquiry("g_1", "Test")

        em.recall_for_goal.assert_called()

    @pytest.mark.asyncio
    async def test_observe_gathers_hypotheses(self):
        hm = MagicMock()
        hm.get_active_hypotheses = AsyncMock(return_value=[
            Hypothesis(statement="H1"),
        ])
        hm.generate_hypotheses = AsyncMock(return_value=[])

        engine = _make_engine(hypothesis_manager=hm)
        await engine.run_inquiry("g_1", "Test")

        hm.get_active_hypotheses.assert_called()


class TestPhaseOrient:
    @pytest.mark.asyncio
    async def test_orient_uses_metacognitor(self):
        mc = MagicMock()
        mc.suggest_next_approach = AsyncMock(
            return_value=ApproachAssessment(
                recommended_approach="Try analysis",
                reasoning="Good approach",
            )
        )

        engine = _make_engine(metacognitor=mc)
        await engine.run_inquiry("g_1", "Test")

        mc.suggest_next_approach.assert_called()

    @pytest.mark.asyncio
    async def test_orient_consults_procedural_memory(self):
        from qe.runtime.procedural_memory import QuestionTemplate

        pm = MagicMock()
        pm.get_best_templates = AsyncMock(return_value=[
            QuestionTemplate(
                pattern="What are the costs?",
                domain="general",
                success_count=5,
                failure_count=1,
            ),
        ])

        engine = _make_engine(procedural_memory=pm)
        await engine.run_inquiry("g_1", "Test")

        pm.get_best_templates.assert_called()

    @pytest.mark.asyncio
    async def test_orient_continues_when_procedural_unavailable(self):
        engine = _make_engine(procedural_memory=None)
        result = await engine.run_inquiry("g_1", "Test")

        assert result.status == "completed"


class TestPhaseQuestion:
    @pytest.mark.asyncio
    async def test_question_generates(self):
        qg = MagicMock()
        qg.generate = AsyncMock(return_value=[
            Question(text="Q1"), Question(text="Q2"),
        ])
        qg.prioritize = AsyncMock(side_effect=lambda g, qs: qs)

        engine = _make_engine(question_generator=qg)
        result = await engine.run_inquiry("g_1", "Test")

        qg.generate.assert_called()
        assert result.total_questions_generated >= 1


class TestPhasePrioritize:
    @pytest.mark.asyncio
    async def test_prioritize_sorts(self):
        qg = MagicMock()
        qg.generate = AsyncMock(return_value=[
            Question(text="Low", expected_info_gain=0.1),
            Question(text="High", expected_info_gain=0.9),
        ])
        qg.prioritize = AsyncMock(side_effect=lambda g, qs: sorted(
            qs, key=lambda q: q.expected_info_gain, reverse=True
        ))

        engine = _make_engine(question_generator=qg)
        await engine.run_inquiry("g_1", "Test")

        qg.prioritize.assert_called()


class TestPhaseInvestigate:
    @pytest.mark.asyncio
    async def test_investigate_no_tools(self):
        engine = _make_engine(tool_registry=None)
        result = await engine.run_inquiry("g_1", "Test")

        # Should complete without error even without tools
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_investigate_calls_tools(self):
        tr = MagicMock()
        tr.get_tool_schemas.return_value = [
            {"type": "function", "function": {"name": "test"}}
        ]

        # Mock litellm (imported inside _phase_investigate)
        with patch("litellm.acompletion") as mock_acompletion:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.tool_calls = None
            mock_response.choices[0].message.content = "Found something"
            mock_acompletion.return_value = mock_response

            engine = _make_engine(tool_registry=tr)
            result = await engine.run_inquiry("g_1", "Test")

            assert result.status == "completed"


class TestInvestigationConfidence:
    """Tests for _assess_investigation_confidence — evidence-quality-based scoring."""

    def test_empty_findings_zero_confidence(self):
        score = InquiryEngine._assess_investigation_confidence("", [], has_tools=True)
        assert score == 0.0

    def test_failure_findings_near_zero(self):
        score = InquiryEngine._assess_investigation_confidence(
            "Investigation failed due to tool error.", [], has_tools=True
        )
        assert score == 0.05

    def test_all_tool_errors_low_confidence(self):
        calls = [
            {"tool": "web_search", "result": "Error: timeout"},
            {"tool": "web_search", "result": "Error: connection refused"},
        ]
        score = InquiryEngine._assess_investigation_confidence(
            "Some findings", calls, has_tools=True
        )
        assert score == 0.1

    def test_mixed_tool_results_moderate_confidence(self):
        calls = [
            {"tool": "web_search", "result": "Error: timeout"},
            {"tool": "web_search", "result": "Found relevant data about markets"},
        ]
        score = InquiryEngine._assess_investigation_confidence(
            "Market data shows growth", calls, has_tools=True
        )
        # 1/2 success ratio → should be moderate, not high
        assert 0.5 < score < 0.8

    def test_all_tools_succeed_high_confidence(self):
        calls = [
            {"tool": "web_search", "result": "Strong evidence found"},
            {"tool": "code_exec", "result": "Analysis confirms hypothesis"},
        ]
        score = InquiryEngine._assess_investigation_confidence(
            "Comprehensive analysis confirms the finding with multiple sources",
            calls, has_tools=True,
        )
        # All successful → high confidence
        assert score > 0.8

    def test_more_failed_calls_lower_confidence_than_fewer(self):
        """5 failed + 1 success should NOT score higher than 1 success alone."""
        many_failures = [
            {"tool": "web_search", "result": "Error: fail"},
            {"tool": "web_search", "result": "Error: fail"},
            {"tool": "web_search", "result": "Error: fail"},
            {"tool": "web_search", "result": "Error: fail"},
            {"tool": "web_search", "result": "Error: fail"},
            {"tool": "web_search", "result": "Found something"},
        ]
        one_success = [
            {"tool": "web_search", "result": "Found something"},
        ]
        score_many = InquiryEngine._assess_investigation_confidence(
            "Some findings", many_failures, has_tools=True
        )
        score_one = InquiryEngine._assess_investigation_confidence(
            "Some findings", one_success, has_tools=True
        )
        # 1/6 success rate should score LOWER than 1/1 success rate
        assert score_many < score_one

    def test_direct_llm_answer_moderate(self):
        score = InquiryEngine._assess_investigation_confidence(
            "LLM provided a direct answer", [], has_tools=True
        )
        assert score == 0.5

    def test_no_tools_low_baseline(self):
        score = InquiryEngine._assess_investigation_confidence(
            "Best guess without tools", [], has_tools=False
        )
        assert score == 0.3


class TestPhaseSynthesize:
    @pytest.mark.asyncio
    async def test_synthesize_dialectic_flow(self):
        de = MagicMock()
        de.full_dialectic = AsyncMock(return_value=DialecticReport(
            original_conclusion="test", revised_confidence=0.75
        ))

        engine = _make_engine(dialectic_engine=de)
        await engine.run_inquiry("g_1", "Test")

        de.full_dialectic.assert_called()

    @pytest.mark.asyncio
    async def test_synthesize_confidence_can_decrease(self):
        """Dialectic counterarguments must be able to LOWER confidence, not just raise it."""
        call_count = 0

        async def _dialectic_with_decreasing_confidence(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return DialecticReport(
                    original_conclusion="test", revised_confidence=0.8
                )
            # Second call: strong counterargument lowers confidence
            return DialecticReport(
                original_conclusion="test", revised_confidence=0.3
            )

        de = MagicMock()
        de.full_dialectic = AsyncMock(side_effect=_dialectic_with_decreasing_confidence)

        qg = MagicMock()
        qg.generate = AsyncMock(return_value=[
            Question(text="Q1"), Question(text="Q2"), Question(text="Q3"),
        ])
        qg.prioritize = AsyncMock(side_effect=lambda g, qs: qs)

        engine = _make_engine(
            dialectic_engine=de,
            question_generator=qg,
            config=InquiryConfig(max_iterations=2, confidence_threshold=0.99),
        )
        result = await engine.run_inquiry("g_1", "Test")

        # After iteration 2, confidence should be 0.3 (decreased from 0.8),
        # NOT 0.8 (the old max() ratchet would have kept it at 0.8)
        assert result.iterations_completed == 2
        # Didn't early-terminate at 0.8 — confidence decreased, so threshold wasn't met
        assert result.termination_reason == "max_iterations"

    @pytest.mark.asyncio
    async def test_synthesize_generates_hypotheses_on_surprise(self):
        ep = MagicMock()
        ep.get_epistemic_state = MagicMock(return_value=None)
        ep.assess_uncertainty = AsyncMock(
            return_value=UncertaintyAssessment(finding_summary="test")
        )
        ep.detect_surprise = AsyncMock(return_value=SurpriseDetection(
            finding="Surprising finding",
            surprise_magnitude=0.8,
        ))
        ep.get_blind_spot_warning = MagicMock(return_value="")

        hm = MagicMock()
        hm.get_active_hypotheses = AsyncMock(return_value=[])
        hm.generate_hypotheses = AsyncMock(return_value=[
            Hypothesis(
                statement="H1",
                falsification_criteria=["Test criterion"],
            ),
        ])
        hm.create_falsification_questions = MagicMock(return_value=[
            Question(text="Falsification Q", question_type="falsification"),
        ])

        engine = _make_engine(
            epistemic_reasoner=ep,
            hypothesis_manager=hm,
        )
        await engine.run_inquiry("g_1", "Test")

        hm.generate_hypotheses.assert_called()


class TestPhaseReflect:
    @pytest.mark.asyncio
    async def test_reflect_drift_detection(self):
        cc = MagicMock()
        drift_report = MagicMock()
        drift_report.similarity = 0.3  # Low similarity = drift
        cc.detect_drift = MagicMock(return_value=drift_report)

        engine = _make_engine(context_curator=cc)
        await engine.run_inquiry("g_1", "Test")

        cc.detect_drift.assert_called()

    @pytest.mark.asyncio
    async def test_reflect_calls_persistence_on_drift(self):
        cc = MagicMock()
        drift_report = MagicMock()
        drift_report.similarity = 0.3  # Low similarity = drift
        cc.detect_drift = MagicMock(return_value=drift_report)

        pe = MagicMock()
        pe.analyze_root_cause = AsyncMock(
            return_value=RootCauseAnalysis(
                failure_summary="Drift detected",
                root_cause="Wrong approach",
            )
        )
        pe.reframe = AsyncMock(
            return_value=ReframingResult(
                original_framing="Test",
                reframing_strategy="inversion",
                reframed_question="What if we tried the opposite?",
                reasoning="Inverting the problem reveals new angles",
            )
        )
        pe.get_relevant_lessons = MagicMock(return_value=[])

        engine = _make_engine(context_curator=cc, persistence_engine=pe)
        await engine.run_inquiry("g_1", "Test")

        pe.analyze_root_cause.assert_called()

    @pytest.mark.asyncio
    async def test_reflect_adds_reframed_question(self):
        cc = MagicMock()
        drift_report = MagicMock()
        drift_report.similarity = 0.3
        cc.detect_drift = MagicMock(return_value=drift_report)

        pe = MagicMock()
        pe.analyze_root_cause = AsyncMock(
            return_value=RootCauseAnalysis(
                failure_summary="Drift",
                root_cause="Stuck",
            )
        )
        pe.reframe = AsyncMock(
            return_value=ReframingResult(
                original_framing="Test",
                reframing_strategy="inversion",
                reframed_question="What if we tried the opposite?",
                reasoning="Inverting reveals new angles",
            )
        )
        pe.get_relevant_lessons = MagicMock(return_value=[])

        engine = _make_engine(context_curator=cc, persistence_engine=pe)
        result = await engine.run_inquiry("g_1", "Test")

        # The reframed question should have been added
        reframed = [q for q in result.question_tree if q.text == "What if we tried the opposite?"]
        assert len(reframed) >= 1

    @pytest.mark.asyncio
    async def test_reflect_skips_persistence_when_on_track(self):
        cc = MagicMock()
        drift_report = MagicMock()
        drift_report.similarity = 0.8  # High similarity = on track
        cc.detect_drift = MagicMock(return_value=drift_report)

        pe = MagicMock()
        pe.analyze_root_cause = AsyncMock()
        pe.reframe = AsyncMock()
        pe.get_relevant_lessons = MagicMock(return_value=[])

        engine = _make_engine(context_curator=cc, persistence_engine=pe)
        await engine.run_inquiry("g_1", "Test")

        pe.analyze_root_cause.assert_not_called()


# ── Phase timings tests ────────────────────────────────────────────────────


class TestPhaseTimings:
    @pytest.mark.asyncio
    async def test_phase_timings_recorded(self):
        engine = _make_engine()
        result = await engine.run_inquiry("g_1", "Test")
        assert result.phase_timings

    @pytest.mark.asyncio
    async def test_all_seven_phases_timed(self):
        engine = _make_engine()
        result = await engine.run_inquiry("g_1", "Test")
        expected = {"observe", "orient", "question", "prioritize",
                    "investigate", "synthesize", "reflect"}
        assert expected == set(result.phase_timings.keys())

    @pytest.mark.asyncio
    async def test_phase_timing_values_positive(self):
        engine = _make_engine()
        result = await engine.run_inquiry("g_1", "Test")
        for phase, stats in result.phase_timings.items():
            assert stats["total_s"] > 0, f"{phase} total_s should be > 0"
            assert stats["avg_s"] > 0, f"{phase} avg_s should be > 0"
            assert stats["max_s"] > 0, f"{phase} max_s should be > 0"
            assert stats["count"] >= 1, f"{phase} count should be >= 1"

    @pytest.mark.asyncio
    async def test_multi_iteration_accumulates_timings(self):
        # Generate 3 questions per iteration so there are always pending ones
        qg = MagicMock()
        qg.generate = AsyncMock(return_value=[
            Question(text="Q1"), Question(text="Q2"), Question(text="Q3"),
        ])
        qg.prioritize = AsyncMock(side_effect=lambda g, qs: qs)

        # Low confidence to avoid early termination
        de = MagicMock()
        de.full_dialectic = AsyncMock(return_value=DialecticReport(
            original_conclusion="test", revised_confidence=0.1,
        ))

        engine = _make_engine(
            question_generator=qg,
            dialectic_engine=de,
            config=InquiryConfig(max_iterations=2, confidence_threshold=0.99),
        )
        result = await engine.run_inquiry("g_1", "Test")

        assert result.iterations_completed == 2
        for phase in ("observe", "orient", "question", "prioritize",
                      "investigate", "synthesize", "reflect"):
            assert result.phase_timings[phase]["count"] == 2.0


# ── Finalize tests ──────────────────────────────────────────────────────────


class TestFinalize:
    @pytest.mark.asyncio
    async def test_finalize_publishes_bus_event(self):
        bus = MagicMock()
        engine = _make_engine(bus=bus)
        await engine.run_inquiry("g_1", "Test")

        # Should have published inquiry.started, phase events, and inquiry.completed
        assert bus.publish.call_count > 0

    @pytest.mark.asyncio
    async def test_finalize_includes_summary(self):
        engine = _make_engine()
        result = await engine.run_inquiry("g_1", "Test goal")

        assert result.inquiry_id.startswith("inq_")
        assert result.goal_id == "g_1"
        assert result.duration_seconds >= 0


# ── Error handling ──────────────────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_exception_during_iteration(self):
        # Raise in question_generator.generate which is not wrapped
        # in a phase-level try/except
        qg = MagicMock()
        qg.generate = AsyncMock(side_effect=RuntimeError("boom"))

        engine = _make_engine(question_generator=qg)
        result = await engine.run_inquiry("g_1", "Test")

        assert result.status == "failed"

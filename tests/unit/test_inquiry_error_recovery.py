"""Tests for structured error recovery in InquiryEngine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.errors import InquiryPhaseError, LLMTokenLimitError
from qe.models.cognition import (
    ApproachAssessment,
    DialecticReport,
    UncertaintyAssessment,
)
from qe.services.inquiry.engine import InquiryEngine
from qe.services.inquiry.schemas import InquiryConfig, Question


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
        "config": InquiryConfig(max_iterations=1, confidence_threshold=0.8),
    }
    defaults.update(overrides)

    if "episodic_memory" not in overrides:
        em = defaults["episodic_memory"]
        em.recall_for_goal = AsyncMock(return_value=[])

    if "metacognitor" not in overrides:
        mc = defaults["metacognitor"]
        mc.suggest_next_approach = AsyncMock(
            return_value=ApproachAssessment(
                recommended_approach="test", reasoning="test",
            )
        )

    if "question_generator" not in overrides:
        qg = defaults["question_generator"]
        qg.generate = AsyncMock(return_value=[
            Question(text="Q1", expected_info_gain=0.8, relevance_to_goal=0.9, novelty_score=0.7),
        ])
        qg.prioritize = AsyncMock(
            side_effect=lambda goal, qs: sorted(
                qs, key=lambda q: q.expected_info_gain, reverse=True
            )
        )

    if "hypothesis_manager" not in overrides:
        hm = defaults["hypothesis_manager"]
        hm.get_active_hypotheses = AsyncMock(return_value=[])

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

    pm = MagicMock()
    pm.get_best_templates = AsyncMock(return_value=[])
    defaults.setdefault("procedural_memory", pm)

    return InquiryEngine(**defaults)


class TestInquiryErrorRecovery:
    @pytest.mark.asyncio
    async def test_transient_error_retried(self):
        """Timeout on first call, succeed on second via question_generator.generate.

        _phase_question does NOT internally catch errors, so
        _run_phase_with_retry handles the retry.
        """
        qg = MagicMock()
        call_count = 0

        async def _generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timeout on first call")
            return [
                Question(
                    text="Q1", expected_info_gain=0.8,
                    relevance_to_goal=0.9, novelty_score=0.7,
                ),
            ]

        qg.generate = AsyncMock(side_effect=_generate)
        qg.prioritize = AsyncMock(
            side_effect=lambda goal, qs: sorted(
                qs, key=lambda q: q.expected_info_gain, reverse=True
            )
        )

        engine = _make_engine(question_generator=qg)
        result = await engine.run_inquiry("g_1", "Test goal")

        assert result.status == "completed"
        assert call_count == 2  # first call failed, second succeeded

    @pytest.mark.asyncio
    async def test_permanent_error_fails_immediately(self):
        """Non-retryable error should not retry."""
        qg = MagicMock()
        call_count = 0

        async def _generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise LLMTokenLimitError("context length exceeded")

        qg.generate = AsyncMock(side_effect=_generate)
        qg.prioritize = AsyncMock(return_value=[])

        engine = _make_engine(question_generator=qg)
        result = await engine.run_inquiry("g_1", "Test goal")

        assert result.status == "failed"
        assert call_count == 1  # no retry for non-retryable error

    @pytest.mark.asyncio
    async def test_error_classified_in_result(self):
        """InquiryPhaseError (retryable) is retried then caught."""
        qg = MagicMock()
        qg.generate = AsyncMock(
            side_effect=InquiryPhaseError("phase failed")
        )
        qg.prioritize = AsyncMock(return_value=[])

        engine = _make_engine(question_generator=qg)
        result = await engine.run_inquiry("g_1", "Test goal")

        # Retryable: retried max_retries=2 times, then raised
        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Always fail — verify retry count matches max_retries + 1."""
        qg = MagicMock()
        call_count = 0

        async def _generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise TimeoutError("always timeout")

        qg.generate = AsyncMock(side_effect=_generate)
        qg.prioritize = AsyncMock(return_value=[])

        engine = _make_engine(question_generator=qg)
        result = await engine.run_inquiry("g_1", "Test goal")

        assert result.status == "failed"
        # max_retries=2, so 3 total attempts (1 original + 2 retries)
        assert call_count == 3

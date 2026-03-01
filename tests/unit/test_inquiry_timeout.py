"""Tests for inquiry timeout."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from qe.models.cognition import (
    ApproachAssessment,
    DialecticReport,
    UncertaintyAssessment,
)
from qe.services.inquiry.engine import InquiryEngine
from qe.services.inquiry.schemas import InquiryConfig, Question


def _make_engine(slow_phase=False, **overrides):
    """Create an InquiryEngine with mocked components.

    If slow_phase=True, question_generator.generate will sleep 1s.
    """
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
    }
    defaults.update(overrides)

    em = defaults["episodic_memory"]
    em.recall_for_goal = AsyncMock(return_value=[])

    mc = defaults["metacognitor"]
    mc.suggest_next_approach = AsyncMock(
        return_value=ApproachAssessment(
            recommended_approach="test", reasoning="test",
        )
    )

    qg = defaults["question_generator"]
    if slow_phase:
        async def _slow_generate(*args, **kwargs):
            await asyncio.sleep(1.0)
            return [
                Question(
                    text="Q1", expected_info_gain=0.8,
                    relevance_to_goal=0.9, novelty_score=0.7,
                ),
            ]
        qg.generate = AsyncMock(side_effect=_slow_generate)
    else:
        qg.generate = AsyncMock(return_value=[
            Question(text="Q1", expected_info_gain=0.8, relevance_to_goal=0.9, novelty_score=0.7),
        ])
    qg.prioritize = AsyncMock(
        side_effect=lambda goal, qs: sorted(qs, key=lambda q: q.expected_info_gain, reverse=True)
    )

    hm = defaults["hypothesis_manager"]
    hm.get_active_hypotheses = AsyncMock(return_value=[])

    ep = defaults["epistemic_reasoner"]
    ep.get_epistemic_state = MagicMock(return_value=None)
    ep.assess_uncertainty = AsyncMock(return_value=UncertaintyAssessment(finding_summary="test"))
    ep.detect_surprise = AsyncMock(return_value=None)
    ep.get_blind_spot_warning = MagicMock(return_value="")

    de = defaults["dialectic_engine"]
    de.full_dialectic = AsyncMock(return_value=DialecticReport(
        original_conclusion="test", revised_confidence=0.6
    ))

    ic = defaults["insight_crystallizer"]
    ic.crystallize = AsyncMock(return_value=None)

    cc = defaults["context_curator"]
    cc.detect_drift = MagicMock(return_value=None)

    pm = MagicMock()
    pm.get_best_templates = AsyncMock(return_value=[])
    defaults.setdefault("procedural_memory", pm)

    return InquiryEngine(**defaults)


class TestInquiryTimeout:
    @pytest.mark.asyncio
    async def test_inquiry_timeout_returns_partial(self):
        """Very short timeout with slow mock should return partial results."""
        cfg = InquiryConfig(
            max_iterations=10,
            confidence_threshold=0.99,
        )
        # Bypass validation to set a very short timeout for testing
        cfg.model_config["validate_assignment"] = False
        object.__setattr__(cfg, "inquiry_timeout_seconds", 0.05)
        engine = _make_engine(slow_phase=True, config=cfg)
        result = await engine.run_inquiry("g_1", "Slow goal")

        assert result.termination_reason == "timeout"
        # Should still return a valid result
        assert result.inquiry_id is not None
        assert result.goal_id == "g_1"

    @pytest.mark.asyncio
    async def test_inquiry_completes_within_timeout(self):
        """Generous timeout — normal completion."""
        cfg = InquiryConfig(
            max_iterations=1,
            confidence_threshold=0.8,
            inquiry_timeout_seconds=30.0,
        )
        engine = _make_engine(config=cfg)
        result = await engine.run_inquiry("g_1", "Normal goal")

        assert result.termination_reason != "timeout"
        assert result.status == "completed"

    def test_inquiry_timeout_config_validation(self):
        """Valid and invalid timeout values."""
        # Valid
        cfg = InquiryConfig(inquiry_timeout_seconds=10.0)
        assert cfg.inquiry_timeout_seconds == 10.0

        cfg = InquiryConfig(inquiry_timeout_seconds=3600.0)
        assert cfg.inquiry_timeout_seconds == 3600.0

        # Invalid — below min
        with pytest.raises(ValidationError):
            InquiryConfig(inquiry_timeout_seconds=5.0)

        # Invalid — above max
        with pytest.raises(ValidationError):
            InquiryConfig(inquiry_timeout_seconds=4000.0)

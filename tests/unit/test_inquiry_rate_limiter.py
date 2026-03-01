"""Tests for inquiry rate limiting."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.models.cognition import (
    ApproachAssessment,
    DialecticReport,
    UncertaintyAssessment,
)
from qe.services.inquiry.engine import InquiryEngine, _InquiryRateLimiter
from qe.services.inquiry.schemas import InquiryConfig, Question


def _make_engine(rate_limiter=None, **overrides):
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
        "rate_limiter": rate_limiter,
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

    pm = defaults.get("procedural_memory")
    if pm is None:
        pm = MagicMock()
        pm.get_best_templates = AsyncMock(return_value=[])
        defaults["procedural_memory"] = pm

    return InquiryEngine(**defaults)


class TestInquiryRateLimiter:
    def test_rate_limiter_allows_under_limit(self):
        """Rate limiter allows requests under the limit."""
        rl = _InquiryRateLimiter(max_concurrent=3, rpm=10)
        # Should allow 10 requests
        for _ in range(10):
            assert rl.try_acquire_rate() is True

    def test_rate_limiter_blocks_over_limit(self):
        """Rate limiter blocks when tokens exhausted."""
        rl = _InquiryRateLimiter(max_concurrent=3, rpm=5)
        # Exhaust all 5 tokens
        for _ in range(5):
            assert rl.try_acquire_rate() is True
        # 6th should fail
        assert rl.try_acquire_rate() is False

    def test_rate_limiter_refills_over_time(self):
        """Tokens refill over time."""
        import time

        rl = _InquiryRateLimiter(max_concurrent=3, rpm=60)  # 1 token/sec
        # Exhaust all tokens
        for _ in range(60):
            rl.try_acquire_rate()
        assert rl.try_acquire_rate() is False

        # Simulate time passing by manipulating _last_refill
        rl._last_refill = time.monotonic() - 2.0  # 2 seconds ago = 2 tokens
        assert rl.try_acquire_rate() is True

    @pytest.mark.asyncio
    async def test_concurrent_inquiry_semaphore(self):
        """Semaphore limits concurrent inquiries."""
        rl = _InquiryRateLimiter(max_concurrent=1, rpm=100)

        engine = _make_engine(rate_limiter=rl)

        # First inquiry should work
        result1 = await engine.run_inquiry("g_1", "Goal 1")
        assert result1.status == "completed"

        # Run two concurrent inquiries — one should complete at a time
        results = await asyncio.gather(
            engine.run_inquiry("g_2", "Goal 2"),
            engine.run_inquiry("g_3", "Goal 3"),
        )
        assert all(r.goal_id in ("g_2", "g_3") for r in results)

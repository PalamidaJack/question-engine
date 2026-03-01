"""Tests for Phase 3 wiring: bus topics, bus schemas, feature flag, engine accessibility,
self_correction Bayes factor, inference HypothesisTemplate."""

from __future__ import annotations

import pytest

from qe.bus.protocol import TOPICS
from qe.bus.schemas import (
    TOPIC_SCHEMAS,
    InquiryBudgetWarningPayload,
    InquiryCompletedPayload,
    InquiryFailedPayload,
    InquiryHypothesisGeneratedPayload,
    InquiryHypothesisUpdatedPayload,
    InquiryInsightGeneratedPayload,
    InquiryInvestigationCompletedPayload,
    InquiryPhaseCompletedPayload,
    InquiryQuestionGeneratedPayload,
    InquiryStartedPayload,
    validate_payload,
)
from qe.runtime.self_correction import SelfCorrectionEngine
from qe.substrate.inference import HypothesisTemplate

# ── Bus Topics ──────────────────────────────────────────────────────────────


INQUIRY_TOPICS = {
    "inquiry.started",
    "inquiry.phase_completed",
    "inquiry.question_generated",
    "inquiry.investigation_completed",
    "inquiry.hypothesis_generated",
    "inquiry.hypothesis_updated",
    "inquiry.insight_generated",
    "inquiry.completed",
    "inquiry.failed",
    "inquiry.budget_warning",
}


class TestInquiryBusTopics:
    def test_inquiry_topics_present(self):
        assert INQUIRY_TOPICS.issubset(TOPICS)

    def test_topic_count_includes_inquiry(self):
        # 89 pre-existing + 10 inquiry = 99
        assert len(TOPICS) >= 99


# ── Bus Schemas ─────────────────────────────────────────────────────────────


class TestInquiryBusSchemas:
    def test_inquiry_started_schema(self):
        payload = InquiryStartedPayload(
            inquiry_id="inq_1", goal_id="g_1", goal="Test"
        )
        assert payload.inquiry_id == "inq_1"

    def test_inquiry_phase_completed_schema(self):
        payload = InquiryPhaseCompletedPayload(
            inquiry_id="inq_1", goal_id="g_1", phase="observe", iteration=0
        )
        assert payload.phase == "observe"

    def test_inquiry_completed_schema(self):
        payload = InquiryCompletedPayload(
            inquiry_id="inq_1", goal_id="g_1", status="completed",
            iterations=5, insights=3, questions_answered=10,
        )
        assert payload.iterations == 5

    def test_question_generated_schema(self):
        payload = InquiryQuestionGeneratedPayload(
            inquiry_id="inq_1", question_id="q_1", text="What?"
        )
        assert payload.question_id == "q_1"

    def test_investigation_completed_schema(self):
        payload = InquiryInvestigationCompletedPayload(
            inquiry_id="inq_1", question_id="q_1"
        )
        assert payload.inquiry_id == "inq_1"

    def test_hypothesis_generated_schema(self):
        payload = InquiryHypothesisGeneratedPayload(
            inquiry_id="inq_1", hypothesis_id="hyp_1", statement="X causes Y"
        )
        assert payload.statement == "X causes Y"

    def test_hypothesis_updated_schema(self):
        payload = InquiryHypothesisUpdatedPayload(
            inquiry_id="inq_1", hypothesis_id="hyp_1", probability=0.8
        )
        assert payload.probability == 0.8

    def test_insight_generated_schema(self):
        payload = InquiryInsightGeneratedPayload(
            inquiry_id="inq_1", insight_id="ins_1", headline="Key finding"
        )
        assert payload.headline == "Key finding"

    def test_failed_schema(self):
        payload = InquiryFailedPayload(inquiry_id="inq_1", iteration=3)
        assert payload.iteration == 3

    def test_budget_warning_schema(self):
        payload = InquiryBudgetWarningPayload(inquiry_id="inq_1", iteration=5)
        assert payload.iteration == 5

    def test_all_inquiry_schemas_registered(self):
        for topic in INQUIRY_TOPICS:
            assert topic in TOPIC_SCHEMAS, f"Missing schema for {topic}"

    def test_validate_payload(self):
        result = validate_payload("inquiry.started", {
            "inquiry_id": "inq_1", "goal_id": "g_1", "goal": "Test"
        })
        assert result is not None
        assert isinstance(result, InquiryStartedPayload)


# ── Self-Correction Bayes Factor ────────────────────────────────────────────


class TestSelfCorrectionBayesFactor:
    @pytest.mark.asyncio
    async def test_bayes_factor_supersedes(self):
        """BF > 10 (log10 > 1.0) → superseded."""
        engine = SelfCorrectionEngine()
        claim = {"claim_id": "c1", "confidence": 0.05}
        challenge = {"claim_id": "c2", "confidence": 0.95}

        result = await engine.evaluate_with_bayes_factor(claim, challenge)
        assert result.action == "superseded"
        assert result.new_confidence == 0.95

    @pytest.mark.asyncio
    async def test_bayes_factor_needs_investigation(self):
        """BF between 3 and 10 → needs_investigation."""
        engine = SelfCorrectionEngine()
        claim = {"claim_id": "c1", "confidence": 0.2}
        challenge = {"claim_id": "c2", "confidence": 0.8}

        result = await engine.evaluate_with_bayes_factor(claim, challenge)
        # BF = 0.8/0.2 = 4.0, log10(4) ≈ 0.60 > 0.48
        assert result.action == "needs_investigation"

    @pytest.mark.asyncio
    async def test_bayes_factor_reinforces(self):
        """BF < 3 → reinforced."""
        engine = SelfCorrectionEngine()
        claim = {"claim_id": "c1", "confidence": 0.6}
        challenge = {"claim_id": "c2", "confidence": 0.7}

        result = await engine.evaluate_with_bayes_factor(claim, challenge)
        # BF = 0.7/0.6 ≈ 1.17, log10 ≈ 0.07 < 0.48
        assert result.action == "reinforced"


# ── Inference HypothesisTemplate ────────────────────────────────────────────


class TestHypothesisTemplate:
    def test_confirmed_hypothesis_boosts(self):
        hyps = [{"hypothesis_id": "hyp_1", "current_probability": 0.96, "status": "confirmed"}]
        template = HypothesisTemplate(hypotheses=hyps)

        claims = [{
            "claim_id": "c1",
            "subject_entity_id": "company_x",
            "predicate": "revenue_trend",
            "object_value": "growing",
            "confidence": 0.7,
            "metadata": {"hypothesis_id": "hyp_1"},
        }]

        inferred = template.match(claims)
        assert len(inferred) == 1
        assert inferred[0].inference_type == "hypothesis_confirmed"
        assert inferred[0].confidence > 0.7  # Boosted

    def test_falsified_hypothesis_weakens(self):
        hyps = [{"hypothesis_id": "hyp_2", "current_probability": 0.03, "status": "falsified"}]
        template = HypothesisTemplate(hypotheses=hyps)

        claims = [{
            "claim_id": "c2",
            "subject_entity_id": "company_y",
            "predicate": "market_share",
            "object_value": "increasing",
            "confidence": 0.8,
            "metadata": {"hypothesis_id": "hyp_2"},
        }]

        inferred = template.match(claims)
        assert len(inferred) == 1
        assert inferred[0].inference_type == "hypothesis_falsified"
        assert inferred[0].confidence < 0.8  # Weakened

    def test_active_hypothesis_no_inference(self):
        hyps = [{"hypothesis_id": "hyp_3", "current_probability": 0.5, "status": "active"}]
        template = HypothesisTemplate(hypotheses=hyps)

        claims = [{
            "claim_id": "c3",
            "subject_entity_id": "x",
            "predicate": "p",
            "object_value": "v",
            "confidence": 0.5,
            "metadata": {"hypothesis_id": "hyp_3"},
        }]

        inferred = template.match(claims)
        assert len(inferred) == 0  # No inference for active hypothesis

    def test_no_hypothesis_metadata(self):
        hyps = [{"hypothesis_id": "hyp_1", "current_probability": 0.96, "status": "confirmed"}]
        template = HypothesisTemplate(hypotheses=hyps)

        claims = [{
            "claim_id": "c1",
            "subject_entity_id": "x",
            "predicate": "p",
            "object_value": "v",
            "confidence": 0.5,
            "metadata": {},
        }]

        inferred = template.match(claims)
        assert len(inferred) == 0

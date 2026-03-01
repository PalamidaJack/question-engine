"""Tests for Inquiry Loop schemas — model construction, validation, serialization."""

from __future__ import annotations

from qe.services.inquiry.schemas import (
    InquiryConfig,
    InquiryResult,
    InquiryState,
    InvestigationResult,
    Question,
    QuestionPriority,
    Reflection,
)


class TestQuestion:
    def test_defaults(self):
        q = Question(text="What is X?")
        assert q.text == "What is X?"
        assert q.question_type == "factual"
        assert q.status == "pending"
        assert q.question_id.startswith("q_")
        assert q.parent_id is None
        assert q.confidence_in_answer == 0.0

    def test_full_construction(self):
        q = Question(
            text="Why does Y cause Z?",
            question_type="causal",
            expected_info_gain=0.9,
            relevance_to_goal=0.8,
            novelty_score=0.7,
            status="answered",
            answer="Because of W.",
            evidence=["source1", "source2"],
            confidence_in_answer=0.85,
            hypothesis_id="hyp_abc",
            iteration_generated=2,
        )
        assert q.question_type == "causal"
        assert q.confidence_in_answer == 0.85
        assert len(q.evidence) == 2


class TestQuestionPriority:
    def test_construction(self):
        qp = QuestionPriority(
            question_id="q_123", priority_score=0.75, reasoning="High relevance"
        )
        assert qp.priority_score == 0.75


class TestInvestigationResult:
    def test_defaults(self):
        ir = InvestigationResult(question_id="q_abc")
        assert ir.question_id == "q_abc"
        assert ir.tool_calls == []
        assert ir.tokens_used == 0
        assert ir.cost_usd == 0.0


class TestReflection:
    def test_defaults(self):
        r = Reflection(iteration=0)
        assert r.on_track is True
        assert r.decision == "continue"

    def test_terminate(self):
        r = Reflection(iteration=5, decision="terminate", reasoning="Confidence met")
        assert r.decision == "terminate"


class TestInquiryConfig:
    def test_defaults(self):
        cfg = InquiryConfig()
        assert cfg.max_iterations == 10
        assert cfg.confidence_threshold == 0.8
        assert cfg.budget_hard_stop_pct == 0.05
        assert cfg.questions_per_iteration == 3


class TestInquiryState:
    def test_defaults(self):
        state = InquiryState()
        assert state.inquiry_id.startswith("inq_")
        assert state.current_phase == "observe"
        assert state.overall_confidence == 0.0

    def test_serialization_roundtrip(self):
        state = InquiryState(
            goal_id="g_123",
            goal_description="Test goal",
            current_iteration=3,
        )
        state.questions.append(Question(text="Q1"))
        data = state.model_dump()
        restored = InquiryState.model_validate(data)
        assert restored.goal_id == "g_123"
        assert len(restored.questions) == 1
        assert restored.questions[0].text == "Q1"


class TestInquiryResult:
    def test_construction(self):
        result = InquiryResult(
            inquiry_id="inq_abc",
            goal_id="g_123",
            status="completed",
            termination_reason="confidence_met",
            iterations_completed=5,
            total_questions_generated=15,
            total_questions_answered=12,
            findings_summary="Summary here.",
            duration_seconds=45.2,
        )
        assert result.status == "completed"
        assert result.termination_reason == "confidence_met"
        assert result.iterations_completed == 5

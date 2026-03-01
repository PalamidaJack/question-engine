"""Tests for QuestionGenerator — mock LLM, verify prompts, priority scoring."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.services.inquiry.question_generator import (
    GeneratedQuestion,
    GeneratedQuestions,
    QuestionGenerator,
)
from qe.services.inquiry.schemas import Question


@pytest.fixture
def mock_llm():
    """Mock the instructor client for QuestionGenerator."""
    with patch("qe.services.inquiry.question_generator.instructor") as mock_inst:
        mock_client = MagicMock()
        mock_inst.from_litellm.return_value = mock_client
        yield mock_client


class TestQuestionGeneratorGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_questions(self, mock_llm):
        mock_llm.chat.completions.create = AsyncMock(
            return_value=GeneratedQuestions(
                questions=[
                    GeneratedQuestion(
                        text="What is revenue?",
                        question_type="factual",
                        expected_info_gain=0.8,
                        relevance_to_goal=0.9,
                        novelty_score=0.7,
                    ),
                    GeneratedQuestion(
                        text="Why did profits decline?",
                        question_type="causal",
                        expected_info_gain=0.7,
                        relevance_to_goal=0.8,
                        novelty_score=0.6,
                    ),
                ]
            )
        )

        gen = QuestionGenerator(model="test-model")
        questions = await gen.generate(
            goal="Analyze company finances",
            n_questions=2,
        )

        assert len(questions) == 2
        assert questions[0].text == "What is revenue?"
        assert questions[0].question_type == "factual"
        assert questions[1].question_type == "causal"

    @pytest.mark.asyncio
    async def test_generate_with_findings(self, mock_llm):
        mock_llm.chat.completions.create = AsyncMock(
            return_value=GeneratedQuestions(
                questions=[
                    GeneratedQuestion(text="Follow-up question", question_type="factual"),
                ]
            )
        )

        gen = QuestionGenerator(model="test-model")
        questions = await gen.generate(
            goal="Analyze X",
            findings_summary="Found A and B",
            asked_questions=["Previous question?"],
            iteration=2,
            max_iterations=10,
        )

        assert len(questions) == 1
        # Verify the prompt includes findings
        call_args = mock_llm.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "Found A and B" in user_msg
        assert "Previous question?" in user_msg

    @pytest.mark.asyncio
    async def test_generate_with_hypotheses(self, mock_llm):
        mock_llm.chat.completions.create = AsyncMock(
            return_value=GeneratedQuestions(
                questions=[
                    GeneratedQuestion(text="Falsification Q", question_type="falsification"),
                ]
            )
        )

        gen = QuestionGenerator(model="test-model")
        questions = await gen.generate(
            goal="Test hypotheses",
            hypotheses_summary="- H1: X is true (p=0.7)",
        )

        assert questions[0].question_type == "falsification"

    @pytest.mark.asyncio
    async def test_generate_invalid_type_defaults_to_factual(self, mock_llm):
        mock_llm.chat.completions.create = AsyncMock(
            return_value=GeneratedQuestions(
                questions=[
                    GeneratedQuestion(text="Q1", question_type="invalid_type"),
                ]
            )
        )

        gen = QuestionGenerator(model="test-model")
        questions = await gen.generate(goal="Test")
        assert questions[0].question_type == "factual"

    @pytest.mark.asyncio
    async def test_generate_sets_iteration(self, mock_llm):
        mock_llm.chat.completions.create = AsyncMock(
            return_value=GeneratedQuestions(
                questions=[GeneratedQuestion(text="Q1")]
            )
        )

        gen = QuestionGenerator(model="test-model")
        questions = await gen.generate(goal="Test", iteration=5)
        assert questions[0].iteration_generated == 5


class TestQuestionGeneratorPrioritize:
    @pytest.mark.asyncio
    async def test_prioritize_sorts_by_score(self, mock_llm):
        gen = QuestionGenerator(model="test-model")
        questions = [
            Question(text="Low", expected_info_gain=0.1, relevance_to_goal=0.1, novelty_score=0.1),
            Question(text="High", expected_info_gain=0.9, relevance_to_goal=0.9, novelty_score=0.9),
            Question(text="Mid", expected_info_gain=0.5, relevance_to_goal=0.5, novelty_score=0.5),
        ]

        sorted_qs = await gen.prioritize("Goal", questions)
        assert sorted_qs[0].text == "High"
        assert sorted_qs[-1].text == "Low"


class TestComputePriorityScore:
    def test_weights(self):
        q = Question(
            text="Test",
            expected_info_gain=1.0,
            relevance_to_goal=0.0,
            novelty_score=0.0,
        )
        score = QuestionGenerator.compute_priority_score(q)
        assert score == pytest.approx(0.4)

    def test_all_ones(self):
        q = Question(
            text="Test",
            expected_info_gain=1.0,
            relevance_to_goal=1.0,
            novelty_score=1.0,
        )
        score = QuestionGenerator.compute_priority_score(q)
        assert score == pytest.approx(1.0)

    def test_all_zeros(self):
        q = Question(
            text="Test",
            expected_info_gain=0.0,
            relevance_to_goal=0.0,
            novelty_score=0.0,
        )
        score = QuestionGenerator.compute_priority_score(q)
        assert score == pytest.approx(0.0)

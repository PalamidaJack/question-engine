"""Tests for HypothesisManager — mock LLM, belief_store delegation, Bayes factor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.services.inquiry.hypothesis import (
    GeneratedHypotheses,
    HypothesisManager,
    HypothesisSpec,
)
from qe.substrate.bayesian_belief import EvidenceRecord, Hypothesis


@pytest.fixture
def mock_llm():
    """Mock the instructor client for HypothesisManager."""
    with patch("qe.services.inquiry.hypothesis.instructor") as mock_inst:
        mock_client = MagicMock()
        mock_inst.from_litellm.return_value = mock_client
        yield mock_client


class TestGenerateHypotheses:
    @pytest.mark.asyncio
    async def test_generate_returns_hypotheses(self, mock_llm):
        mock_llm.chat.completions.create = AsyncMock(
            return_value=GeneratedHypotheses(
                hypotheses=[
                    HypothesisSpec(
                        statement="X causes Y",
                        falsification_criteria=["Find case where X but not Y"],
                        prior_probability=0.6,
                    ),
                    HypothesisSpec(
                        statement="Z causes Y instead",
                        falsification_criteria=["Find case where Z but not Y"],
                        prior_probability=0.4,
                    ),
                ]
            )
        )

        mgr = HypothesisManager(belief_store=None, model="test-model")
        hyps = await mgr.generate_hypotheses(
            goal="Understand Y",
            contradictions=["X correlates with Y but so does Z"],
        )

        assert len(hyps) == 2
        assert hyps[0].statement == "X causes Y"
        assert hyps[0].prior_probability == 0.6
        assert hyps[0].status == "active"

    @pytest.mark.asyncio
    async def test_generate_with_belief_store(self, mock_llm):
        mock_llm.chat.completions.create = AsyncMock(
            return_value=GeneratedHypotheses(
                hypotheses=[
                    HypothesisSpec(statement="H1", prior_probability=0.5),
                ]
            )
        )

        mock_store = MagicMock()
        mock_store.store_hypothesis = AsyncMock(
            side_effect=lambda h: h  # Return as-is
        )

        mgr = HypothesisManager(belief_store=mock_store, model="test-model")
        hyps = await mgr.generate_hypotheses(goal="Test")

        assert len(hyps) == 1
        mock_store.store_hypothesis.assert_called_once()


class TestCreateFalsificationQuestions:
    def test_creates_questions_from_criteria(self):
        h = Hypothesis(
            statement="X causes Y",
            falsification_criteria=["A is false", "B is false"],
        )

        mgr = HypothesisManager()
        questions = mgr.create_falsification_questions(h)

        assert len(questions) == 2
        assert all(q.question_type == "falsification" for q in questions)
        assert all(q.hypothesis_id == h.hypothesis_id for q in questions)
        assert "A is false" in questions[0].text

    def test_empty_criteria(self):
        h = Hypothesis(statement="No criteria")
        mgr = HypothesisManager()
        questions = mgr.create_falsification_questions(h)
        assert questions == []


class TestUpdateWithEvidence:
    @pytest.mark.asyncio
    async def test_update_via_belief_store(self, mock_llm):
        updated_hyp = Hypothesis(
            hypothesis_id="hyp_1",
            statement="H1",
            current_probability=0.8,
        )
        mock_store = MagicMock()
        mock_store.update_hypothesis = AsyncMock(return_value=updated_hyp)

        mgr = HypothesisManager(belief_store=mock_store, model="test-model")
        evidence = EvidenceRecord(source="test", supports=True, strength=0.7)
        result = await mgr.update_with_evidence("hyp_1", evidence)

        assert result.current_probability == 0.8
        mock_store.update_hypothesis.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_local_fallback(self, mock_llm):
        mgr = HypothesisManager(belief_store=None, model="test-model")

        # First generate a local hypothesis
        mock_llm.chat.completions.create = AsyncMock(
            return_value=GeneratedHypotheses(
                hypotheses=[HypothesisSpec(statement="H1", prior_probability=0.5)]
            )
        )
        hyps = await mgr.generate_hypotheses(goal="Test")
        hyp_id = hyps[0].hypothesis_id

        # Update with supporting evidence
        evidence = EvidenceRecord(source="test", supports=True, strength=0.8)
        result = await mgr.update_with_evidence(hyp_id, evidence)

        assert result.current_probability > 0.5

    @pytest.mark.asyncio
    async def test_update_local_not_found_raises(self, mock_llm):
        mgr = HypothesisManager(belief_store=None, model="test-model")

        evidence = EvidenceRecord(source="test", supports=True, strength=0.5)
        with pytest.raises(ValueError, match="not found"):
            await mgr.update_with_evidence("nonexistent", evidence)


class TestGetActiveHypotheses:
    @pytest.mark.asyncio
    async def test_get_active_via_belief_store(self, mock_llm):
        mock_store = MagicMock()
        mock_store.get_active_hypotheses = AsyncMock(return_value=[
            Hypothesis(statement="Active H"),
        ])

        mgr = HypothesisManager(belief_store=mock_store, model="test-model")
        active = await mgr.get_active_hypotheses()
        assert len(active) == 1


class TestBayesFactor:
    def test_strong_evidence(self):
        h_a = Hypothesis(statement="A", current_probability=0.95)
        h_b = Hypothesis(statement="B", current_probability=0.05)
        bf = HypothesisManager.compute_bayes_factor(h_a, h_b)
        assert bf == pytest.approx(19.0)  # 0.95/0.05

    def test_inconclusive(self):
        h_a = Hypothesis(statement="A", current_probability=0.5)
        h_b = Hypothesis(statement="B", current_probability=0.5)
        bf = HypothesisManager.compute_bayes_factor(h_a, h_b)
        assert bf == pytest.approx(1.0)

    def test_evidence_for_b(self):
        h_a = Hypothesis(statement="A", current_probability=0.1)
        h_b = Hypothesis(statement="B", current_probability=0.9)
        bf = HypothesisManager.compute_bayes_factor(h_a, h_b)
        assert bf < 1.0  # Evidence for B

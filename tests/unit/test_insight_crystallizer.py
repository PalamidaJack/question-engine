"""Tests for InsightCrystallizer — novelty gate + mechanism extraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.cognition import (
    ActionabilityResult,
    MechanismExplanation,
    NoveltyAssessment,
)
from qe.services.inquiry.insight import InsightCrystallizer


@pytest.fixture
def crystallizer():
    return InsightCrystallizer(model="test-model")


# ── Novelty Assessment (Mocked LLM) ─────────────────────────────────────


class TestNoveltyAssessment:
    async def test_novel_finding(self, crystallizer):
        mock_novelty = NoveltyAssessment(
            finding="Grid infrastructure is mispriced",
            is_novel=True,
            novelty_type="new_connection",
            why_novel="Links sector classification to pricing",
            who_would_find_this_surprising="Portfolio managers",
        )

        mock_create = AsyncMock(return_value=mock_novelty)
        with patch("qe.services.inquiry.insight.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await crystallizer.assess_novelty(
                "Grid infrastructure is mispriced", "financial"
            )

        assert result.is_novel is True
        assert result.novelty_type == "new_connection"

    async def test_not_novel_finding(self, crystallizer):
        mock_novelty = NoveltyAssessment(
            finding="Tech stocks are volatile",
            is_novel=False,
            novelty_type="not_novel",
        )

        mock_create = AsyncMock(return_value=mock_novelty)
        with patch("qe.services.inquiry.insight.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await crystallizer.assess_novelty(
                "Tech stocks are volatile"
            )

        assert result.is_novel is False

    async def test_novelty_with_belief_store(self, crystallizer):
        mock_belief_store = AsyncMock()
        mock_belief_store.get_beliefs_for_entity = AsyncMock(return_value=[
            MagicMock(
                claim=MagicMock(predicate="growth", object_value="20%"),
                posterior=0.8,
            ),
        ])
        crystallizer._belief_store = mock_belief_store

        mock_novelty = NoveltyAssessment(
            finding="test", is_novel=True, novelty_type="contradicts_consensus",
        )

        mock_create = AsyncMock(return_value=mock_novelty)
        with patch("qe.services.inquiry.insight.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            await crystallizer.assess_novelty(
                "Revenue declined", entity_id="company_x"
            )

        # Should have queried belief store
        mock_belief_store.get_beliefs_for_entity.assert_called_once_with("company_x")


# ── Mechanism Extraction (Mocked LLM) ────────────────────────────────────


class TestMechanismExtraction:
    async def test_extract_mechanism(self, crystallizer):
        mock_mechanism = MechanismExplanation(
            what_happens="Grid stocks underperform indices",
            why_it_happens="Sector misclassification",
            how_it_works="ETF classification lumps grid with solar",
            key_causal_links=["Classification → allocation → pricing"],
            confidence_in_mechanism=0.7,
        )

        mock_create = AsyncMock(return_value=mock_mechanism)
        with patch("qe.services.inquiry.insight.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await crystallizer.extract_mechanism(
                "Grid infrastructure is mispriced"
            )

        assert result.confidence_in_mechanism == 0.7
        assert len(result.key_causal_links) == 1


# ── Actionability Scoring (Mocked LLM) ──────────────────────────────────


class TestActionabilityScoring:
    async def test_score_actionability(self, crystallizer):
        mock_action = ActionabilityResult(
            score=0.8,
            who_can_act="Portfolio managers",
            what_action="Overweight grid ETFs",
            time_horizon="weeks",
        )

        mock_create = AsyncMock(return_value=mock_action)
        with patch("qe.services.inquiry.insight.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await crystallizer.score_actionability(
                "Grid is mispriced", "Sector classification issue"
            )

        assert result.score == 0.8
        assert result.who_can_act == "Portfolio managers"


# ── Cross-Domain Connections (Mocked LLM) ────────────────────────────────


class TestCrossDomain:
    async def test_find_connections(self, crystallizer):
        class MockResult:
            connections = [
                "Healthcare has similar classification issues",
                "Real estate sector had analogous repricing in 2015",
            ]

        mock_create = AsyncMock(return_value=MockResult())
        with patch("qe.services.inquiry.insight.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await crystallizer.find_cross_domain_connections(
                "Grid mispriced due to classification"
            )

        assert len(result) == 2


# ── Full Crystallization Pipeline ────────────────────────────────────────


class TestCrystallize:
    async def test_novel_finding_crystallizes(self, crystallizer):
        """Novel finding should produce a CrystallizedInsight."""
        call_idx = 0

        async def mock_create(**kwargs):
            nonlocal call_idx
            call_idx += 1
            rm = kwargs.get("response_model")

            if rm == NoveltyAssessment:
                return NoveltyAssessment(
                    finding="Grid mispriced",
                    is_novel=True,
                    novelty_type="new_connection",
                )
            if rm == MechanismExplanation:
                return MechanismExplanation(
                    what_happens="Underperformance",
                    why_it_happens="Misclassification",
                    how_it_works="ETF grouping",
                    confidence_in_mechanism=0.7,
                )
            if rm == ActionabilityResult:
                return ActionabilityResult(
                    score=0.8,
                    who_can_act="PMs",
                    what_action="Overweight",
                    time_horizon="weeks",
                )
            # Cross-domain connections
            return MagicMock(connections=["Healthcare analogy"])

        with patch("qe.services.inquiry.insight.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=mock_create
            )
            mock_inst.from_litellm.return_value = mock_client

            result = await crystallizer.crystallize(
                goal_id="g1",
                finding="Grid infrastructure is mispriced",
                evidence="ETF data analysis",
                original_question="Find investment opportunities",
                dialectic_survived=True,
            )

        assert result is not None
        assert result.headline == "Grid infrastructure is mispriced"
        assert result.novelty.is_novel is True
        assert result.mechanism.confidence_in_mechanism == 0.7
        assert result.actionability_score == 0.8
        assert result.dialectic_survivor is True
        assert len(result.cross_domain_connections) == 1
        assert result.provenance.original_question == "Find investment opportunities"

    async def test_not_novel_returns_none(self, crystallizer):
        """Non-novel finding should be rejected (return None)."""
        mock_novelty = NoveltyAssessment(
            finding="Tech is volatile",
            is_novel=False,
            novelty_type="not_novel",
        )

        mock_create = AsyncMock(return_value=mock_novelty)
        with patch("qe.services.inquiry.insight.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await crystallizer.crystallize(
                goal_id="g1",
                finding="Tech stocks are volatile",
            )

        assert result is None

    async def test_long_headline_truncated(self, crystallizer):
        long_finding = "A" * 200

        async def mock_create(**kwargs):
            rm = kwargs.get("response_model")
            if rm == NoveltyAssessment:
                return NoveltyAssessment(
                    finding=long_finding, is_novel=True,
                    novelty_type="new_connection",
                )
            if rm == MechanismExplanation:
                return MechanismExplanation(
                    what_happens="X", why_it_happens="Y",
                    how_it_works="Z", confidence_in_mechanism=0.5,
                )
            if rm == ActionabilityResult:
                return ActionabilityResult(score=0.5)
            return MagicMock(connections=[])

        with patch("qe.services.inquiry.insight.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=mock_create
            )
            mock_inst.from_litellm.return_value = mock_client

            result = await crystallizer.crystallize(
                goal_id="g1", finding=long_finding
            )

        assert result is not None
        assert len(result.headline) == 120
        assert result.headline.endswith("...")


# ── Status ───────────────────────────────────────────────────────────────


class TestStatus:
    def test_status(self, crystallizer):
        s = crystallizer.status()
        assert s["model"] == "test-model"

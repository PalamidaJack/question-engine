"""Tests for EpistemicReasoner — what we know vs. don't know."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.cognition import (
    AbsenceDetection,
    SurpriseDetection,
    UncertaintyAssessment,
)
from qe.runtime.epistemic_reasoner import EpistemicReasoner


@pytest.fixture
def reasoner():
    return EpistemicReasoner(model="test-model")


# ── State Management ─────────────────────────────────────────────────────


class TestStateManagement:
    def test_create_state(self, reasoner):
        state = reasoner.get_or_create_state("g1")
        assert state.goal_id == "g1"
        assert state.known_facts == []

    def test_get_existing_state(self, reasoner):
        s1 = reasoner.get_or_create_state("g1")
        s1.overall_confidence = "high"
        s2 = reasoner.get_or_create_state("g1")
        assert s2.overall_confidence == "high"

    def test_register_unknown(self, reasoner):
        reasoner.get_or_create_state("g1")
        ku = reasoner.register_unknown(
            "g1", "What is the debt ratio?", "No financial access"
        )
        assert ku.question == "What is the debt ratio?"
        state = reasoner.get_epistemic_state("g1")
        assert len(state.known_unknowns) == 1

    def test_register_unknown_creates_state(self, reasoner):
        ku = reasoner.register_unknown("g_new", "question?", "reason")
        assert ku.question == "question?"

    def test_resolve_unknown(self, reasoner):
        ku = reasoner.register_unknown("g1", "question?", "reason")
        reasoner.resolve_unknown("g1", ku.unknown_id)
        state = reasoner.get_epistemic_state("g1")
        assert len(state.known_unknowns) == 0

    def test_resolve_unknown_nonexistent_goal(self, reasoner):
        # Should not raise
        reasoner.resolve_unknown("nonexistent", "unk_123")

    def test_clear_goal(self, reasoner):
        reasoner.register_unknown("g1", "q?", "r")
        reasoner.clear_goal("g1")
        assert "g1" not in reasoner._states

    def test_clear_nonexistent_goal(self, reasoner):
        reasoner.clear_goal("nonexistent")  # should not raise


# ── Blind Spot Warnings ──────────────────────────────────────────────────


class TestBlindSpotWarnings:
    def test_no_unknowns_warning(self, reasoner):
        reasoner.get_or_create_state("g1")
        warning = reasoner.get_blind_spot_warning("g1")
        assert "No known unknowns" in warning

    def test_no_absences_warning(self, reasoner):
        reasoner.get_or_create_state("g1")
        warning = reasoner.get_blind_spot_warning("g1")
        assert "No absences" in warning

    def test_all_high_confidence_warning(self, reasoner):
        state = reasoner.get_or_create_state("g1")
        state.known_facts.append(UncertaintyAssessment(
            finding_summary="Finding 1", confidence_level="high",
        ))
        state.known_facts.append(UncertaintyAssessment(
            finding_summary="Finding 2", confidence_level="very_high",
        ))
        warning = reasoner.get_blind_spot_warning("g1")
        assert "confirmation bias" in warning

    def test_no_warning_when_healthy(self, reasoner):
        state = reasoner.get_or_create_state("g1")
        state.known_unknowns.append(
            reasoner.register_unknown("g1", "q?", "r")
        )
        state.absences.append(AbsenceDetection(
            expected_data="data", why_expected="expected",
        ))
        state.known_facts.append(UncertaintyAssessment(
            finding_summary="F1", confidence_level="moderate",
        ))
        warning = reasoner.get_blind_spot_warning("g1")
        assert warning == ""


# ── Absence Detection (Mocked LLM) ──────────────────────────────────────


class TestAbsenceDetection:
    async def test_empty_results_auto_flags(self, reasoner):
        """Empty results should create an automatic absence."""
        mock_absences = [
            AbsenceDetection(
                expected_data="Market cap data",
                why_expected="Required for valuation",
                significance="high",
            )
        ]

        class MockResult:
            absences = mock_absences

        mock_create = AsyncMock(return_value=MockResult())
        with patch("qe.runtime.epistemic_reasoner.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            await reasoner.detect_absences(
                "g1", "Find valuations", "Search for market cap", []
            )

        # Both auto-generated and LLM-generated absences
        state = reasoner.get_epistemic_state("g1")
        assert len(state.absences) >= 2  # 1 auto + 1 from LLM

    async def test_non_empty_results(self, reasoner):
        """Non-empty results should only have LLM absences."""
        mock_create = AsyncMock(return_value=MagicMock(absences=[]))
        with patch("qe.runtime.epistemic_reasoner.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            await reasoner.detect_absences(
                "g1", "Find data", "Search", [{"data": "found"}]
            )

        state = reasoner.get_epistemic_state("g1")
        assert len(state.absences) == 0  # No auto, no LLM results


# ── Uncertainty Assessment (Mocked LLM) ──────────────────────────────────


class TestUncertaintyAssessment:
    async def test_assess_adds_to_state(self, reasoner):
        mock_assessment = UncertaintyAssessment(
            finding_summary="Revenue grew 20%",
            confidence_level="moderate",
            evidence_quality="secondary",
            potential_biases=["survivorship bias"],
        )

        mock_create = AsyncMock(return_value=mock_assessment)
        with patch("qe.runtime.epistemic_reasoner.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await reasoner.assess_uncertainty(
                "g1", "Revenue grew 20%"
            )

        assert result.confidence_level == "moderate"
        state = reasoner.get_epistemic_state("g1")
        assert len(state.known_facts) == 1


# ── Surprise Detection (Mocked LLM) ──────────────────────────────────────


class TestSurpriseDetection:
    async def test_no_belief_store(self, reasoner):
        """Without belief store, should return None."""
        result = await reasoner.detect_surprise("g1", "entity_x", "finding")
        assert result is None

    async def test_surprise_above_threshold(self, reasoner):
        mock_belief_store = AsyncMock()
        mock_belief_store.get_beliefs_for_entity = AsyncMock(return_value=[
            MagicMock(
                claim=MagicMock(
                    claim_id="c1",
                    predicate="revenue",
                    object_value="growth",
                ),
                posterior=0.8,
            ),
        ])
        reasoner._belief_store = mock_belief_store

        mock_surprise = SurpriseDetection(
            finding="Revenue dropped",
            expected_instead="Revenue growth",
            surprise_magnitude=0.8,
        )

        mock_create = AsyncMock(return_value=mock_surprise)
        with patch("qe.runtime.epistemic_reasoner.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await reasoner.detect_surprise(
                "g1", "entity_x", "Revenue dropped"
            )

        assert result is not None
        assert result.surprise_magnitude == 0.8
        state = reasoner.get_epistemic_state("g1")
        assert len(state.surprises) == 1

    async def test_surprise_below_threshold(self, reasoner):
        mock_belief_store = AsyncMock()
        mock_belief_store.get_beliefs_for_entity = AsyncMock(return_value=[
            MagicMock(
                claim=MagicMock(
                    claim_id="c1", predicate="p", object_value="v",
                ),
                posterior=0.5,
            ),
        ])
        reasoner._belief_store = mock_belief_store

        mock_surprise = SurpriseDetection(
            finding="Mild change",
            surprise_magnitude=0.1,
        )

        mock_create = AsyncMock(return_value=mock_surprise)
        with patch("qe.runtime.epistemic_reasoner.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await reasoner.detect_surprise(
                "g1", "entity_x", "Mild change"
            )

        assert result is None

    async def test_no_beliefs_returns_none(self, reasoner):
        mock_belief_store = AsyncMock()
        mock_belief_store.get_beliefs_for_entity = AsyncMock(return_value=[])
        reasoner._belief_store = mock_belief_store

        result = await reasoner.detect_surprise("g1", "unknown", "finding")
        assert result is None


# ── Status ───────────────────────────────────────────────────────────────


class TestStatus:
    def test_status_empty(self, reasoner):
        s = reasoner.status()
        assert s["active_goals"] == 0
        assert s["total_unknowns"] == 0

    def test_status_with_data(self, reasoner):
        reasoner.register_unknown("g1", "q?", "r")
        reasoner.register_unknown("g1", "q2?", "r2")
        s = reasoner.status()
        assert s["active_goals"] == 1
        assert s["total_unknowns"] == 2

"""Tests for DialecticEngine — adversarial self-critique."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.cognition import (
    AssumptionChallenge,
    Counterargument,
    PerspectiveAnalysis,
)
from qe.services.inquiry.dialectic import PERSPECTIVE_SETS, DialecticEngine


@pytest.fixture
def engine():
    return DialecticEngine(model="test-model")


# ── Perspective Sets ─────────────────────────────────────────────────────


class TestPerspectiveSets:
    def test_financial_perspectives(self):
        assert "bullish analyst" in PERSPECTIVE_SETS["financial"]
        assert "bearish analyst" in PERSPECTIVE_SETS["financial"]

    def test_general_perspectives(self):
        assert "contrarian" in PERSPECTIVE_SETS["general"]

    def test_scientific_perspectives(self):
        assert "critic" in PERSPECTIVE_SETS["scientific"]

    def test_technology_perspectives(self):
        assert "skeptic" in PERSPECTIVE_SETS["technology"]


# ── Challenge (Mocked LLM) ──────────────────────────────────────────────


class TestChallenge:
    async def test_challenge_returns_counterarguments(self, engine):
        mock_counters = [
            Counterargument(
                target_claim="Market is undervalued",
                counterargument="High P/E ratios suggest otherwise",
                strength="strong",
                evidence_needed_to_resolve="Historical P/E comparison",
                concession="Low interest rates do support valuations",
            ),
            Counterargument(
                target_claim="Market is undervalued",
                counterargument="Earnings estimates may be inflated",
                strength="moderate",
            ),
        ]

        class MockResult:
            counterarguments = mock_counters

        mock_create = AsyncMock(return_value=MockResult())
        with patch("qe.services.inquiry.dialectic.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await engine.challenge(
                "g1", "Market is undervalued"
            )

        assert len(result) == 2
        assert result[0].strength == "strong"


# ── Perspective Rotation (Mocked LLM) ────────────────────────────────────


class TestPerspectiveRotation:
    async def test_rotates_through_perspectives(self, engine):
        async def make_analysis(**kwargs):
            return PerspectiveAnalysis(
                perspective_name="",
                key_observations=["Key observation"],
                overall_assessment="Cautious",
            )

        with patch("qe.services.inquiry.dialectic.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=make_analysis
            )
            mock_inst.from_litellm.return_value = mock_client

            result = await engine.rotate_perspectives(
                "g1",
                "Market trends",
                domain="general",
                custom_perspectives=["optimist", "pessimist"],
            )

        assert len(result) == 2
        assert result[0].perspective_name == "optimist"
        assert result[1].perspective_name == "pessimist"


# ── Assumption Surfacing (Mocked LLM) ────────────────────────────────────


class TestAssumptionSurfacing:
    async def test_surfaces_assumptions(self, engine):
        mock_assumptions = [
            AssumptionChallenge(
                assumption="Growth continues",
                is_explicit=True,
                challenge="Recession risk",
                what_if_wrong="Entire analysis invalid",
                testable=True,
                test_method="Check leading indicators",
            ),
            AssumptionChallenge(
                assumption="Data is accurate",
                is_explicit=False,
                challenge="Source reliability unknown",
                testable=True,
            ),
        ]

        class MockResult:
            assumptions = mock_assumptions

        mock_create = AsyncMock(return_value=MockResult())
        with patch("qe.services.inquiry.dialectic.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await engine.surface_assumptions(
                "g1",
                "Market will grow 10% next year",
                ["Growth continues"],
            )

        assert len(result) == 2
        hidden = [a for a in result if not a.is_explicit]
        assert len(hidden) == 1


# ── Red Team (Mocked LLM) ───────────────────────────────────────────────


class TestRedTeam:
    async def test_red_team_attacks(self, engine):
        mock_counter = Counterargument(
            target_claim="Grid infrastructure is mispriced",
            counterargument="No structural catalyst for rerating",
            strength="strong",
        )

        mock_create = AsyncMock(return_value=mock_counter)
        with patch("qe.services.inquiry.dialectic.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await engine.red_team(
                "g1", "Grid infrastructure is mispriced"
            )

        assert result.strength == "strong"


# ── Full Dialectic (Mocked LLM) ─────────────────────────────────────────


class TestFullDialectic:
    async def test_full_dialectic_pipeline(self, engine):
        # Mock challenge
        mock_counters = [
            Counterargument(
                target_claim="X", counterargument="Not X",
                strength="strong",
                evidence_needed_to_resolve="Check Y",
            ),
        ]
        # Mock perspectives
        mock_perspective = PerspectiveAnalysis(
            perspective_name="", key_observations=["obs"],
        )
        # Mock assumptions
        mock_assumptions = [
            AssumptionChallenge(
                assumption="Hidden A", is_explicit=False,
                challenge="May not hold",
                testable=True, test_method="Test method",
            ),
            AssumptionChallenge(
                assumption="Hidden B", is_explicit=False,
                challenge="Questionable",
            ),
            AssumptionChallenge(
                assumption="Hidden C", is_explicit=False,
                challenge="Unlikely",
            ),
        ]

        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            rm = kwargs.get("response_model")
            if rm and hasattr(rm, "__annotations__"):
                if "counterarguments" in rm.__annotations__:
                    return MagicMock(counterarguments=mock_counters)
                if "assumptions" in rm.__annotations__:
                    return MagicMock(assumptions=mock_assumptions)
            return mock_perspective

        with patch("qe.services.inquiry.dialectic.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
            mock_inst.from_litellm.return_value = mock_client

            report = await engine.full_dialectic(
                "g1",
                "Market is undervalued",
                domain="general",
            )

        assert len(report.counterarguments) == 1
        assert len(report.assumptions_challenged) == 3
        # Strong counter + >2 hidden assumptions → should investigate
        assert report.should_investigate_further is True
        # Confidence revised down from 0.7
        assert report.revised_confidence < 0.7
        assert "strong counterarguments" in report.synthesis

    async def test_confidence_no_strong_counters(self, engine):
        """No strong counters and few hidden assumptions → confidence stays high."""
        mock_counters = [
            Counterargument(
                target_claim="X", counterargument="Weak",
                strength="weak",
            ),
        ]

        async def mock_create(**kwargs):
            rm = kwargs.get("response_model")
            if rm and hasattr(rm, "__annotations__"):
                if "counterarguments" in rm.__annotations__:
                    return MagicMock(counterarguments=mock_counters)
                if "assumptions" in rm.__annotations__:
                    return MagicMock(assumptions=[])
            return PerspectiveAnalysis(perspective_name="")

        with patch("qe.services.inquiry.dialectic.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
            mock_inst.from_litellm.return_value = mock_client

            report = await engine.full_dialectic("g1", "conclusion")

        assert report.revised_confidence == 0.7
        assert report.should_investigate_further is False


# ── Status ───────────────────────────────────────────────────────────────


class TestStatus:
    def test_status(self, engine):
        s = engine.status()
        assert s["model"] == "test-model"

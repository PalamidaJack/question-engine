"""Tests for Prompt Evolution wiring — component integration + bus topics/schemas."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from qe.bus.protocol import TOPICS
from qe.bus.schemas import (
    TOPIC_SCHEMAS,
    PromptOutcomeRecordedPayload,
    PromptVariantCreatedPayload,
    PromptVariantDeactivatedPayload,
    PromptVariantSelectedPayload,
    validate_payload,
)
from qe.optimization.prompt_registry import PromptRegistry, register_all_baselines

# ── Bus Topics ───────────────────────────────────────────────────────────


class TestPromptTopics:
    def test_variant_selected_in_topics(self):
        assert "prompt.variant_selected" in TOPICS

    def test_outcome_recorded_in_topics(self):
        assert "prompt.outcome_recorded" in TOPICS

    def test_variant_created_in_topics(self):
        assert "prompt.variant_created" in TOPICS

    def test_variant_deactivated_in_topics(self):
        assert "prompt.variant_deactivated" in TOPICS


# ── Bus Schemas ──────────────────────────────────────────────────────────


class TestPromptSchemas:
    def test_variant_selected_schema(self):
        p = PromptVariantSelectedPayload(
            slot_key="dialectic.challenge.user",
            variant_id="v1",
            is_baseline=False,
        )
        assert p.slot_key == "dialectic.challenge.user"

    def test_outcome_recorded_schema(self):
        p = PromptOutcomeRecordedPayload(
            slot_key="dialectic.challenge.user",
            variant_id="v1",
            success=True,
            quality_score=0.8,
        )
        assert p.success is True

    def test_variant_created_schema(self):
        p = PromptVariantCreatedPayload(
            slot_key="dialectic.challenge.user",
            variant_id="v1",
            parent_variant_id="base1",
            strategy="rephrase",
        )
        assert p.strategy == "rephrase"

    def test_variant_deactivated_schema(self):
        p = PromptVariantDeactivatedPayload(
            slot_key="dialectic.challenge.user",
            variant_id="v1",
            reason="low performance",
        )
        assert p.reason == "low performance"

    def test_schemas_in_registry(self):
        assert "prompt.variant_selected" in TOPIC_SCHEMAS
        assert "prompt.outcome_recorded" in TOPIC_SCHEMAS
        assert "prompt.variant_created" in TOPIC_SCHEMAS
        assert "prompt.variant_deactivated" in TOPIC_SCHEMAS

    def test_validate_payload_works(self):
        result = validate_payload("prompt.variant_selected", {
            "slot_key": "test", "variant_id": "v1", "is_baseline": True,
        })
        assert result is not None
        assert result.slot_key == "test"


# ── Component Integration — DialecticEngine ──────────────────────────────


class TestDialecticEngineWithRegistry:
    async def test_challenge_uses_registry(self):
        from qe.models.cognition import Counterargument
        from qe.services.inquiry.dialectic import DialecticEngine

        reg = PromptRegistry(enabled=False)
        register_all_baselines(reg)
        reg.record_outcome = MagicMock()

        engine = DialecticEngine(model="test-model", prompt_registry=reg)

        class MockResult:
            counterarguments = [
                Counterargument(
                    target_claim="X",
                    counterargument="Not X",
                    strength="moderate",
                )
            ]

        with patch("qe.services.inquiry.dialectic.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=MockResult())
            mock_inst.from_litellm.return_value = mock_client

            result = await engine.challenge("g1", "Market is bullish")
            assert len(result) == 1
            reg.record_outcome.assert_called_once()

    async def test_challenge_without_registry(self):
        from qe.models.cognition import Counterargument
        from qe.services.inquiry.dialectic import DialecticEngine

        engine = DialecticEngine(model="test-model")

        class MockResult:
            counterarguments = [
                Counterargument(
                    target_claim="X",
                    counterargument="Not X",
                    strength="moderate",
                )
            ]

        with patch("qe.services.inquiry.dialectic.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=MockResult())
            mock_inst.from_litellm.return_value = mock_client

            result = await engine.challenge("g1", "Market is bullish")
            assert len(result) == 1


# ── Component Integration — InsightCrystallizer ──────────────────────────


class TestInsightCrystallizerWithRegistry:
    async def test_assess_novelty_uses_registry(self):
        from qe.models.cognition import NoveltyAssessment
        from qe.services.inquiry.insight import InsightCrystallizer

        reg = PromptRegistry(enabled=False)
        register_all_baselines(reg)
        reg.record_outcome = MagicMock()

        ic = InsightCrystallizer(model="test-model", prompt_registry=reg)

        mock_novelty = NoveltyAssessment(
            finding="test finding",
            is_novel=True,
            novelty_type="new_connection",
            who_would_find_this_surprising="experts",
        )

        with patch("qe.services.inquiry.insight.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_novelty)
            mock_inst.from_litellm.return_value = mock_client

            result = await ic.assess_novelty("test finding")
            assert result.is_novel is True
            reg.record_outcome.assert_called_once()


# ── Component Integration — Metacognitor ─────────────────────────────────


class TestMetacognitorWithRegistry:
    async def test_suggest_approach_uses_registry(self):
        from qe.models.cognition import ApproachAssessment
        from qe.runtime.metacognitor import Metacognitor

        reg = PromptRegistry(enabled=False)
        register_all_baselines(reg)
        reg.record_outcome = MagicMock()

        meta = Metacognitor(model="test-model", prompt_registry=reg)

        mock_assessment = ApproachAssessment(
            recommended_approach="Try web search",
            reasoning="Standard approach",
        )

        with patch("qe.runtime.metacognitor.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_assessment)
            mock_inst.from_litellm.return_value = mock_client

            result = await meta.suggest_next_approach("g1", "research topic")
            assert result.recommended_approach == "Try web search"
            reg.record_outcome.assert_called_once()


# ── Component Integration — EpistemicReasoner ────────────────────────────


class TestEpistemicReasonerWithRegistry:
    async def test_assess_uncertainty_uses_registry(self):
        from qe.models.cognition import UncertaintyAssessment
        from qe.runtime.epistemic_reasoner import EpistemicReasoner

        reg = PromptRegistry(enabled=False)
        register_all_baselines(reg)
        reg.record_outcome = MagicMock()

        er = EpistemicReasoner(model="test-model", prompt_registry=reg)

        mock_assessment = UncertaintyAssessment(
            finding_summary="some finding",
            confidence_level="moderate",
            evidence_quality="secondary",
        )

        with patch("qe.runtime.epistemic_reasoner.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_assessment)
            mock_inst.from_litellm.return_value = mock_client

            result = await er.assess_uncertainty("g1", "some finding")
            assert result.confidence_level == "moderate"
            reg.record_outcome.assert_called_once()


# ── Component Integration — PersistenceEngine ────────────────────────────


class TestPersistenceEngineWithRegistry:
    async def test_root_cause_uses_registry(self):
        from qe.models.cognition import RootCauseAnalysis, RootCauseLink
        from qe.runtime.persistence_engine import PersistenceEngine

        reg = PromptRegistry(enabled=False)
        register_all_baselines(reg)
        reg.record_outcome = MagicMock()

        pe = PersistenceEngine(model="test-model", prompt_registry=reg)

        mock_analysis = RootCauseAnalysis(
            failure_summary="test failure",
            chain=[
                RootCauseLink(level=1, question="Why?", answer="Because", confidence=0.8),
                RootCauseLink(level=2, question="Why?", answer="Root", confidence=0.7),
                RootCauseLink(level=3, question="Why?", answer="Deep", confidence=0.6),
            ],
            root_cause="Deep issue",
            lesson_learned="Fix it",
            prevention_strategy="Better testing",
        )

        with patch("qe.runtime.persistence_engine.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_analysis)
            mock_inst.from_litellm.return_value = mock_client

            result = await pe.analyze_root_cause("g1", "failure")
            assert result.root_cause == "Deep issue"
            reg.record_outcome.assert_called_once()


# ── Component Integration — QuestionGenerator ────────────────────────────


class TestQuestionGeneratorWithRegistry:
    async def test_generate_uses_registry(self):
        from qe.services.inquiry.question_generator import (
            GeneratedQuestions,
            QuestionGenerator,
        )

        reg = PromptRegistry(enabled=False)
        register_all_baselines(reg)
        reg.record_outcome = MagicMock()

        qg = QuestionGenerator(model="test-model", prompt_registry=reg)

        mock_result = GeneratedQuestions(questions=[])

        with patch("qe.services.inquiry.question_generator.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_result)
            mock_inst.from_litellm.return_value = mock_client
            qg._client = mock_client

            result = await qg.generate("test goal")
            assert isinstance(result, list)
            reg.record_outcome.assert_called_once()


# ── Component Integration — HypothesisManager ────────────────────────────


class TestHypothesisManagerWithRegistry:
    async def test_generate_hypotheses_uses_registry(self):
        from qe.services.inquiry.hypothesis import (
            GeneratedHypotheses,
            HypothesisManager,
        )

        reg = PromptRegistry(enabled=False)
        register_all_baselines(reg)
        reg.record_outcome = MagicMock()

        hm = HypothesisManager(model="test-model", prompt_registry=reg)

        mock_result = GeneratedHypotheses(hypotheses=[])

        with patch("qe.services.inquiry.hypothesis.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_result)
            mock_inst.from_litellm.return_value = mock_client
            hm._client = mock_client

            result = await hm.generate_hypotheses("test goal")
            assert isinstance(result, list)
            reg.record_outcome.assert_called_once()


# ── Feature Flag ─────────────────────────────────────────────────────────


class TestPromptEvolutionFeatureFlag:
    def test_flag_defined(self):
        from qe.runtime.feature_flags import FeatureFlagStore

        store = FeatureFlagStore()
        store.define("prompt_evolution", enabled=False, description="test")
        assert store.is_enabled("prompt_evolution") is False

        store.enable("prompt_evolution")
        assert store.is_enabled("prompt_evolution") is True

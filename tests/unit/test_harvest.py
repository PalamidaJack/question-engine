"""Tests for the HarvestService."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.config import HarvestConfig
from qe.models.envelope import Envelope
from qe.services.harvest.service import _PROFILE_PROMPTS, HarvestService

# ── Helpers ──────────────────────────────────────────────────────────────


@dataclass
class FakeModel:
    model_id: str = "test/model-a"
    provider: str = "test"
    base_model_name: str = "model-a"
    quality_tier: str = "fast"
    is_free: bool = True
    status: str = "active"


@dataclass
class FakeClaim:
    claim_id: str = "clm_test1"
    subject_entity_id: str = "Python"
    predicate: str = "is"
    object_value: str = "a programming language"
    confidence: float = 0.3
    source_service_id: str = "test"
    source_envelope_ids: list = None
    tags: list = None

    def __post_init__(self):
        if self.source_envelope_ids is None:
            self.source_envelope_ids = []
        if self.tags is None:
            self.tags = []


@dataclass
class FakeKnownUnknown:
    unknown_id: str = "ku_1"
    question: str = "What is quantum entanglement?"
    why_unknown: str = "Complex physics topic"
    importance: str = "high"


@dataclass
class FakeEpistemicState:
    goal_id: str = "goal_1"
    known_unknowns: list = None

    def __post_init__(self):
        if self.known_unknowns is None:
            self.known_unknowns = [FakeKnownUnknown()]


@dataclass
class FakeResponse:
    success: bool = True
    response: str = (
        '{"verdict": "agree", "confidence": 0.8, '
        '"reasoning": "correct"}'
    )
    model_id: str = "test/model"
    provider: str = "test"
    model_name: str = "model"
    latency_ms: float = 100.0
    error: str | None = None
    tokens_used: int = 50


@dataclass
class FakeMassResult:
    prompt: str = ""
    responses: list = None
    total_models: int = 3
    successful: int = 3
    failed: int = 0
    total_time_ms: float = 300.0

    def __post_init__(self):
        if self.responses is None:
            self.responses = [FakeResponse() for _ in range(3)]


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def bus():
    return MemoryBus()


@pytest.fixture()
def mock_substrate():
    sub = AsyncMock()
    sub.get_claims = AsyncMock(return_value=[])
    return sub


@pytest.fixture()
def mock_discovery():
    disc = MagicMock()
    disc.get_available_models = MagicMock(return_value=[])
    disc.record_call = MagicMock()
    return disc


@pytest.fixture()
def mock_executor():
    ex = AsyncMock()
    ex.quick_query = AsyncMock(return_value=FakeMassResult())
    return ex


@pytest.fixture()
def mock_reasoner():
    r = MagicMock()
    r._states = {}
    return r


@pytest.fixture()
def mock_procedural():
    p = AsyncMock()
    p.record_sequence_outcome = AsyncMock()
    return p


@pytest.fixture()
def config():
    return HarvestConfig(
        poll_interval_seconds=1,  # Fast for tests
        cycle_timeout_seconds=10,
    )


@pytest.fixture()
def service(
    bus, mock_substrate, mock_discovery, mock_executor,
    mock_reasoner, mock_procedural, config,
):
    return HarvestService(
        bus=bus,
        substrate=mock_substrate,
        discovery=mock_discovery,
        mass_executor=mock_executor,
        epistemic_reasoner=mock_reasoner,
        procedural_memory=mock_procedural,
        config=config,
    )


# ── Config tests ─────────────────────────────────────────────────────────


class TestConfig:
    def test_defaults(self):
        cfg = HarvestConfig()
        assert cfg.enabled is False
        assert cfg.poll_interval_seconds == 1800
        assert cfg.max_claims_per_cycle == 10
        assert cfg.consensus_model_count == 5

    def test_custom_values(self):
        cfg = HarvestConfig(
            enabled=True,
            poll_interval_seconds=600,
            adversarial_confidence_threshold=0.9,
        )
        assert cfg.enabled is True
        assert cfg.poll_interval_seconds == 600
        assert cfg.adversarial_confidence_threshold == 0.9

    def test_validation_poll_interval(self):
        with pytest.raises(ValueError):
            HarvestConfig(poll_interval_seconds=0)

    def test_validation_consensus_count(self):
        with pytest.raises(ValueError):
            HarvestConfig(consensus_model_count=1)

    def test_validation_confidence_threshold(self):
        with pytest.raises(ValueError):
            HarvestConfig(adversarial_confidence_threshold=1.5)


# ── Lifecycle tests ──────────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_poll_task(self, service):
        await service.start()
        assert service._running is True
        assert service._poll_task is not None
        await service.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_poll_task(self, service):
        await service.start()
        await service.stop()
        assert service._running is False

    @pytest.mark.asyncio
    async def test_stop_without_start(self, service):
        await service.stop()
        assert service._running is False

    @pytest.mark.asyncio
    async def test_status_default(self, service):
        s = service.status()
        assert s["running"] is False
        assert s["cycles_completed"] == 0
        assert s["last_cycle_at"] is None
        assert s["last_mode"] is None
        assert s["models_profiled"] == 0
        assert s["claims_processed"] == 0


# ── Mode selection tests ─────────────────────────────────────────────────


class TestModeSelection:
    @pytest.mark.asyncio
    async def test_premium_sprint_priority(
        self, service, mock_discovery,
    ):
        mock_discovery.get_available_models.return_value = [
            FakeModel(quality_tier="powerful"),
        ]
        mode = await service._select_mode()
        assert mode == "premium_sprint"

    @pytest.mark.asyncio
    async def test_model_profile_when_unprofiled(
        self, service, mock_discovery,
    ):
        mock_discovery.get_available_models.side_effect = lambda **kw: (
            [] if kw.get("tier") == "powerful" else [FakeModel()]
        )
        mode = await service._select_mode()
        assert mode == "model_profile"

    @pytest.mark.asyncio
    async def test_consensus_when_low_confidence(
        self, service, mock_discovery, mock_substrate,
    ):
        mock_discovery.get_available_models.return_value = []
        service._model_profiles["test/model-a"] = {"overall_score": 0.8}
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.3),
        ]
        mode = await service._select_mode()
        assert mode == "consensus_validate"

    @pytest.mark.asyncio
    async def test_adversarial_when_high_confidence(
        self, service, mock_discovery, mock_substrate,
    ):
        mock_discovery.get_available_models.return_value = []
        service._model_profiles["test/model-a"] = {"overall_score": 0.8}
        mock_substrate.get_claims.side_effect = [
            [],  # low-confidence call
            [FakeClaim(confidence=0.9)],  # high-confidence call
        ]
        mode = await service._select_mode()
        assert mode == "adversarial_challenge"

    @pytest.mark.asyncio
    async def test_knowledge_gap_when_unknowns(
        self, service, mock_discovery, mock_substrate,
        mock_reasoner,
    ):
        mock_discovery.get_available_models.return_value = []
        service._model_profiles["test/model-a"] = {"overall_score": 0.8}
        mock_substrate.get_claims.return_value = []
        mock_reasoner._states = {"g1": FakeEpistemicState()}
        mode = await service._select_mode()
        assert mode == "knowledge_gap"

    @pytest.mark.asyncio
    async def test_no_mode_when_nothing(
        self, service, mock_discovery, mock_substrate,
        mock_reasoner,
    ):
        mock_discovery.get_available_models.return_value = []
        service._model_profiles["test/model-a"] = {"overall_score": 0.8}
        mock_substrate.get_claims.return_value = []
        mock_reasoner._states = {}
        mode = await service._select_mode()
        assert mode is None


# ── Consensus validate tests ─────────────────────────────────────────────


class TestConsensusValidate:
    @pytest.mark.asyncio
    async def test_validates_claims(
        self, service, mock_substrate, mock_executor,
    ):
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.3),
        ]
        result = await service._mode_consensus_validate()
        assert result["validated_count"] == 1
        assert mock_executor.quick_query.called

    @pytest.mark.asyncio
    async def test_skips_processed_claims(self, service, mock_substrate):
        claim = FakeClaim()
        service._processed_claim_ids.add(claim.claim_id)
        mock_substrate.get_claims.return_value = [claim]
        result = await service._mode_consensus_validate()
        assert result["validated_count"] == 0

    @pytest.mark.asyncio
    async def test_respects_max_claims(
        self, service, mock_substrate, mock_executor,
    ):
        claims = [
            FakeClaim(claim_id=f"clm_{i}", confidence=0.3)
            for i in range(20)
        ]
        mock_substrate.get_claims.return_value = claims
        service._config.max_claims_per_cycle = 3
        result = await service._mode_consensus_validate()
        assert result["total_claims"] == 3

    @pytest.mark.asyncio
    async def test_publishes_observation(
        self, service, mock_substrate, mock_executor, bus,
    ):
        published = []
        bus.subscribe(
            "observations.structured",
            lambda e: published.append(e),
        )
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.3),
        ]
        await service._mode_consensus_validate()
        await asyncio.sleep(0.05)
        assert len(published) >= 1

    @pytest.mark.asyncio
    async def test_publishes_claim_validated(
        self, service, mock_substrate, mock_executor, bus,
    ):
        published = []
        bus.subscribe(
            "harvest.claim_validated",
            lambda e: published.append(e),
        )
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.3),
        ]
        await service._mode_consensus_validate()
        await asyncio.sleep(0.05)
        assert len(published) == 1

    @pytest.mark.asyncio
    async def test_handles_executor_error(
        self, service, mock_substrate, mock_executor,
    ):
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.3),
        ]
        mock_executor.quick_query.side_effect = RuntimeError("fail")
        result = await service._mode_consensus_validate()
        assert result["validated_count"] == 0


# ── Adversarial challenge tests ──────────────────────────────────────────


class TestAdversarialChallenge:
    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.litellm")
    async def test_challenges_claims(
        self, mock_litellm, service, mock_substrate,
        mock_discovery,
    ):
        mock_discovery.get_available_models.return_value = [FakeModel()]
        service._model_profiles["test/model-a"] = {
            "overall_score": 0.8,
        }
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.9),
        ]
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = (
            '{"challenge_valid": true}'
        )
        mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
        result = await service._mode_adversarial_challenge()
        assert result["challenged_count"] == 1

    @pytest.mark.asyncio
    async def test_no_models_available(
        self, service, mock_substrate, mock_discovery,
    ):
        mock_discovery.get_available_models.return_value = []
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.9),
        ]
        result = await service._mode_adversarial_challenge()
        assert result["challenged_count"] == 0
        assert result["error"] == "no_models_available"

    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.litellm")
    async def test_records_model_call(
        self, mock_litellm, service, mock_substrate,
        mock_discovery,
    ):
        mock_discovery.get_available_models.return_value = [FakeModel()]
        service._model_profiles["test/model-a"] = {
            "overall_score": 0.8,
        }
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.9),
        ]
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "challenge"
        mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
        await service._mode_adversarial_challenge()
        mock_discovery.record_call.assert_called()

    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.litellm")
    async def test_handles_litellm_error(
        self, mock_litellm, service, mock_substrate,
        mock_discovery,
    ):
        mock_discovery.get_available_models.return_value = [FakeModel()]
        service._model_profiles["test/model-a"] = {
            "overall_score": 0.8,
        }
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.9),
        ]
        mock_litellm.acompletion = AsyncMock(
            side_effect=RuntimeError("timeout"),
        )
        result = await service._mode_adversarial_challenge()
        assert result["challenged_count"] == 0


# ── Knowledge gap tests ──────────────────────────────────────────────────


class TestKnowledgeGap:
    @pytest.mark.asyncio
    async def test_explores_unknowns(
        self, service, mock_executor, mock_reasoner,
    ):
        mock_reasoner._states = {"g1": FakeEpistemicState()}
        result = await service._mode_knowledge_gap()
        assert result["explored_count"] == 1
        assert mock_executor.quick_query.called

    @pytest.mark.asyncio
    async def test_sorts_by_importance(
        self, service, mock_executor, mock_reasoner,
    ):
        critical = FakeKnownUnknown(
            unknown_id="ku_c", importance="critical",
        )
        low = FakeKnownUnknown(
            unknown_id="ku_l", importance="low",
        )
        state = FakeEpistemicState(known_unknowns=[low, critical])
        mock_reasoner._states = {"g1": state}
        service._config.max_claims_per_cycle = 1
        await service._mode_knowledge_gap()
        prompt_arg = mock_executor.quick_query.call_args[0][0]
        assert "quantum entanglement" in prompt_arg

    @pytest.mark.asyncio
    async def test_handles_executor_error(
        self, service, mock_executor, mock_reasoner,
    ):
        mock_reasoner._states = {"g1": FakeEpistemicState()}
        mock_executor.quick_query.side_effect = RuntimeError("fail")
        result = await service._mode_knowledge_gap()
        assert result["explored_count"] == 0


# ── Premium sprint tests ─────────────────────────────────────────────────


class TestPremiumSprint:
    @pytest.mark.asyncio
    async def test_no_powerful_models(self, service, mock_discovery):
        mock_discovery.get_available_models.return_value = []
        result = await service._mode_premium_sprint()
        assert result["sprint_count"] == 0

    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.litellm")
    async def test_routes_unknowns(
        self, mock_litellm, service, mock_discovery,
        mock_reasoner, mock_substrate,
    ):
        mock_discovery.get_available_models.return_value = [
            FakeModel(quality_tier="powerful"),
        ]
        mock_reasoner._states = {"g1": FakeEpistemicState()}
        mock_substrate.get_claims.return_value = []
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Expert analysis..."
        mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
        result = await service._mode_premium_sprint()
        assert result["sprint_count"] == 1

    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.litellm")
    async def test_routes_low_confidence_claims(
        self, mock_litellm, service, mock_discovery,
        mock_reasoner, mock_substrate,
    ):
        mock_discovery.get_available_models.return_value = [
            FakeModel(quality_tier="powerful"),
        ]
        mock_reasoner._states = {}
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.2),
        ]
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Analysis"
        mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
        result = await service._mode_premium_sprint()
        assert result["sprint_count"] == 1

    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.litellm")
    async def test_records_discovery_call(
        self, mock_litellm, service, mock_discovery,
        mock_reasoner, mock_substrate,
    ):
        mock_discovery.get_available_models.return_value = [
            FakeModel(quality_tier="powerful"),
        ]
        mock_reasoner._states = {"g1": FakeEpistemicState()}
        mock_substrate.get_claims.return_value = []
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "answer"
        mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
        await service._mode_premium_sprint()
        mock_discovery.record_call.assert_called()


# ── Model profile tests ─────────────────────────────────────────────────


class TestModelProfile:
    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.litellm")
    async def test_profiles_unprofiled_models(
        self, mock_litellm, service, mock_discovery,
    ):
        mock_discovery.get_available_models.return_value = [FakeModel()]
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "9"
        mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
        result = await service._mode_model_profile()
        assert result["profiled_count"] == 1
        assert "test/model-a" in service._model_profiles

    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.litellm")
    async def test_max_3_per_cycle(
        self, mock_litellm, service, mock_discovery,
    ):
        models = [FakeModel(model_id=f"test/m{i}") for i in range(5)]
        mock_discovery.get_available_models.return_value = models
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "answer"
        mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
        result = await service._mode_model_profile()
        assert result["profiled_count"] == 3

    @pytest.mark.asyncio
    async def test_skips_already_profiled(
        self, service, mock_discovery,
    ):
        service._model_profiles["test/model-a"] = {
            "overall_score": 0.8,
        }
        mock_discovery.get_available_models.return_value = [FakeModel()]
        result = await service._mode_model_profile()
        assert result["profiled_count"] == 0

    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.litellm")
    async def test_publishes_model_profiled(
        self, mock_litellm, service, mock_discovery, bus,
    ):
        published = []
        bus.subscribe(
            "harvest.model_profiled",
            lambda e: published.append(e),
        )
        mock_discovery.get_available_models.return_value = [FakeModel()]
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "answer"
        mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
        await service._mode_model_profile()
        await asyncio.sleep(0.05)
        assert len(published) == 1

    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.litellm")
    async def test_handles_model_error(
        self, mock_litellm, service, mock_discovery,
    ):
        mock_discovery.get_available_models.return_value = [FakeModel()]
        mock_litellm.acompletion = AsyncMock(
            side_effect=RuntimeError("model down"),
        )
        result = await service._mode_model_profile()
        assert result["profiled_count"] == 1
        profile = service._model_profiles["test/model-a"]
        assert profile["tests_passed"] == 0


# ── Helper tests ─────────────────────────────────────────────────────────


class TestHelpers:
    def test_compute_consensus_agree_majority(self):
        responses = [
            FakeResponse(
                response='{"verdict": "agree", "confidence": 0.9}',
            ),
            FakeResponse(
                response='{"verdict": "agree", "confidence": 0.8}',
            ),
            FakeResponse(
                response='{"verdict": "disagree", "confidence": 0.7}',
            ),
        ]
        result = HarvestService._compute_consensus(responses)
        assert result["verdict"] == "agree"
        assert result["agree_count"] == 2
        assert result["disagree_count"] == 1

    def test_compute_consensus_disagree_majority(self):
        responses = [
            FakeResponse(
                response='{"verdict": "disagree", "confidence": 0.8}',
            ),
            FakeResponse(
                response='{"verdict": "disagree", "confidence": 0.9}',
            ),
            FakeResponse(
                response='{"verdict": "agree", "confidence": 0.5}',
            ),
        ]
        result = HarvestService._compute_consensus(responses)
        assert result["verdict"] == "disagree"

    def test_compute_consensus_empty(self):
        result = HarvestService._compute_consensus([])
        assert result["verdict"] == "inconclusive"
        assert result["total"] == 0

    def test_compute_consensus_fallback_text(self):
        responses = [
            FakeResponse(response="I agree with this claim"),
            FakeResponse(response="I agree"),
        ]
        result = HarvestService._compute_consensus(responses)
        assert result["verdict"] == "agree"
        assert result["agree_count"] == 2

    def test_score_profile_reasoning_correct(self):
        score = HarvestService._score_profile_response(
            "reasoning", "The answer is 9 sheep", "9",
        )
        assert score == 1.0

    def test_score_profile_reasoning_wrong(self):
        score = HarvestService._score_profile_response(
            "reasoning", "The answer is 17", "9",
        )
        assert score == 0.2

    def test_score_profile_factual_correct(self):
        score = HarvestService._score_profile_response(
            "factual", "Canberra", "Canberra",
        )
        assert score == 1.0

    def test_score_profile_factual_wrong(self):
        score = HarvestService._score_profile_response(
            "factual", "Sydney", "Canberra",
        )
        assert score == 0.0

    def test_score_profile_creative_good(self):
        haiku = (
            "Silicon minds think\n"
            "Processing data streams flow\n"
            "AI dreams take shape"
        )
        score = HarvestService._score_profile_response(
            "creative", haiku, "",
        )
        assert score == 1.0

    def test_score_profile_structured_valid_json(self):
        resp = '{"name": "apple", "type": "fruit", "count": 3}'
        score = HarvestService._score_profile_response(
            "structured_output", resp, "",
        )
        assert score == 1.0

    def test_score_profile_structured_invalid(self):
        score = HarvestService._score_profile_response(
            "structured_output", "not json", "",
        )
        assert score == 0.0

    def test_score_profile_instruction_good(self):
        score = HarvestService._score_profile_response(
            "instruction_following", "2, 3, 5", "",
        )
        assert score == 1.0

    def test_score_profile_instruction_partial(self):
        score = HarvestService._score_profile_response(
            "instruction_following", "2, 3, 5, 7", "",
        )
        assert score == 0.3

    def test_score_profile_empty(self):
        score = HarvestService._score_profile_response(
            "reasoning", "", "9",
        )
        assert score == 0.0

    def test_pick_best_model_prefers_profiled(
        self, service, mock_discovery,
    ):
        model_a = FakeModel(model_id="test/a")
        model_b = FakeModel(model_id="test/b")
        mock_discovery.get_available_models.return_value = [
            model_a, model_b,
        ]
        service._model_profiles["test/b"] = {"overall_score": 0.9}
        best = service._pick_best_model()
        assert best.model_id == "test/b"

    def test_pick_best_model_none(self, service, mock_discovery):
        mock_discovery.get_available_models.return_value = []
        assert service._pick_best_model() is None


# ── Bus event handler tests ──────────────────────────────────────────────


class TestBusHandlers:
    @pytest.mark.asyncio
    async def test_on_models_discovered_clears_stale(self, service):
        service._model_profiles = {
            "gone/model": {"overall_score": 0.5},
        }
        envelope = Envelope(
            topic="models.discovered",
            source_service_id="discovery",
            payload={"models": [{"model_id": "new/model"}]},
        )
        await service._on_models_discovered(envelope)
        assert "gone/model" not in service._model_profiles

    @pytest.mark.asyncio
    async def test_on_models_discovered_keeps_active(self, service):
        service._model_profiles = {
            "active/model": {"overall_score": 0.8},
        }
        envelope = Envelope(
            topic="models.discovered",
            source_service_id="discovery",
            payload={"models": [{"model_id": "active/model"}]},
        )
        await service._on_models_discovered(envelope)
        assert "active/model" in service._model_profiles

    @pytest.mark.asyncio
    async def test_on_tiers_updated(self, service):
        envelope = Envelope(
            topic="models.tiers_updated",
            source_service_id="discovery",
            payload={
                "changes": [{"tier": "fast", "model": "new/model"}],
            },
        )
        await service._on_tiers_updated(envelope)


# ── Cycle tests ──────────────────────────────────────────────────────────


class TestCycle:
    @pytest.mark.asyncio
    async def test_cycle_skipped_when_flag_disabled(
        self, service, bus,
    ):
        started = []
        bus.subscribe(
            "harvest.cycle_started",
            lambda e: started.append(e),
        )
        await service._run_cycle()
        await asyncio.sleep(0.05)
        # Flag disabled → returns early, no events published
        assert len(started) == 0
        assert service._cycles_completed == 0

    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.get_flag_store")
    async def test_cycle_runs_when_enabled(
        self, mock_flags, service, mock_discovery, bus,
    ):
        mock_flags.return_value.is_enabled.return_value = True
        mock_discovery.get_available_models.return_value = []
        service._model_profiles["test/model-a"] = {}
        published = []
        bus.subscribe(
            "harvest.cycle_started",
            lambda e: published.append(e),
        )
        await service._run_cycle()
        await asyncio.sleep(0.05)
        assert len(published) == 1

    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.get_flag_store")
    async def test_cycle_records_procedural_memory(
        self, mock_flags, service, mock_discovery,
        mock_substrate, mock_executor, mock_procedural,
    ):
        mock_flags.return_value.is_enabled.return_value = True
        mock_discovery.get_available_models.return_value = []
        service._model_profiles["test/model-a"] = {}
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.3),
        ]
        await service._run_cycle()
        mock_procedural.record_sequence_outcome.assert_called_once()

    @pytest.mark.asyncio
    @patch("qe.services.harvest.service.get_flag_store")
    async def test_cycle_increments_counter(
        self, mock_flags, service, mock_discovery,
        mock_substrate, mock_executor,
    ):
        mock_flags.return_value.is_enabled.return_value = True
        # Provide low-confidence claims so consensus_validate runs
        mock_discovery.get_available_models.return_value = []
        service._model_profiles["test/model-a"] = {}
        mock_substrate.get_claims.return_value = [
            FakeClaim(confidence=0.3),
        ]
        await service._run_cycle()
        assert service._cycles_completed == 1


# ── Profile prompts sanity ───────────────────────────────────────────────


class TestProfilePrompts:
    def test_five_prompts(self):
        assert len(_PROFILE_PROMPTS) == 5

    def test_all_have_required_keys(self):
        for p in _PROFILE_PROMPTS:
            assert "category" in p
            assert "prompt" in p
            assert "expected_answer" in p

    def test_unique_categories(self):
        cats = [p["category"] for p in _PROFILE_PROMPTS]
        assert len(cats) == len(set(cats))

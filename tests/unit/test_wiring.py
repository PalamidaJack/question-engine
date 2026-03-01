"""Tests for Phase 1+2 wiring: bus topics, substrate property, cache swap, shared state."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.bus.protocol import TOPICS
from qe.models.genome import Blueprint, CapabilityDeclaration, ModelPreference
from qe.runtime.context_curator import ContextCurator
from qe.runtime.episodic_memory import EpisodicMemory
from qe.runtime.epistemic_reasoner import EpistemicReasoner
from qe.runtime.metacognitor import Metacognitor
from qe.runtime.persistence_engine import PersistenceEngine
from qe.runtime.service import BaseService
from qe.services.inquiry.dialectic import DialecticEngine
from qe.services.inquiry.insight import InsightCrystallizer
from qe.substrate.bayesian_belief import BayesianBeliefStore


@pytest.fixture(autouse=True)
def _reset_shared_state():
    """Reset BaseService class-level state between tests."""
    BaseService._shared_episodic_memory = None
    BaseService._shared_bayesian_belief = None
    BaseService._shared_context_curator = None
    BaseService._shared_metacognitor = None
    BaseService._shared_epistemic_reasoner = None
    BaseService._shared_dialectic_engine = None
    BaseService._shared_persistence_engine = None
    BaseService._shared_insight_crystallizer = None
    yield
    BaseService._shared_episodic_memory = None
    BaseService._shared_bayesian_belief = None
    BaseService._shared_context_curator = None
    BaseService._shared_metacognitor = None
    BaseService._shared_epistemic_reasoner = None
    BaseService._shared_dialectic_engine = None
    BaseService._shared_persistence_engine = None
    BaseService._shared_insight_crystallizer = None


# ── Bus Topics ──────────────────────────────────────────────────────────────


class TestCognitiveBusTopics:
    """Verify all 12 cognitive bus topics are registered."""

    COGNITIVE_TOPICS = {
        "cognitive.approach_selected",
        "cognitive.approach_exhausted",
        "cognitive.absence_detected",
        "cognitive.surprise_detected",
        "cognitive.uncertainty_assessed",
        "cognitive.dialectic_completed",
        "cognitive.assumption_surfaced",
        "cognitive.root_cause_analyzed",
        "cognitive.reframe_suggested",
        "cognitive.lesson_learned",
        "cognitive.insight_crystallized",
        "cognitive.capability_gap",
    }

    def test_cognitive_topics_present(self):
        assert self.COGNITIVE_TOPICS.issubset(TOPICS)

    def test_topic_count_includes_cognitive(self):
        # 77 pre-existing + 12 cognitive = 89
        assert len(TOPICS) >= 89


# ── Substrate Property ──────────────────────────────────────────────────────


class TestSubstrateBayesianBelief:
    """Verify lazy bayesian_belief property on Substrate."""

    def test_returns_bayesian_belief_store(self, tmp_path):
        from qe.substrate import Substrate

        db = str(tmp_path / "test.db")
        sub = Substrate(db_path=db)
        result = sub.bayesian_belief
        assert isinstance(result, BayesianBeliefStore)

    def test_caches_instance(self, tmp_path):
        from qe.substrate import Substrate

        db = str(tmp_path / "test.db")
        sub = Substrate(db_path=db)
        first = sub.bayesian_belief
        second = sub.bayesian_belief
        assert first is second


# ── EngramCache Swap ────────────────────────────────────────────────────────


class TestEngramCacheSwap:
    """Verify _call_llm uses EngramCache (get_exact/put_exact)."""

    @pytest.mark.asyncio
    async def test_call_llm_uses_engram_cache_hit(self):
        """Cache hit via get_exact returns cached response without LLM call."""
        from pydantic import BaseModel

        class _DummyResponse(BaseModel):
            answer: str = "cached"

        bp = Blueprint(
            service_id="test-svc",
            display_name="Test",
            version="1.0",
            system_prompt="test",
            model_preference=ModelPreference(tier="balanced"),
            capabilities=CapabilityDeclaration(),
        )

        class _Svc(BaseService):
            async def handle_response(self, envelope, response):
                pass

            def get_response_schema(self, topic):
                return _DummyResponse

        svc = _Svc(bp, MagicMock(), None)
        cached_resp = _DummyResponse(answer="from_cache")

        mock_cache = MagicMock()
        mock_cache.make_key.return_value = "test_key"
        mock_cache.get_exact.return_value = cached_resp

        msgs = [{"role": "user", "content": "hi"}]
        with patch("qe.runtime.service.get_engram_cache", return_value=mock_cache):
            result = await svc._call_llm("gpt-4o", msgs, _DummyResponse)

        mock_cache.get_exact.assert_called_once_with("test_key")
        assert result is cached_resp

    @pytest.mark.asyncio
    async def test_call_llm_uses_engram_cache_miss(self):
        """Cache miss calls LLM and stores via put_exact."""
        from pydantic import BaseModel

        class _DummyResponse(BaseModel):
            answer: str = "fresh"

        bp = Blueprint(
            service_id="test-svc",
            display_name="Test",
            version="1.0",
            system_prompt="test",
            model_preference=ModelPreference(tier="balanced"),
            capabilities=CapabilityDeclaration(),
        )

        class _Svc(BaseService):
            async def handle_response(self, envelope, response):
                pass

            def get_response_schema(self, topic):
                return _DummyResponse

        svc = _Svc(bp, MagicMock(), None)
        llm_resp = _DummyResponse(answer="from_llm")

        mock_cache = MagicMock()
        mock_cache.make_key.return_value = "test_key"
        mock_cache.get_exact.return_value = None

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=llm_resp)

        with (
            patch("qe.runtime.service.get_engram_cache", return_value=mock_cache),
            patch("qe.runtime.service.instructor") as mock_instructor,
            patch("qe.runtime.service.get_rate_limiter") as mock_rl,
        ):
            mock_instructor.from_litellm.return_value = mock_client
            mock_rl.return_value.acquire = AsyncMock()
            msgs = [{"role": "user", "content": "hi"}]
            result = await svc._call_llm("gpt-4o", msgs, _DummyResponse)

        mock_cache.put_exact.assert_called_once_with("test_key", llm_resp, "gpt-4o")
        assert result is llm_resp


# ── Shared State Classmethods ───────────────────────────────────────────────


class TestSharedState:
    """Verify all 8 set_* classmethods work correctly."""

    def test_set_episodic_memory(self):
        mem = MagicMock(spec=EpisodicMemory)
        BaseService.set_episodic_memory(mem)
        assert BaseService._shared_episodic_memory is mem

    def test_set_bayesian_belief(self):
        store = MagicMock(spec=BayesianBeliefStore)
        BaseService.set_bayesian_belief(store)
        assert BaseService._shared_bayesian_belief is store

    def test_set_context_curator(self):
        curator = MagicMock(spec=ContextCurator)
        BaseService.set_context_curator(curator)
        assert BaseService._shared_context_curator is curator

    def test_set_metacognitor(self):
        meta = MagicMock(spec=Metacognitor)
        BaseService.set_metacognitor(meta)
        assert BaseService._shared_metacognitor is meta

    def test_set_epistemic_reasoner(self):
        reasoner = MagicMock(spec=EpistemicReasoner)
        BaseService.set_epistemic_reasoner(reasoner)
        assert BaseService._shared_epistemic_reasoner is reasoner

    def test_set_dialectic_engine(self):
        engine = MagicMock(spec=DialecticEngine)
        BaseService.set_dialectic_engine(engine)
        assert BaseService._shared_dialectic_engine is engine

    def test_set_persistence_engine(self):
        engine = MagicMock(spec=PersistenceEngine)
        BaseService.set_persistence_engine(engine)
        assert BaseService._shared_persistence_engine is engine

    def test_set_insight_crystallizer(self):
        cryst = MagicMock(spec=InsightCrystallizer)
        BaseService.set_insight_crystallizer(cryst)
        assert BaseService._shared_insight_crystallizer is cryst

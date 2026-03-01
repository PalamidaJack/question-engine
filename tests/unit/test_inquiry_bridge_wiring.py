"""Wiring tests for InquiryBridge — bus topics, schemas, readiness fields."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from qe.bus.protocol import TOPICS
from qe.bus.schemas import (
    TOPIC_SCHEMAS,
    BridgeStrategyOutcomePayload,
    validate_payload,
)
from qe.runtime.knowledge_loop import KnowledgeLoop
from qe.runtime.readiness import ReadinessState, reset_readiness

# ── Bus Topic Registration ───────────────────────────────────────────────


class TestBusTopicRegistration:
    def test_bridge_topic_in_topics(self):
        assert "bridge.strategy_outcome_recorded" in TOPICS

    def test_bridge_topic_in_schemas(self):
        assert "bridge.strategy_outcome_recorded" in TOPIC_SCHEMAS

    def test_schema_validates(self):
        payload = validate_payload("bridge.strategy_outcome_recorded", {
            "strategy_name": "breadth_first",
            "goal_id": "g_1",
            "success": True,
            "insights_count": 3,
        })
        assert isinstance(payload, BridgeStrategyOutcomePayload)
        assert payload.strategy_name == "breadth_first"
        assert payload.success is True

    def test_schema_defaults(self):
        payload = BridgeStrategyOutcomePayload(strategy_name="test")
        assert payload.goal_id == ""
        assert payload.success is False
        assert payload.insights_count == 0

    def test_schema_requires_strategy_name(self):
        with pytest.raises(ValidationError):
            BridgeStrategyOutcomePayload()


# ── Readiness Fields ─────────────────────────────────────────────────────


class TestReadinessFields:
    def test_knowledge_loop_ready_field_exists(self):
        state = ReadinessState()
        assert hasattr(state, "knowledge_loop_ready")
        assert state.knowledge_loop_ready is False

    def test_strategy_loop_ready_field_exists(self):
        state = ReadinessState()
        assert hasattr(state, "strategy_loop_ready")
        assert state.strategy_loop_ready is False

    def test_cognitive_layer_ready_field_exists(self):
        state = ReadinessState()
        assert hasattr(state, "cognitive_layer_ready")
        assert state.cognitive_layer_ready is False

    def test_to_dict_has_loops_section(self):
        state = ReadinessState()
        d = state.to_dict()
        assert "loops" in d
        assert "knowledge_loop_ready" in d["loops"]
        assert "strategy_loop_ready" in d["loops"]
        assert "cognitive_layer_ready" in d["loops"]
        assert "inquiry_engine_ready" in d["loops"]

    def test_mark_ready_sets_loop_fields(self):
        state = reset_readiness()
        state.mark_ready("knowledge_loop_ready")
        assert state.knowledge_loop_ready is True
        state.mark_ready("strategy_loop_ready")
        assert state.strategy_loop_ready is True


# ── KnowledgeLoop trigger_consolidation ──────────────────────────────────


class TestTriggerConsolidation:
    @pytest.mark.asyncio
    async def test_trigger_consolidation_runs(self):
        """trigger_consolidation calls _consolidate when running."""
        from qe.runtime.episodic_memory import EpisodicMemory
        from qe.runtime.procedural_memory import ProceduralMemory
        from qe.substrate.bayesian_belief import BayesianBeliefStore

        episodic = AsyncMock(spec=EpisodicMemory)
        episodic.recall = AsyncMock(return_value=[])
        belief = AsyncMock(spec=BayesianBeliefStore)
        belief.get_active_hypotheses = AsyncMock(return_value=[])
        procedural = AsyncMock(spec=ProceduralMemory)
        procedural.get_best_templates = AsyncMock(return_value=[])
        procedural.get_best_sequences = AsyncMock(return_value=[])

        loop = KnowledgeLoop(
            episodic_memory=episodic,
            belief_store=belief,
            procedural_memory=procedural,
            consolidation_interval=9999,
        )
        loop._running = True

        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop.trigger_consolidation()

        episodic.recall.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_consolidation_noop_when_stopped(self):
        """trigger_consolidation is a no-op when loop is not running."""
        from qe.runtime.episodic_memory import EpisodicMemory
        from qe.runtime.procedural_memory import ProceduralMemory
        from qe.substrate.bayesian_belief import BayesianBeliefStore

        episodic = AsyncMock(spec=EpisodicMemory)
        episodic.recall = AsyncMock(return_value=[])
        belief = AsyncMock(spec=BayesianBeliefStore)
        procedural = AsyncMock(spec=ProceduralMemory)

        loop = KnowledgeLoop(
            episodic_memory=episodic,
            belief_store=belief,
            procedural_memory=procedural,
        )
        # Not started, _running is False
        await loop.trigger_consolidation()
        episodic.recall.assert_not_called()

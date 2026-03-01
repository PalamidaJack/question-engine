"""Tests for Phase 4 wiring — bus topics, schemas, and integration."""

from __future__ import annotations

from qe.bus.protocol import TOPICS
from qe.bus.schemas import (
    TOPIC_SCHEMAS,
    PoolHealthCheckPayload,
    PoolScaleExecutedPayload,
    PoolScaleRecommendedPayload,
    StrategyEvaluatedPayload,
    StrategySelectedPayload,
    StrategySwitchRequestedPayload,
    validate_payload,
)


class TestStrategyTopics:
    """Verify all 6 strategy/pool topics are registered."""

    def test_strategy_selected_in_topics(self):
        assert "strategy.selected" in TOPICS

    def test_strategy_switch_requested_in_topics(self):
        assert "strategy.switch_requested" in TOPICS

    def test_strategy_evaluated_in_topics(self):
        assert "strategy.evaluated" in TOPICS

    def test_pool_scale_recommended_in_topics(self):
        assert "pool.scale_recommended" in TOPICS

    def test_pool_scale_executed_in_topics(self):
        assert "pool.scale_executed" in TOPICS

    def test_pool_health_check_in_topics(self):
        assert "pool.health_check" in TOPICS

    def test_topic_count_at_least_105(self):
        assert len(TOPICS) >= 105


class TestStrategySchemas:
    """Verify all 6 schema models construct correctly."""

    def test_strategy_selected_payload(self):
        p = StrategySelectedPayload(
            strategy_name="breadth_first",
            agent_id="a1",
            reason="Thompson sample",
        )
        assert p.strategy_name == "breadth_first"

    def test_strategy_switch_requested_payload(self):
        p = StrategySwitchRequestedPayload(
            from_strategy="breadth_first",
            to_strategy="depth_first",
            reason="low success rate",
        )
        assert p.from_strategy == "breadth_first"
        assert p.to_strategy == "depth_first"

    def test_strategy_evaluated_payload(self):
        p = StrategyEvaluatedPayload(
            strategy_name="hypothesis_driven",
            alpha=5.0,
            beta=2.0,
            sample_count=5,
        )
        assert p.alpha == 5.0

    def test_pool_scale_recommended_payload(self):
        p = PoolScaleRecommendedPayload(
            profile_name="aggressive",
            agents_count=5,
            model_tier="balanced",
            reasoning="high success rate",
        )
        assert p.profile_name == "aggressive"

    def test_pool_scale_executed_payload(self):
        p = PoolScaleExecutedPayload(
            profile_name="minimal",
            agents_before=3,
            agents_after=1,
        )
        assert p.agents_before == 3
        assert p.agents_after == 1

    def test_pool_health_check_payload(self):
        p = PoolHealthCheckPayload(
            total_agents=3,
            active_agents=2,
            avg_success_rate=0.85,
            avg_load_pct=0.4,
        )
        assert p.avg_success_rate == 0.85


class TestSchemaRegistration:
    """Verify schemas are registered in TOPIC_SCHEMAS."""

    def test_strategy_selected_registered(self):
        assert "strategy.selected" in TOPIC_SCHEMAS

    def test_strategy_switch_requested_registered(self):
        assert "strategy.switch_requested" in TOPIC_SCHEMAS

    def test_strategy_evaluated_registered(self):
        assert "strategy.evaluated" in TOPIC_SCHEMAS

    def test_pool_scale_recommended_registered(self):
        assert "pool.scale_recommended" in TOPIC_SCHEMAS

    def test_pool_scale_executed_registered(self):
        assert "pool.scale_executed" in TOPIC_SCHEMAS

    def test_pool_health_check_registered(self):
        assert "pool.health_check" in TOPIC_SCHEMAS


class TestPayloadValidation:
    """Verify payload validation round-trip."""

    def test_validate_strategy_selected(self):
        result = validate_payload("strategy.selected", {
            "strategy_name": "breadth_first",
            "agent_id": "a1",
            "reason": "test",
        })
        assert result is not None
        assert result.strategy_name == "breadth_first"

    def test_validate_pool_health_check(self):
        result = validate_payload("pool.health_check", {
            "total_agents": 5,
            "active_agents": 3,
            "avg_success_rate": 0.9,
            "avg_load_pct": 0.6,
        })
        assert result is not None
        assert result.total_agents == 5

"""Tests for strategy models and defaults."""

from __future__ import annotations

from qe.runtime.strategy_models import (
    DEFAULT_PROFILES,
    DEFAULT_STRATEGIES,
    ScaleProfile,
    StrategyConfig,
    StrategyOutcome,
    StrategySnapshot,
)


class TestStrategyConfig:
    def test_defaults(self):
        cfg = StrategyConfig(name="test")
        assert cfg.question_batch_size == 3
        assert cfg.max_depth == 5
        assert cfg.exploration_rate == 0.2
        assert cfg.preferred_model_tier == "balanced"
        assert cfg.arena_enabled is False

    def test_arena_enabled(self):
        cfg = StrategyConfig(name="arena_test", arena_enabled=True)
        assert cfg.arena_enabled is True

    def test_custom_values(self):
        cfg = StrategyConfig(
            name="custom",
            description="desc",
            question_batch_size=10,
            max_depth=20,
            exploration_rate=0.5,
            preferred_model_tier="fast",
        )
        assert cfg.name == "custom"
        assert cfg.question_batch_size == 10


class TestScaleProfile:
    def test_defaults(self):
        p = ScaleProfile(name="test")
        assert p.min_agents == 1
        assert p.max_agents == 3
        assert p.target_success_rate == 0.7
        assert p.max_cost_per_goal_usd == 1.0

    def test_custom_values(self):
        p = ScaleProfile(
            name="big",
            min_agents=5,
            max_agents=10,
            model_tier="fast",
            target_success_rate=0.9,
            max_cost_per_goal_usd=5.0,
        )
        assert p.max_agents == 10


class TestStrategyOutcome:
    def test_construction(self):
        o = StrategyOutcome(
            strategy_name="depth_first",
            goal_id="g1",
            agent_id="a1",
            success=True,
            duration_s=1.5,
            cost_usd=0.01,
            insights_count=3,
        )
        assert o.success is True
        assert o.insights_count == 3

    def test_serialization_roundtrip(self):
        o = StrategyOutcome(
            strategy_name="breadth_first",
            goal_id="g2",
            success=False,
        )
        data = o.model_dump()
        o2 = StrategyOutcome.model_validate(data)
        assert o2.strategy_name == "breadth_first"
        assert o2.success is False


class TestStrategySnapshot:
    def test_defaults(self):
        s = StrategySnapshot(strategy_name="test")
        assert s.alpha == 1.0
        assert s.beta == 1.0
        assert s.sample_count == 0

    def test_alpha_beta(self):
        s = StrategySnapshot(
            strategy_name="hyp",
            alpha=10.0,
            beta=5.0,
            avg_cost=0.05,
            avg_duration=2.3,
            sample_count=13,
        )
        assert s.alpha == 10.0
        assert s.beta == 5.0


class TestDefaultStrategies:
    def test_strategies_populated(self):
        assert len(DEFAULT_STRATEGIES) == 4

    def test_strategy_names(self):
        expected = {
            "breadth_first",
            "depth_first",
            "hypothesis_driven",
            "iterative_refinement",
        }
        assert set(DEFAULT_STRATEGIES.keys()) == expected

    def test_all_are_strategy_config(self):
        for s in DEFAULT_STRATEGIES.values():
            assert isinstance(s, StrategyConfig)


class TestDefaultProfiles:
    def test_profiles_populated(self):
        assert len(DEFAULT_PROFILES) == 3

    def test_profile_names(self):
        expected = {"minimal", "balanced", "aggressive"}
        assert set(DEFAULT_PROFILES.keys()) == expected

    def test_all_are_scale_profile(self):
        for p in DEFAULT_PROFILES.values():
            assert isinstance(p, ScaleProfile)

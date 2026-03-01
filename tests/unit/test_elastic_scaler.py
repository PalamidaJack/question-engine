"""Tests for ElasticScaler."""

from __future__ import annotations

import pytest

from qe.runtime.cognitive_agent_pool import CognitiveAgentPool
from qe.runtime.strategy_evolver import ElasticScaler
from qe.runtime.strategy_models import DEFAULT_PROFILES, ScaleProfile


class TestProfileRecommendation:
    def test_low_budget_returns_minimal(self):
        scaler = ElasticScaler()
        profile = scaler.recommend_profile(
            pool_stats={"agents": []},
            budget_pct=0.15,
        )
        assert profile.name == "minimal"

    def test_high_success_returns_aggressive(self):
        scaler = ElasticScaler()
        profile = scaler.recommend_profile(
            pool_stats={
                "agents": [
                    {"success_rate": 0.9},
                    {"success_rate": 0.95},
                ]
            },
            budget_pct=0.8,
        )
        assert profile.name == "aggressive"

    def test_default_returns_balanced(self):
        scaler = ElasticScaler()
        profile = scaler.recommend_profile(
            pool_stats={
                "agents": [
                    {"success_rate": 0.7},
                    {"success_rate": 0.6},
                ]
            },
            budget_pct=0.5,
        )
        assert profile.name == "balanced"

    def test_empty_agents_returns_balanced(self):
        scaler = ElasticScaler()
        profile = scaler.recommend_profile(
            pool_stats={"agents": []},
            budget_pct=0.5,
        )
        assert profile.name == "balanced"

    def test_budget_takes_priority(self):
        """Even with high success, low budget should return minimal."""
        scaler = ElasticScaler()
        profile = scaler.recommend_profile(
            pool_stats={
                "agents": [{"success_rate": 0.95}]
            },
            budget_pct=0.1,
        )
        assert profile.name == "minimal"


class TestApplyProfile:
    @pytest.mark.asyncio
    async def test_apply_spawns_agents(self):
        pool = CognitiveAgentPool(max_agents=5)
        scaler = ElasticScaler(agent_pool=pool)
        profile = ScaleProfile(
            name="test",
            max_agents=3,
            model_tier="balanced",
        )
        await scaler.apply_profile(profile)
        assert pool.pool_status()["total_agents"] == 3

    @pytest.mark.asyncio
    async def test_apply_retires_excess(self):
        pool = CognitiveAgentPool(max_agents=5)
        # Spawn 4 agents first
        for _ in range(4):
            await pool.spawn_agent()

        scaler = ElasticScaler(agent_pool=pool)
        profile = ScaleProfile(name="shrink", max_agents=2)
        await scaler.apply_profile(profile)
        assert pool.pool_status()["total_agents"] == 2

    @pytest.mark.asyncio
    async def test_apply_noop_when_at_target(self):
        pool = CognitiveAgentPool(max_agents=5)
        for _ in range(3):
            await pool.spawn_agent()

        scaler = ElasticScaler(agent_pool=pool)
        profile = ScaleProfile(name="match", max_agents=3)
        await scaler.apply_profile(profile)
        assert pool.pool_status()["total_agents"] == 3

    @pytest.mark.asyncio
    async def test_apply_no_pool_noop(self):
        scaler = ElasticScaler(agent_pool=None)
        profile = ScaleProfile(name="test", max_agents=3)
        await scaler.apply_profile(profile)  # Should not raise


class TestCurrentProfile:
    def test_default_profile_is_balanced(self):
        scaler = ElasticScaler()
        assert scaler.current_profile_name() == "balanced"

    @pytest.mark.asyncio
    async def test_profile_tracks_after_apply(self):
        pool = CognitiveAgentPool(max_agents=5)
        scaler = ElasticScaler(agent_pool=pool)
        profile = DEFAULT_PROFILES["minimal"]
        await scaler.apply_profile(profile)
        assert scaler.current_profile_name() == "minimal"

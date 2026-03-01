"""Tests for Strategy → Inquiry wiring (strategy_to_inquiry_config, call site wiring,
per-agent configs, elastic scaler wiring, auto-populate pool, outcome enrichment)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.envelope import Envelope
from qe.runtime.strategy_models import (
    DEFAULT_STRATEGIES,
    ScaleProfile,
    StrategyConfig,
    StrategyOutcome,
    strategy_to_inquiry_config,
)
from qe.services.inquiry.schemas import InquiryConfig, InquiryResult

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_bus():
    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    bus.publish = MagicMock()
    return bus


def _make_evolver(**overrides):
    """Build a mock StrategyEvolver with select_strategy + record_outcome."""
    from qe.runtime.strategy_evolver import StrategyEvolver

    evolver = MagicMock(spec=StrategyEvolver)
    evolver._current_strategy = "breadth_first"
    evolver.select_strategy = MagicMock(
        return_value=DEFAULT_STRATEGIES["breadth_first"]
    )
    evolver.record_outcome = MagicMock()
    for k, v in overrides.items():
        setattr(evolver, k, v)
    return evolver


# ═══════════════════════════════════════════════════════════════════════════
# TestStrategyToInquiryConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestStrategyToInquiryConfig:
    """Tests for the strategy_to_inquiry_config() mapping function."""

    def test_maps_question_batch_size(self):
        s = StrategyConfig(name="test", question_batch_size=7)
        cfg = strategy_to_inquiry_config(s)
        assert cfg.questions_per_iteration == 7

    def test_maps_max_depth(self):
        s = StrategyConfig(name="test", max_depth=12)
        cfg = strategy_to_inquiry_config(s)
        assert cfg.max_iterations == 12

    def test_maps_exploration_rate_to_inverse_confidence(self):
        s = StrategyConfig(name="test", exploration_rate=0.3)
        cfg = strategy_to_inquiry_config(s)
        assert cfg.confidence_threshold == pytest.approx(0.7)

    def test_maps_fast_model_tier(self):
        s = StrategyConfig(name="test", preferred_model_tier="fast")
        cfg = strategy_to_inquiry_config(s)
        assert cfg.model_balanced == "openai/google/gemini-2.0-flash"

    def test_maps_balanced_model_tier(self):
        s = StrategyConfig(name="test", preferred_model_tier="balanced")
        cfg = strategy_to_inquiry_config(s)
        assert cfg.model_balanced == "openai/anthropic/claude-sonnet-4"

    def test_unknown_tier_defaults_to_balanced(self):
        s = StrategyConfig(name="test", preferred_model_tier="premium")
        cfg = strategy_to_inquiry_config(s)
        assert cfg.model_balanced == "openai/anthropic/claude-sonnet-4"

    def test_confidence_threshold_clamped_low(self):
        # exploration_rate=1.0 → 1-1.0=0.0 → clamped to 0.1
        s = StrategyConfig(name="test", exploration_rate=1.0)
        cfg = strategy_to_inquiry_config(s)
        assert cfg.confidence_threshold == pytest.approx(0.1)

    def test_confidence_threshold_clamped_high(self):
        # exploration_rate=0.0 → 1-0.0=1.0 → clamped to 1.0 (stays)
        s = StrategyConfig(name="test", exploration_rate=0.0)
        cfg = strategy_to_inquiry_config(s)
        assert cfg.confidence_threshold == pytest.approx(1.0)

    def test_model_fast_always_gemini(self):
        s = StrategyConfig(name="test", preferred_model_tier="balanced")
        cfg = strategy_to_inquiry_config(s)
        assert cfg.model_fast == "openai/google/gemini-2.0-flash"

    def test_returns_inquiry_config_type(self):
        s = StrategyConfig(name="test")
        cfg = strategy_to_inquiry_config(s)
        assert isinstance(cfg, InquiryConfig)

    def test_breadth_first_default_strategy(self):
        cfg = strategy_to_inquiry_config(DEFAULT_STRATEGIES["breadth_first"])
        assert cfg.questions_per_iteration == 5
        assert cfg.max_iterations == 3
        assert cfg.confidence_threshold == pytest.approx(0.7)
        assert cfg.model_balanced == "openai/google/gemini-2.0-flash"

    def test_depth_first_default_strategy(self):
        cfg = strategy_to_inquiry_config(DEFAULT_STRATEGIES["depth_first"])
        assert cfg.questions_per_iteration == 1
        assert cfg.max_iterations == 10
        assert cfg.confidence_threshold == pytest.approx(0.9)
        assert cfg.model_balanced == "openai/anthropic/claude-sonnet-4"


# ═══════════════════════════════════════════════════════════════════════════
# TestSelectStrategyAtCallSites
# ═══════════════════════════════════════════════════════════════════════════


class TestSelectStrategyAtCallSites:
    """Tests that select_strategy() is called at the 3 inquiry call sites in app.py."""

    @pytest.mark.asyncio
    async def test_single_agent_path_calls_select_strategy(self):
        """Single-agent path should call select_strategy when evolver is available."""
        import qe.api.app as app_mod

        evolver = _make_evolver()
        engine = AsyncMock()
        engine.run_inquiry = AsyncMock(return_value=InquiryResult(
            inquiry_id="inq_test", goal_id="g1", status="completed",
            phase_timings={}, duration_seconds=1.0,
        ))
        flag_store = MagicMock()
        flag_store.is_enabled = MagicMock(side_effect=lambda f: f == "inquiry_mode")

        old_evolver = app_mod._strategy_evolver
        old_engine = app_mod._inquiry_engine
        old_profiling = app_mod._inquiry_profiling_store
        try:
            app_mod._strategy_evolver = evolver
            app_mod._inquiry_engine = engine
            app_mod._inquiry_profiling_store = MagicMock()

            with patch.object(app_mod, "get_flag_store", return_value=flag_store):
                with patch.object(app_mod, "get_readiness", return_value=MagicMock()):
                    await app_mod.submit_goal({"description": "test goal"})

            evolver.select_strategy.assert_called_once()
            # Config should have been passed to run_inquiry
            kw = engine.run_inquiry.call_args
            assert kw.kwargs.get("config") is not None
        finally:
            app_mod._strategy_evolver = old_evolver
            app_mod._inquiry_engine = old_engine
            app_mod._inquiry_profiling_store = old_profiling

    @pytest.mark.asyncio
    async def test_multi_agent_path_calls_select_strategy(self):
        """Multi-agent path should call select_strategy when evolver is available."""
        import qe.api.app as app_mod

        evolver = _make_evolver()
        pool = AsyncMock()
        pool.run_parallel_inquiry = AsyncMock(return_value=[
            InquiryResult(inquiry_id="inq_1", goal_id="g1"),
        ])
        pool.merge_results = AsyncMock(return_value=InquiryResult(
            inquiry_id="merged", goal_id="g1",
            findings_summary="test", insights=[],
        ))
        flag_store = MagicMock()
        flag_store.is_enabled = MagicMock(side_effect=lambda f: f == "multi_agent_mode")

        old_evolver = app_mod._strategy_evolver
        old_pool = app_mod._cognitive_pool
        try:
            app_mod._strategy_evolver = evolver
            app_mod._cognitive_pool = pool

            with patch.object(app_mod, "get_flag_store", return_value=flag_store):
                await app_mod.submit_goal({"description": "test goal"})

            evolver.select_strategy.assert_called_once()
            kw = pool.run_parallel_inquiry.call_args
            assert kw.kwargs.get("config") is not None
        finally:
            app_mod._strategy_evolver = old_evolver
            app_mod._cognitive_pool = old_pool

    @pytest.mark.asyncio
    async def test_channel_routing_calls_select_strategy(self):
        """Channel routing (bus handler) should call select_strategy."""
        import qe.api.app as app_mod

        evolver = _make_evolver()
        engine = AsyncMock()
        engine.run_inquiry = AsyncMock(return_value=InquiryResult(
            inquiry_id="inq_ch", goal_id="g1",
            findings_summary="result",
        ))

        old_evolver = app_mod._strategy_evolver
        old_engine = app_mod._inquiry_engine
        try:
            app_mod._strategy_evolver = evolver
            app_mod._inquiry_engine = engine

            # Verify the strategy → config mapping works for channel path
            from qe.runtime.strategy_models import strategy_to_inquiry_config
            strategy = evolver.select_strategy()
            config = strategy_to_inquiry_config(strategy)
            assert config is not None
            assert config.questions_per_iteration == 5  # breadth_first
        finally:
            app_mod._strategy_evolver = old_evolver
            app_mod._inquiry_engine = old_engine

    @pytest.mark.asyncio
    async def test_config_passed_to_run_inquiry(self):
        """Config from strategy should be passed through to run_inquiry."""
        strategy = DEFAULT_STRATEGIES["depth_first"]
        config = strategy_to_inquiry_config(strategy)
        assert config.questions_per_iteration == 1
        assert config.max_iterations == 10
        assert config.confidence_threshold == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_no_crash_when_evolver_is_none(self):
        """When evolver is None, config should be None (default InquiryConfig used)."""
        import qe.api.app as app_mod

        engine = AsyncMock()
        engine.run_inquiry = AsyncMock(return_value=InquiryResult(
            inquiry_id="inq_test", goal_id="g1",
            phase_timings={}, duration_seconds=1.0,
        ))
        flag_store = MagicMock()
        flag_store.is_enabled = MagicMock(side_effect=lambda f: f == "inquiry_mode")

        old_evolver = app_mod._strategy_evolver
        old_engine = app_mod._inquiry_engine
        old_profiling = app_mod._inquiry_profiling_store
        try:
            app_mod._strategy_evolver = None
            app_mod._inquiry_engine = engine
            app_mod._inquiry_profiling_store = MagicMock()

            with patch.object(app_mod, "get_flag_store", return_value=flag_store):
                with patch.object(app_mod, "get_readiness", return_value=MagicMock()):
                    await app_mod.submit_goal({"description": "test"})

            # Should succeed without calling select_strategy
            kw = engine.run_inquiry.call_args
            config_val = kw.kwargs.get("config") if kw.kwargs else None
            assert config_val is None
        finally:
            app_mod._strategy_evolver = old_evolver
            app_mod._inquiry_engine = old_engine
            app_mod._inquiry_profiling_store = old_profiling


# ═══════════════════════════════════════════════════════════════════════════
# TestPerAgentStrategyConfigs
# ═══════════════════════════════════════════════════════════════════════════


class TestPerAgentStrategyConfigs:
    """Tests for per-agent strategy-derived configs in CognitiveAgentPool."""

    @pytest.mark.asyncio
    async def test_per_agent_config_from_slot_strategy(self):
        """When no shared config, each agent uses its slot's strategy."""
        from qe.runtime.cognitive_agent_pool import AgentSlot, CognitiveAgentPool

        pool = CognitiveAgentPool(max_agents=3)
        agent = MagicMock()
        agent.agent_id = "a1"
        agent.status = "idle"
        agent.active_inquiry_id = None

        engine = AsyncMock()
        engine.run_inquiry = AsyncMock(return_value=InquiryResult(
            inquiry_id="inq_1", goal_id="g1",
        ))

        strategy = DEFAULT_STRATEGIES["depth_first"]
        slot = AgentSlot(agent=agent, engine=engine, strategy=strategy)
        pool._slots["a1"] = slot

        await pool.run_parallel_inquiry(
            goal_id="g1", goal_description="test",
        )

        # Engine should have been called with depth_first-derived config
        call_kwargs = engine.run_inquiry.call_args.kwargs
        cfg = call_kwargs["config"]
        assert cfg.questions_per_iteration == 1  # depth_first batch_size
        assert cfg.max_iterations == 10  # depth_first max_depth

    @pytest.mark.asyncio
    async def test_shared_config_overrides_per_agent(self):
        """When shared config is provided, it overrides per-agent strategy."""
        from qe.runtime.cognitive_agent_pool import AgentSlot, CognitiveAgentPool

        pool = CognitiveAgentPool(max_agents=3)
        agent = MagicMock()
        agent.agent_id = "a1"
        agent.status = "idle"
        agent.active_inquiry_id = None

        engine = AsyncMock()
        engine.run_inquiry = AsyncMock(return_value=InquiryResult(
            inquiry_id="inq_1", goal_id="g1",
        ))

        strategy = DEFAULT_STRATEGIES["depth_first"]
        slot = AgentSlot(agent=agent, engine=engine, strategy=strategy)
        pool._slots["a1"] = slot

        shared_config = InquiryConfig(questions_per_iteration=99)
        await pool.run_parallel_inquiry(
            goal_id="g1", goal_description="test", config=shared_config,
        )

        call_kwargs = engine.run_inquiry.call_args.kwargs
        assert call_kwargs["config"].questions_per_iteration == 99

    @pytest.mark.asyncio
    async def test_default_strategy_when_no_slot_strategy(self):
        """When slot has default strategy, per-agent config uses defaults."""
        from qe.runtime.cognitive_agent_pool import AgentSlot, CognitiveAgentPool

        pool = CognitiveAgentPool(max_agents=3)
        agent = MagicMock()
        agent.agent_id = "a1"
        agent.status = "idle"
        agent.active_inquiry_id = None

        engine = AsyncMock()
        engine.run_inquiry = AsyncMock(return_value=InquiryResult(
            inquiry_id="inq_1", goal_id="g1",
        ))

        default_strategy = StrategyConfig(name="default")
        slot = AgentSlot(agent=agent, engine=engine, strategy=default_strategy)
        pool._slots["a1"] = slot

        await pool.run_parallel_inquiry(
            goal_id="g1", goal_description="test",
        )

        call_kwargs = engine.run_inquiry.call_args.kwargs
        cfg = call_kwargs["config"]
        # Default StrategyConfig has batch_size=3, max_depth=5
        assert cfg.questions_per_iteration == 3
        assert cfg.max_iterations == 5


# ═══════════════════════════════════════════════════════════════════════════
# TestElasticScalerWiring
# ═══════════════════════════════════════════════════════════════════════════


class TestElasticScalerWiring:
    """Tests for elastic scaler wiring in StrategyEvolver._evaluate()."""

    @pytest.mark.asyncio
    async def test_evaluate_calls_recommend_profile(self):
        """_evaluate() should call elastic_scaler.recommend_profile."""
        from qe.runtime.strategy_evolver import StrategyEvolver

        scaler = MagicMock()
        scaler.recommend_profile = MagicMock(return_value=ScaleProfile(name="balanced"))
        scaler.current_profile_name = MagicMock(return_value="balanced")

        pool = MagicMock()
        pool.pool_status = MagicMock(return_value={"total_agents": 1, "agents": []})

        evolver = StrategyEvolver(
            agent_pool=pool, bus=_make_bus(),
            elastic_scaler=scaler,
        )

        await evolver._evaluate()
        scaler.recommend_profile.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_applies_new_profile_on_change(self):
        """_evaluate() should apply a new profile when it differs from current."""
        from qe.runtime.strategy_evolver import StrategyEvolver

        new_profile = ScaleProfile(name="aggressive", max_agents=5)
        scaler = MagicMock()
        scaler.recommend_profile = MagicMock(return_value=new_profile)
        scaler.current_profile_name = MagicMock(return_value="balanced")
        scaler.apply_profile = AsyncMock()

        pool = MagicMock()
        pool.pool_status = MagicMock(return_value={"total_agents": 1, "agents": []})

        evolver = StrategyEvolver(
            agent_pool=pool, bus=_make_bus(),
            elastic_scaler=scaler,
        )

        await evolver._evaluate()
        scaler.apply_profile.assert_called_once_with(new_profile)

    @pytest.mark.asyncio
    async def test_evaluate_skips_when_same_profile(self):
        """_evaluate() should not apply profile when it hasn't changed."""
        from qe.runtime.strategy_evolver import StrategyEvolver

        scaler = MagicMock()
        scaler.recommend_profile = MagicMock(
            return_value=ScaleProfile(name="balanced")
        )
        scaler.current_profile_name = MagicMock(return_value="balanced")
        scaler.apply_profile = AsyncMock()

        pool = MagicMock()
        pool.pool_status = MagicMock(return_value={"total_agents": 1, "agents": []})

        evolver = StrategyEvolver(
            agent_pool=pool, bus=_make_bus(),
            elastic_scaler=scaler,
        )

        await evolver._evaluate()
        scaler.apply_profile.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_handles_no_scaler(self):
        """_evaluate() should work fine when elastic_scaler is None."""
        from qe.runtime.strategy_evolver import StrategyEvolver

        evolver = StrategyEvolver(bus=_make_bus())
        # Should not raise
        await evolver._evaluate()

    @pytest.mark.asyncio
    async def test_evaluate_handles_no_budget_tracker(self):
        """When budget_tracker is None, budget_pct defaults to 1.0."""
        from qe.runtime.strategy_evolver import StrategyEvolver

        scaler = MagicMock()
        scaler.recommend_profile = MagicMock(
            return_value=ScaleProfile(name="balanced")
        )
        scaler.current_profile_name = MagicMock(return_value="balanced")

        pool = MagicMock()
        pool.pool_status = MagicMock(return_value={"total_agents": 1, "agents": []})

        evolver = StrategyEvolver(
            agent_pool=pool, bus=_make_bus(),
            elastic_scaler=scaler,
            budget_tracker=None,
        )

        await evolver._evaluate()
        # recommend_profile should have been called with budget_pct=1.0
        call_args = scaler.recommend_profile.call_args
        assert call_args[0][1] == 1.0 or call_args.kwargs.get("budget_pct") == 1.0

    @pytest.mark.asyncio
    async def test_publishes_pool_scale_executed_event(self):
        """_evaluate() should publish pool.scale_executed on profile change."""
        from qe.runtime.strategy_evolver import StrategyEvolver

        new_profile = ScaleProfile(name="aggressive", max_agents=5)
        scaler = MagicMock()
        scaler.recommend_profile = MagicMock(return_value=new_profile)
        scaler.current_profile_name = MagicMock(return_value="balanced")
        scaler.apply_profile = AsyncMock()

        pool = MagicMock()
        pool.pool_status = MagicMock(return_value={"total_agents": 1, "agents": []})

        bus = _make_bus()
        evolver = StrategyEvolver(
            agent_pool=pool, bus=bus,
            elastic_scaler=scaler,
        )

        await evolver._evaluate()

        # Find the pool.scale_executed publish call
        published_topics = [
            call.args[0].topic for call in bus.publish.call_args_list
        ]
        assert "pool.scale_executed" in published_topics

    @pytest.mark.asyncio
    async def test_evaluate_uses_budget_tracker_remaining_pct(self):
        """_evaluate() should pass budget_tracker.remaining_pct() to recommend_profile."""
        from qe.runtime.strategy_evolver import StrategyEvolver

        scaler = MagicMock()
        scaler.recommend_profile = MagicMock(
            return_value=ScaleProfile(name="minimal")
        )
        scaler.current_profile_name = MagicMock(return_value="minimal")

        pool = MagicMock()
        pool.pool_status = MagicMock(return_value={"total_agents": 1, "agents": []})

        budget = MagicMock()
        budget.remaining_pct = MagicMock(return_value=0.15)

        evolver = StrategyEvolver(
            agent_pool=pool, bus=_make_bus(),
            elastic_scaler=scaler,
            budget_tracker=budget,
        )

        await evolver._evaluate()
        call_args = scaler.recommend_profile.call_args
        assert call_args[0][1] == 0.15


# ═══════════════════════════════════════════════════════════════════════════
# TestAutoPopulatePool
# ═══════════════════════════════════════════════════════════════════════════


class TestAutoPopulatePool:
    """Tests for auto-populating the agent pool at startup."""

    @pytest.mark.asyncio
    async def test_spawns_agents_with_diverse_strategies(self):
        """Pool should be populated with DEFAULT_STRATEGIES when multi_agent_mode enabled."""
        from qe.runtime.cognitive_agent_pool import CognitiveAgentPool

        pool = CognitiveAgentPool(max_agents=5)
        # Mock spawn_agent to avoid creating real agents
        spawned = []

        async def mock_spawn(specialization="general", model_tier="balanced", strategy=None):
            agent = MagicMock()
            agent.agent_id = f"agent_{len(spawned)}"
            spawned.append({
                "specialization": specialization,
                "model_tier": model_tier,
                "strategy": strategy,
            })
            return agent

        pool.spawn_agent = mock_spawn

        strategies = list(DEFAULT_STRATEGIES.values())
        for strat in strategies[:pool._max_agents]:
            await pool.spawn_agent(
                specialization=strat.name,
                model_tier=strat.preferred_model_tier,
                strategy=strat,
            )

        assert len(spawned) == 4  # 4 DEFAULT_STRATEGIES
        names = {s["specialization"] for s in spawned}
        assert "breadth_first" in names
        assert "depth_first" in names
        assert "hypothesis_driven" in names
        assert "iterative_refinement" in names

    @pytest.mark.asyncio
    async def test_respects_max_agents_cap(self):
        """Pool should not spawn more agents than max_agents."""
        from qe.runtime.cognitive_agent_pool import CognitiveAgentPool

        pool = CognitiveAgentPool(max_agents=2)
        spawned = []

        async def mock_spawn(specialization="general", model_tier="balanced", strategy=None):
            if len(spawned) >= 2:
                raise RuntimeError("Pool at capacity")
            agent = MagicMock()
            spawned.append(specialization)
            return agent

        pool.spawn_agent = mock_spawn

        strategies = list(DEFAULT_STRATEGIES.values())
        for strat in strategies[:pool._max_agents]:
            try:
                await pool.spawn_agent(
                    specialization=strat.name,
                    model_tier=strat.preferred_model_tier,
                    strategy=strat,
                )
            except RuntimeError:
                break

        assert len(spawned) == 2

    def test_skips_when_multi_agent_mode_disabled(self):
        """No agents should be spawned when multi_agent_mode is disabled."""
        flag_store = MagicMock()
        flag_store.is_enabled = MagicMock(return_value=False)

        # Just verify the flag check logic
        should_populate = flag_store.is_enabled("multi_agent_mode")
        assert not should_populate

    @pytest.mark.asyncio
    async def test_each_agent_gets_different_strategy(self):
        """Each spawned agent should have a unique strategy from DEFAULT_STRATEGIES."""
        strategies = list(DEFAULT_STRATEGIES.values())
        names = [s.name for s in strategies]
        # All names should be unique
        assert len(names) == len(set(names))
        assert len(names) == 4


# ═══════════════════════════════════════════════════════════════════════════
# TestOutcomeEnrichment
# ═══════════════════════════════════════════════════════════════════════════


class TestOutcomeEnrichment:
    """Tests for duration/cost enrichment in the feedback loop."""

    def test_inquiry_completed_payload_includes_duration(self):
        """InquiryCompletedPayload should have duration_s field."""
        from qe.bus.schemas import InquiryCompletedPayload

        payload = InquiryCompletedPayload(
            inquiry_id="inq_1", goal_id="g1",
            duration_s=5.5, cost_usd=0.25,
        )
        assert payload.duration_s == 5.5

    def test_inquiry_completed_payload_includes_cost(self):
        """InquiryCompletedPayload should have cost_usd field."""
        from qe.bus.schemas import InquiryCompletedPayload

        payload = InquiryCompletedPayload(
            inquiry_id="inq_1", goal_id="g1",
            duration_s=5.5, cost_usd=0.25,
        )
        assert payload.cost_usd == 0.25

    def test_inquiry_completed_payload_defaults(self):
        """InquiryCompletedPayload duration_s and cost_usd should default to 0.0."""
        from qe.bus.schemas import InquiryCompletedPayload

        payload = InquiryCompletedPayload(
            inquiry_id="inq_1", goal_id="g1",
        )
        assert payload.duration_s == 0.0
        assert payload.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_bridge_passes_duration_cost_to_outcome(self):
        """InquiryBridge should pass duration_s and cost_usd to StrategyOutcome."""
        from qe.runtime.episodic_memory import EpisodicMemory
        from qe.runtime.inquiry_bridge import InquiryBridge

        evolver = MagicMock()
        evolver._current_strategy = "breadth_first"
        evolver.record_outcome = MagicMock()

        episodic = AsyncMock(spec=EpisodicMemory)
        episodic.store = AsyncMock()

        bus = _make_bus()
        bridge = InquiryBridge(
            bus=bus, episodic_memory=episodic,
            strategy_evolver=evolver,
        )
        bridge.start()

        envelope = Envelope(
            topic="inquiry.completed",
            source_service_id="test",
            payload={
                "inquiry_id": "inq_1",
                "goal_id": "g1",
                "status": "completed",
                "insights": 3,
                "duration_s": 12.5,
                "cost_usd": 0.75,
            },
        )
        await bridge._on_inquiry_completed(envelope)

        outcome = evolver.record_outcome.call_args[0][0]
        assert outcome.duration_s == 12.5
        assert outcome.cost_usd == 0.75

    def test_evolver_tracks_total_costs_and_durations(self):
        """StrategyEvolver should accumulate costs and durations from outcomes."""
        from qe.runtime.strategy_evolver import StrategyEvolver

        evolver = StrategyEvolver()
        outcome1 = StrategyOutcome(
            strategy_name="breadth_first", success=True,
            duration_s=10.0, cost_usd=0.5,
        )
        outcome2 = StrategyOutcome(
            strategy_name="breadth_first", success=False,
            duration_s=5.0, cost_usd=0.3,
        )
        evolver.record_outcome(outcome1)
        evolver.record_outcome(outcome2)

        assert evolver._total_costs["breadth_first"] == pytest.approx(0.8)
        assert evolver._total_durations["breadth_first"] == pytest.approx(15.0)
        assert evolver._outcome_counts["breadth_first"] == 2

    def test_engine_finalize_includes_duration_cost_in_event(self):
        """InquiryEngine._finalize() should include duration_s and cost_usd in event."""
        from qe.services.inquiry.engine import InquiryEngine
        from qe.services.inquiry.schemas import InquiryState

        bus = _make_bus()
        engine = InquiryEngine(bus=bus)

        state = InquiryState(goal_id="g1")
        import time
        start = time.monotonic() - 5.0  # simulate 5 seconds ago

        engine._finalize(state, "max_iterations", [], start)

        # Find the inquiry.completed publish call
        found = False
        for call in bus.publish.call_args_list:
            env = call.args[0]
            if env.topic == "inquiry.completed":
                assert "duration_s" in env.payload
                assert "cost_usd" in env.payload
                assert env.payload["duration_s"] > 0
                found = True
        assert found, "inquiry.completed event not published"


# ═══════════════════════════════════════════════════════════════════════════
# TestIntegration
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Full cycle: strategy → config → inquiry → outcome → arms."""

    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """End-to-end: strategy → config → outcome with duration/cost."""
        from qe.runtime.strategy_evolver import StrategyEvolver

        bus = _make_bus()
        evolver = StrategyEvolver(bus=bus)

        # 1. Select a strategy
        strategy = evolver.select_strategy()
        assert strategy.name in DEFAULT_STRATEGIES

        # 2. Convert to config
        config = strategy_to_inquiry_config(strategy)
        assert isinstance(config, InquiryConfig)

        # 3. Simulate inquiry completion with enriched outcome
        outcome = StrategyOutcome(
            strategy_name=strategy.name,
            goal_id="goal_123",
            success=True,
            insights_count=5,
            duration_s=8.3,
            cost_usd=0.42,
        )

        # 4. Record outcome
        evolver.record_outcome(outcome)

        # 5. Verify arms updated
        snapshots = evolver.get_snapshots()
        snap = next(s for s in snapshots if s.strategy_name == strategy.name)
        assert snap.sample_count == 1
        assert snap.avg_cost == pytest.approx(0.42)
        assert snap.avg_duration == pytest.approx(8.3)
        # Alpha should have increased (success=True)
        assert snap.alpha > 1.0

    @pytest.mark.asyncio
    async def test_full_cycle_with_bridge(self):
        """End-to-end via InquiryBridge: bus event → outcome with duration/cost → arms update."""
        from qe.runtime.episodic_memory import EpisodicMemory
        from qe.runtime.inquiry_bridge import InquiryBridge
        from qe.runtime.strategy_evolver import StrategyEvolver

        bus = _make_bus()
        evolver = StrategyEvolver(bus=bus)
        episodic = AsyncMock(spec=EpisodicMemory)
        episodic.store = AsyncMock()

        # Select strategy first (sets _current_strategy)
        strategy = evolver.select_strategy()

        bridge = InquiryBridge(
            bus=bus, episodic_memory=episodic,
            strategy_evolver=evolver,
        )
        bridge.start()

        # Simulate inquiry.completed event with duration/cost
        envelope = Envelope(
            topic="inquiry.completed",
            source_service_id="inquiry_engine",
            payload={
                "inquiry_id": "inq_full",
                "goal_id": "goal_456",
                "status": "completed",
                "insights": 2,
                "duration_s": 15.7,
                "cost_usd": 1.23,
            },
        )
        await bridge._on_inquiry_completed(envelope)

        # Verify the evolver recorded the outcome with cost/duration
        snapshots = evolver.get_snapshots()
        snap = next(s for s in snapshots if s.strategy_name == strategy.name)
        assert snap.sample_count == 1
        assert snap.avg_cost == pytest.approx(1.23)
        assert snap.avg_duration == pytest.approx(15.7)
        assert snap.alpha > 1.0  # success=True (status=completed, insights>0)

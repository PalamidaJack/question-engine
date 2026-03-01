"""Tests for StrategyEvolver."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qe.runtime.strategy_evolver import StrategyEvolver
from qe.runtime.strategy_models import (
    DEFAULT_STRATEGIES,
    StrategyConfig,
    StrategyOutcome,
)


class TestStrategySelection:
    def test_select_returns_strategy_config(self):
        evolver = StrategyEvolver()
        result = evolver.select_strategy()
        assert isinstance(result, StrategyConfig)
        assert result.name in DEFAULT_STRATEGIES

    def test_known_priors_pick_winner(self):
        """With heavily biased priors, Thompson should pick the winner."""
        strategies = {
            "good": StrategyConfig(name="good"),
            "bad": StrategyConfig(name="bad"),
        }
        evolver = StrategyEvolver(strategies=strategies)
        # Give "good" lots of successes
        for _ in range(50):
            evolver._arms["good"].update(True)
        # Give "bad" lots of failures
        for _ in range(50):
            evolver._arms["bad"].update(False)

        wins = {"good": 0, "bad": 0}
        for _ in range(100):
            result = evolver.select_strategy()
            wins[result.name] += 1

        assert wins["good"] > 80

    def test_exploration_with_uniform_priors(self):
        """With uniform priors, both strategies should be picked sometimes."""
        strategies = {
            "a": StrategyConfig(name="a"),
            "b": StrategyConfig(name="b"),
        }
        evolver = StrategyEvolver(strategies=strategies)

        picks = {"a": 0, "b": 0}
        for _ in range(100):
            result = evolver.select_strategy()
            picks[result.name] += 1

        # Both should be picked at least a few times
        assert picks["a"] > 10
        assert picks["b"] > 10


class TestOutcomeRecording:
    def test_record_success_updates_alpha(self):
        evolver = StrategyEvolver()
        outcome = StrategyOutcome(
            strategy_name="breadth_first",
            success=True,
            cost_usd=0.01,
            duration_s=1.5,
        )
        evolver.record_outcome(outcome)
        arm = evolver._arms["breadth_first"]
        assert arm.alpha == 2.0
        assert arm.beta == 1.0

    def test_record_failure_updates_beta(self):
        evolver = StrategyEvolver()
        outcome = StrategyOutcome(
            strategy_name="depth_first",
            success=False,
            cost_usd=0.02,
            duration_s=2.0,
        )
        evolver.record_outcome(outcome)
        arm = evolver._arms["depth_first"]
        assert arm.alpha == 1.0
        assert arm.beta == 2.0

    def test_record_unknown_strategy_creates_arm(self):
        evolver = StrategyEvolver()
        outcome = StrategyOutcome(
            strategy_name="new_strategy",
            success=True,
        )
        evolver.record_outcome(outcome)
        assert "new_strategy" in evolver._arms
        assert evolver._arms["new_strategy"].alpha == 2.0

    def test_cost_duration_tracking(self):
        evolver = StrategyEvolver()
        evolver.record_outcome(StrategyOutcome(
            strategy_name="breadth_first",
            success=True,
            cost_usd=0.10,
            duration_s=5.0,
        ))
        evolver.record_outcome(StrategyOutcome(
            strategy_name="breadth_first",
            success=False,
            cost_usd=0.20,
            duration_s=3.0,
        ))
        assert evolver._outcome_counts["breadth_first"] == 2
        assert evolver._total_costs["breadth_first"] == pytest.approx(0.30)
        assert evolver._total_durations["breadth_first"] == pytest.approx(8.0)


class TestSnapshots:
    def test_initial_snapshots(self):
        evolver = StrategyEvolver()
        snaps = evolver.get_snapshots()
        assert len(snaps) == len(DEFAULT_STRATEGIES)
        for s in snaps:
            assert s.alpha == 1.0
            assert s.beta == 1.0
            assert s.sample_count == 0

    def test_snapshots_after_outcomes(self):
        evolver = StrategyEvolver()
        evolver.record_outcome(StrategyOutcome(
            strategy_name="breadth_first",
            success=True,
            cost_usd=0.10,
            duration_s=2.0,
        ))
        snaps = evolver.get_snapshots()
        bf_snap = next(s for s in snaps if s.strategy_name == "breadth_first")
        assert bf_snap.alpha == 2.0
        assert bf_snap.sample_count == 1
        assert bf_snap.avg_cost == pytest.approx(0.10)
        assert bf_snap.avg_duration == pytest.approx(2.0)


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        evolver = StrategyEvolver(update_interval_s=0.05)
        await evolver.start()
        assert evolver._running is True
        assert evolver._loop_task is not None
        await evolver.stop()
        assert evolver._running is False

    @pytest.mark.asyncio
    async def test_double_start_noop(self):
        evolver = StrategyEvolver(update_interval_s=0.05)
        await evolver.start()
        task1 = evolver._loop_task
        await evolver.start()  # should not create a new task
        assert evolver._loop_task is task1
        await evolver.stop()

    @pytest.mark.asyncio
    async def test_evaluation_publishes_events(self):
        bus = MagicMock()
        evolver = StrategyEvolver(bus=bus, update_interval_s=0.05)
        await evolver._evaluate()
        assert bus.publish.called

    @pytest.mark.asyncio
    async def test_strategy_switch_on_low_performance(self):
        bus = MagicMock()
        evolver = StrategyEvolver(bus=bus, success_threshold=0.5)
        evolver._current_strategy = "breadth_first"
        # Make breadth_first have poor performance
        for _ in range(10):
            evolver._arms["breadth_first"].update(False)

        await evolver._evaluate()
        # Should have published strategy.switch_requested
        calls = list(bus.publish.call_args_list)
        topics = [c[0][0].topic for c in calls]
        assert "strategy.switch_requested" in topics

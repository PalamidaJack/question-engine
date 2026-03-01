"""StrategyEvolver + ElasticScaler — Phase 4 strategy loop.

StrategyEvolver monitors agent performance and selects inquiry
strategies via Thompson sampling. ElasticScaler auto-scales
the agent pool based on budget and workload.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from qe.runtime.routing_optimizer import BetaArm
from qe.runtime.strategy_models import (
    DEFAULT_PROFILES,
    DEFAULT_STRATEGIES,
    ScaleProfile,
    StrategyConfig,
    StrategyOutcome,
    StrategySnapshot,
)

log = logging.getLogger(__name__)


class StrategyEvolver:
    """Monitors agents and selects strategies via Thompson sampling.

    The outermost of the three nested loops, operating on the
    hours-days timescale.
    """

    def __init__(
        self,
        agent_pool: Any = None,
        procedural_memory: Any = None,
        bus: Any = None,
        strategies: dict[str, StrategyConfig] | None = None,
        update_interval_s: float = 60.0,
        success_threshold: float = 0.5,
    ) -> None:
        strats = strategies or DEFAULT_STRATEGIES
        self._strategies = dict(strats)
        self._arms: dict[str, BetaArm] = {
            name: BetaArm() for name in self._strategies
        }
        self._agent_pool = agent_pool
        self._procedural = procedural_memory
        self._bus = bus
        self._update_interval_s = update_interval_s
        self._success_threshold = success_threshold
        self._running = False
        self._loop_task: asyncio.Task[Any] | None = None
        self._current_strategy: str | None = None

        # Outcome tracking for cost/duration averaging
        self._outcome_counts: dict[str, int] = dict.fromkeys(
            self._strategies, 0
        )
        self._total_costs: dict[str, float] = dict.fromkeys(
            self._strategies, 0.0
        )
        self._total_durations: dict[str, float] = dict.fromkeys(
            self._strategies, 0.0
        )

    def select_strategy(self) -> StrategyConfig:
        """Select a strategy using Thompson sampling.

        Samples from BetaArm for each strategy and returns
        the one with the highest sampled value.
        """
        best_name: str | None = None
        best_sample = -1.0

        for name, arm in self._arms.items():
            sample = arm.sample()
            if sample > best_sample:
                best_sample = sample
                best_name = name

        assert best_name is not None
        self._current_strategy = best_name

        self._publish("strategy.selected", {
            "strategy_name": best_name,
            "agent_id": "",
            "reason": f"Thompson sample={best_sample:.3f}",
        })

        return self._strategies[best_name]

    def record_outcome(self, outcome: StrategyOutcome) -> None:
        """Update the arm's alpha/beta based on outcome."""
        name = outcome.strategy_name
        if name not in self._arms:
            self._arms[name] = BetaArm()
            self._outcome_counts[name] = 0
            self._total_costs[name] = 0.0
            self._total_durations[name] = 0.0

        self._arms[name].update(outcome.success)
        self._outcome_counts[name] += 1
        self._total_costs[name] += outcome.cost_usd
        self._total_durations[name] += outcome.duration_s

    def get_snapshots(self) -> list[StrategySnapshot]:
        """Return Thompson arm state for all strategies."""
        snapshots = []
        for name, arm in self._arms.items():
            count = self._outcome_counts.get(name, 0)
            avg_cost = (
                self._total_costs.get(name, 0.0) / count
                if count > 0
                else 0.0
            )
            avg_duration = (
                self._total_durations.get(name, 0.0) / count
                if count > 0
                else 0.0
            )
            snapshots.append(
                StrategySnapshot(
                    strategy_name=name,
                    alpha=arm.alpha,
                    beta=arm.beta,
                    avg_cost=avg_cost,
                    avg_duration=avg_duration,
                    sample_count=count,
                )
            )
        return snapshots

    async def start(self) -> None:
        """Start the background evaluation loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._evaluation_loop())
        log.info("strategy_evolver.started")

    async def stop(self) -> None:
        """Stop the background evaluation loop."""
        self._running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass
        self._loop_task = None
        log.info("strategy_evolver.stopped")

    async def _evaluation_loop(self) -> None:
        """Periodic loop: check pool health, evaluate strategies."""
        while self._running:
            try:
                await asyncio.sleep(self._update_interval_s)
                await self._evaluate()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("strategy_evolver.evaluation_error")

    async def _evaluate(self) -> None:
        """Single evaluation cycle."""
        snapshots = self.get_snapshots()

        for snap in snapshots:
            self._publish("strategy.evaluated", {
                "strategy_name": snap.strategy_name,
                "alpha": snap.alpha,
                "beta": snap.beta,
                "sample_count": snap.sample_count,
            })

        # Check if current strategy needs switching
        if self._current_strategy and self._current_strategy in self._arms:
            arm = self._arms[self._current_strategy]
            if arm.sample_count > 0 and arm.mean < self._success_threshold:
                old = self._current_strategy
                new_strategy = self.select_strategy()
                self._publish("strategy.switch_requested", {
                    "agent_id": "",
                    "from_strategy": old,
                    "to_strategy": new_strategy.name,
                    "reason": f"Success rate {arm.mean:.2f} < {self._success_threshold}",
                })

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish a bus event if bus is available."""
        if self._bus is None:
            return
        try:
            from qe.models.envelope import Envelope
            self._bus.publish(
                Envelope(
                    topic=topic,
                    source_service_id="strategy_evolver",
                    payload=payload,
                )
            )
        except Exception:
            log.debug("strategy_evolver.publish_failed topic=%s", topic)


class ElasticScaler:
    """Auto-scales the cognitive agent pool based on budget and workload.

    Uses deterministic rules (no LLM) to recommend and apply
    scale profiles.
    """

    def __init__(
        self,
        agent_pool: Any = None,
        budget_tracker: Any = None,
        profiles: dict[str, ScaleProfile] | None = None,
    ) -> None:
        self._agent_pool = agent_pool
        self._budget_tracker = budget_tracker
        self._profiles = profiles or dict(DEFAULT_PROFILES)
        self._current_profile_name: str = "balanced"

    def recommend_profile(
        self,
        pool_stats: dict[str, Any],
        budget_pct: float,
    ) -> ScaleProfile:
        """Recommend a scaling profile based on pool stats and budget.

        Rules:
        - budget < 20% → minimal
        - avg success_rate > 85% → aggressive
        - else → balanced
        """
        if budget_pct < 0.2:
            return self._profiles["minimal"]

        # Calculate average success rate from pool stats
        agents = pool_stats.get("agents", [])
        if agents:
            avg_success = sum(
                a.get("success_rate", 0.0) for a in agents
            ) / len(agents)
        else:
            avg_success = 0.0

        if avg_success > 0.85:
            return self._profiles["aggressive"]

        return self._profiles["balanced"]

    async def apply_profile(self, profile: ScaleProfile) -> None:
        """Spawn or retire agents to match the profile's agent count.

        Adjusts the pool to have exactly max_agents agents.
        """
        if self._agent_pool is None:
            return

        self._current_profile_name = profile.name
        current_count = self._agent_pool.pool_status()["total_agents"]
        target = profile.max_agents

        if current_count < target:
            # Spawn agents to reach target
            for _ in range(target - current_count):
                try:
                    await self._agent_pool.spawn_agent(
                        specialization="general",
                        model_tier=profile.model_tier,
                    )
                except RuntimeError:
                    break  # Pool at capacity
        elif current_count > target:
            # Retire excess agents
            slots = list(self._agent_pool._slots.keys())
            to_retire = slots[target:]
            for aid in to_retire:
                await self._agent_pool.retire_agent(aid)

    def current_profile_name(self) -> str:
        """Return the name of the currently active profile."""
        return self._current_profile_name

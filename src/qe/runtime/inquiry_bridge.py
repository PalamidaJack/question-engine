"""InquiryBridge — Cross-loop glue connecting Inquiry, Knowledge, and Strategy.

Subscribes to inquiry bus events and orchestrates feedback between
the three nested loops: stores episodic memories, records strategy
outcomes, and triggers knowledge consolidation on inquiry completion.
"""

from __future__ import annotations

import logging
from typing import Any

from qe.models.envelope import Envelope
from qe.runtime.episodic_memory import Episode, EpisodicMemory

log = logging.getLogger(__name__)


class InquiryBridge:
    """Lightweight glue that wires inquiry events to knowledge + strategy loops."""

    def __init__(
        self,
        bus: Any,
        episodic_memory: EpisodicMemory,
        strategy_evolver: Any | None = None,
        knowledge_loop: Any | None = None,
    ) -> None:
        self._bus = bus
        self._episodic = episodic_memory
        self._evolver = strategy_evolver
        self._knowledge_loop = knowledge_loop

        self._running = False
        self._episodes_stored = 0
        self._outcomes_recorded = 0
        self._consolidations_triggered = 0

        # Keep handler refs for unsubscribe
        self._handlers: dict[str, Any] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Subscribe to inquiry bus topics."""
        if self._running:
            return
        self._running = True

        topics = {
            "inquiry.started": self._on_inquiry_started,
            "inquiry.completed": self._on_inquiry_completed,
            "inquiry.failed": self._on_inquiry_failed,
            "inquiry.insight_generated": self._on_insight_generated,
        }
        for topic, handler in topics.items():
            self._bus.subscribe(topic, handler)
            self._handlers[topic] = handler

        log.info("inquiry_bridge.started")

    async def stop(self) -> None:
        """Unsubscribe from all topics."""
        for topic, handler in self._handlers.items():
            try:
                self._bus.unsubscribe(topic, handler)
            except Exception:
                log.debug("inquiry_bridge.unsubscribe_failed topic=%s", topic)
        self._handlers.clear()
        self._running = False
        log.info("inquiry_bridge.stopped")

    def status(self) -> dict[str, Any]:
        """Return monitoring snapshot."""
        return {
            "running": self._running,
            "episodes_stored": self._episodes_stored,
            "outcomes_recorded": self._outcomes_recorded,
            "consolidations_triggered": self._consolidations_triggered,
        }

    # ── Event Handlers ────────────────────────────────────────────────

    async def _on_inquiry_started(self, envelope: Envelope) -> None:
        """Store observation episode when an inquiry begins."""
        payload = envelope.payload
        try:
            episode = Episode(
                episode_type="observation",
                goal_id=payload.get("goal_id", ""),
                inquiry_id=payload.get("inquiry_id", ""),
                summary=f"Inquiry started: {payload.get('goal', '')}",
                content={"event": "inquiry.started", **payload},
            )
            await self._episodic.store(episode)
            self._episodes_stored += 1
        except Exception:
            log.debug("inquiry_bridge.started_episode_failed")

    async def _on_inquiry_completed(self, envelope: Envelope) -> None:
        """Store synthesis episode, record strategy outcome, trigger consolidation."""
        payload = envelope.payload
        goal_id = payload.get("goal_id", "")
        status = payload.get("status", "completed")
        insights = payload.get("insights", 0)

        # 1. Store synthesis episode
        try:
            episode = Episode(
                episode_type="synthesis",
                goal_id=goal_id,
                inquiry_id=payload.get("inquiry_id", ""),
                summary=(
                    f"Inquiry completed: status={status}, "
                    f"iterations={payload.get('iterations', 0)}, "
                    f"insights={insights}"
                ),
                content={"event": "inquiry.completed", **payload},
            )
            await self._episodic.store(episode)
            self._episodes_stored += 1
        except Exception:
            log.debug("inquiry_bridge.completed_episode_failed")

        # 2. Record strategy outcome
        if self._evolver is not None and self._evolver._current_strategy:
            try:
                from qe.runtime.strategy_models import StrategyOutcome

                outcome = StrategyOutcome(
                    strategy_name=self._evolver._current_strategy,
                    goal_id=goal_id,
                    success=(status == "completed" and insights > 0),
                    insights_count=insights,
                    duration_s=payload.get("duration_s", 0.0),
                    cost_usd=payload.get("cost_usd", 0.0),
                )
                self._evolver.record_outcome(outcome)
                self._outcomes_recorded += 1

                self._publish("bridge.strategy_outcome_recorded", {
                    "strategy_name": outcome.strategy_name,
                    "goal_id": goal_id,
                    "success": outcome.success,
                    "insights_count": insights,
                })
            except Exception:
                log.debug("inquiry_bridge.outcome_record_failed")

        # 3. Trigger knowledge consolidation
        if self._knowledge_loop is not None:
            try:
                await self._knowledge_loop.trigger_consolidation()
                self._consolidations_triggered += 1
            except Exception:
                log.debug("inquiry_bridge.consolidation_trigger_failed")

    async def _on_inquiry_failed(self, envelope: Envelope) -> None:
        """Store failure episode and record negative strategy outcome."""
        payload = envelope.payload
        goal_id = payload.get("goal_id", "")

        # Store observation episode
        try:
            episode = Episode(
                episode_type="observation",
                goal_id=goal_id,
                inquiry_id=payload.get("inquiry_id", ""),
                summary=f"Inquiry failed at iteration {payload.get('iteration', 0)}",
                content={"event": "inquiry.failed", **payload},
            )
            await self._episodic.store(episode)
            self._episodes_stored += 1
        except Exception:
            log.debug("inquiry_bridge.failed_episode_failed")

        # Record negative outcome
        if self._evolver is not None and self._evolver._current_strategy:
            try:
                from qe.runtime.strategy_models import StrategyOutcome

                outcome = StrategyOutcome(
                    strategy_name=self._evolver._current_strategy,
                    goal_id=goal_id,
                    success=False,
                    insights_count=0,
                )
                self._evolver.record_outcome(outcome)
                self._outcomes_recorded += 1

                self._publish("bridge.strategy_outcome_recorded", {
                    "strategy_name": outcome.strategy_name,
                    "goal_id": goal_id,
                    "success": False,
                    "insights_count": 0,
                })
            except Exception:
                log.debug("inquiry_bridge.failed_outcome_record_failed")

    async def _on_insight_generated(self, envelope: Envelope) -> None:
        """Store synthesis episode with insight headline."""
        payload = envelope.payload
        try:
            episode = Episode(
                episode_type="synthesis",
                goal_id=payload.get("goal_id", ""),
                inquiry_id=payload.get("inquiry_id", ""),
                summary=f"Insight: {payload.get('headline', '')}",
                content={"event": "inquiry.insight_generated", **payload},
            )
            await self._episodic.store(episode)
            self._episodes_stored += 1
        except Exception:
            log.debug("inquiry_bridge.insight_episode_failed")

    # ── Helpers ───────────────────────────────────────────────────────

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish a bus event if bus is available."""
        if self._bus is None:
            return
        try:
            self._bus.publish(
                Envelope(
                    topic=topic,
                    source_service_id="inquiry_bridge",
                    payload=payload,
                )
            )
        except Exception:
            log.debug("inquiry_bridge.publish_failed topic=%s", topic)

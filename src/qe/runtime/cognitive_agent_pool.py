"""CognitiveAgentPool — multi-agent parallel inquiry execution.

Wraps AgentPool with inquiry-specific multi-agent execution,
including agent lifecycle management, parallel inquiry fan-out,
and result merging.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from qe.runtime.agent_pool import AgentPool, AgentRecord
from qe.runtime.cognitive_agent import CognitiveAgent
from qe.runtime.strategy_models import StrategyConfig
from qe.services.inquiry.schemas import InquiryConfig, InquiryResult

log = logging.getLogger(__name__)


@dataclass
class AgentSlot:
    """Runtime state for one cognitive agent."""

    agent: CognitiveAgent
    engine: Any  # InquiryEngine instance
    strategy: StrategyConfig
    current_task: asyncio.Task[Any] | None = None


# Type alias for engine factory callables
EngineFactory = Callable[[], Any]


class CognitiveAgentPool:
    """Multi-agent pool for parallel inquiry execution.

    Manages cognitive agent lifecycle, fans out inquiries
    across multiple agents, and merges results.
    """

    def __init__(
        self,
        bus: Any = None,
        max_agents: int = 3,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self._slots: dict[str, AgentSlot] = {}
        self._pool = AgentPool()
        self._bus = bus
        self._max_agents = max_agents
        self._engine_factory = engine_factory
        self._semaphore = asyncio.Semaphore(max_agents)

    async def spawn_agent(
        self,
        specialization: str = "general",
        model_tier: str = "balanced",
        strategy: StrategyConfig | None = None,
    ) -> CognitiveAgent:
        """Create a cognitive agent with a dedicated InquiryEngine.

        Returns the newly created CognitiveAgent.
        """
        if len(self._slots) >= self._max_agents:
            raise RuntimeError(
                f"Pool at capacity ({self._max_agents} agents)"
            )

        agent = CognitiveAgent(
            specialization=specialization,
            model_preference=model_tier,
            persona=f"{specialization} researcher",
        )

        # Create dedicated engine via factory or None placeholder
        engine = (
            self._engine_factory()
            if self._engine_factory is not None
            else None
        )

        strat = strategy or StrategyConfig(name="default")

        slot = AgentSlot(
            agent=agent,
            engine=engine,
            strategy=strat,
        )
        self._slots[agent.agent_id] = slot

        # Register in underlying AgentPool
        self._pool.register(
            AgentRecord(
                agent_id=agent.agent_id,
                service_id=f"cognitive_{specialization}",
                capabilities=set(agent.capabilities) if agent.capabilities else set(),
                task_types={"research", "analysis", "synthesis"},
                model_tier=model_tier,
            )
        )

        agent.status = "idle"
        log.info(
            "cognitive_pool.spawned agent_id=%s specialization=%s",
            agent.agent_id,
            specialization,
        )
        return agent

    async def retire_agent(self, agent_id: str) -> bool:
        """Retire an agent, cancelling any active task.

        Returns True if the agent was found and retired.
        """
        slot = self._slots.pop(agent_id, None)
        if slot is None:
            return False

        # Cancel active task if running
        if slot.current_task is not None and not slot.current_task.done():
            slot.current_task.cancel()
            try:
                await slot.current_task
            except (asyncio.CancelledError, Exception):
                pass

        slot.agent.status = "retired"
        self._pool.deregister(agent_id)
        log.info("cognitive_pool.retired agent_id=%s", agent_id)
        return True

    async def run_parallel_inquiry(
        self,
        goal_id: str,
        goal_description: str,
        agent_ids: list[str] | None = None,
        config: InquiryConfig | None = None,
    ) -> list[InquiryResult]:
        """Fan out inquiry across specified agents in parallel.

        Uses asyncio.gather with semaphore-based concurrency control.
        Returns list of InquiryResult from each agent.
        """
        ids = agent_ids or list(self._slots.keys())
        if not ids:
            return []

        cfg = config or InquiryConfig()

        async def _run_one(aid: str) -> InquiryResult | None:
            slot = self._slots.get(aid)
            if slot is None or slot.engine is None:
                return None

            async with self._semaphore:
                slot.agent.status = "active"
                slot.agent.active_inquiry_id = goal_id
                try:
                    result = await slot.engine.run_inquiry(
                        goal_id=goal_id,
                        goal_description=goal_description,
                        config=cfg,
                    )
                    return result
                except Exception:
                    log.exception(
                        "cognitive_pool.inquiry_failed agent_id=%s goal_id=%s",
                        aid,
                        goal_id,
                    )
                    return None
                finally:
                    slot.agent.status = "idle"
                    slot.agent.active_inquiry_id = None

        tasks = [_run_one(aid) for aid in ids]
        raw_results = await asyncio.gather(*tasks, return_exceptions=False)
        return [r for r in raw_results if r is not None]

    async def merge_results(
        self, results: list[InquiryResult]
    ) -> InquiryResult:
        """Merge multiple InquiryResults into one.

        - Union of insights
        - Best (longest) findings_summary
        - Max confidence (highest overall_confidence is not on InquiryResult,
          so we use best status)
        - Summed cost
        """
        if not results:
            return InquiryResult(
                inquiry_id=f"merged_{uuid.uuid4().hex[:8]}",
                goal_id="",
                status="completed",
            )

        # Union of all insights
        all_insights: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for r in results:
            for ins in r.insights:
                iid = ins.get("insight_id", "")
                if iid not in seen_ids:
                    all_insights.append(ins)
                    seen_ids.add(iid)

        # Best findings_summary — take the longest
        best_summary = max(
            results, key=lambda r: len(r.findings_summary)
        ).findings_summary

        # Sum costs
        total_cost = sum(r.total_cost_usd for r in results)

        # Max iterations
        max_iterations = max(r.iterations_completed for r in results)

        # Total questions
        total_questions = sum(r.total_questions_generated for r in results)
        total_answered = sum(r.total_questions_answered for r in results)

        # Max duration
        max_duration = max(r.duration_seconds for r in results)

        # Best status (completed > failed)
        best_status = "completed" if any(
            r.status == "completed" for r in results
        ) else "failed"

        return InquiryResult(
            inquiry_id=f"merged_{uuid.uuid4().hex[:8]}",
            goal_id=results[0].goal_id,
            status=best_status,
            termination_reason=results[0].termination_reason,
            iterations_completed=max_iterations,
            total_questions_generated=total_questions,
            total_questions_answered=total_answered,
            findings_summary=best_summary,
            insights=all_insights,
            total_cost_usd=total_cost,
            duration_seconds=max_duration,
        )

    def get_slot(self, agent_id: str) -> AgentSlot | None:
        """Get the slot for an agent."""
        return self._slots.get(agent_id)

    def active_agents(self) -> list[AgentSlot]:
        """Return all active agent slots."""
        return [
            slot for slot in self._slots.values()
            if slot.agent.status == "active"
        ]

    def pool_status(self) -> dict[str, Any]:
        """Monitoring dict for the cognitive agent pool."""
        return {
            "total_agents": len(self._slots),
            "active_agents": len(self.active_agents()),
            "max_agents": self._max_agents,
            "agents": [
                {
                    "agent_id": slot.agent.agent_id,
                    "specialization": slot.agent.specialization,
                    "status": slot.agent.status,
                    "strategy": slot.strategy.name,
                    "model_tier": slot.agent.model_preference,
                }
                for slot in self._slots.values()
            ],
        }

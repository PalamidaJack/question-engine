"""Agent pool with capability-based routing and performance tracking.

Provides ``AgentRecord`` for per-agent runtime state, ``AgentPool`` for
routing dispatched tasks to the best-fit agent, and
``AgentPerformanceTracker`` for persisting metrics to SQLite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qe.models.goal import Subtask

log = logging.getLogger(__name__)

# Scoring weights
_W_SUCCESS = 0.4
_W_AVAILABILITY = 0.3
_W_TIER = 0.2
_W_LATENCY = 0.1


@dataclass
class AgentRecord:
    """Runtime state for a registered agent."""

    agent_id: str
    service_id: str = ""
    capabilities: set[str] = field(default_factory=set)
    task_types: set[str] = field(default_factory=set)
    model_tier: str = "balanced"
    max_concurrency: int = 5
    current_load: int = 0

    # Performance stats
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0  # optimistic default
        return self.tasks_completed / total

    @property
    def avg_latency_ms(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return self.total_latency_ms / self.tasks_completed

    @property
    def available_slots(self) -> int:
        return max(0, self.max_concurrency - self.current_load)

    @property
    def load_pct(self) -> float:
        if self.max_concurrency == 0:
            return 1.0
        return self.current_load / self.max_concurrency


class AgentPool:
    """Core routing engine for multi-agent orchestration."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentRecord] = {}
        self._tracker: AgentPerformanceTracker | None = None

    def set_tracker(self, tracker: AgentPerformanceTracker) -> None:
        self._tracker = tracker

    def register(self, record: AgentRecord) -> None:
        """Register an agent."""
        self._agents[record.agent_id] = record
        log.info(
            "agent_pool.registered agent_id=%s capabilities=%s task_types=%s",
            record.agent_id,
            record.capabilities,
            record.task_types,
        )

    def deregister(self, agent_id: str) -> None:
        """Remove an agent."""
        self._agents.pop(agent_id, None)
        log.info("agent_pool.deregistered agent_id=%s", agent_id)

    def get(self, agent_id: str) -> AgentRecord | None:
        return self._agents.get(agent_id)

    def all_agents(self) -> list[AgentRecord]:
        return list(self._agents.values())

    def select_agent(self, subtask: Subtask) -> AgentRecord | None:
        """Select the best agent for a subtask based on capability and scoring."""
        candidates: list[AgentRecord] = []

        for agent in self._agents.values():
            # Filter: task_type must match
            if subtask.task_type not in agent.task_types:
                continue
            # Filter: capabilities must cover tools_required
            if subtask.tools_required and not set(subtask.tools_required).issubset(
                agent.capabilities
            ):
                continue
            # Filter: must have available capacity
            if agent.available_slots <= 0:
                continue
            candidates.append(agent)

        if not candidates:
            return None

        # Score each candidate
        max_latency = max(
            (a.avg_latency_ms for a in candidates if a.avg_latency_ms > 0),
            default=1.0,
        )

        best: AgentRecord | None = None
        best_score = -1.0

        for agent in candidates:
            normalized_latency = (
                agent.avg_latency_ms / max_latency if max_latency > 0 else 0.0
            )
            tier_bonus = 1.0 if agent.model_tier == subtask.model_tier else 0.0

            score = (
                _W_SUCCESS * agent.success_rate
                + _W_AVAILABILITY * (1.0 - agent.load_pct)
                + _W_TIER * tier_bonus
                + _W_LATENCY * (1.0 - normalized_latency)
            )

            if score > best_score:
                best_score = score
                best = agent

        return best

    def acquire(self, agent_id: str) -> bool:
        """Increment load for an agent. Returns False if at capacity."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return False
        if agent.current_load >= agent.max_concurrency:
            return False
        agent.current_load += 1
        return True

    def release(self, agent_id: str) -> None:
        """Decrement load for an agent."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        agent.current_load = max(0, agent.current_load - 1)

    def record_completion(
        self,
        agent_id: str,
        latency_ms: float,
        cost_usd: float,
        success: bool,
    ) -> None:
        """Update stats for an agent and flush to tracker."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        if success:
            agent.tasks_completed += 1
        else:
            agent.tasks_failed += 1
        agent.total_latency_ms += latency_ms
        agent.total_cost_usd += cost_usd

        if self._tracker is not None:
            self._tracker.record(agent_id, success, latency_ms, cost_usd)

    def status(self) -> dict[str, Any]:
        """Monitoring dict for API."""
        agents_info: list[dict[str, Any]] = []
        for agent in self._agents.values():
            agents_info.append(
                {
                    "agent_id": agent.agent_id,
                    "service_id": agent.service_id,
                    "capabilities": sorted(agent.capabilities),
                    "task_types": sorted(agent.task_types),
                    "model_tier": agent.model_tier,
                    "current_load": agent.current_load,
                    "max_concurrency": agent.max_concurrency,
                    "success_rate": agent.success_rate,
                    "avg_latency_ms": agent.avg_latency_ms,
                }
            )
        return {
            "total_agents": len(self._agents),
            "agents": agents_info,
        }


@dataclass
class _MetricRecord:
    """Queued metric entry for async persistence."""

    agent_id: str
    success: bool
    latency_ms: float
    cost_usd: float
    recorded_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


class AgentPerformanceTracker:
    """Persists per-agent metrics to SQLite."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path
        self._queue: list[_MetricRecord] = []

    def record(
        self,
        agent_id: str,
        success: bool,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        """Queue a metric for async persistence."""
        self._queue.append(
            _MetricRecord(
                agent_id=agent_id,
                success=success,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
            )
        )

    async def flush(self) -> None:
        """Batch write queued metrics to SQLite."""
        if not self._queue or self._db_path is None:
            self._queue.clear()
            return

        import aiosqlite

        records = self._queue[:]
        self._queue.clear()

        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.executemany(
                    "INSERT INTO agent_metrics "
                    "(agent_id, recorded_at, success, latency_ms, cost_usd) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (r.agent_id, r.recorded_at, r.success, r.latency_ms, r.cost_usd)
                        for r in records
                    ],
                )
                # Upsert agent_summary
                for r in records:
                    completed_inc = 1 if r.success else 0
                    failed_inc = 0 if r.success else 1
                    await db.execute(
                        "INSERT INTO agent_summary (agent_id, tasks_completed, tasks_failed, "
                        "total_latency_ms, total_cost_usd, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(agent_id) DO UPDATE SET "
                        "tasks_completed = tasks_completed + ?, "
                        "tasks_failed = tasks_failed + ?, "
                        "total_latency_ms = total_latency_ms + ?, "
                        "total_cost_usd = total_cost_usd + ?, "
                        "updated_at = ?",
                        (
                            r.agent_id,
                            completed_inc,
                            failed_inc,
                            r.latency_ms,
                            r.cost_usd,
                            r.recorded_at,
                            completed_inc,
                            failed_inc,
                            r.latency_ms,
                            r.cost_usd,
                            r.recorded_at,
                        ),
                    )
                await db.commit()
        except Exception:
            log.exception("Failed to flush agent metrics")

    async def load_agent_stats(self, agent_id: str) -> dict[str, Any] | None:
        """Load persisted stats for seeding an AgentRecord on startup."""
        if self._db_path is None:
            return None

        import aiosqlite

        try:
            async with aiosqlite.connect(self._db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM agent_summary WHERE agent_id = ?",
                    (agent_id,),
                )
                row = await cursor.fetchone()
                if row is None:
                    return None
                return dict(row)
        except Exception:
            log.exception("Failed to load agent stats for %s", agent_id)
            return None

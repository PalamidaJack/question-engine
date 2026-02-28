"""Planner service: decomposes goals into executable subtask DAGs."""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

import instructor
import litellm
from dotenv import load_dotenv

from qe.models.goal import ExecutionContract, GoalDecomposition, GoalState, Subtask
from qe.runtime.budget import BudgetTracker
from qe.services.planner.schemas import DecompositionOutput

load_dotenv()

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a task decomposition planner. Given a goal, you decompose it into \
the minimum number of subtasks needed to achieve it.

PROBLEM REPRESENTATION PROTOCOL (complete this first):

1. RESTATE: What is the core problem in one sentence?
2. ACTUAL NEED: What actually needs to happen (may differ from the literal request)?
3. CONSTRAINTS: What are the hard limits?
4. SUCCESS CRITERIA: What does a correct solution look like?
5. PROBLEM TYPE: well_defined / ill_defined / wicked

Then decompose into subtasks. Each subtask should:
- Do exactly one thing
- Have clear success criteria
- Specify which other subtasks it depends on (by index)
- Use the cheapest model tier that can handle it

Available task types: research, analysis, fact_check, synthesis, \
code_execution, web_search, document_generation
Available model tiers: fast, balanced, powerful, local
"""


class PlannerService:
    """Decomposes goals into executable subtask DAGs."""

    def __init__(
        self,
        bus: Any,
        substrate: Any,
        budget_tracker: BudgetTracker | None = None,
        model: str = "gpt-4o",
    ) -> None:
        self.bus = bus
        self.substrate = substrate
        self.budget_tracker = budget_tracker
        self._model = model

    # ── Pattern memory helpers ─────────────────────────────────────────────

    _PATTERN_PREFIX = "goal_pattern:"
    _PATTERN_MIN_SIMILARITY = 0.85

    async def _find_cached_pattern(
        self, goal_description: str
    ) -> DecompositionOutput | None:
        """Search for a similar past decomposition in the embedding store."""
        if self.substrate is None:
            return None
        try:
            results = await self.substrate.embeddings.search(
                goal_description,
                top_k=5,
                min_similarity=self._PATTERN_MIN_SIMILARITY,
            )
            for r in results:
                if not r.id.startswith(self._PATTERN_PREFIX):
                    continue
                raw = r.metadata.get("decomposition")
                if raw:
                    return DecompositionOutput.model_validate_json(raw)
        except Exception:
            log.debug("planner.pattern_search_failed", exc_info=True)
        return None

    async def _store_pattern(
        self, goal_description: str, output: DecompositionOutput
    ) -> None:
        """Persist a successful decomposition so future similar goals can reuse it."""
        if self.substrate is None:
            return
        try:
            key = hashlib.sha256(goal_description.encode()).hexdigest()[:16]
            pattern_id = f"{self._PATTERN_PREFIX}{key}"
            await self.substrate.embeddings.store(
                id=pattern_id,
                text=goal_description,
                metadata={"decomposition": output.model_dump_json()},
            )
            log.debug("planner.pattern_stored id=%s", pattern_id)
        except Exception:
            log.debug("planner.pattern_store_failed", exc_info=True)

    # ── Core decomposition ─────────────────────────────────────────────────

    async def decompose(self, goal_description: str) -> GoalState:
        """Decompose a goal into a subtask DAG.

        Returns a GoalState with status='executing' and a filled decomposition.
        Checks pattern memory first; falls through to LLM on cache miss.
        """
        goal_id = f"goal_{uuid.uuid4().hex[:12]}"

        log.info(
            "planner.decompose goal_id=%s description=%s",
            goal_id,
            goal_description[:100],
        )

        # Try pattern memory first
        cached = await self._find_cached_pattern(goal_description)
        if cached is not None:
            output = cached
            log.info("planner.pattern_hit goal_id=%s", goal_id)
        else:
            # Call LLM for decomposition
            client = instructor.from_litellm(litellm.acompletion)
            output = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": goal_description},
                ],
                response_model=DecompositionOutput,
            )

            # Record cost
            if self.budget_tracker:
                cost = litellm.completion_cost(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": goal_description},
                    ],
                    completion="",
                )
                self.budget_tracker.record_cost(
                    self._model, cost, service_id="planner"
                )

            # Store pattern for future reuse
            await self._store_pattern(goal_description, output)

        # Convert LLM output to internal models
        subtasks = []
        subtask_ids = []
        for plan in output.subtasks:
            sub = Subtask(
                description=plan.description,
                task_type=plan.task_type,
                model_tier=plan.model_tier,
                contract=ExecutionContract(
                    timeout_seconds=120,
                    max_retries=3,
                ),
            )
            subtask_ids.append(sub.subtask_id)
            subtasks.append(sub)

        # Resolve dependency indices to subtask_ids
        for i, plan in enumerate(output.subtasks):
            for dep_idx in plan.depends_on_indices:
                if 0 <= dep_idx < len(subtask_ids) and dep_idx != i:
                    subtasks[i].depends_on.append(subtask_ids[dep_idx])

        decomposition = GoalDecomposition(
            goal_id=goal_id,
            original_description=goal_description,
            strategy=output.strategy,
            subtasks=subtasks,
            assumptions=output.assumptions,
            estimated_time_seconds=output.estimated_time_seconds,
        )

        state = GoalState(
            goal_id=goal_id,
            description=goal_description,
            status="executing",
            decomposition=decomposition,
            subtask_states={s.subtask_id: "pending" for s in subtasks},
        )

        log.info(
            "planner.decomposed goal_id=%s subtasks=%d strategy=%s",
            goal_id,
            len(subtasks),
            output.strategy[:80],
        )

        return state

    def get_ready_subtasks(self, state: GoalState) -> list[Subtask]:
        """Return subtasks whose dependencies are all completed."""
        if not state.decomposition:
            return []

        ready = []
        for subtask in state.decomposition.subtasks:
            if state.subtask_states.get(subtask.subtask_id) != "pending":
                continue
            deps_met = all(
                state.subtask_states.get(dep_id) == "completed"
                for dep_id in subtask.depends_on
            )
            if deps_met:
                ready.append(subtask)
        return ready

"""Planner service: decomposes goals into executable subtask DAGs."""

from __future__ import annotations

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

    async def decompose(self, goal_description: str) -> GoalState:
        """Decompose a goal into a subtask DAG.

        Returns a GoalState with status='executing' and a filled decomposition.
        """
        goal_id = f"goal_{uuid.uuid4().hex[:12]}"

        log.info(
            "planner.decompose goal_id=%s description=%s",
            goal_id,
            goal_description[:100],
        )

        # Call LLM for decomposition
        client = instructor.from_litellm(litellm.acompletion)
        output: DecompositionOutput = await client.chat.completions.create(
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
            self.budget_tracker.record_cost(self._model, cost, service_id="planner")

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

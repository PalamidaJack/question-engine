"""Composable Workflow Skills + Multi-Skill Chaining.

Upgrade playbooks to DAG skill templates that can invoke tools, LLM calls,
or other skills.  Chain skills with automatic data passing.
Gated behind ``composable_skills`` and ``skill_chaining`` flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SkillStep:
    """A single step in a skill workflow."""

    step_id: str
    action: str  # tool_call | llm_call | skill_invoke
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    output_key: str = ""  # Key to store result in context


@dataclass
class Skill:
    """A composable workflow skill (DAG of steps)."""

    skill_id: str
    name: str
    description: str = ""
    steps: list[SkillStep] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "steps": len(self.steps),
            "tags": self.tags,
            "version": self.version,
        }


@dataclass
class SkillExecutionResult:
    """Result from executing a skill or skill chain."""

    skill_id: str
    success: bool
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    steps_completed: int = 0


class SkillEngine:
    """Executes composable skills and skill chains."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._handlers: dict[str, Any] = {}

    def register_skill(self, skill: Skill) -> None:
        self._skills[skill.skill_id] = skill

    def register_handler(
        self, action: str, handler: Any,
    ) -> None:
        """Register a handler for a step action type."""
        self._handlers[action] = handler

    def get_skill(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    async def execute(
        self,
        skill_id: str,
        inputs: dict[str, Any] | None = None,
    ) -> SkillExecutionResult:
        """Execute a single skill."""
        skill = self._skills.get(skill_id)
        if not skill:
            return SkillExecutionResult(
                skill_id=skill_id, success=False,
                errors=[f"Skill not found: {skill_id}"],
            )

        context = dict(inputs or {})
        completed = 0
        errors: list[str] = []

        # Topological sort by depends_on
        executed: set[str] = set()
        remaining = list(skill.steps)

        max_iterations = len(remaining) * 2
        iteration = 0
        while remaining and iteration < max_iterations:
            iteration += 1
            progress = False
            for step in list(remaining):
                if all(
                    d in executed for d in step.depends_on
                ):
                    try:
                        result = await self._execute_step(
                            step, context
                        )
                        if step.output_key:
                            context[step.output_key] = result
                        executed.add(step.step_id)
                        remaining.remove(step)
                        completed += 1
                        progress = True
                    except Exception as e:
                        errors.append(
                            f"Step {step.step_id}: {e}"
                        )
                        remaining.remove(step)
                        progress = True
            if not progress:
                errors.append("Deadlock: unresolvable deps")
                break

        return SkillExecutionResult(
            skill_id=skill_id,
            success=len(errors) == 0,
            outputs=context,
            errors=errors,
            steps_completed=completed,
        )

    async def chain(
        self,
        skill_ids: list[str],
        initial_inputs: dict[str, Any] | None = None,
    ) -> list[SkillExecutionResult]:
        """Chain multiple skills — output of one feeds into next."""
        results: list[SkillExecutionResult] = []
        context = dict(initial_inputs or {})

        for sid in skill_ids:
            result = await self.execute(sid, context)
            results.append(result)
            if not result.success:
                break
            # Pass outputs forward
            context.update(result.outputs)

        return results

    async def _execute_step(
        self, step: SkillStep, context: dict[str, Any],
    ) -> Any:
        handler = self._handlers.get(step.action)
        if handler is None:
            raise ValueError(
                f"No handler for action: {step.action}"
            )
        # Resolve params from context
        resolved_params = {}
        for k, v in step.params.items():
            if isinstance(v, str) and v.startswith("$"):
                key = v[1:]
                resolved_params[k] = context.get(key, v)
            else:
                resolved_params[k] = v
        import asyncio
        result = handler(**resolved_params)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    def list_skills(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._skills.values()]

    def stats(self) -> dict[str, Any]:
        return {
            "total_skills": len(self._skills),
            "handlers": list(self._handlers.keys()),
        }

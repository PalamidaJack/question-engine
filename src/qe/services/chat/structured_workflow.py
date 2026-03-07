"""Structured design-plan-execute workflow and RPI methodology.

Provides a 3-phase workflow (Design→Plan→Execute) with explicit
entry/exit criteria.  Extended by RPI (Research-Plan-Implement)
with confidence tags.

Gated behind ``structured_workflow`` and ``rpi_methodology`` flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

log = logging.getLogger(__name__)


class WorkflowPhase(StrEnum):
    DESIGN = "design"
    PLAN = "plan"
    EXECUTE = "execute"
    # RPI extensions
    RESEARCH = "research"
    IMPLEMENT = "implement"


@dataclass
class PhaseResult:
    """Result from completing a workflow phase."""

    phase: str
    outputs: dict[str, Any] = field(default_factory=dict)
    exit_criteria_met: bool = True
    issues: list[str] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class StructuredWorkflow:
    """A 3-phase Design→Plan→Execute workflow."""

    goal: str
    current_phase: WorkflowPhase = WorkflowPhase.DESIGN
    phases: dict[str, PhaseResult] = field(default_factory=dict)
    completed: bool = False

    def advance(self, result: PhaseResult) -> WorkflowPhase | None:
        """Record phase result and advance to next phase.

        Returns the next phase, or None if workflow is complete.
        """
        self.phases[result.phase] = result

        if not result.exit_criteria_met:
            log.warning(
                "structured_workflow.exit_criteria_not_met phase=%s issues=%s",
                result.phase,
                result.issues,
            )
            return WorkflowPhase(result.phase)  # Stay in same phase

        phase_order = [WorkflowPhase.DESIGN, WorkflowPhase.PLAN, WorkflowPhase.EXECUTE]
        try:
            idx = phase_order.index(self.current_phase)
            if idx + 1 < len(phase_order):
                self.current_phase = phase_order[idx + 1]
                return self.current_phase
        except ValueError:
            pass

        self.completed = True
        return None

    def get_phase_prompt(self) -> str:
        """Return the prompt addendum for the current phase."""
        prompts = {
            WorkflowPhase.DESIGN: (
                "[PHASE: DESIGN]\n"
                "Define the approach, identify requirements, and outline the solution. "
                "Exit criteria: clear requirements, identified risks, chosen approach."
            ),
            WorkflowPhase.PLAN: (
                "[PHASE: PLAN]\n"
                "Break down the approach into concrete steps with dependencies. "
                "Exit criteria: step-by-step plan, estimated effort, risk mitigations."
            ),
            WorkflowPhase.EXECUTE: (
                "[PHASE: EXECUTE]\n"
                "Implement the plan step by step. Verify each step before proceeding. "
                "Exit criteria: all steps completed, outputs verified."
            ),
        }
        return prompts.get(self.current_phase, "")

    def status(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "current_phase": self.current_phase,
            "completed": self.completed,
            "phases_done": list(self.phases.keys()),
        }


@dataclass
class RPIWorkflow(StructuredWorkflow):
    """Research-Plan-Implement workflow with confidence tags.

    Extends StructuredWorkflow with confidence-tagged outputs.
    """

    def __post_init__(self):
        self.current_phase = WorkflowPhase.RESEARCH

    def advance(self, result: PhaseResult) -> WorkflowPhase | None:
        self.phases[result.phase] = result

        if not result.exit_criteria_met:
            return WorkflowPhase(result.phase)

        phase_order = [
            WorkflowPhase.RESEARCH,
            WorkflowPhase.PLAN,
            WorkflowPhase.IMPLEMENT,
        ]
        try:
            idx = phase_order.index(self.current_phase)
            if idx + 1 < len(phase_order):
                self.current_phase = phase_order[idx + 1]
                return self.current_phase
        except ValueError:
            pass

        self.completed = True
        return None

    def get_phase_prompt(self) -> str:
        prompts = {
            WorkflowPhase.RESEARCH: (
                "[PHASE: RESEARCH]\n"
                "Gather information, explore the problem space, identify constraints. "
                "Tag each finding with confidence: "
                "HIGH (verified), MEDIUM (likely), LOW (uncertain). "
                "Exit criteria: sufficient understanding to plan, key unknowns identified."
            ),
            WorkflowPhase.PLAN: (
                "[PHASE: PLAN]\n"
                "Design the implementation based on research findings. "
                "Address HIGH-confidence findings directly, mitigate LOW-confidence risks. "
                "Exit criteria: actionable plan with clear steps."
            ),
            WorkflowPhase.IMPLEMENT: (
                "[PHASE: IMPLEMENT]\n"
                "Execute the plan. Verify each step against research findings. "
                "Tag implementation confidence for each component. "
                "Exit criteria: all components implemented and verified."
            ),
        }
        return prompts.get(self.current_phase, "")

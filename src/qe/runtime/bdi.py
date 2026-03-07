"""BDI Mental State Tracking — Beliefs, Desires, Intentions.

Assembles agent mental state from substrate beliefs, goals (desires),
and active workflows (intentions) into a coherent context for LLM calls.
Gated behind ``bdi_tracking`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Belief:
    """A belief derived from the substrate."""

    entity: str
    predicate: str
    value: str
    confidence: float = 0.5
    source: str = ""


@dataclass
class Desire:
    """A desire/goal the agent wants to achieve."""

    goal_id: str
    description: str
    priority: float = 0.5
    status: str = "active"  # active | achieved | abandoned


@dataclass
class Intention:
    """An active plan/workflow the agent is executing."""

    workflow_id: str
    description: str
    current_step: str = ""
    progress: float = 0.0  # 0.0 to 1.0


@dataclass
class MentalState:
    """Complete BDI mental state snapshot."""

    beliefs: list[Belief] = field(default_factory=list)
    desires: list[Desire] = field(default_factory=list)
    intentions: list[Intention] = field(default_factory=list)

    def context_prompt(self) -> str:
        """Generate a prompt section from the mental state."""
        sections: list[str] = []

        if self.beliefs:
            lines = ["## Current Beliefs"]
            for b in self.beliefs[:10]:
                lines.append(
                    f"- {b.entity} {b.predicate}: "
                    f"{b.value} (conf: {b.confidence:.1f})"
                )
            sections.append("\n".join(lines))

        if self.desires:
            lines = ["## Active Goals"]
            for d in self.desires:
                if d.status == "active":
                    lines.append(
                        f"- [{d.priority:.1f}] {d.description}"
                    )
            sections.append("\n".join(lines))

        if self.intentions:
            lines = ["## Active Plans"]
            for i in self.intentions:
                lines.append(
                    f"- {i.description} "
                    f"(step: {i.current_step}, "
                    f"progress: {i.progress:.0%})"
                )
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def summary(self) -> dict[str, Any]:
        return {
            "beliefs": len(self.beliefs),
            "desires": len(self.desires),
            "intentions": len(self.intentions),
            "active_goals": sum(
                1 for d in self.desires if d.status == "active"
            ),
        }


class BDITracker:
    """Tracks and assembles BDI mental state."""

    def __init__(self) -> None:
        self._beliefs: list[Belief] = []
        self._desires: list[Desire] = []
        self._intentions: list[Intention] = []

    def add_belief(self, belief: Belief) -> None:
        self._beliefs.append(belief)

    def add_desire(self, desire: Desire) -> None:
        self._desires.append(desire)

    def add_intention(self, intention: Intention) -> None:
        self._intentions.append(intention)

    def update_belief(
        self, entity: str, predicate: str, value: str,
        confidence: float = 0.5,
    ) -> None:
        """Update or add a belief."""
        for b in self._beliefs:
            if b.entity == entity and b.predicate == predicate:
                b.value = value
                b.confidence = confidence
                return
        self._beliefs.append(
            Belief(entity, predicate, value, confidence)
        )

    def achieve_goal(self, goal_id: str) -> None:
        for d in self._desires:
            if d.goal_id == goal_id:
                d.status = "achieved"
                return

    def abandon_goal(self, goal_id: str) -> None:
        for d in self._desires:
            if d.goal_id == goal_id:
                d.status = "abandoned"
                return

    def update_intention_progress(
        self, workflow_id: str, progress: float,
        current_step: str = "",
    ) -> None:
        for i in self._intentions:
            if i.workflow_id == workflow_id:
                i.progress = progress
                if current_step:
                    i.current_step = current_step
                return

    def get_state(self) -> MentalState:
        return MentalState(
            beliefs=list(self._beliefs),
            desires=list(self._desires),
            intentions=list(self._intentions),
        )

    def clear(self) -> None:
        self._beliefs.clear()
        self._desires.clear()
        self._intentions.clear()

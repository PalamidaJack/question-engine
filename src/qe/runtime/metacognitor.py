"""Metacognitor — self-awareness and approach management for the Cognitive Layer.

Maintains a model of the system's own capabilities, reasons about gaps,
suggests creative tool combinations, and tracks all attempted approaches
as a tree. When stuck, fundamentally reframes the problem instead of retrying.
"""

from __future__ import annotations

import logging
from typing import Any

import instructor
import litellm
from pydantic import BaseModel

from qe.models.cognition import (
    ApproachAssessment,
    ApproachNode,
    CapabilityGap,
    CapabilityProfile,
    ToolCombinationSuggestion,
)
from qe.runtime.episodic_memory import Episode, EpisodicMemory

log = logging.getLogger(__name__)


_APPROACH_PROMPT = """\
You are a metacognitive reasoning module for an AI research system.

The system is working on the following goal:
{goal_description}

AVAILABLE CAPABILITIES:
{capabilities_summary}

CAPABILITY GAPS:
{gaps_summary}

APPROACHES ALREADY TRIED (and their outcomes):
{tried_approaches}

CURRENT FAILURE (if any):
{current_failure}

Your task: Recommend the NEXT approach to try. Do NOT suggest anything \
already tried. Think creatively:
- Can existing tools be combined in unusual ways?
- Can the problem be reframed?
- Is there an indirect way to get the answer?
- Should we change what we're looking for entirely?

If all reasonable approaches are exhausted, say so explicitly.
"""

_TOOL_COMBINATION_PROMPT = """\
You are a creative problem solver for an AI system.

PROBLEM: {problem}
AVAILABLE TOOLS: {tools}
CONSTRAINT: Standard approaches have failed. We need creative tool \
combinations.

Suggest unconventional ways to combine these tools to solve the problem.
Think like a resourceful analyst who has to work with limited tools.
Examples of creative combinations:
- "Web search found nothing" → "Use code execution to compute from \
public data"
- "Can't access proprietary DB" → "Search SEC filings + news for the \
same data"
- "Direct query too broad" → "Search for specific sub-examples and \
aggregate"
"""


class Metacognitor:
    """System self-awareness: capabilities, gaps, and approach management.

    Solo-agent mode: one Metacognitor per inquiry.
    Multi-agent mode: shared across agents via MemoryBus broadcasts.
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory | None = None,
        model: str = "openai/google/gemini-2.0-flash",
    ) -> None:
        self._episodic = episodic_memory
        self._model = model

        # Capability registry
        self._capabilities: dict[str, CapabilityProfile] = {}
        self._gaps: list[CapabilityGap] = []

        # Per-goal approach trees
        self._approach_trees: dict[str, dict[str, ApproachNode]] = {}
        self._approach_roots: dict[str, str] = {}
        self._current_nodes: dict[str, str] = {}

    # -------------------------------------------------------------------
    # Capability Management
    # -------------------------------------------------------------------

    def register_capability(self, profile: CapabilityProfile) -> None:
        """Register a tool/capability the system has access to."""
        self._capabilities[profile.tool_name] = profile

    def register_capabilities_bulk(
        self,
        tool_names: list[str],
        descriptions: dict[str, str] | None = None,
    ) -> None:
        """Bulk register capabilities."""
        desc = descriptions or {}
        for name in tool_names:
            self._capabilities[name] = CapabilityProfile(
                tool_name=name,
                description=desc.get(name, f"Tool: {name}"),
            )

    def identify_gap(
        self,
        description: str,
        workaround: str = "",
        severity: str = "degraded",
    ) -> CapabilityGap:
        """Record a capability gap discovered during execution."""
        gap = CapabilityGap(
            description=description,
            workaround=workaround,
            severity=severity,
        )
        self._gaps.append(gap)
        return gap

    def get_capabilities_summary(self) -> str:
        """Formatted string of capabilities for LLM prompts."""
        lines = []
        for cap in self._capabilities.values():
            limits = ", ".join(cap.limitations) if cap.limitations else "none"
            lines.append(
                f"- {cap.tool_name}: {cap.description} (limits: {limits})"
            )
        return "\n".join(lines) if lines else "No tools registered."

    def get_gaps_summary(self) -> str:
        """Formatted string of known gaps for LLM prompts."""
        lines = []
        for gap in self._gaps:
            wa = f" Workaround: {gap.workaround}" if gap.workaround else ""
            lines.append(f"- [{gap.severity}] {gap.description}{wa}")
        return "\n".join(lines) if lines else "No known gaps."

    # -------------------------------------------------------------------
    # Approach Tree Management
    # -------------------------------------------------------------------

    def init_approach_tree(
        self,
        goal_id: str,
        initial_approach: str,
    ) -> ApproachNode:
        """Initialize the approach tree for a goal."""
        root = ApproachNode(
            approach_description=initial_approach,
            status="in_progress",
        )
        self._approach_trees[goal_id] = {root.node_id: root}
        self._approach_roots[goal_id] = root.node_id
        self._current_nodes[goal_id] = root.node_id
        return root

    def record_approach_outcome(
        self,
        goal_id: str,
        success: bool,
        failure_reason: str = "",
    ) -> None:
        """Record the outcome of the current approach."""
        tree = self._approach_trees.get(goal_id, {})
        current_id = self._current_nodes.get(goal_id)
        if not current_id or current_id not in tree:
            return
        node = tree[current_id]
        node.status = "succeeded" if success else "failed"
        node.failure_reason = failure_reason

    def add_child_approach(
        self,
        goal_id: str,
        approach_description: str,
    ) -> ApproachNode:
        """Add a child approach under the current node (drilling deeper)."""
        tree = self._approach_trees.get(goal_id, {})
        current_id = self._current_nodes.get(goal_id)
        child = ApproachNode(
            approach_description=approach_description,
            parent_id=current_id,
            status="in_progress",
        )
        tree[child.node_id] = child
        if current_id and current_id in tree:
            tree[current_id].children.append(child.node_id)
        self._current_nodes[goal_id] = child.node_id
        return child

    def add_sibling_approach(
        self,
        goal_id: str,
        approach_description: str,
    ) -> ApproachNode:
        """Add a sibling approach (backtrack to parent, try alternative)."""
        tree = self._approach_trees.get(goal_id, {})
        current_id = self._current_nodes.get(goal_id)
        if not current_id or current_id not in tree:
            return self.add_child_approach(goal_id, approach_description)
        parent_id = tree[current_id].parent_id
        if parent_id is None:
            parent_id = current_id
        sibling = ApproachNode(
            approach_description=approach_description,
            parent_id=parent_id,
            status="in_progress",
        )
        tree[sibling.node_id] = sibling
        if parent_id in tree:
            tree[parent_id].children.append(sibling.node_id)
        self._current_nodes[goal_id] = sibling.node_id
        return sibling

    def get_tried_approaches_summary(self, goal_id: str) -> str:
        """Format all tried approaches for LLM context."""
        tree = self._approach_trees.get(goal_id, {})
        if not tree:
            return "No approaches tried yet."
        icons = {
            "succeeded": "[OK]",
            "failed": "[FAIL]",
            "in_progress": "[...]",
            "untried": "[?]",
            "abandoned": "[X]",
        }
        lines = []
        for node in tree.values():
            icon = icons.get(node.status, "[?]")
            reason = f" -- {node.failure_reason}" if node.failure_reason else ""
            lines.append(f"  {icon} {node.approach_description}{reason}")
        return "\n".join(lines)

    def has_untried_approaches(self, goal_id: str) -> bool:
        """Check if there are untried approaches available."""
        tree = self._approach_trees.get(goal_id, {})
        return any(n.status == "untried" for n in tree.values())

    def all_approaches_exhausted(self, goal_id: str) -> bool:
        """Check if all approaches have been tried and failed."""
        tree = self._approach_trees.get(goal_id, {})
        if not tree:
            return False
        return all(
            n.status in ("failed", "abandoned") for n in tree.values()
        )

    def get_current_node(self, goal_id: str) -> ApproachNode | None:
        """Get the currently active approach node."""
        tree = self._approach_trees.get(goal_id, {})
        current_id = self._current_nodes.get(goal_id)
        if current_id and current_id in tree:
            return tree[current_id]
        return None

    def approach_count(self, goal_id: str) -> int:
        """Total number of approaches in the tree."""
        return len(self._approach_trees.get(goal_id, {}))

    # -------------------------------------------------------------------
    # LLM-Powered Reasoning
    # -------------------------------------------------------------------

    async def suggest_next_approach(
        self,
        goal_id: str,
        goal_description: str,
        current_failure: str = "",
    ) -> ApproachAssessment:
        """Use LLM to reason about the next approach to try."""
        prompt = _APPROACH_PROMPT.format(
            goal_description=goal_description,
            capabilities_summary=self.get_capabilities_summary(),
            gaps_summary=self.get_gaps_summary(),
            tried_approaches=self.get_tried_approaches_summary(goal_id),
            current_failure=current_failure or "None",
        )

        client = instructor.from_litellm(litellm.acompletion)
        assessment = await client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a metacognitive reasoning module.",
                },
                {"role": "user", "content": prompt},
            ],
            response_model=ApproachAssessment,
        )

        if self._episodic:
            await self._episodic.store(
                Episode(
                    goal_id=goal_id,
                    episode_type="observation",
                    content={
                        "phase": "metacognition",
                        "assessment": assessment.model_dump(),
                    },
                    summary=(
                        f"Metacognitor: suggested "
                        f"'{assessment.recommended_approach}'"
                    ),
                    relevance_to_goal=0.9,
                )
            )

        return assessment

    async def suggest_tool_combinations(
        self,
        problem: str,
    ) -> list[ToolCombinationSuggestion]:
        """Suggest creative tool combinations for a difficult problem."""
        tools_str = "\n".join(
            f"- {name}: {cap.description}"
            for name, cap in self._capabilities.items()
        )
        prompt = _TOOL_COMBINATION_PROMPT.format(
            problem=problem,
            tools=tools_str,
        )

        class ToolCombinations(BaseModel):
            suggestions: list[ToolCombinationSuggestion]

        client = instructor.from_litellm(litellm.acompletion)
        result = await client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative problem-solving module.",
                },
                {"role": "user", "content": prompt},
            ],
            response_model=ToolCombinations,
        )
        return result.suggestions

    # -------------------------------------------------------------------
    # Cleanup & Status
    # -------------------------------------------------------------------

    def clear_goal(self, goal_id: str) -> None:
        """Clean up state for a completed goal."""
        self._approach_trees.pop(goal_id, None)
        self._approach_roots.pop(goal_id, None)
        self._current_nodes.pop(goal_id, None)

    def status(self) -> dict[str, Any]:
        """Monitoring snapshot."""
        return {
            "registered_capabilities": len(self._capabilities),
            "known_gaps": len(self._gaps),
            "active_approach_trees": len(self._approach_trees),
        }

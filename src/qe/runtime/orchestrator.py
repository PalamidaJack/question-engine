"""Tool orchestrator — handoff rules for intelligent tool selection.

Defines rules like "if compare requested → use swarm", "if tool X
failed → try tool Y".  Gated behind ``orchestrator_handoff`` flag.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class HandoffRule:
    """A single orchestration rule."""

    name: str
    trigger: str  # regex pattern matching user intent or tool result
    action: str  # "route_to", "fallback", "escalate", "swarm"
    target: str  # tool name, tier, or swarm config
    priority: int = 0
    description: str = ""

    def matches(self, context: str) -> bool:
        return bool(re.search(self.trigger, context, re.IGNORECASE))


# Built-in rules
DEFAULT_RULES: list[HandoffRule] = [
    HandoffRule(
        name="compare_to_swarm",
        trigger=r"\b(compare|versus|vs\.?|difference between)\b",
        action="swarm",
        target="parallel_comparison",
        priority=10,
        description="Route comparison queries to parallel swarm",
    ),
    HandoffRule(
        name="web_search_fallback",
        trigger=r"web_search.*failed|web_search.*error",
        action="fallback",
        target="web_fetch",
        priority=5,
        description="Fall back to web_fetch if web_search fails",
    ),
    HandoffRule(
        name="code_error_escalate",
        trigger=r"code_execute.*error|syntax.*error|runtime.*error",
        action="escalate",
        target="powerful",
        priority=8,
        description="Escalate to powerful tier on code errors",
    ),
    HandoffRule(
        name="deep_research",
        trigger=r"\b(deep dive|thorough|comprehensive|in-depth)\b",
        action="route_to",
        target="deep_research",
        priority=7,
        description="Route thorough research requests to deep_research tool",
    ),
]


class ToolOrchestrator:
    """Evaluates handoff rules against context to determine tool routing."""

    def __init__(self, rules: list[HandoffRule] | None = None) -> None:
        self._rules = sorted(
            rules or list(DEFAULT_RULES),
            key=lambda r: r.priority,
            reverse=True,
        )

    def add_rule(self, rule: HandoffRule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def evaluate(self, context: str) -> HandoffRule | None:
        """Return the highest-priority matching rule, or None."""
        for rule in self._rules:
            if rule.matches(context):
                log.debug("orchestrator.match rule=%s action=%s", rule.name, rule.action)
                return rule
        return None

    def evaluate_all(self, context: str) -> list[HandoffRule]:
        """Return all matching rules in priority order."""
        return [r for r in self._rules if r.matches(context)]

    def list_rules(self) -> list[dict[str, Any]]:
        return [
            {
                "name": r.name,
                "trigger": r.trigger,
                "action": r.action,
                "target": r.target,
                "priority": r.priority,
                "description": r.description,
            }
            for r in self._rules
        ]

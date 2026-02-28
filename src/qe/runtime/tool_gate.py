"""Security policy enforcement for tool invocations."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class GateDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


class GateResult(BaseModel):
    """Result of a tool gate validation."""

    decision: GateDecision
    reason: str = ""
    policy_name: str = ""


class SecurityPolicy(BaseModel):
    """A security policy rule."""

    name: str
    description: str = ""
    blocked_domains: list[str] = Field(default_factory=list)
    max_calls_per_goal: int = 0
    require_hil: bool = False

    class Config:
        arbitrary_types_allowed = True


class ToolGate:
    """Security policy enforcement for all tool invocations.

    Runs as code-level enforcement, not prompt-level — cannot be
    bypassed by prompt injection.
    """

    def __init__(self, policies: list[SecurityPolicy] | None = None) -> None:
        self._policies = policies or []
        self._call_counts: dict[str, dict[str, int]] = {}  # goal_id -> {tool_name -> count}

    def validate(
        self,
        tool_name: str,
        params: dict[str, Any],
        capabilities: set[str],
        goal_id: str = "",
    ) -> GateResult:
        """Validate a tool call against all security policies.

        Returns ALLOW, DENY, or ESCALATE.
        """
        # 1. Capability check
        required_cap = self._get_required_capability(tool_name)
        if required_cap and required_cap not in capabilities:
            log.warning(
                "tool_gate.deny capability_not_declared tool=%s required=%s",
                tool_name,
                required_cap,
            )
            return GateResult(
                decision=GateDecision.DENY,
                reason=f"capability '{required_cap}' not declared",
                policy_name="capability_check",
            )

        # 2. Policy-specific checks
        for policy in self._policies:
            # Check require_hil
            if policy.require_hil and tool_name in policy.name:
                return GateResult(
                    decision=GateDecision.ESCALATE,
                    reason=f"Tool '{tool_name}' requires human approval",
                    policy_name=policy.name,
                )

            # Check domain blocking for web tools
            if policy.blocked_domains and tool_name in ("web_fetch", "web_search"):
                url = params.get("url", "")
                query = params.get("query", "")
                for domain in policy.blocked_domains:
                    if domain in url or domain in query:
                        return GateResult(
                            decision=GateDecision.DENY,
                            reason=f"Domain blocked: {domain}",
                            policy_name=policy.name,
                        )

            # Check rate limiting
            if policy.max_calls_per_goal > 0 and goal_id:
                goal_counts = self._call_counts.setdefault(goal_id, {})
                count = goal_counts.get(tool_name, 0)
                if count >= policy.max_calls_per_goal:
                    return GateResult(
                        decision=GateDecision.DENY,
                        reason=f"Rate limit exceeded: {count}/{policy.max_calls_per_goal}",
                        policy_name=policy.name,
                    )

        # 3. Path sandboxing for file operations
        if tool_name in ("file_read", "file_write"):
            path = params.get("path", "")
            if ".." in path or path.startswith("/"):
                return GateResult(
                    decision=GateDecision.DENY,
                    reason=f"Path escapes sandbox: {path}",
                    policy_name="sandbox_check",
                )

        # Record the call
        if goal_id:
            goal_counts = self._call_counts.setdefault(goal_id, {})
            goal_counts[tool_name] = goal_counts.get(tool_name, 0) + 1

        return GateResult(decision=GateDecision.ALLOW)

    def _get_required_capability(self, tool_name: str) -> str | None:
        """Map tool names to required capabilities."""
        capability_map = {
            "web_search": "web_search",
            "web_fetch": "web_search",
            "file_read": "file_read",
            "file_write": "file_write",
            "code_execute": "code_execute",
            "browser_navigate": "browser_control",
        }
        return capability_map.get(tool_name)

    def reset_counts(self, goal_id: str) -> None:
        """Reset call counts for a goal (on goal completion)."""
        self._call_counts.pop(goal_id, None)

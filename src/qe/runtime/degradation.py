"""Graceful degradation: formal policy for fallbacks when components fail.

Defines what features are available at each degradation level and
provides health-based automatic feature gating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

log = logging.getLogger(__name__)


class DegradationLevel(IntEnum):
    """System degradation levels, from healthy to critical."""

    HEALTHY = 0  # all features available
    DEGRADED = 1  # some features limited
    FALLBACK = 2  # using backup providers/local models
    MINIMAL = 3  # core only, no LLM, read-only
    OFFLINE = 4  # system down, maintenance mode


@dataclass
class FeatureFlag:
    """A controllable feature with degradation behavior."""

    name: str
    enabled: bool = True
    min_level: DegradationLevel = DegradationLevel.HEALTHY
    description: str = ""


# ── Default feature definitions ────────────────────────────────────────────

_DEFAULT_FEATURES: dict[str, FeatureFlag] = {
    "llm_calls": FeatureFlag(
        "llm_calls",
        description="Cloud LLM API calls",
        min_level=DegradationLevel.DEGRADED,
    ),
    "local_models": FeatureFlag(
        "local_models",
        description="Local Ollama model calls",
        min_level=DegradationLevel.FALLBACK,
    ),
    "goal_execution": FeatureFlag(
        "goal_execution",
        description="Goal decomposition and dispatch",
        min_level=DegradationLevel.DEGRADED,
    ),
    "web_search": FeatureFlag(
        "web_search",
        description="External web search tool",
        min_level=DegradationLevel.HEALTHY,
    ),
    "claim_extraction": FeatureFlag(
        "claim_extraction",
        description="Extract claims from observations",
        min_level=DegradationLevel.DEGRADED,
    ),
    "chat": FeatureFlag(
        "chat",
        description="Interactive chat with LLM",
        min_level=DegradationLevel.DEGRADED,
    ),
    "query_answering": FeatureFlag(
        "query_answering",
        description="Question answering against belief ledger",
        min_level=DegradationLevel.FALLBACK,
    ),
    "hil_approvals": FeatureFlag(
        "hil_approvals",
        description="Human-in-the-loop approvals",
        min_level=DegradationLevel.MINIMAL,
    ),
    "read_api": FeatureFlag(
        "read_api",
        description="Read-only API endpoints",
        min_level=DegradationLevel.MINIMAL,
    ),
    "health_checks": FeatureFlag(
        "health_checks",
        description="System health monitoring",
        min_level=DegradationLevel.OFFLINE,
    ),
}


@dataclass
class FallbackChain:
    """Ordered fallback options for a capability."""

    capability: str
    chain: list[str]  # ordered model/service names to try
    current_index: int = 0

    @property
    def current(self) -> str | None:
        if self.current_index < len(self.chain):
            return self.chain[self.current_index]
        return None

    def advance(self) -> str | None:
        """Move to next fallback. Returns new current or None if exhausted."""
        self.current_index += 1
        return self.current

    def reset(self) -> None:
        self.current_index = 0


# Default fallback chains for LLM model selection
_DEFAULT_FALLBACK_CHAINS: dict[str, list[str]] = {
    "powerful": ["claude-sonnet-4-20250514", "gpt-4o", "gpt-4o-mini", "ollama/qwen3"],
    "balanced": ["gpt-4o", "gpt-4o-mini", "ollama/qwen3"],
    "fast": ["gpt-4o-mini", "ollama/llama3.2"],
}


class DegradationPolicy:
    """Central policy engine for graceful degradation."""

    def __init__(self) -> None:
        self._level = DegradationLevel.HEALTHY
        self._features: dict[str, FeatureFlag] = {
            k: FeatureFlag(
                name=v.name,
                enabled=v.enabled,
                min_level=v.min_level,
                description=v.description,
            )
            for k, v in _DEFAULT_FEATURES.items()
        }
        self._fallback_chains: dict[str, FallbackChain] = {
            name: FallbackChain(capability=name, chain=list(chain))
            for name, chain in _DEFAULT_FALLBACK_CHAINS.items()
        }
        self._overrides: dict[str, bool] = {}  # manual kill switches

    @property
    def level(self) -> DegradationLevel:
        return self._level

    def set_level(self, level: DegradationLevel) -> None:
        """Set degradation level. Automatically gates features."""
        old = self._level
        self._level = level
        if old != level:
            log.warning(
                "degradation.level_changed from=%s to=%s",
                old.name,
                level.name,
            )

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is available at current degradation level."""
        # Manual override takes precedence
        if feature in self._overrides:
            return self._overrides[feature]

        ff = self._features.get(feature)
        if ff is None:
            return True  # unknown features default to enabled

        return ff.enabled and self._level <= ff.min_level

    def disable_feature(self, feature: str) -> None:
        """Manually kill-switch a feature regardless of level."""
        self._overrides[feature] = False
        log.warning("degradation.feature_disabled feature=%s", feature)

    def enable_feature(self, feature: str) -> None:
        """Remove manual override, restoring level-based gating."""
        self._overrides.pop(feature, None)
        log.info("degradation.feature_enabled feature=%s", feature)

    def get_fallback(self, tier: str) -> str | None:
        """Get current fallback model for a tier."""
        chain = self._fallback_chains.get(tier)
        return chain.current if chain else None

    def advance_fallback(self, tier: str) -> str | None:
        """Advance to next fallback for a tier (e.g., after provider error)."""
        chain = self._fallback_chains.get(tier)
        if chain:
            next_model = chain.advance()
            if next_model:
                log.warning(
                    "degradation.fallback_advanced tier=%s model=%s",
                    tier,
                    next_model,
                )
            return next_model
        return None

    def reset_fallbacks(self) -> None:
        """Reset all fallback chains to primary."""
        for chain in self._fallback_chains.values():
            chain.reset()
        log.info("degradation.fallbacks_reset")

    def status(self) -> dict[str, Any]:
        """Return full degradation status for monitoring."""
        return {
            "level": self._level.name,
            "features": {
                name: {
                    "enabled": self.is_feature_enabled(name),
                    "overridden": name in self._overrides,
                    "min_level": ff.min_level.name,
                }
                for name, ff in self._features.items()
            },
            "fallback_chains": {
                name: {
                    "current": chain.current,
                    "index": chain.current_index,
                    "exhausted": chain.current is None,
                }
                for name, chain in self._fallback_chains.items()
            },
        }

    def assess_from_health(self, health_report: dict[str, Any]) -> DegradationLevel:
        """Compute degradation level from a Doctor health report.

        Maps health check results to a degradation level.
        """
        checks = health_report.get("checks", [])
        fail_count = sum(1 for c in checks if c.get("status") == "fail")
        warn_count = sum(1 for c in checks if c.get("status") == "warn")

        if fail_count >= 3:
            return DegradationLevel.MINIMAL
        if fail_count >= 1:
            return DegradationLevel.FALLBACK
        if warn_count >= 2:
            return DegradationLevel.DEGRADED
        return DegradationLevel.HEALTHY

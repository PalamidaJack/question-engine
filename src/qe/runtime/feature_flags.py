"""General-purpose runtime feature flags.

Distinct from the degradation system, which provides system-level
kill-switches. Feature flags support named flags with enable/disable,
percentage rollouts, and context-based targeting.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class FeatureFlagDef:
    """Definition of a feature flag."""

    name: str
    enabled: bool = False
    description: str = ""
    rollout_pct: float = 100.0  # 0-100, percentage of contexts that see it
    targeting: dict[str, list[str]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "description": self.description,
            "rollout_pct": self.rollout_pct,
            "targeting": self.targeting,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class FeatureFlagStore:
    """In-memory feature flag store with evaluation engine.

    Supports:
    - Simple enable/disable flags
    - Percentage rollouts (deterministic by context key)
    - Context-based targeting (e.g., user_id, goal_id)
    - Default values for undefined flags

    Usage:
        store = FeatureFlagStore()
        store.define("new_planner", enabled=True, rollout_pct=50)
        store.add_targeting("new_planner", "user_id", ["alice", "bob"])

        if store.is_enabled("new_planner", {"user_id": "alice"}):
            # use new planner
    """

    def __init__(self) -> None:
        self._flags: dict[str, FeatureFlagDef] = {}
        self._evaluation_log: list[dict[str, Any]] = []
        self._max_log = 1000

    def define(
        self,
        name: str,
        *,
        enabled: bool = False,
        description: str = "",
        rollout_pct: float = 100.0,
    ) -> FeatureFlagDef:
        """Define or update a feature flag."""
        if name in self._flags:
            flag = self._flags[name]
            flag.enabled = enabled
            flag.description = description or flag.description
            flag.rollout_pct = rollout_pct
            flag.updated_at = time.time()
        else:
            flag = FeatureFlagDef(
                name=name,
                enabled=enabled,
                description=description,
                rollout_pct=rollout_pct,
            )
            self._flags[name] = flag

        log.debug(
            "feature_flag.defined name=%s enabled=%s rollout=%.0f%%",
            name,
            enabled,
            rollout_pct,
        )
        return flag

    def is_enabled(
        self,
        name: str,
        context: dict[str, str] | None = None,
        default: bool = False,
    ) -> bool:
        """Evaluate whether a feature flag is enabled for the given context.

        Evaluation order:
        1. Flag must exist (returns default if not)
        2. Flag must be globally enabled
        3. If targeting rules exist, context must match at least one
        4. Rollout percentage check (deterministic hash of flag + context key)
        """
        flag = self._flags.get(name)
        if flag is None:
            return default

        if not flag.enabled:
            self._log_eval(name, context, False, "disabled")
            return False

        # Check targeting rules
        if flag.targeting and context:
            matched = False
            for key, allowed_values in flag.targeting.items():
                ctx_value = context.get(key)
                if ctx_value and ctx_value in allowed_values:
                    matched = True
                    break
            if not matched:
                self._log_eval(name, context, False, "targeting_miss")
                return False
        elif flag.targeting and not context:
            # Has targeting rules but no context — deny
            self._log_eval(name, context, False, "no_context")
            return False

        # Percentage rollout check
        if flag.rollout_pct < 100.0:
            if not self._in_rollout(name, context, flag.rollout_pct):
                self._log_eval(name, context, False, "rollout_excluded")
                return False

        self._log_eval(name, context, True, "enabled")
        return True

    def _in_rollout(
        self,
        name: str,
        context: dict[str, str] | None,
        pct: float,
    ) -> bool:
        """Deterministic rollout check using hash of flag name + context."""
        # Build a stable key for hashing
        ctx_key = ""
        if context:
            # Use first context value for consistent bucketing
            for v in sorted(context.values()):
                ctx_key = v
                break

        hash_input = f"{name}:{ctx_key}".encode()
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)  # noqa: S324
        bucket = hash_val % 100
        return bucket < pct

    def _log_eval(
        self,
        name: str,
        context: dict[str, str] | None,
        result: bool,
        reason: str,
    ) -> None:
        """Record flag evaluation for debugging."""
        self._evaluation_log.append({
            "flag": name,
            "context": context,
            "result": result,
            "reason": reason,
            "timestamp": time.time(),
        })
        if len(self._evaluation_log) > self._max_log:
            self._evaluation_log = self._evaluation_log[-self._max_log:]

    def add_targeting(
        self,
        name: str,
        key: str,
        values: list[str],
    ) -> bool:
        """Add targeting rules to a flag. Returns False if flag not found."""
        flag = self._flags.get(name)
        if flag is None:
            return False
        flag.targeting[key] = values
        flag.updated_at = time.time()
        log.debug(
            "feature_flag.targeting name=%s key=%s values=%s",
            name,
            key,
            values,
        )
        return True

    def clear_targeting(self, name: str) -> bool:
        """Remove all targeting rules from a flag."""
        flag = self._flags.get(name)
        if flag is None:
            return False
        flag.targeting.clear()
        flag.updated_at = time.time()
        return True

    def enable(self, name: str) -> bool:
        """Enable a flag. Returns False if flag not found."""
        flag = self._flags.get(name)
        if flag is None:
            return False
        flag.enabled = True
        flag.updated_at = time.time()
        log.info("feature_flag.enabled name=%s", name)
        return True

    def disable(self, name: str) -> bool:
        """Disable a flag. Returns False if flag not found."""
        flag = self._flags.get(name)
        if flag is None:
            return False
        flag.enabled = False
        flag.updated_at = time.time()
        log.info("feature_flag.disabled name=%s", name)
        return True

    def delete(self, name: str) -> bool:
        """Remove a flag entirely."""
        return self._flags.pop(name, None) is not None

    def get(self, name: str) -> FeatureFlagDef | None:
        """Get flag definition by name."""
        return self._flags.get(name)

    def list_flags(self) -> list[dict[str, Any]]:
        """Return all flag definitions."""
        return [f.to_dict() for f in self._flags.values()]

    def evaluation_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent evaluation log entries."""
        return list(reversed(self._evaluation_log[-limit:]))

    def stats(self) -> dict[str, Any]:
        """Return feature flag statistics."""
        total = len(self._flags)
        enabled = sum(1 for f in self._flags.values() if f.enabled)
        return {
            "total_flags": total,
            "enabled_flags": enabled,
            "disabled_flags": total - enabled,
            "total_evaluations": len(self._evaluation_log),
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_flag_store: FeatureFlagStore | None = None


def get_flag_store() -> FeatureFlagStore:
    global _flag_store
    if _flag_store is None:
        _flag_store = FeatureFlagStore()
    return _flag_store


def reset_flag_store() -> None:
    """Reset flag store (for testing)."""
    global _flag_store
    _flag_store = None

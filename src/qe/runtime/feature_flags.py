"""General-purpose runtime feature flags.

Distinct from the degradation system, which provides system-level
kill-switches. Feature flags support named flags with enable/disable,
percentage rollouts, and context-based targeting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class FlagConflictError(Exception):
    def __init__(self, flag: str, conflicting_flag: str) -> None:
        self.flag = flag
        self.conflicting_flag = conflicting_flag
        super().__init__(
            f"Cannot enable '{flag}': conflicts with"
            f" enabled flag '{conflicting_flag}'"
        )


MATURITY_LEVELS = ("experimental", "preview", "stable", "deprecated")


@dataclass
class FeatureFlagDef:
    name: str
    enabled: bool = False
    description: str = ""
    rollout_pct: float = 100.0
    targeting: dict[str, list[str]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    category: str = "general"
    default: bool = False
    risk: str = "low"
    risk_note: str = ""
    requires: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    effects: dict = field(default_factory=dict)
    recommended: bool = False
    experimental: bool = False
    maturity: str = "stable"  # experimental | preview | stable | deprecated

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "description": self.description,
            "rollout_pct": self.rollout_pct,
            "targeting": self.targeting,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "category": self.category,
            "default": self.default,
            "risk": self.risk,
            "risk_note": self.risk_note,
            "requires": self.requires,
            "conflicts": self.conflicts,
            "effects": self.effects,
            "recommended": self.recommended,
            "experimental": self.experimental,
            "maturity": self.maturity,
        }


PRESETS: dict[str, list[str]] = {
    "conservative": [
        "chat_llm_recovery",
        "parallel_tool_calls",
        "task_aware_routing",
        "subagent_cache",
        "llm_health_check",
        "stable_prompt_prefix",
        "artifact_handles",
    ],
    "recommended": [
        "chat_llm_recovery",
        "parallel_tool_calls",
        "task_aware_routing",
        "subagent_cache",
        "llm_health_check",
        "stable_prompt_prefix",
        "artifact_handles",
        "proactive_recall",
        "knowledge_consolidation",
        "goal_orchestration",
        "inquiry_mode",
        "harvest_service",
    ],
    "full_power": [
        "chat_llm_recovery",
        "parallel_tool_calls",
        "task_aware_routing",
        "subagent_cache",
        "llm_health_check",
        "stable_prompt_prefix",
        "artifact_handles",
        "proactive_recall",
        "knowledge_consolidation",
        "goal_orchestration",
        "inquiry_mode",
        "harvest_service",
        "recitation_pattern",
        "tool_masking",
        "multi_agent_mode",
        "competitive_arena",
        "innovation_scout",
    ],
    "experimental": [
        "chat_llm_recovery",
        "parallel_tool_calls",
        "task_aware_routing",
        "subagent_cache",
        "llm_health_check",
        "stable_prompt_prefix",
        "artifact_handles",
        "proactive_recall",
        "knowledge_consolidation",
        "goal_orchestration",
        "inquiry_mode",
        "harvest_service",
        "recitation_pattern",
        "tool_masking",
        "multi_agent_mode",
        "competitive_arena",
        "innovation_scout",
        "prompt_evolution",
        "pattern_breaking",
    ],
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

    def __init__(
        self,
        persistence_path: Path | None = None,
    ) -> None:
        self._flags: dict[str, FeatureFlagDef] = {}
        self._evaluation_log: list[dict[str, Any]] = []
        self._max_log = 1000
        self._persistence_path = persistence_path

    # ── Persistence ───────────────────────────────────────────────────

    def _auto_save(self) -> None:
        if self._persistence_path is not None:
            self.save(self._persistence_path)

    def save(self, path: Path) -> None:
        data: dict[str, Any] = {}
        for name, flag in self._flags.items():
            data[name] = flag.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        raw = json.loads(path.read_text())
        for name, saved in raw.items():
            if name in self._flags:
                self._flags[name].enabled = saved.get(
                    "enabled", self._flags[name].enabled
                )
            else:
                self._flags[name] = FeatureFlagDef(
                    name=name,
                    enabled=saved.get("enabled", False),
                    description=saved.get("description", ""),
                    rollout_pct=saved.get("rollout_pct", 100.0),
                    targeting=saved.get("targeting", {}),
                    created_at=saved.get(
                        "created_at", time.time()
                    ),
                    updated_at=saved.get(
                        "updated_at", time.time()
                    ),
                    category=saved.get("category", "general"),
                    default=saved.get("default", False),
                    risk=saved.get("risk", "low"),
                    risk_note=saved.get("risk_note", ""),
                    requires=saved.get("requires", []),
                    conflicts=saved.get("conflicts", []),
                    effects=saved.get("effects", {}),
                    recommended=saved.get("recommended", False),
                    experimental=saved.get(
                        "experimental", False
                    ),
                    maturity=saved.get("maturity", "stable"),
                )

    # ── Define / Enable / Disable ─────────────────────────────────────

    def define(
        self,
        name: str,
        *,
        enabled: bool = False,
        description: str = "",
        rollout_pct: float = 100.0,
        category: str = "general",
        default: bool = False,
        risk: str = "low",
        risk_note: str = "",
        requires: list[str] | None = None,
        conflicts: list[str] | None = None,
        effects: dict | None = None,
        recommended: bool = False,
        experimental: bool = False,
        maturity: str = "stable",
    ) -> FeatureFlagDef:
        if name in self._flags:
            flag = self._flags[name]
            flag.enabled = enabled
            flag.description = description or flag.description
            flag.rollout_pct = rollout_pct
            flag.category = category
            flag.default = default
            flag.risk = risk
            flag.risk_note = risk_note
            flag.requires = requires if requires is not None else flag.requires
            flag.conflicts = (
                conflicts if conflicts is not None else flag.conflicts
            )
            flag.effects = effects if effects is not None else flag.effects
            flag.recommended = recommended
            flag.experimental = experimental
            flag.maturity = maturity
            flag.updated_at = time.time()
        else:
            flag = FeatureFlagDef(
                name=name,
                enabled=enabled,
                description=description,
                rollout_pct=rollout_pct,
                category=category,
                default=default,
                risk=risk,
                risk_note=risk_note,
                requires=requires or [],
                conflicts=conflicts or [],
                effects=effects or {},
                recommended=recommended,
                experimental=experimental,
                maturity=maturity,
            )
            self._flags[name] = flag

        log.debug(
            "feature_flag.defined name=%s enabled=%s rollout=%.0f%%",
            name,
            enabled,
            rollout_pct,
        )
        self._auto_save()
        return flag

    def _check_conflicts(self, name: str) -> None:
        flag = self._flags.get(name)
        if flag is None:
            return
        for conflict in flag.conflicts:
            other = self._flags.get(conflict)
            if other is not None and other.enabled:
                raise FlagConflictError(name, conflict)
        for other_name, other_flag in self._flags.items():
            if other_name == name:
                continue
            if other_flag.enabled and name in other_flag.conflicts:
                raise FlagConflictError(name, other_name)

    def enable(self, name: str) -> bool:
        flag = self._flags.get(name)
        if flag is None:
            return False
        self._check_conflicts(name)
        flag.enabled = True
        flag.updated_at = time.time()
        log.info("feature_flag.enabled name=%s", name)
        self._auto_save()
        return True

    def disable(self, name: str) -> bool:
        flag = self._flags.get(name)
        if flag is None:
            return False
        flag.enabled = False
        flag.updated_at = time.time()
        log.info("feature_flag.disabled name=%s", name)
        self._auto_save()
        return True

    # ── Dependency / Conflict helpers ─────────────────────────────────

    def enable_with_deps(self, name: str) -> dict[str, list[str]]:
        flag = self._flags.get(name)
        if flag is None:
            return {"enabled": [], "already_on": []}

        enabled: list[str] = []
        already_on: list[str] = []

        to_enable = self._resolve_deps(name)
        for dep in to_enable:
            dep_flag = self._flags.get(dep)
            if dep_flag is None:
                continue
            if dep_flag.enabled:
                already_on.append(dep)
            else:
                self._check_conflicts(dep)
                dep_flag.enabled = True
                dep_flag.updated_at = time.time()
                enabled.append(dep)
                log.info("feature_flag.enabled name=%s", dep)

        self._auto_save()
        return {"enabled": enabled, "already_on": already_on}

    def _resolve_deps(self, name: str) -> list[str]:
        ordered: list[str] = []
        visited: set[str] = set()

        def _walk(n: str) -> None:
            if n in visited:
                return
            visited.add(n)
            flag = self._flags.get(n)
            if flag is None:
                return
            for req in flag.requires:
                _walk(req)
            ordered.append(n)

        _walk(name)
        return ordered

    def disable_with_check(self, name: str) -> dict[str, Any]:
        flag = self._flags.get(name)
        if flag is None:
            return {"disabled": name, "warnings": []}

        warnings: list[str] = []
        for other_name, other_flag in self._flags.items():
            if other_name == name:
                continue
            if other_flag.enabled and name in other_flag.requires:
                warnings.append(f"flag {other_name} requires this")

        flag.enabled = False
        flag.updated_at = time.time()
        log.info("feature_flag.disabled name=%s", name)
        self._auto_save()
        return {"disabled": name, "warnings": warnings}

    # ── Presets ────────────────────────────────────────────────────────

    def apply_preset(self, preset_name: str) -> dict[str, Any]:
        if preset_name not in PRESETS:
            return {"error": f"unknown preset: {preset_name}"}

        target = set(PRESETS[preset_name])
        enabled_list: list[str] = []
        disabled_list: list[str] = []

        for name, flag in self._flags.items():
            should_be_on = name in target
            if should_be_on and not flag.enabled:
                self._check_conflicts(name)
                flag.enabled = True
                flag.updated_at = time.time()
                enabled_list.append(name)
            elif not should_be_on and flag.enabled:
                flag.enabled = False
                flag.updated_at = time.time()
                disabled_list.append(name)

        self._auto_save()
        return {
            "preset": preset_name,
            "enabled": enabled_list,
            "disabled": disabled_list,
        }

    def current_preset(self) -> str | None:
        active = {
            n for n, f in self._flags.items() if f.enabled
        }
        for preset_name in reversed(list(PRESETS)):
            if active == set(PRESETS[preset_name]):
                return preset_name
        return None

    # ── Evaluation ────────────────────────────────────────────────────

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
                self._log_eval(
                    name, context, False, "targeting_miss"
                )
                return False
        elif flag.targeting and not context:
            self._log_eval(name, context, False, "no_context")
            return False

        # Percentage rollout check
        if flag.rollout_pct < 100.0:
            if not self._in_rollout(
                name, context, flag.rollout_pct
            ):
                self._log_eval(
                    name, context, False, "rollout_excluded"
                )
                return False

        self._log_eval(name, context, True, "enabled")
        return True

    def _in_rollout(
        self,
        name: str,
        context: dict[str, str] | None,
        pct: float,
    ) -> bool:
        ctx_key = ""
        if context:
            for v in sorted(context.values()):
                ctx_key = v
                break

        hash_input = f"{name}:{ctx_key}".encode()
        hash_val = int(
            hashlib.md5(hash_input).hexdigest(), 16  # noqa: S324
        )
        bucket = hash_val % 100
        return bucket < pct

    def _log_eval(
        self,
        name: str,
        context: dict[str, str] | None,
        result: bool,
        reason: str,
    ) -> None:
        self._evaluation_log.append({
            "flag": name,
            "context": context,
            "result": result,
            "reason": reason,
            "timestamp": time.time(),
        })
        if len(self._evaluation_log) > self._max_log:
            self._evaluation_log = self._evaluation_log[
                -self._max_log:
            ]

    # ── Targeting ─────────────────────────────────────────────────────

    def add_targeting(
        self,
        name: str,
        key: str,
        values: list[str],
    ) -> bool:
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
        flag = self._flags.get(name)
        if flag is None:
            return False
        flag.targeting.clear()
        flag.updated_at = time.time()
        return True

    # ── Query ─────────────────────────────────────────────────────────

    def delete(self, name: str) -> bool:
        return self._flags.pop(name, None) is not None

    def get(self, name: str) -> FeatureFlagDef | None:
        return self._flags.get(name)

    def list_flags(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for flag in self._flags.values():
            cat = flag.category
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(flag.to_dict())
        return grouped

    def categories(self) -> list[str]:
        cats: list[str] = []
        seen: set[str] = set()
        for flag in self._flags.values():
            if flag.category not in seen:
                seen.add(flag.category)
                cats.append(flag.category)
        return cats

    def evaluation_log(
        self, limit: int = 100
    ) -> list[dict[str, Any]]:
        return list(reversed(self._evaluation_log[-limit:]))

    def list_by_maturity(self, maturity: str) -> list[FeatureFlagDef]:
        """Return all flags with the given maturity level."""
        return [f for f in self._flags.values() if f.maturity == maturity]

    def stats(self) -> dict[str, Any]:
        total = len(self._flags)
        enabled = sum(
            1 for f in self._flags.values() if f.enabled
        )
        maturity_counts = {}
        for f in self._flags.values():
            maturity_counts[f.maturity] = maturity_counts.get(f.maturity, 0) + 1
        return {
            "total_flags": total,
            "enabled_flags": enabled,
            "disabled_flags": total - enabled,
            "total_evaluations": len(self._evaluation_log),
            "maturity": maturity_counts,
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_flag_store: FeatureFlagStore | None = None


def get_flag_store() -> FeatureFlagStore:
    global _flag_store
    if _flag_store is None:
        _flag_store = FeatureFlagStore()
    return _flag_store


def reset_flag_store() -> None:
    global _flag_store
    _flag_store = None

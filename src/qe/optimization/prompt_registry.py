"""Prompt Registry — Thompson sampling over prompt variants.

Sits between cognitive components and their prompt templates, enabling
A/B testing and automated evolution of prompts. When disabled (default),
returns the original hardcoded prompt with zero overhead.

Uses BetaArm from routing_optimizer for Thompson sampling.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import aiosqlite
from pydantic import BaseModel, Field

from qe.runtime.routing_optimizer import BetaArm

log = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────


class PromptVariant(BaseModel):
    """A prompt variant registered for a slot."""

    variant_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    slot_key: str
    content: str
    is_baseline: bool = False
    rollout_pct: float = 100.0
    parent_variant_id: str | None = None
    active: bool = True
    alpha: float = 1.0
    beta: float = 1.0


class PromptOutcome(BaseModel):
    """Outcome of a prompt variant invocation."""

    variant_id: str
    slot_key: str
    success: bool
    quality_score: float = 0.0
    latency_ms: int = 0
    error: str = ""


# ── Registry ──────────────────────────────────────────────────────────────


class PromptRegistry:
    """Registry for prompt variants with Thompson sampling selection.

    Hot path (get_prompt) is pure Python with no I/O.
    Persistence is async (initialize/persist).
    """

    def __init__(
        self,
        db_path: str | None = None,
        bus: Any = None,
        enabled: bool = False,
    ) -> None:
        self._db_path = db_path
        self._bus = bus
        self._enabled = enabled

        # slot_key -> baseline content string
        self._baselines: dict[str, str] = {}
        # slot_key -> list of PromptVariant
        self._variants: dict[str, list[PromptVariant]] = {}
        # variant_id -> BetaArm
        self._arms: dict[str, BetaArm] = {}
        # outcome buffer for persistence
        self._outcome_buffer: list[PromptOutcome] = []

    async def initialize(self) -> None:
        """Create tables and load existing variants from SQLite."""
        if not self._db_path:
            return

        async with aiosqlite.connect(self._db_path) as db:
            # Apply migration inline (idempotent)
            migration = (
                "CREATE TABLE IF NOT EXISTS prompt_variants ("
                "    variant_id TEXT PRIMARY KEY,"
                "    slot_key TEXT NOT NULL,"
                "    content TEXT NOT NULL,"
                "    is_baseline INTEGER NOT NULL DEFAULT 0,"
                "    rollout_pct REAL NOT NULL DEFAULT 100.0,"
                "    parent_variant_id TEXT,"
                "    active INTEGER NOT NULL DEFAULT 1,"
                "    alpha REAL NOT NULL DEFAULT 1.0,"
                "    beta REAL NOT NULL DEFAULT 1.0,"
                "    created_at TEXT NOT NULL"
                ");"
            )
            await db.executescript(migration)

            migration2 = (
                "CREATE TABLE IF NOT EXISTS prompt_outcomes ("
                "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "    variant_id TEXT NOT NULL,"
                "    slot_key TEXT NOT NULL,"
                "    success INTEGER NOT NULL,"
                "    quality_score REAL NOT NULL DEFAULT 0.0,"
                "    latency_ms INTEGER NOT NULL DEFAULT 0,"
                "    error TEXT NOT NULL DEFAULT '',"
                "    recorded_at TEXT NOT NULL"
                ");"
            )
            await db.executescript(migration2)

            # Load existing variants
            async with db.execute(
                "SELECT variant_id, slot_key, content, is_baseline, "
                "rollout_pct, parent_variant_id, active, alpha, beta "
                "FROM prompt_variants WHERE active = 1"
            ) as cursor:
                async for row in cursor:
                    variant = PromptVariant(
                        variant_id=row[0],
                        slot_key=row[1],
                        content=row[2],
                        is_baseline=bool(row[3]),
                        rollout_pct=row[4],
                        parent_variant_id=row[5],
                        active=bool(row[6]),
                        alpha=row[7],
                        beta=row[8],
                    )
                    self._variants.setdefault(variant.slot_key, []).append(variant)
                    self._arms[variant.variant_id] = BetaArm(
                        alpha=variant.alpha, beta=variant.beta
                    )
                    if variant.is_baseline:
                        self._baselines[variant.slot_key] = variant.content

        log.info(
            "prompt_registry.initialized slots=%d variants=%d",
            len(self._baselines),
            sum(len(v) for v in self._variants.values()),
        )

    def register_baseline(self, slot_key: str, content: str) -> PromptVariant:
        """Register the baseline (hardcoded) prompt for a slot.

        Idempotent: if already registered with same content, returns existing.
        """
        if slot_key in self._baselines:
            # Already registered — find and return existing variant
            for v in self._variants.get(slot_key, []):
                if v.is_baseline:
                    return v

        variant = PromptVariant(
            variant_id=f"base_{hashlib.sha256(slot_key.encode()).hexdigest()[:10]}",
            slot_key=slot_key,
            content=content,
            is_baseline=True,
            rollout_pct=100.0,
        )
        self._baselines[slot_key] = content
        self._variants.setdefault(slot_key, []).append(variant)
        self._arms[variant.variant_id] = BetaArm(alpha=variant.alpha, beta=variant.beta)
        return variant

    def add_variant(
        self,
        slot_key: str,
        content: str,
        rollout_pct: float = 10.0,
        parent_variant_id: str | None = None,
        strategy: str = "manual",
    ) -> PromptVariant:
        """Add a new prompt variant for a slot."""
        variant = PromptVariant(
            slot_key=slot_key,
            content=content,
            rollout_pct=rollout_pct,
            parent_variant_id=parent_variant_id,
        )
        self._variants.setdefault(slot_key, []).append(variant)
        self._arms[variant.variant_id] = BetaArm(alpha=variant.alpha, beta=variant.beta)

        if self._bus:
            self._bus.publish_sync(
                "prompt.variant_created",
                {
                    "slot_key": slot_key,
                    "variant_id": variant.variant_id,
                    "parent_variant_id": parent_variant_id or "",
                    "strategy": strategy,
                },
            )

        return variant

    def get_prompt(
        self,
        slot_key: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Get the prompt content for a slot key.

        Returns (content, variant_id). Hot path — no I/O.

        When disabled, returns the baseline prompt.
        When enabled, Thompson-samples among eligible variants.
        """
        if not self._enabled:
            baseline = self._baselines.get(slot_key)
            if baseline is None:
                raise KeyError(f"No baseline registered for slot: {slot_key}")
            return baseline, "baseline"

        eligible = [
            v
            for v in self._variants.get(slot_key, [])
            if v.active and self._in_rollout(v, context)
        ]
        if not eligible:
            baseline = self._baselines.get(slot_key)
            if baseline is None:
                raise KeyError(f"No baseline registered for slot: {slot_key}")
            return baseline, "baseline"

        best = max(eligible, key=lambda v: self._arms[v.variant_id].sample())

        if self._bus:
            self._bus.publish_sync(
                "prompt.variant_selected",
                {
                    "slot_key": slot_key,
                    "variant_id": best.variant_id,
                    "is_baseline": best.is_baseline,
                },
            )

        return best.content, best.variant_id

    def record_outcome(
        self,
        variant_id: str,
        slot_key: str,
        success: bool,
        quality_score: float = 0.0,
    ) -> None:
        """Record the outcome of a prompt invocation.

        Updates the BetaArm and buffers for persistence.
        """
        if variant_id == "baseline":
            return

        arm = self._arms.get(variant_id)
        if arm is not None:
            arm.update(success)

        outcome = PromptOutcome(
            variant_id=variant_id,
            slot_key=slot_key,
            success=success,
            quality_score=quality_score,
        )
        self._outcome_buffer.append(outcome)

        if self._bus:
            self._bus.publish_sync(
                "prompt.outcome_recorded",
                {
                    "slot_key": slot_key,
                    "variant_id": variant_id,
                    "success": success,
                    "quality_score": quality_score,
                },
            )

    async def persist(self) -> None:
        """Save variant arms and buffered outcomes to SQLite."""
        if not self._db_path:
            return

        async with aiosqlite.connect(self._db_path) as db:
            # Update variant alpha/beta
            for slot_variants in self._variants.values():
                for v in slot_variants:
                    arm = self._arms.get(v.variant_id)
                    if arm:
                        await db.execute(
                            "INSERT OR REPLACE INTO prompt_variants "
                            "(variant_id, slot_key, content, is_baseline, rollout_pct, "
                            "parent_variant_id, active, alpha, beta, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                v.variant_id,
                                v.slot_key,
                                v.content,
                                int(v.is_baseline),
                                v.rollout_pct,
                                v.parent_variant_id,
                                int(v.active),
                                arm.alpha,
                                arm.beta,
                                datetime.now(UTC).isoformat(),
                            ),
                        )

            # Flush outcome buffer
            for o in self._outcome_buffer:
                await db.execute(
                    "INSERT INTO prompt_outcomes "
                    "(variant_id, slot_key, success, quality_score, latency_ms, "
                    "error, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        o.variant_id,
                        o.slot_key,
                        int(o.success),
                        o.quality_score,
                        o.latency_ms,
                        o.error,
                        datetime.now(UTC).isoformat(),
                    ),
                )

            await db.commit()

        self._outcome_buffer.clear()
        log.info("prompt_registry.persisted")

    async def deactivate_variant(self, variant_id: str) -> None:
        """Deactivate a variant (remove from selection)."""
        for slot_variants in self._variants.values():
            for v in slot_variants:
                if v.variant_id == variant_id:
                    v.active = False

                    if self._bus:
                        self._bus.publish_sync(
                            "prompt.variant_deactivated",
                            {
                                "slot_key": v.slot_key,
                                "variant_id": variant_id,
                                "reason": "manual",
                            },
                        )
                    return

    def promote_variant(self, variant_id: str, new_rollout_pct: float) -> bool:
        """Increase a variant's rollout percentage. Returns False if not found."""
        for slot_variants in self._variants.values():
            for v in slot_variants:
                if v.variant_id == variant_id:
                    v.rollout_pct = new_rollout_pct
                    return True
        return False

    def get_slot_stats(self, slot_key: str) -> list[dict[str, Any]]:
        """Get statistics for all variants in a slot."""
        stats: list[dict[str, Any]] = []
        for v in self._variants.get(slot_key, []):
            arm = self._arms.get(v.variant_id)
            stats.append({
                "variant_id": v.variant_id,
                "is_baseline": v.is_baseline,
                "active": v.active,
                "rollout_pct": v.rollout_pct,
                "alpha": arm.alpha if arm else 1.0,
                "beta": arm.beta if arm else 1.0,
                "mean": arm.mean if arm else 0.5,
                "sample_count": arm.sample_count if arm else 0,
            })
        return stats

    def get_best_variant(self, slot_key: str) -> PromptVariant | None:
        """Get the variant with the highest posterior mean for a slot."""
        variants = self._variants.get(slot_key, [])
        active = [v for v in variants if v.active]
        if not active:
            return None
        return max(
            active,
            key=lambda v: self._arms[v.variant_id].mean,
        )

    def status(self) -> dict[str, Any]:
        """Monitoring snapshot."""
        total_variants = sum(len(vs) for vs in self._variants.values())
        active_variants = sum(
            sum(1 for v in vs if v.active) for vs in self._variants.values()
        )
        return {
            "enabled": self._enabled,
            "slots": len(self._baselines),
            "total_variants": total_variants,
            "active_variants": active_variants,
            "pending_outcomes": len(self._outcome_buffer),
            "slot_keys": sorted(self._baselines.keys()),
        }

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _in_rollout(
        variant: PromptVariant,
        context: dict[str, Any] | None,
    ) -> bool:
        """Check if a variant is within its rollout percentage.

        Uses deterministic hashing for consistent assignment.
        """
        if variant.rollout_pct >= 100.0:
            return True
        if variant.rollout_pct <= 0.0:
            return False

        # Deterministic hash based on variant_id + context
        seed = variant.variant_id
        if context:
            seed += str(sorted(context.items()))
        h = int(hashlib.md5(seed.encode()).hexdigest(), 16)  # noqa: S324
        bucket = (h % 100) + 1  # 1-100
        return bucket <= variant.rollout_pct


# ── Baseline Registration ─────────────────────────────────────────────────


def register_all_baselines(registry: PromptRegistry) -> None:
    """Register all hardcoded prompts from cognitive components as baselines.

    Centralizes baseline knowledge so the registry knows every slot.
    """
    # --- DialecticEngine ---
    from qe.services.inquiry.dialectic import (
        _ASSUMPTION_SURFACING_PROMPT,
        _DEVILS_ADVOCATE_PROMPT,
        _PERSPECTIVE_PROMPT,
        _RED_TEAM_PROMPT,
    )

    registry.register_baseline("dialectic.challenge.system", (
        "You are a devil's advocate. "
        "You MUST argue against the conclusion."
    ))
    registry.register_baseline("dialectic.challenge.user", _DEVILS_ADVOCATE_PROMPT)
    registry.register_baseline("dialectic.perspectives.system", (
        "You are a {perspective_name}. Stay in character."
    ))
    registry.register_baseline("dialectic.perspectives.user", _PERSPECTIVE_PROMPT)
    registry.register_baseline("dialectic.assumptions.system", (
        "You are an assumption-detection module."
    ))
    registry.register_baseline("dialectic.assumptions.user", _ASSUMPTION_SURFACING_PROMPT)
    registry.register_baseline("dialectic.red_team.system", (
        "You are a red team analyst. "
        "You MUST attack the finding."
    ))
    registry.register_baseline("dialectic.red_team.user", _RED_TEAM_PROMPT)

    # --- InsightCrystallizer ---
    from qe.services.inquiry.insight import (
        _ACTIONABILITY_PROMPT,
        _CROSS_DOMAIN_PROMPT,
        _MECHANISM_PROMPT,
        _NOVELTY_PROMPT,
    )

    registry.register_baseline("insight.novelty.system", (
        "You are a strict novelty assessor. "
        "Most findings are NOT novel."
    ))
    registry.register_baseline("insight.novelty.user", _NOVELTY_PROMPT)
    registry.register_baseline("insight.mechanism.system", (
        "You are a causal reasoning module. Be specific."
    ))
    registry.register_baseline("insight.mechanism.user", _MECHANISM_PROMPT)
    registry.register_baseline("insight.actionability.system", (
        "You are an actionability assessor."
    ))
    registry.register_baseline("insight.actionability.user", _ACTIONABILITY_PROMPT)
    registry.register_baseline("insight.cross_domain.system", (
        "You find structural analogies across domains."
    ))
    registry.register_baseline("insight.cross_domain.user", _CROSS_DOMAIN_PROMPT)

    # --- Metacognitor ---
    from qe.runtime.metacognitor import (
        _APPROACH_PROMPT,
        _TOOL_COMBINATION_PROMPT,
    )

    registry.register_baseline("metacognitor.approach.system", (
        "You are a metacognitive reasoning module."
    ))
    registry.register_baseline("metacognitor.approach.user", _APPROACH_PROMPT)
    registry.register_baseline("metacognitor.tool_combo.system", (
        "You are a creative problem-solving module."
    ))
    registry.register_baseline("metacognitor.tool_combo.user", _TOOL_COMBINATION_PROMPT)

    # --- EpistemicReasoner ---
    from qe.runtime.epistemic_reasoner import (
        _ABSENCE_DETECTION_PROMPT,
        _SURPRISE_PROMPT,
        _UNCERTAINTY_PROMPT,
    )

    registry.register_baseline("epistemic.absence.system", (
        "You are an epistemic reasoning module."
    ))
    registry.register_baseline("epistemic.absence.user", _ABSENCE_DETECTION_PROMPT)
    registry.register_baseline("epistemic.uncertainty.system", (
        "You are an epistemic reasoning module."
    ))
    registry.register_baseline("epistemic.uncertainty.user", _UNCERTAINTY_PROMPT)
    registry.register_baseline("epistemic.surprise.system", (
        "You are a surprise detection module."
    ))
    registry.register_baseline("epistemic.surprise.user", _SURPRISE_PROMPT)

    # --- PersistenceEngine ---
    from qe.runtime.persistence_engine import (
        _REFRAME_PROMPT,
        _ROOT_CAUSE_PROMPT,
    )

    registry.register_baseline("persistence.root_cause.system", (
        "You are a root cause analysis expert."
    ))
    registry.register_baseline("persistence.root_cause.user", _ROOT_CAUSE_PROMPT)
    registry.register_baseline("persistence.reframe.system", (
        "You are a creative problem reframing module."
    ))
    registry.register_baseline("persistence.reframe.user", _REFRAME_PROMPT)

    # --- QuestionGenerator ---
    registry.register_baseline("question_gen.generate.system", (
        "You are a research question generator. Generate precise, "
        "non-overlapping questions that maximize information gain. "
        "Later iterations should focus on gaps and surprises. "
        "If hypotheses are present, include falsification questions."
    ))

    # --- HypothesisManager ---
    registry.register_baseline("hypothesis.generate.system", (
        "You are a hypothesis generator following Popperian "
        "philosophy. Every hypothesis MUST be falsifiable. "
        "Generate competing hypotheses that cover different "
        "explanations for the same observations."
    ))

    log.info(
        "prompt_registry.baselines_registered count=%d",
        len(registry._baselines),
    )

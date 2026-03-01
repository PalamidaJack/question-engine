"""PromptMutator — LLM-powered prompt variant generation.

Uses LLM calls to auto-generate prompt variants via four mutation
strategies: rephrase, elaborate, simplify, restructure. Monitors
variant performance, auto-rolls back failures, and promotes winners.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import instructor
import litellm
from pydantic import BaseModel

from qe.optimization.prompt_registry import PromptRegistry, PromptVariant
from qe.runtime.feature_flags import get_flag_store

log = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────


class MutatedPrompt(BaseModel):
    """LLM response model for a mutated prompt."""

    mutated_content: str
    mutation_rationale: str
    preserved_format_keys: list[str]


# ── Constants ─────────────────────────────────────────────────────────────

MUTATION_STRATEGIES: dict[str, str] = {
    "rephrase": (
        "Rephrase the prompt using different wording while preserving "
        "the exact same meaning and all {format_keys}. "
        "Change vocabulary and sentence structure."
    ),
    "elaborate": (
        "Elaborate on the prompt by adding more specific instructions "
        "and clarifying expectations. Preserve all {format_keys}. "
        "Add detail that improves task performance."
    ),
    "simplify": (
        "Simplify the prompt by removing unnecessary words and making "
        "instructions more direct. Preserve all {format_keys}. "
        "Shorter is better if meaning is preserved."
    ),
    "restructure": (
        "Restructure the prompt by reordering instructions for better "
        "logical flow. Preserve all {format_keys}. "
        "Use numbered steps or clear sections if appropriate."
    ),
}

_MUTATION_SYSTEM_PROMPT = (
    "You are a prompt engineering expert. Your task is to mutate a prompt "
    "template used in an AI system. You MUST preserve ALL {placeholders} "
    "exactly as they appear — do not rename, remove, or add placeholders. "
    "Stay within 2x the original prompt length. "
    "Return your mutation as structured output."
)


def _build_mutation_user_prompt(
    original: str,
    strategy: str,
    strategy_instruction: str,
    slot_key: str,
) -> str:
    """Format the user prompt for the LLM mutation call."""
    return (
        f"Slot: {slot_key}\n"
        f"Strategy: {strategy}\n"
        f"Instruction: {strategy_instruction}\n\n"
        f"Original prompt:\n---\n{original}\n---\n\n"
        "Produce a mutated version of this prompt."
    )


# ── Helpers ───────────────────────────────────────────────────────────────


def _extract_format_keys(template: str) -> set[str]:
    """Extract {placeholder} keys from a template string."""
    return set(re.findall(r"\{(\w+)\}", template))


def _validate_mutation(original: str, mutated: str) -> bool:
    """Check that all format keys from original are present in mutated."""
    original_keys = _extract_format_keys(original)
    mutated_keys = _extract_format_keys(mutated)
    return original_keys <= mutated_keys


# ── PromptMutator ─────────────────────────────────────────────────────────


class PromptMutator:
    """LLM-powered prompt variant generator with auto-rollback and promotion.

    Follows the StrategyEvolver lifecycle pattern: start/stop with a
    background evaluation loop.
    """

    def __init__(
        self,
        registry: PromptRegistry,
        bus: Any = None,
        model: str = "gpt-4o-mini",
        eval_interval: float = 300.0,
        min_samples: int = 20,
        rollback_threshold: float = 0.3,
        promote_threshold: float = 0.6,
        max_variants_per_slot: int = 5,
        max_mutations_per_cycle: int = 3,
        initial_rollout_pct: float = 10.0,
        promoted_rollout_pct: float = 50.0,
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._model = model
        self._eval_interval = eval_interval
        self._min_samples = min_samples
        self._rollback_threshold = rollback_threshold
        self._promote_threshold = promote_threshold
        self._max_variants_per_slot = max_variants_per_slot
        self._max_mutations_per_cycle = max_mutations_per_cycle
        self._initial_rollout_pct = initial_rollout_pct
        self._promoted_rollout_pct = promoted_rollout_pct

        self._running = False
        self._loop_task: asyncio.Task[Any] | None = None

        # Counters
        self._mutations_total = 0
        self._rollbacks_total = 0
        self._promotions_total = 0
        self._last_cycle_at: float | None = None

        # Strategy rotation index
        self._strategy_index = 0

    def start(self) -> None:
        """Start the background evaluation loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._evaluation_loop())
        log.info("prompt_mutator.started")

    async def stop(self) -> None:
        """Stop the background evaluation loop."""
        self._running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass
        self._loop_task = None
        log.info("prompt_mutator.stopped")

    async def _evaluation_loop(self) -> None:
        """Periodic loop: evaluate variants and mutate."""
        while self._running:
            try:
                await asyncio.sleep(self._eval_interval)
                await self._evaluate()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("prompt_mutator.evaluation_error")

    async def _evaluate(self) -> None:
        """Single evaluation cycle — three phases per slot."""
        if not get_flag_store().is_enabled("prompt_evolution"):
            return

        status = self._registry.status()
        slot_keys = status["slot_keys"]

        rollbacks = 0
        promotions = 0
        mutations = 0
        mutations_this_cycle = 0

        for slot_key in slot_keys:
            stats = self._registry.get_slot_stats(slot_key)

            # Phase 1: Rollback low performers
            for s in stats:
                if s["is_baseline"] or not s["active"]:
                    continue
                if (
                    s["sample_count"] >= self._min_samples
                    and s["mean"] < self._rollback_threshold
                ):
                    await self._registry.deactivate_variant(s["variant_id"])
                    rollbacks += 1
                    self._rollbacks_total += 1
                    log.info(
                        "prompt_mutator.rollback slot=%s variant=%s mean=%.3f",
                        slot_key,
                        s["variant_id"],
                        s["mean"],
                    )

            # Phase 2: Promote high performers
            for s in stats:
                if s["is_baseline"] or not s["active"]:
                    continue
                if (
                    s["sample_count"] >= self._min_samples
                    and s["mean"] > self._promote_threshold
                    and s["rollout_pct"] < self._promoted_rollout_pct
                ):
                    old_pct = s["rollout_pct"]
                    self._registry.promote_variant(
                        s["variant_id"], self._promoted_rollout_pct
                    )
                    promotions += 1
                    self._promotions_total += 1
                    self._publish(
                        "prompt.variant_promoted",
                        {
                            "slot_key": slot_key,
                            "variant_id": s["variant_id"],
                            "old_rollout_pct": old_pct,
                            "new_rollout_pct": self._promoted_rollout_pct,
                        },
                    )
                    log.info(
                        "prompt_mutator.promoted slot=%s variant=%s %.0f%%->%.0f%%",
                        slot_key,
                        s["variant_id"],
                        old_pct,
                        self._promoted_rollout_pct,
                    )

            # Phase 3: Mutate if room
            active_count = sum(1 for s in stats if s["active"])
            if (
                active_count < self._max_variants_per_slot
                and mutations_this_cycle < self._max_mutations_per_cycle
            ):
                # Pick best-performing parent
                best_parent = None
                best_mean = -1.0
                for s in stats:
                    if s["active"] and s["mean"] > best_mean:
                        best_mean = s["mean"]
                        best_parent = s

                if best_parent is not None:
                    # Find the actual variant object
                    parent_variant = self._find_variant(
                        slot_key, best_parent["variant_id"]
                    )
                    if parent_variant is not None:
                        strategy = self._next_strategy()
                        result = await self._mutate_variant(
                            slot_key, parent_variant, strategy
                        )
                        if result is not None:
                            mutations += 1
                            mutations_this_cycle += 1
                            self._mutations_total += 1

        # Persist at end of cycle
        await self._registry.persist()
        self._last_cycle_at = time.time()

        # Publish cycle summary
        self._publish(
            "prompt.mutation_cycle_completed",
            {
                "slots_evaluated": len(slot_keys),
                "variants_created": mutations,
                "variants_rolled_back": rollbacks,
                "variants_promoted": promotions,
            },
        )

        log.info(
            "prompt_mutator.cycle slots=%d mutations=%d rollbacks=%d promotions=%d",
            len(slot_keys),
            mutations,
            rollbacks,
            promotions,
        )

    async def _mutate_variant(
        self,
        slot_key: str,
        parent_variant: PromptVariant,
        strategy: str,
    ) -> PromptVariant | None:
        """Use LLM to generate a mutated prompt variant."""
        strategy_instruction = MUTATION_STRATEGIES[strategy]
        user_prompt = _build_mutation_user_prompt(
            original=parent_variant.content,
            strategy=strategy,
            strategy_instruction=strategy_instruction,
            slot_key=slot_key,
        )

        try:
            client = instructor.from_litellm(litellm.acompletion)
            result = await client.chat.completions.create(
                model=self._model,
                response_model=MutatedPrompt,
                messages=[
                    {"role": "system", "content": _MUTATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.9,
            )
        except Exception:
            log.exception(
                "prompt_mutator.llm_error slot=%s strategy=%s", slot_key, strategy
            )
            return None

        # Validate format keys preserved
        if not _validate_mutation(parent_variant.content, result.mutated_content):
            log.warning(
                "prompt_mutator.invalid_mutation slot=%s strategy=%s missing_keys",
                slot_key,
                strategy,
            )
            return None

        # Add to registry
        variant = self._registry.add_variant(
            slot_key=slot_key,
            content=result.mutated_content,
            rollout_pct=self._initial_rollout_pct,
            parent_variant_id=parent_variant.variant_id,
            strategy=strategy,
        )

        log.info(
            "prompt_mutator.created slot=%s variant=%s strategy=%s",
            slot_key,
            variant.variant_id,
            strategy,
        )
        return variant

    def _find_variant(
        self, slot_key: str, variant_id: str
    ) -> PromptVariant | None:
        """Find a variant object by slot_key and variant_id."""
        for v in self._registry._variants.get(slot_key, []):
            if v.variant_id == variant_id:
                return v
        return None

    def _next_strategy(self) -> str:
        """Rotate through mutation strategies."""
        strategies = list(MUTATION_STRATEGIES.keys())
        strategy = strategies[self._strategy_index % len(strategies)]
        self._strategy_index += 1
        return strategy

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish a bus event if bus is available."""
        if self._bus is None:
            return
        try:
            self._bus.publish_sync(topic, payload)
        except Exception:
            log.debug("prompt_mutator.publish_failed topic=%s", topic)

    def status(self) -> dict[str, Any]:
        """Monitoring snapshot."""
        return {
            "running": self._running,
            "mutations_total": self._mutations_total,
            "rollbacks_total": self._rollbacks_total,
            "promotions_total": self._promotions_total,
            "last_cycle_at": self._last_cycle_at,
        }

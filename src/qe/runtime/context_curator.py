"""Context Curator — Tier 0 working memory with anti-drift mechanisms.

Replaces ContextManager with relevance-scored, goal-anchored context assembly.
Every LLM call receives the smallest possible high-signal token set.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

SLOT_CATEGORIES = Literal[
    "goal_anchor",
    "observation",
    "question",
    "finding",
    "constraint",
    "hypothesis",
    "progress",
    "background",
    "procedural",
]

# Token budget allocation per category (sums to 1.0)
DEFAULT_BUDGET_ALLOCATION: dict[str, float] = {
    "goal_anchor": 0.20,
    "constraint": 0.00,      # Shares with goal_anchor
    "finding": 0.30,
    "background": 0.25,
    "question": 0.08,
    "hypothesis": 0.07,
    "procedural": 0.10,
    "observation": 0.00,     # Shares with finding
    "progress": 0.00,        # Shares with finding
}

# Section-level budget allocation for LLM context assembly (#71)
DEFAULT_SECTION_BUDGET: dict[str, float] = {
    "system": 0.20,       # System prompt + identity
    "knowledge": 0.30,    # Retrieved claims, graph context
    "memory": 0.20,       # Episodic + procedural memory
    "history": 0.25,      # Conversation history
    "margin": 0.05,       # Safety margin for response
}


class WorkingMemorySlot(BaseModel):
    """A single item in working memory with relevance scoring."""

    slot_id: str
    content: str
    category: SLOT_CATEGORIES
    relevance_score: float = 0.5  # 0.0-1.0, recomputed each cycle
    inserted_at: float = Field(default_factory=time.monotonic)
    last_accessed: float = Field(default_factory=time.monotonic)
    token_count: int = 0
    source: str = ""  # which tier/operation produced this


class WorkingMemoryState(BaseModel):
    """Full state of Tier 0 working memory for a goal."""

    goal_id: str
    goal_anchor: str  # ALWAYS present, NEVER evicted
    slots: list[WorkingMemorySlot] = Field(default_factory=list)
    max_tokens: int = 4096
    current_tokens: int = 0


class DriftReport(BaseModel):
    """Result of drift detection."""

    drift_score: float  # 0.0 = on track, 1.0 = total drift
    on_track: bool
    recommendation: Literal["continue", "refocus", "replan"]


# ---------------------------------------------------------------------------
# Context Curator
# ---------------------------------------------------------------------------


class ContextCurator:
    """Manages Tier 0 working memory. Replaces ContextManager.

    Key differences from ContextManager:
    - Items have relevance scores (cosine similarity to goal)
    - Goal anchor is ALWAYS in context, NEVER evicted
    - Token budget is allocated by category
    - Drift detection via embedding similarity
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        budget_allocation: dict[str, float] | None = None,
        drift_threshold: float = 0.3,
        embed_fn=None,
    ) -> None:
        self._max_tokens = max_tokens
        self._budget = budget_allocation or DEFAULT_BUDGET_ALLOCATION
        self._drift_threshold = drift_threshold
        self._embed_fn = embed_fn  # async (text) -> list[float]
        self._states: dict[str, WorkingMemoryState] = {}

    def get_or_create_state(
        self,
        goal_id: str,
        goal_description: str,
    ) -> WorkingMemoryState:
        """Get or create working memory state for a goal."""
        if goal_id not in self._states:
            self._states[goal_id] = WorkingMemoryState(
                goal_id=goal_id,
                goal_anchor=goal_description,
                max_tokens=self._max_tokens,
            )
        return self._states[goal_id]

    def add_slot(
        self,
        goal_id: str,
        slot_id: str,
        content: str,
        category: SLOT_CATEGORIES,
        relevance_score: float = 0.5,
        source: str = "",
    ) -> None:
        """Add an item to working memory with relevance scoring."""
        state = self._states.get(goal_id)
        if state is None:
            raise ValueError(f"No working memory state for goal {goal_id}")

        token_count = self._estimate_tokens(content)
        slot = WorkingMemorySlot(
            slot_id=slot_id,
            content=content,
            category=category,
            relevance_score=relevance_score,
            token_count=token_count,
            source=source,
        )
        state.slots.append(slot)
        state.current_tokens += token_count

        # Evict low-relevance slots if over budget
        self._enforce_budget(state)

    def build_context(
        self,
        goal_id: str,
        system_prompt: str,
        constitution: str | None = None,
        user_message: str = "",
    ) -> list[dict[str, str]]:
        """Build optimal LLM context from current working memory.

        1. System prompt is always first
        2. Constitution (if present) is always second (immutable)
        3. Goal anchor is ALWAYS included
        4. Remaining slots are packed by relevance within category budgets
        5. User message is always last
        """
        state = self._states.get(goal_id)
        if state is None:
            # Fallback: just system prompt + user message
            messages = [{"role": "system", "content": system_prompt}]
            if constitution:
                messages.append({
                    "role": "system",
                    "content": f"[CONSTITUTION — IMMUTABLE SAFETY CONSTRAINTS]\n{constitution}",
                })
            if user_message:
                messages.append({"role": "user", "content": user_message})
            return messages

        messages: list[dict[str, str]] = []

        # 1. System prompt (always first)
        messages.append({"role": "system", "content": system_prompt})

        # 2. Constitution (immutable, never truncated)
        if constitution:
            messages.append({
                "role": "system",
                "content": f"[CONSTITUTION — IMMUTABLE SAFETY CONSTRAINTS]\n{constitution}",
            })

        # 3. Goal anchor (ALWAYS included, NEVER evicted)
        messages.append({
            "role": "system",
            "content": f"[GOAL ANCHOR — DO NOT LOSE SIGHT OF THIS]\n{state.goal_anchor}",
        })

        # 4. Pack remaining slots by relevance within category budgets
        packed = self._pack_slots(state)
        for slot in packed:
            is_system = slot.category in ("background", "procedural", "constraint")
            role = "system" if is_system else "user"
            prefix = f"[{slot.category.upper()}]"
            messages.append({
                "role": role,
                "content": f"{prefix} {slot.content}",
            })
            slot.last_accessed = time.monotonic()

        # 5. User message (always last)
        if user_message:
            messages.append({"role": "user", "content": user_message})

        return messages

    async def score_relevance(
        self,
        item: str,
        goal: str,
    ) -> float:
        """Compute semantic relevance via embedding cosine similarity."""
        if self._embed_fn is None:
            # Fallback: simple word overlap (Jaccard, like v1 but as fallback)
            return self._word_overlap(item, goal)

        goal_emb = await self._embed_fn(goal)
        item_emb = await self._embed_fn(item)
        return self._cosine_similarity(goal_emb, item_emb)

    async def detect_drift(
        self,
        goal_id: str,
    ) -> DriftReport:
        """Compute how far current working memory has drifted from goal.

        Returns DriftReport with score and recommendation.
        """
        state = self._states.get(goal_id)
        if state is None:
            return DriftReport(drift_score=0.0, on_track=True, recommendation="continue")

        # Combine all finding/observation slots into a summary
        findings = " ".join(
            s.content for s in state.slots
            if s.category in ("finding", "observation", "progress")
        )
        if not findings:
            return DriftReport(drift_score=0.0, on_track=True, recommendation="continue")

        similarity = await self.score_relevance(findings, state.goal_anchor)
        drift_score = 1.0 - similarity

        if drift_score > 0.5:
            recommendation = "replan"
        elif drift_score > self._drift_threshold:
            recommendation = "refocus"
        else:
            recommendation = "continue"

        return DriftReport(
            drift_score=round(drift_score, 4),
            on_track=drift_score <= self._drift_threshold,
            recommendation=recommendation,
        )

    async def refocus(self, goal_id: str) -> int:
        """Evict low-relevance items when drift is detected.

        Re-scores all slots against the goal and removes those below threshold.
        Returns number of slots evicted.
        """
        state = self._states.get(goal_id)
        if state is None:
            return 0

        evicted = 0
        survivors: list[WorkingMemorySlot] = []
        for slot in state.slots:
            if slot.category == "goal_anchor":
                survivors.append(slot)
                continue

            new_score = await self.score_relevance(slot.content, state.goal_anchor)
            slot.relevance_score = new_score
            if new_score >= 0.2:
                survivors.append(slot)
            else:
                evicted += 1

        state.slots = survivors
        state.current_tokens = sum(s.token_count for s in survivors)
        return evicted

    def clear_goal(self, goal_id: str) -> None:
        """Clean up working memory for a completed/failed goal."""
        self._states.pop(goal_id, None)

    def status(self) -> dict[str, Any]:
        """Monitoring snapshot."""
        return {
            "active_goals": len(self._states),
            "total_slots": sum(len(s.slots) for s in self._states.values()),
            "total_tokens": sum(s.current_tokens for s in self._states.values()),
        }

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _pack_slots(self, state: WorkingMemoryState) -> list[WorkingMemorySlot]:
        """Pack slots into the remaining token budget by category priority."""
        # Calculate available tokens (subtract system prompt + goal anchor estimate)
        overhead = self._estimate_tokens(state.goal_anchor) + 200  # system prompt estimate
        available = state.max_tokens - overhead
        if available <= 0:
            return []

        # Group slots by budget category
        category_groups: dict[str, list[WorkingMemorySlot]] = {}
        for slot in state.slots:
            cat = self._budget_category(slot.category)
            category_groups.setdefault(cat, []).append(slot)

        # Sort each group by relevance (descending)
        for cat in category_groups:
            category_groups[cat].sort(key=lambda s: s.relevance_score, reverse=True)

        # Allocate tokens per category
        packed: list[WorkingMemorySlot] = []
        tokens_used = 0

        for cat, budget_pct in sorted(self._budget.items(), key=lambda x: x[1], reverse=True):
            if budget_pct <= 0 or cat not in category_groups:
                continue

            cat_budget = int(available * budget_pct)
            cat_used = 0

            for slot in category_groups[cat]:
                if cat_used + slot.token_count > cat_budget:
                    continue
                if tokens_used + slot.token_count > available:
                    break
                packed.append(slot)
                cat_used += slot.token_count
                tokens_used += slot.token_count

        # Sort by relevance for final ordering
        packed.sort(key=lambda s: s.relevance_score, reverse=True)
        return packed

    def _enforce_budget(self, state: WorkingMemoryState) -> None:
        """Evict lowest-relevance non-anchor slots when over token budget."""
        while state.current_tokens > state.max_tokens and state.slots:
            # Find lowest relevance non-anchor slot
            candidates = [
                (i, s) for i, s in enumerate(state.slots)
                if s.category != "goal_anchor"
            ]
            if not candidates:
                break

            worst_idx, worst = min(candidates, key=lambda x: x[1].relevance_score)
            state.current_tokens -= worst.token_count
            state.slots.pop(worst_idx)

    @staticmethod
    def _budget_category(category: str) -> str:
        """Map slot category to budget category."""
        mapping = {
            "goal_anchor": "goal_anchor",
            "constraint": "goal_anchor",
            "finding": "finding",
            "observation": "finding",
            "progress": "finding",
            "background": "background",
            "question": "question",
            "hypothesis": "hypothesis",
            "procedural": "procedural",
        }
        return mapping.get(category, "finding")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)

    @staticmethod
    def _word_overlap(a: str, b: str) -> float:
        """Simple word-level Jaccard similarity as embedding fallback."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    # -------------------------------------------------------------------
    # #71 — Section-level token budget management
    # -------------------------------------------------------------------

    def assemble_sections(
        self,
        *,
        max_tokens: int = 8192,
        system_content: str = "",
        knowledge_items: list[str] | None = None,
        memory_items: list[str] | None = None,
        history_messages: list[dict[str, str]] | None = None,
        section_budget: dict[str, float] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble LLM context with explicit per-section token budgets.

        Sections (primacy → recency placement):
        1. system (20%) — system prompt, identity (primacy)
        2. knowledge (30%) — retrieved claims, graph context
        3. memory (20%) — episodic + procedural memory
        4. history (25%) — conversation history (recency)
        5. margin (5%) — reserved for response generation

        Items within each section are truncated to fit their budget.
        """
        budget = section_budget or DEFAULT_SECTION_BUDGET
        messages: list[dict[str, str]] = []

        # 1. System section (primacy position)
        sys_budget = int(max_tokens * budget.get("system", 0.20))
        if system_content:
            trimmed = self._trim_to_tokens(system_content, sys_budget)
            messages.append({"role": "system", "content": trimmed})

        # 2. Knowledge section
        know_budget = int(max_tokens * budget.get("knowledge", 0.30))
        if knowledge_items:
            packed = self._pack_items(knowledge_items, know_budget)
            if packed:
                messages.append({
                    "role": "system",
                    "content": "[KNOWLEDGE]\n" + "\n---\n".join(packed),
                })

        # 3. Memory section
        mem_budget = int(max_tokens * budget.get("memory", 0.20))
        if memory_items:
            packed = self._pack_items(memory_items, mem_budget)
            if packed:
                messages.append({
                    "role": "system",
                    "content": "[MEMORY]\n" + "\n---\n".join(packed),
                })

        # 4. History section (recency position — most recent messages)
        hist_budget = int(max_tokens * budget.get("history", 0.25))
        if history_messages:
            trimmed_hist = self._trim_history(history_messages, hist_budget)
            messages.extend(trimmed_hist)

        return messages

    def _trim_to_tokens(self, text: str, max_tokens: int) -> str:
        """Trim text to fit within max_tokens (approximate)."""
        est = self._estimate_tokens(text)
        if est <= max_tokens:
            return text
        # Approximate: 4 chars per token
        max_chars = max_tokens * 4
        return text[:max_chars] + "..."

    def _pack_items(self, items: list[str], budget_tokens: int) -> list[str]:
        """Pack as many items as possible within the token budget."""
        packed: list[str] = []
        used = 0
        for item in items:
            tokens = self._estimate_tokens(item)
            if used + tokens > budget_tokens:
                break
            packed.append(item)
            used += tokens
        return packed

    def _trim_history(
        self, messages: list[dict[str, str]], budget_tokens: int
    ) -> list[dict[str, str]]:
        """Keep the most recent messages that fit within budget."""
        result: list[dict[str, str]] = []
        used = 0
        # Walk backwards (most recent first)
        for msg in reversed(messages):
            tokens = self._estimate_tokens(msg.get("content", ""))
            if used + tokens > budget_tokens:
                break
            result.append(msg)
            used += tokens
        result.reverse()
        return result

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── Context Degradation Detection (#69) ──────────────────────────

    def check_context_health(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 128_000,
    ) -> dict[str, Any]:
        """Detect context degradation issues.

        Checks:
        - Lost-in-the-middle: important info buried in middle
        - Context size approaching limit
        - Repetition/redundancy
        - Attention scatter (too many topics)

        Gated behind ``context_health_check`` feature flag.
        """
        issues: list[str] = []
        total_tokens = sum(
            self._estimate_tokens(m.get("content", ""))
            for m in messages
        )

        # Size check
        usage_pct = total_tokens / max(max_tokens, 1)
        if usage_pct > 0.9:
            issues.append(
                f"Context nearly full: {usage_pct:.0%} used"
            )
        elif usage_pct > 0.7:
            issues.append(
                f"Context usage high: {usage_pct:.0%} used"
            )

        # Repetition detection
        seen_texts: set[str] = set()
        duplicates = 0
        for m in messages:
            text = m.get("content", "")[:200].lower().strip()
            if text in seen_texts:
                duplicates += 1
            seen_texts.add(text)
        if duplicates > 2:
            issues.append(
                f"Redundancy detected: {duplicates} duplicate messages"
            )

        # Topic scatter (count unique "topics" via word frequency)
        all_words: list[str] = []
        for m in messages:
            all_words.extend(
                m.get("content", "").lower().split()
            )
        unique_ratio = (
            len(set(all_words)) / max(len(all_words), 1)
        )
        if unique_ratio > 0.8 and len(messages) > 10:
            issues.append("High topic scatter detected")

        health_score = 1.0 - min(len(issues) * 0.2, 0.8)

        remediation: list[str] = []
        if usage_pct > 0.7:
            remediation.append("Compact conversation history")
        if duplicates > 2:
            remediation.append("Remove duplicate messages")
        if unique_ratio > 0.8:
            remediation.append("Focus on fewer topics")

        return {
            "health_score": round(health_score, 2),
            "total_tokens": total_tokens,
            "usage_pct": round(usage_pct, 3),
            "issues": issues,
            "remediation": remediation,
        }

    # ── Tiered Context Loading (#83) ─────────────────────────────────

    def tiered_context(
        self,
        query: str,
        system_prompt: str,
        claims: list[Any] | None = None,
        history: list[dict[str, str]] | None = None,
        *,
        tier: str = "auto",
        query_complexity: float | None = None,
    ) -> dict[str, Any]:
        """Load context at L0/L1/L2 tiers to reduce token cost.

        L0: System prompt only (simple greetings, acks)
        L1: System + top-5 claims + recent 3 history messages
        L2: Full context assembly (all claims, full history)

        Gated behind ``tiered_context_loading`` feature flag.
        """
        if tier == "auto":
            tier = self._auto_select_tier(
                query, query_complexity
            )

        if tier == "L0":
            return {
                "tier": "L0",
                "messages": [
                    {"role": "system", "content": system_prompt},
                ],
                "claims_included": 0,
                "history_included": 0,
            }

        if tier == "L1":
            top_claims = (claims or [])[:5]
            recent_history = (history or [])[-3:]
            claim_text = "\n".join(
                f"- {getattr(c, 'predicate', '')}: "
                f"{getattr(c, 'object_value', '')}"
                for c in top_claims
            ) if top_claims else ""
            msgs: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
            ]
            if claim_text:
                msgs.append({
                    "role": "system",
                    "content": f"Relevant knowledge:\n{claim_text}",
                })
            msgs.extend(recent_history)
            return {
                "tier": "L1",
                "messages": msgs,
                "claims_included": len(top_claims),
                "history_included": len(recent_history),
            }

        # L2: full assembly
        all_claims = claims or []
        all_history = history or []
        claim_text = "\n".join(
            f"- {getattr(c, 'predicate', '')}: "
            f"{getattr(c, 'object_value', '')}"
            for c in all_claims
        ) if all_claims else ""
        msgs = [
            {"role": "system", "content": system_prompt},
        ]
        if claim_text:
            msgs.append({
                "role": "system",
                "content": f"Knowledge base:\n{claim_text}",
            })
        msgs.extend(all_history)
        return {
            "tier": "L2",
            "messages": msgs,
            "claims_included": len(all_claims),
            "history_included": len(all_history),
        }

    def _auto_select_tier(
        self,
        query: str,
        complexity: float | None = None,
    ) -> str:
        """Auto-select context tier based on query complexity."""
        if complexity is not None:
            if complexity < 0.2:
                return "L0"
            if complexity < 0.5:
                return "L1"
            return "L2"

        # Heuristic: short queries → L0, medium → L1, complex → L2
        words = query.split()
        if len(words) <= 3:
            return "L0"
        if len(words) <= 10:
            return "L1"
        return "L2"

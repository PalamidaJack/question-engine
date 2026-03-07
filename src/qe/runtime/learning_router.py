"""Learning Loop — 5-stage knowledge improvement cycle.

Stages: Retrieve → Judge → Distill → Consolidate → Route
Gated behind ``learning_loop`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class LearningPattern:
    """A distilled pattern from past interactions."""

    pattern_id: str
    description: str
    frequency: int = 1
    quality_score: float = 0.5
    domain: str = "general"
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "frequency": self.frequency,
            "quality_score": round(self.quality_score, 3),
            "domain": self.domain,
        }


class LearningLoop:
    """5-stage learning loop for continuous improvement."""

    def __init__(self) -> None:
        self._patterns: dict[str, LearningPattern] = {}
        self._next_id = 0

    def _gen_id(self) -> str:
        self._next_id += 1
        return f"pat_{self._next_id:04d}"

    # Stage 1: Retrieve past patterns
    def retrieve(
        self, domain: str = "general", top_k: int = 10,
    ) -> list[LearningPattern]:
        """Retrieve relevant patterns from the store."""
        candidates = [
            p for p in self._patterns.values()
            if p.domain == domain or domain == "general"
        ]
        candidates.sort(
            key=lambda p: p.quality_score * p.frequency,
            reverse=True,
        )
        return candidates[:top_k]

    # Stage 2: Judge quality
    def judge(
        self, pattern: LearningPattern, outcome: bool,
    ) -> float:
        """Update pattern quality based on outcome."""
        alpha = 0.1
        new_score = 1.0 if outcome else 0.0
        pattern.quality_score = (
            alpha * new_score
            + (1 - alpha) * pattern.quality_score
        )
        pattern.frequency += 1
        return pattern.quality_score

    # Stage 3: Distill new pattern
    def distill(
        self,
        description: str,
        domain: str = "general",
        quality: float = 0.5,
    ) -> LearningPattern:
        """Create a new pattern from an observation."""
        pid = self._gen_id()
        pattern = LearningPattern(
            pattern_id=pid,
            description=description,
            domain=domain,
            quality_score=quality,
        )
        self._patterns[pid] = pattern
        return pattern

    # Stage 4: Consolidate (merge similar patterns)
    def consolidate(self) -> int:
        """Merge similar patterns and prune low-quality ones."""
        pruned = 0
        to_remove: list[str] = []
        for pid, pattern in self._patterns.items():
            if (
                pattern.quality_score < 0.1
                and pattern.frequency > 5
            ):
                to_remove.append(pid)
                pruned += 1
        for pid in to_remove:
            del self._patterns[pid]
        return pruned

    # Stage 5: Route (suggest based on patterns)
    def suggest(
        self, context: str, domain: str = "general",
    ) -> list[dict[str, Any]]:
        """Suggest actions based on learned patterns."""
        patterns = self.retrieve(domain)
        context_lower = context.lower()
        suggestions = []
        for p in patterns:
            # Simple keyword match
            words = p.description.lower().split()
            overlap = sum(
                1 for w in words if w in context_lower
            )
            if overlap > 0 or p.quality_score > 0.7:
                suggestions.append({
                    "pattern_id": p.pattern_id,
                    "description": p.description,
                    "relevance": overlap / max(len(words), 1),
                    "quality": p.quality_score,
                })
        suggestions.sort(
            key=lambda s: s["quality"], reverse=True
        )
        return suggestions[:5]

    def list_patterns(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self._patterns.values()]

    def stats(self) -> dict[str, Any]:
        return {
            "total_patterns": len(self._patterns),
            "avg_quality": (
                sum(p.quality_score for p in self._patterns.values())
                / max(len(self._patterns), 1)
            ),
        }

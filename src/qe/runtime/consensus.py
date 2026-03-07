"""Swarm Consensus — voting and agreement protocols for swarm results.

Implements majority voting, weighted consensus, and unanimous agreement
for high-confidence outputs.
Gated behind ``swarm_consensus`` feature flag.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ConsensusVote:
    """A single vote from a swarm agent."""

    agent_id: str
    answer: str
    confidence: float = 0.5
    weight: float = 1.0


@dataclass
class ConsensusResult:
    """Result of a consensus protocol."""

    winner: str
    method: str  # majority | weighted | unanimous
    agreement_score: float
    votes: list[ConsensusVote] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class ConsensusEngine:
    """Consensus protocols for swarm agent results."""

    def majority_vote(
        self, votes: list[ConsensusVote],
    ) -> ConsensusResult:
        """Simple majority vote — most common answer wins."""
        if not votes:
            return ConsensusResult(
                winner="", method="majority",
                agreement_score=0.0,
            )

        counts = Counter(v.answer for v in votes)
        winner, count = counts.most_common(1)[0]
        agreement = count / len(votes)

        return ConsensusResult(
            winner=winner,
            method="majority",
            agreement_score=round(agreement, 3),
            votes=votes,
            details={
                "vote_counts": dict(counts),
                "total_votes": len(votes),
            },
        )

    def weighted_vote(
        self, votes: list[ConsensusVote],
    ) -> ConsensusResult:
        """Weighted vote — answers weighted by confidence * weight."""
        if not votes:
            return ConsensusResult(
                winner="", method="weighted",
                agreement_score=0.0,
            )

        weighted_counts: dict[str, float] = {}
        for v in votes:
            w = v.confidence * v.weight
            weighted_counts[v.answer] = (
                weighted_counts.get(v.answer, 0.0) + w
            )

        total_weight = sum(weighted_counts.values())
        winner = max(weighted_counts, key=weighted_counts.get)
        agreement = (
            weighted_counts[winner] / total_weight
            if total_weight > 0 else 0.0
        )

        return ConsensusResult(
            winner=winner,
            method="weighted",
            agreement_score=round(agreement, 3),
            votes=votes,
            details={
                "weighted_scores": {
                    k: round(v, 3)
                    for k, v in weighted_counts.items()
                },
            },
        )

    def unanimous(
        self, votes: list[ConsensusVote],
    ) -> ConsensusResult:
        """Unanimous agreement — all agents must agree."""
        if not votes:
            return ConsensusResult(
                winner="", method="unanimous",
                agreement_score=0.0,
            )

        answers = {v.answer for v in votes}
        if len(answers) == 1:
            winner = answers.pop()
            return ConsensusResult(
                winner=winner,
                method="unanimous",
                agreement_score=1.0,
                votes=votes,
                details={"unanimous": True},
            )

        # No unanimity — fall back to weighted
        result = self.weighted_vote(votes)
        result.method = "unanimous"
        result.details["unanimous"] = False
        return result

    def auto_select(
        self, votes: list[ConsensusVote],
        *, require_high_confidence: bool = False,
    ) -> ConsensusResult:
        """Auto-select consensus method based on vote distribution.

        - If all agree → unanimous
        - If high confidence required → weighted
        - Otherwise → majority
        """
        if not votes:
            return ConsensusResult(
                winner="", method="auto",
                agreement_score=0.0,
            )

        answers = {v.answer for v in votes}
        if len(answers) == 1:
            return self.unanimous(votes)
        if require_high_confidence:
            return self.weighted_vote(votes)
        return self.majority_vote(votes)

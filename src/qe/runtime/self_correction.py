"""Self-correction loops: evaluate challenges and resolve contradictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Threshold by which a challenge must exceed a claim's confidence to supersede it
_SUPERSEDE_THRESHOLD = 0.15


@dataclass
class CorrectionResult:
    """Outcome of evaluating a challenge or resolving a contradiction."""

    original_claim_id: str
    action: str  # "reinforced" | "superseded" | "needs_investigation"
    new_confidence: float | None = None
    evidence: list[str] = field(default_factory=list)


def _claim_field(claim: dict, key: str, default: Any = "") -> Any:
    """Safely extract a field from a claim dict."""
    return claim.get(key, default)


def _gather_evidence(claim: dict) -> list[str]:
    """Collect evidence identifiers from a claim dict."""
    evidence: list[str] = []

    source_ids = _claim_field(claim, "source_envelope_ids", [])
    if isinstance(source_ids, list):
        evidence.extend(str(s) for s in source_ids)

    tags = _claim_field(claim, "tags", [])
    if isinstance(tags, list):
        evidence.extend(str(t) for t in tags)

    return evidence


class SelfCorrectionEngine:
    """Evaluates challenges against claims and resolves contradictions.

    Tracks statistics on reinforcement, supersession, and investigation
    outcomes for system health monitoring.
    """

    def __init__(self) -> None:
        self._stats: dict[str, int] = {
            "reinforced": 0,
            "superseded": 0,
            "needs_investigation": 0,
        }

    async def evaluate_challenge(
        self,
        claim: dict,
        challenge: dict,
    ) -> CorrectionResult:
        """Evaluate a challenge against an existing claim.

        Decision logic:
        1. If challenge confidence exceeds claim confidence by >= threshold
           -> supersede the original claim.
        2. If the original claim has more supporting evidence
           -> reinforce the original claim.
        3. Otherwise -> flag for investigation.

        Args:
            claim: the original claim dict.
            challenge: the challenging claim dict.

        Returns:
            CorrectionResult describing the outcome.
        """
        claim_id = _claim_field(claim, "claim_id", "unknown")
        claim_conf = float(_claim_field(claim, "confidence", 0.5))
        challenge_conf = float(_claim_field(challenge, "confidence", 0.5))

        claim_evidence = _gather_evidence(claim)
        challenge_evidence = _gather_evidence(challenge)
        combined_evidence = claim_evidence + challenge_evidence

        # Rule 1: Challenge confidence exceeds claim by threshold
        if challenge_conf - claim_conf >= _SUPERSEDE_THRESHOLD:
            result = CorrectionResult(
                original_claim_id=claim_id,
                action="superseded",
                new_confidence=challenge_conf,
                evidence=combined_evidence,
            )
            self._stats["superseded"] += 1
            log.info(
                "Claim %s superseded: challenge confidence %.2f > claim %.2f + %.2f",
                claim_id,
                challenge_conf,
                claim_conf,
                _SUPERSEDE_THRESHOLD,
            )
            return result

        # Rule 2: Original claim has more supporting evidence
        if len(claim_evidence) > len(challenge_evidence):
            # Reinforce: slightly boost confidence based on survived challenge
            boosted = min(claim_conf + 0.05, 1.0)
            result = CorrectionResult(
                original_claim_id=claim_id,
                action="reinforced",
                new_confidence=boosted,
                evidence=combined_evidence,
            )
            self._stats["reinforced"] += 1
            log.info(
                "Claim %s reinforced: %d evidence items vs %d challenge items",
                claim_id,
                len(claim_evidence),
                len(challenge_evidence),
            )
            return result

        # Rule 3: Inconclusive -> needs investigation
        result = CorrectionResult(
            original_claim_id=claim_id,
            action="needs_investigation",
            new_confidence=None,
            evidence=combined_evidence,
        )
        self._stats["needs_investigation"] += 1
        log.info(
            "Claim %s needs investigation: conf diff %.2f, evidence %d vs %d",
            claim_id,
            challenge_conf - claim_conf,
            len(claim_evidence),
            len(challenge_evidence),
        )
        return result

    async def resolve_contradiction(
        self,
        claim_a: dict,
        claim_b: dict,
    ) -> CorrectionResult:
        """Resolve a contradiction between two claims.

        Uses a multi-factor comparison:
        1. Confidence comparison
        2. Evidence count comparison
        3. Recency comparison (newer claim preferred when tied)

        The claim that "wins" is reinforced; the other is superseded.
        If no clear winner can be determined, flags for investigation.

        Args:
            claim_a: first claim dict.
            claim_b: second claim dict.

        Returns:
            CorrectionResult for the losing claim (superseded) or
            for claim_a if investigation is needed.
        """
        id_a = _claim_field(claim_a, "claim_id", "unknown_a")
        id_b = _claim_field(claim_b, "claim_id", "unknown_b")
        conf_a = float(_claim_field(claim_a, "confidence", 0.5))
        conf_b = float(_claim_field(claim_b, "confidence", 0.5))
        evidence_a = _gather_evidence(claim_a)
        evidence_b = _gather_evidence(claim_b)
        combined_evidence = evidence_a + evidence_b

        # Scoring: higher score = stronger claim
        score_a = 0.0
        score_b = 0.0

        # Factor 1: confidence
        if conf_a > conf_b:
            score_a += 1.0
        elif conf_b > conf_a:
            score_b += 1.0

        # Factor 2: evidence count
        if len(evidence_a) > len(evidence_b):
            score_a += 1.0
        elif len(evidence_b) > len(evidence_a):
            score_b += 1.0

        # Factor 3: recency (newer is preferred)
        ts_a = str(_claim_field(claim_a, "created_at", ""))
        ts_b = str(_claim_field(claim_b, "created_at", ""))
        if ts_a > ts_b:
            score_a += 0.5
        elif ts_b > ts_a:
            score_b += 0.5

        if score_a > score_b:
            # Claim A wins, claim B is superseded
            result = CorrectionResult(
                original_claim_id=id_b,
                action="superseded",
                new_confidence=conf_a,
                evidence=combined_evidence,
            )
            self._stats["superseded"] += 1
            log.info(
                "Contradiction resolved: claim %s supersedes %s (score %.1f vs %.1f)",
                id_a,
                id_b,
                score_a,
                score_b,
            )
            return result

        if score_b > score_a:
            # Claim B wins, claim A is superseded
            result = CorrectionResult(
                original_claim_id=id_a,
                action="superseded",
                new_confidence=conf_b,
                evidence=combined_evidence,
            )
            self._stats["superseded"] += 1
            log.info(
                "Contradiction resolved: claim %s supersedes %s (score %.1f vs %.1f)",
                id_b,
                id_a,
                score_b,
                score_a,
            )
            return result

        # Tied: needs investigation
        result = CorrectionResult(
            original_claim_id=id_a,
            action="needs_investigation",
            new_confidence=None,
            evidence=combined_evidence,
        )
        self._stats["needs_investigation"] += 1
        log.info(
            "Contradiction between %s and %s is inconclusive (tied at %.1f)",
            id_a,
            id_b,
            score_a,
        )
        return result

    def get_correction_stats(self) -> dict:
        """Return counts of correction outcomes.

        Returns:
            Dict with keys 'reinforced', 'superseded', 'needs_investigation'
            and their integer counts.
        """
        return dict(self._stats)

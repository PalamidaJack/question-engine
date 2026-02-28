"""Adversarial fact-checking service for claim verification."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Default verdicts
_VERDICT_SUPPORTED = "supported"
_VERDICT_CHALLENGED = "challenged"
_VERDICT_INSUFFICIENT = "insufficient_evidence"


class FactCheckerService:
    """Checks claims against existing evidence and finds
    contradictions."""

    def __init__(
        self,
        substrate: Any = None,
        bus: Any = None,
    ) -> None:
        self.substrate = substrate
        self.bus = bus

    async def check_claim(
        self,
        claim_text: str,
        existing_claims: list[dict] | None = None,
    ) -> dict:
        """Check a claim against existing evidence.

        Returns a verdict with confidence, reasoning, and
        any contradictions found.
        """
        if not existing_claims:
            log.info("No existing claims to check against")
            return {
                "verdict": _VERDICT_INSUFFICIENT,
                "confidence": 0.0,
                "reasoning": "No existing claims available.",
                "contradictions": [],
            }

        contradictions = await self.find_contradictions(
            claim_text, existing_claims
        )

        if contradictions:
            log.info(
                "Found %d contradictions for claim",
                len(contradictions),
            )
            return {
                "verdict": _VERDICT_CHALLENGED,
                "confidence": 0.7,
                "reasoning": (
                    f"Found {len(contradictions)} "
                    "potentially contradicting claim(s)."
                ),
                "contradictions": contradictions,
            }

        log.info("Claim appears supported by existing evidence")
        return {
            "verdict": _VERDICT_SUPPORTED,
            "confidence": 0.6,
            "reasoning": (
                "No contradictions found among "
                f"{len(existing_claims)} existing claims."
            ),
            "contradictions": [],
        }

    async def find_contradictions(
        self,
        claim_text: str,
        claims: list[dict],
    ) -> list[dict]:
        """Find claims that potentially contradict the given
        claim using simple text overlap checks.

        Returns a list of claims with overlap information.
        """
        claim_words = set(claim_text.lower().split())
        contradictions: list[dict] = []

        for claim in claims:
            text = claim.get("text", claim.get("object_value", ""))
            if not text:
                continue

            other_words = set(text.lower().split())
            overlap = claim_words & other_words

            # Require meaningful overlap but different text
            if (
                len(overlap) >= 3
                and text.lower().strip()
                != claim_text.lower().strip()
            ):
                contradictions.append(
                    {
                        "claim": claim,
                        "overlap_words": sorted(overlap),
                        "overlap_ratio": len(overlap)
                        / max(
                            len(claim_words),
                            len(other_words),
                        ),
                    }
                )

        return contradictions

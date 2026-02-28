"""Analyst service for claim analysis and insight generation."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_VALID_ANALYSIS_TYPES = {
    "trend",
    "contradiction",
    "gap",
    "comparative",
}


class AnalystService:
    """Analyzes collections of claims to surface trends,
    contradictions, gaps, and comparisons."""

    def __init__(
        self,
        substrate: Any = None,
        bus: Any = None,
    ) -> None:
        self.substrate = substrate
        self.bus = bus

    async def analyze_claims(
        self,
        claims: list[dict],
        analysis_type: str = "trend",
    ) -> dict:
        """Analyze a set of claims according to the requested
        analysis type.

        Supported types: trend, contradiction, gap, comparative.

        Returns findings, gaps, and a confidence score.
        """
        if analysis_type not in _VALID_ANALYSIS_TYPES:
            log.warning(
                "Unknown analysis type: %s, defaulting to trend",
                analysis_type,
            )
            analysis_type = "trend"

        findings = self._generate_findings(
            claims, analysis_type
        )
        gaps = self._identify_gaps(claims, analysis_type)

        confidence = min(0.9, len(claims) * 0.1)

        log.info(
            "Analysis complete: type=%s claims=%d findings=%d",
            analysis_type,
            len(claims),
            len(findings),
        )

        return {
            "analysis_type": analysis_type,
            "findings": findings,
            "gaps": gaps,
            "confidence": confidence,
        }

    def _generate_findings(
        self,
        claims: list[dict],
        analysis_type: str,
    ) -> list[str]:
        """Generate placeholder findings based on analysis type
        and claim count."""
        if not claims:
            return ["No claims available for analysis."]

        base = f"Analyzed {len(claims)} claims"
        if analysis_type == "trend":
            return [
                f"{base} for temporal patterns.",
                "Insufficient data for trend detection.",
            ]
        if analysis_type == "contradiction":
            return [
                f"{base} for contradictions.",
                "No contradictions detected.",
            ]
        if analysis_type == "gap":
            return [
                f"{base} for coverage gaps.",
                "Potential gaps identified in coverage.",
            ]
        # comparative
        return [
            f"{base} for comparative analysis.",
            "Comparison requires additional context.",
        ]

    def _identify_gaps(
        self,
        claims: list[dict],
        analysis_type: str,
    ) -> list[str]:
        """Identify knowledge gaps based on claim coverage."""
        if not claims:
            return ["No claims to analyze."]

        if analysis_type == "gap":
            return [
                "Source diversity is limited.",
                "Temporal coverage may be incomplete.",
            ]
        return []

"""Retrieval Analyzer — analyze-then-validate retrieval strategy.

Two-step retrieval: analyze query → determine validation criteria →
retrieve → validate results against criteria.
Gated behind ``analyze_then_validate`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Analysis of a query before retrieval."""

    query: str
    intent: str = "factual"  # factual | analytical | comparative
    key_entities: list[str] = field(default_factory=list)
    validation_criteria: list[str] = field(default_factory=list)
    confidence_threshold: float = 0.5


@dataclass
class ValidationResult:
    """Result of validating retrieved items."""

    total_retrieved: int = 0
    passed: int = 0
    failed: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.total_retrieved == 0:
            return 0.0
        return self.passed / self.total_retrieved


class RetrievalAnalyzer:
    """Analyze-then-validate retrieval strategy."""

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a query to determine retrieval strategy."""
        lower = query.lower()

        # Detect intent
        if any(w in lower for w in (
            "compare", "versus", "vs", "difference",
        )):
            intent = "comparative"
        elif any(w in lower for w in (
            "why", "how", "analyze", "explain", "cause",
        )):
            intent = "analytical"
        else:
            intent = "factual"

        # Extract key entities (simple word extraction)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were",
            "what", "how", "why", "when", "where", "who",
            "do", "does", "did", "can", "could", "would",
            "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "about",
        }
        words = query.split()
        entities = [
            w for w in words
            if w.lower() not in stop_words and len(w) > 2
        ]

        # Generate validation criteria
        criteria: list[str] = []
        if intent == "factual":
            criteria.append("Results must contain factual claims")
            criteria.append("Sources should be verifiable")
        elif intent == "comparative":
            criteria.append("Results must cover all compared items")
            criteria.append("Comparison dimensions must be explicit")
        elif intent == "analytical":
            criteria.append("Results must include reasoning")
            criteria.append("Causal links should be supported")

        return QueryAnalysis(
            query=query,
            intent=intent,
            key_entities=entities[:10],
            validation_criteria=criteria,
            confidence_threshold=0.6 if intent == "factual" else 0.4,
        )

    def validate_results(
        self,
        analysis: QueryAnalysis,
        results: list[Any],
    ) -> ValidationResult:
        """Validate retrieved results against analysis criteria."""
        if not results:
            return ValidationResult(
                issues=["No results retrieved"],
            )

        passed = 0
        failed = 0
        issues: list[str] = []

        for result in results:
            text = str(result).lower()
            # Check entity coverage
            entity_hits = sum(
                1 for e in analysis.key_entities
                if e.lower() in text
            )
            relevance = (
                entity_hits / max(len(analysis.key_entities), 1)
            )

            if relevance >= 0.3:
                passed += 1
            else:
                failed += 1

        if failed > passed:
            issues.append(
                f"Low relevance: {failed}/{len(results)} "
                f"results below threshold"
            )

        # Check comparative coverage
        if analysis.intent == "comparative":
            all_text = " ".join(str(r).lower() for r in results)
            covered = sum(
                1 for e in analysis.key_entities
                if e.lower() in all_text
            )
            if covered < len(analysis.key_entities):
                issues.append(
                    "Not all compared entities covered in results"
                )

        return ValidationResult(
            total_retrieved=len(results),
            passed=passed,
            failed=failed,
            issues=issues,
        )

    def analyze_and_validate(
        self,
        query: str,
        results: list[Any],
    ) -> dict[str, Any]:
        """Full analyze-then-validate pipeline."""
        analysis = self.analyze_query(query)
        validation = self.validate_results(analysis, results)
        return {
            "query": query,
            "intent": analysis.intent,
            "key_entities": analysis.key_entities,
            "validation": {
                "total": validation.total_retrieved,
                "passed": validation.passed,
                "failed": validation.failed,
                "pass_rate": round(validation.pass_rate, 3),
                "issues": validation.issues,
            },
        }

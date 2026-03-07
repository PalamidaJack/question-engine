"""Swarm Coordinator — domain-specialized parallel agent swarms.

Assigns domain expertise to swarm agents, coordinates parallel execution,
and performs weighted merge of results.
Gated behind ``domain_swarms`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class DomainExpert:
    """A domain-specialized agent profile for swarm assignment."""

    name: str
    domain: str
    expertise_keywords: list[str] = field(default_factory=list)
    weight: float = 1.0
    description: str = ""


BUILTIN_EXPERTS: list[DomainExpert] = [
    DomainExpert(
        name="tech_expert", domain="technology",
        expertise_keywords=[
            "software", "code", "api", "database", "cloud",
            "programming", "algorithm",
        ],
        description="Technology and software engineering",
    ),
    DomainExpert(
        name="science_expert", domain="science",
        expertise_keywords=[
            "research", "hypothesis", "experiment", "data",
            "analysis", "study", "evidence",
        ],
        description="Scientific research and analysis",
    ),
    DomainExpert(
        name="business_expert", domain="business",
        expertise_keywords=[
            "market", "strategy", "revenue", "customer",
            "growth", "competition", "roi",
        ],
        description="Business strategy and analysis",
    ),
    DomainExpert(
        name="general_expert", domain="general",
        expertise_keywords=[],
        weight=0.5,
        description="General-purpose fallback",
    ),
]


@dataclass
class SwarmResult:
    """Result from a single swarm agent."""

    expert_name: str
    domain: str
    result: str
    confidence: float = 0.5
    weight: float = 1.0


class SwarmCoordinator:
    """Coordinates domain-specialized parallel swarms."""

    def __init__(
        self, experts: list[DomainExpert] | None = None,
    ) -> None:
        self._experts = experts or list(BUILTIN_EXPERTS)

    def assign_experts(
        self, query: str, *, max_agents: int = 3,
    ) -> list[DomainExpert]:
        """Select domain experts based on query content."""
        query_lower = query.lower()
        scored: list[tuple[float, DomainExpert]] = []
        for expert in self._experts:
            if not expert.expertise_keywords:
                scored.append((0.1, expert))
                continue
            hits = sum(
                1 for kw in expert.expertise_keywords
                if kw in query_lower
            )
            relevance = hits / len(expert.expertise_keywords)
            scored.append((relevance, expert))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [
            e for _, e in scored[:max_agents] if _ > 0
        ]
        # Always include at least one expert
        if not selected and scored:
            selected = [scored[0][1]]
        return selected

    def weighted_merge(
        self, results: list[SwarmResult],
    ) -> dict[str, Any]:
        """Merge swarm results with confidence-weighted scoring."""
        if not results:
            return {
                "merged_text": "",
                "total_confidence": 0.0,
                "contributors": [],
            }

        total_weight = sum(
            r.confidence * r.weight for r in results
        )
        parts: list[str] = []
        contributors: list[dict[str, Any]] = []

        for r in results:
            parts.append(
                f"[{r.domain}] {r.result}"
            )
            contributors.append({
                "expert": r.expert_name,
                "domain": r.domain,
                "confidence": r.confidence,
                "weight": r.weight,
            })

        avg_confidence = (
            total_weight / len(results) if results else 0.0
        )

        return {
            "merged_text": "\n\n".join(parts),
            "total_confidence": round(avg_confidence, 3),
            "contributors": contributors,
            "agent_count": len(results),
        }

    def list_experts(self) -> list[dict[str, str]]:
        return [
            {
                "name": e.name,
                "domain": e.domain,
                "description": e.description,
            }
            for e in self._experts
        ]

"""Contradiction Cascade — blast radius analysis for claim retraction.

When a claim is retracted, trace dependent claims and show the cascade
before confirming.  Gated behind ``contradiction_cascade`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CascadeNode:
    """A single node in a retraction cascade."""

    claim_id: str
    entity: str
    predicate: str
    depth: int = 0
    impact: str = "direct"  # direct | indirect


@dataclass
class CascadeResult:
    """Result of a cascade analysis."""

    root_claim_id: str
    affected: list[CascadeNode] = field(default_factory=list)
    total_affected: int = 0
    max_depth: int = 0

    def summary(self) -> str:
        if not self.affected:
            return "No dependent claims affected."
        return (
            f"Retracting {self.root_claim_id} would affect "
            f"{self.total_affected} claim(s) across "
            f"{self.max_depth} level(s) of dependency."
        )


class ContradictionCascade:
    """Analyzes blast radius of claim retraction."""

    def __init__(self) -> None:
        self._dependency_graph: dict[str, list[str]] = {}

    def build_dependencies(self, claims: list[Any]) -> None:
        """Build dependency graph from claims.

        A claim X depends on Y if Y's entity appears in X's
        object_value (i.e., X references Y's subject).
        """
        self._dependency_graph.clear()
        entity_to_claims: dict[str, list[str]] = {}

        for claim in claims:
            entity = getattr(claim, "subject_entity_id", "")
            cid = getattr(claim, "claim_id", "")
            if entity and cid:
                entity_to_claims.setdefault(entity, []).append(cid)

        for claim in claims:
            cid = getattr(claim, "claim_id", "")
            obj_val = str(getattr(claim, "object_value", ""))
            if not cid:
                continue
            # Check if this claim references entities from other claims
            for entity, dep_cids in entity_to_claims.items():
                if entity.lower() in obj_val.lower() and entity:
                    for dep_cid in dep_cids:
                        if dep_cid != cid:
                            self._dependency_graph.setdefault(
                                dep_cid, []
                            ).append(cid)

    def analyze(
        self,
        claim_id: str,
        claims: list[Any] | None = None,
        *,
        max_depth: int = 5,
    ) -> CascadeResult:
        """Analyze the blast radius of retracting a claim.

        If claims are provided, rebuilds the dependency graph first.
        """
        if claims is not None:
            self.build_dependencies(claims)

        affected: list[CascadeNode] = []
        visited: set[str] = {claim_id}
        queue: list[tuple[str, int]] = [(claim_id, 0)]
        max_d = 0

        # Build claim lookup for metadata
        claim_map: dict[str, Any] = {}
        if claims:
            for c in claims:
                cid = getattr(c, "claim_id", "")
                if cid:
                    claim_map[cid] = c

        while queue:
            current, depth = queue.pop(0)
            if depth > max_depth:
                continue

            dependents = self._dependency_graph.get(current, [])
            for dep_cid in dependents:
                if dep_cid in visited:
                    continue
                visited.add(dep_cid)
                dep_claim = claim_map.get(dep_cid)
                node = CascadeNode(
                    claim_id=dep_cid,
                    entity=getattr(dep_claim, "subject_entity_id", "")
                    if dep_claim else "",
                    predicate=getattr(dep_claim, "predicate", "")
                    if dep_claim else "",
                    depth=depth + 1,
                    impact="direct" if depth == 0 else "indirect",
                )
                affected.append(node)
                max_d = max(max_d, depth + 1)
                queue.append((dep_cid, depth + 1))

        return CascadeResult(
            root_claim_id=claim_id,
            affected=affected,
            total_affected=len(affected),
            max_depth=max_d,
        )

    def preview_retraction(
        self, claim_id: str, claims: list[Any],
    ) -> dict[str, Any]:
        """Return a user-friendly preview of retraction impact."""
        result = self.analyze(claim_id, claims)
        return {
            "root": claim_id,
            "affected_count": result.total_affected,
            "max_depth": result.max_depth,
            "affected_claims": [
                {
                    "claim_id": n.claim_id,
                    "entity": n.entity,
                    "depth": n.depth,
                    "impact": n.impact,
                }
                for n in result.affected
            ],
            "summary": result.summary(),
        }

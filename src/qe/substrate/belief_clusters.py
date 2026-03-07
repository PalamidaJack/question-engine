"""Belief Clusters — group similar claims and detect causal chains.

Clusters semantically similar claims at ingestion time and detects
A→B→C causal chains.  Gated behind ``belief_clustering`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ClaimCluster:
    """A group of semantically related claims."""

    cluster_id: str
    label: str
    claim_ids: list[str] = field(default_factory=list)
    centroid_text: str = ""
    confidence: float = 0.0

    def size(self) -> int:
        return len(self.claim_ids)


@dataclass
class CausalLink:
    """A directed causal relationship between two claims."""

    source_id: str
    target_id: str
    relation: str = "causes"
    strength: float = 0.5


@dataclass
class CausalChain:
    """An ordered chain of causal links (A→B→C)."""

    chain_id: str
    links: list[CausalLink] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.links)

    def claim_ids(self) -> list[str]:
        ids: list[str] = []
        for link in self.links:
            if link.source_id not in ids:
                ids.append(link.source_id)
            if link.target_id not in ids:
                ids.append(link.target_id)
        return ids


class BeliefClusterEngine:
    """Clusters claims by semantic similarity and detects causal chains."""

    def __init__(self) -> None:
        self._clusters: dict[str, ClaimCluster] = {}
        self._claim_to_cluster: dict[str, str] = {}
        self._causal_links: list[CausalLink] = []
        self._next_id = 0

    def _gen_id(self) -> str:
        self._next_id += 1
        return f"clst_{self._next_id:04d}"

    def cluster_claims(
        self,
        claims: list[Any],
        *,
        similarity_threshold: float = 0.5,
    ) -> list[ClaimCluster]:
        """Cluster claims by entity overlap (heuristic).

        Uses entity/predicate overlap as a proxy for semantic similarity.
        """
        groups: dict[str, list[Any]] = {}
        for claim in claims:
            entity = getattr(claim, "subject_entity_id", "unknown")
            key = entity.lower().strip()
            groups.setdefault(key, []).append(claim)

        clusters: list[ClaimCluster] = []
        for entity_key, group in groups.items():
            if len(group) < 2:
                continue
            cid = self._gen_id()
            claim_ids = [
                getattr(c, "claim_id", str(i))
                for i, c in enumerate(group)
            ]
            cluster = ClaimCluster(
                cluster_id=cid,
                label=entity_key,
                claim_ids=claim_ids,
                centroid_text=entity_key,
                confidence=min(len(group) / 10.0, 1.0),
            )
            self._clusters[cid] = cluster
            for cid_item in claim_ids:
                self._claim_to_cluster[cid_item] = cid
            clusters.append(cluster)

        return clusters

    def get_cluster(self, cluster_id: str) -> ClaimCluster | None:
        return self._clusters.get(cluster_id)

    def get_cluster_for_claim(self, claim_id: str) -> ClaimCluster | None:
        cid = self._claim_to_cluster.get(claim_id)
        return self._clusters.get(cid) if cid else None

    def detect_causal_chains(
        self, claims: list[Any], *, max_chain_length: int = 5,
    ) -> list[CausalChain]:
        """Detect causal chains from claim predicates.

        Looks for claims with causal predicates (causes, leads_to, etc.)
        and builds directed chains.
        """
        causal_predicates = {
            "causes", "leads_to", "results_in",
            "enables", "triggers", "produces",
        }

        # Build adjacency from causal claims
        adjacency: dict[str, list[CausalLink]] = {}
        for claim in claims:
            pred = getattr(claim, "predicate", "")
            if pred.lower().replace(" ", "_") in causal_predicates:
                source = getattr(claim, "subject_entity_id", "")
                target = getattr(claim, "object_value", "")
                if source and target:
                    link = CausalLink(
                        source_id=source, target_id=target,
                        relation=pred,
                        strength=getattr(claim, "confidence", 0.5),
                    )
                    self._causal_links.append(link)
                    adjacency.setdefault(source, []).append(link)

        # Walk chains using DFS
        chains: list[CausalChain] = []
        chain_counter = 0
        visited_starts: set[str] = set()

        for start in adjacency:
            if start in visited_starts:
                continue
            visited_starts.add(start)
            # DFS from this node
            stack: list[tuple[str, list[CausalLink]]] = [
                (start, [])
            ]
            while stack:
                node, path = stack.pop()
                if len(path) >= max_chain_length:
                    continue
                if node in adjacency:
                    for link in adjacency[node]:
                        new_path = path + [link]
                        if len(new_path) >= 2:
                            chain_counter += 1
                            chains.append(CausalChain(
                                chain_id=f"chain_{chain_counter:04d}",
                                links=list(new_path),
                            ))
                        # Avoid cycles
                        visited_nodes = {
                            lk.source_id for lk in new_path
                        }
                        if link.target_id not in visited_nodes:
                            stack.append(
                                (link.target_id, new_path)
                            )

        return chains

    def list_clusters(self) -> list[dict[str, Any]]:
        return [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "size": c.size(),
                "confidence": c.confidence,
            }
            for c in self._clusters.values()
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "total_clusters": len(self._clusters),
            "total_causal_links": len(self._causal_links),
            "clustered_claims": len(self._claim_to_cluster),
        }

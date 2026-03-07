"""Graph-based knowledge retrieval over the belief ledger.

Builds an in-memory entity-claim graph and provides 1-hop and 2-hop
traversal for contextual knowledge assembly.  Gated behind the
``graph_knowledge_retrieval`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphEdge:
    """Directed edge: subject --predicate--> object."""

    claim_id: str
    subject: str
    predicate: str
    object_value: str
    confidence: float


@dataclass
class GraphNode:
    """An entity node with outgoing and incoming edges."""

    entity: str
    outgoing: list[GraphEdge] = field(default_factory=list)
    incoming: list[GraphEdge] = field(default_factory=list)


class KnowledgeGraph:
    """In-memory entity-claim graph built from the belief ledger.

    Supports:
    - ``build()`` to load/refresh from a claim list.
    - ``neighbors(entity, hops=1)`` for 1-hop or 2-hop traversal.
    - ``graph_context(entity)`` for a formatted context block.
    - ``subgraph(entities)`` for a multi-entity slice.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._built = False

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def build(self, claims: list[Any]) -> None:
        """Build the graph from a list of claim objects.

        Each claim must have: claim_id, subject_entity_id, predicate,
        object_value, confidence.
        """
        self._nodes.clear()
        self._edges.clear()

        for claim in claims:
            subj = getattr(claim, "subject_entity_id", None) or ""
            pred = getattr(claim, "predicate", None) or ""
            obj = getattr(claim, "object_value", None) or ""
            cid = getattr(claim, "claim_id", None) or ""
            conf = getattr(claim, "confidence", 0.5)

            if not subj or not pred:
                continue

            edge = GraphEdge(
                claim_id=cid,
                subject=subj,
                predicate=pred,
                object_value=obj,
                confidence=conf,
            )
            self._edges.append(edge)

            # Ensure subject node exists
            if subj not in self._nodes:
                self._nodes[subj] = GraphNode(entity=subj)
            self._nodes[subj].outgoing.append(edge)

            # If the object looks like an entity reference, add incoming edge
            obj_norm = obj.strip().lower()
            if obj_norm and not obj_norm[0].isdigit() and len(obj_norm) < 200:
                if obj_norm not in self._nodes:
                    self._nodes[obj_norm] = GraphNode(entity=obj_norm)
                self._nodes[obj_norm].incoming.append(edge)

        self._built = True
        log.info(
            "knowledge_graph.built nodes=%d edges=%d",
            len(self._nodes),
            len(self._edges),
        )

    def neighbors(
        self,
        entity: str,
        *,
        hops: int = 1,
        min_confidence: float = 0.0,
    ) -> list[GraphEdge]:
        """Return edges reachable within *hops* from *entity*."""
        entity_key = entity.strip().lower()
        if entity_key not in self._nodes:
            return []

        visited_entities: set[str] = set()
        result_edges: list[GraphEdge] = []

        frontier = {entity_key}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for ent in frontier:
                if ent in visited_entities:
                    continue
                visited_entities.add(ent)
                node = self._nodes.get(ent)
                if node is None:
                    continue
                for edge in node.outgoing:
                    if edge.confidence >= min_confidence:
                        result_edges.append(edge)
                        obj_key = edge.object_value.strip().lower()
                        if obj_key not in visited_entities:
                            next_frontier.add(obj_key)
                for edge in node.incoming:
                    if edge.confidence >= min_confidence:
                        result_edges.append(edge)
                        if edge.subject not in visited_entities:
                            next_frontier.add(edge.subject)
            frontier = next_frontier

        # Deduplicate by claim_id
        seen: set[str] = set()
        unique: list[GraphEdge] = []
        for e in result_edges:
            if e.claim_id not in seen:
                seen.add(e.claim_id)
                unique.append(e)

        return sorted(unique, key=lambda e: e.confidence, reverse=True)

    def graph_context(
        self,
        entity: str,
        *,
        hops: int = 2,
        max_edges: int = 20,
        min_confidence: float = 0.1,
    ) -> str:
        """Return a formatted context block for LLM injection.

        Produces a human-readable summary of the entity's neighborhood.
        """
        edges = self.neighbors(entity, hops=hops, min_confidence=min_confidence)
        if not edges:
            return ""

        edges = edges[:max_edges]
        lines = [f"### Knowledge Graph — {entity}"]
        for e in edges:
            lines.append(
                f"- {e.subject} → {e.predicate} → {e.object_value} "
                f"(confidence: {e.confidence:.2f})"
            )
        return "\n".join(lines)

    def subgraph(self, entities: list[str]) -> dict[str, Any]:
        """Return a JSON-serializable subgraph for multiple entities."""
        all_edges: list[GraphEdge] = []
        for ent in entities:
            all_edges.extend(self.neighbors(ent, hops=1))

        # Deduplicate
        seen: set[str] = set()
        unique: list[GraphEdge] = []
        for e in all_edges:
            if e.claim_id not in seen:
                seen.add(e.claim_id)
                unique.append(e)

        nodes_set: set[str] = set()
        edge_list: list[dict[str, Any]] = []
        for e in unique:
            nodes_set.add(e.subject)
            nodes_set.add(e.object_value.strip().lower())
            edge_list.append({
                "claim_id": e.claim_id,
                "subject": e.subject,
                "predicate": e.predicate,
                "object_value": e.object_value,
                "confidence": e.confidence,
            })

        return {
            "nodes": sorted(nodes_set),
            "edges": edge_list,
            "node_count": len(nodes_set),
            "edge_count": len(edge_list),
        }

    def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        return {
            "built": self._built,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
        }

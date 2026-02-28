"""MAGMA Multi-Graph Memory: unified fan-out query across 4 co-indexed graphs.

Graphs:
  1. Semantic  — embedding similarity via EmbeddingStore
  2. Temporal  — event ordering via EventLog
  3. Entity    — entity facts via EntityResolver + MemoryStore
  4. Causal    — causation chains via EventLog causation_id links

A single ``multi_query()`` fans out across all enabled graphs in parallel,
then merges and ranks results by a weighted relevance score.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

log = logging.getLogger(__name__)

GraphName = Literal["semantic", "temporal", "entity", "causal"]

ALL_GRAPHS: frozenset[GraphName] = frozenset(
    ["semantic", "temporal", "entity", "causal"]
)


@dataclass
class GraphResult:
    """A single result from one of the four graphs."""

    graph: GraphName
    id: str
    text: str
    score: float  # 0.0–1.0 relevance within this graph
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MergedResult:
    """A result after cross-graph merge with composite score."""

    id: str
    text: str
    composite_score: float
    graph_scores: dict[GraphName, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# Default weights — callers can override per query
_DEFAULT_WEIGHTS: dict[GraphName, float] = {
    "semantic": 0.40,
    "temporal": 0.20,
    "entity": 0.25,
    "causal": 0.15,
}


class MultiGraphQuery:
    """Fan-out query engine across MAGMA's four graphs.

    Accepts references to the existing QE data stores rather than
    owning them — zero duplication of infrastructure.
    """

    def __init__(
        self,
        *,
        embeddings: Any | None = None,
        event_log: Any | None = None,
        belief_ledger: Any | None = None,
        entity_resolver: Any | None = None,
        memory_store: Any | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._event_log = event_log
        self._belief_ledger = belief_ledger
        self._entity_resolver = entity_resolver
        self._memory_store = memory_store

    # ── Public API ───────────────────────────────────────────────────

    async def multi_query(
        self,
        query: str,
        *,
        graphs: frozenset[GraphName] | None = None,
        weights: dict[GraphName, float] | None = None,
        top_k: int = 10,
        entity_ids: list[str] | None = None,
        time_window_hours: int = 168,  # 1 week default
    ) -> list[MergedResult]:
        """Query multiple graphs in parallel and return merged results.

        Args:
            query: Natural language query string.
            graphs: Which graphs to query (default: all available).
            weights: Per-graph weight for composite scoring.
            top_k: Max results to return after merge.
            entity_ids: Optional entity filter for entity graph.
            time_window_hours: How far back to look for temporal/causal.
        """
        active_graphs = graphs or ALL_GRAPHS
        w = {**_DEFAULT_WEIGHTS, **(weights or {})}

        # Launch all graph queries in parallel
        tasks: dict[GraphName, asyncio.Task] = {}
        if "semantic" in active_graphs and self._embeddings:
            tasks["semantic"] = asyncio.create_task(
                self._query_semantic(query, top_k=top_k * 2)
            )
        if "temporal" in active_graphs and self._event_log:
            tasks["temporal"] = asyncio.create_task(
                self._query_temporal(query, hours=time_window_hours, limit=top_k * 2)
            )
        if "entity" in active_graphs and (self._entity_resolver or self._memory_store):
            tasks["entity"] = asyncio.create_task(
                self._query_entity(query, entity_ids=entity_ids or [])
            )
        if "causal" in active_graphs and self._event_log:
            tasks["causal"] = asyncio.create_task(
                self._query_causal(query, hours=time_window_hours, limit=top_k * 2)
            )

        if not tasks:
            return []

        # Gather results
        graph_results: dict[GraphName, list[GraphResult]] = {}
        for graph_name, task in tasks.items():
            try:
                graph_results[graph_name] = await task
            except Exception:
                log.exception("magma.%s query failed", graph_name)
                graph_results[graph_name] = []

        # Merge across graphs
        return self._merge(graph_results, w, top_k)

    def available_graphs(self) -> list[GraphName]:
        """Return which graphs have backing stores configured."""
        available: list[GraphName] = []
        if self._embeddings:
            available.append("semantic")
        if self._event_log:
            available.append("temporal")
            available.append("causal")
        if self._entity_resolver or self._memory_store:
            available.append("entity")
        return available

    # ── Graph Queries ────────────────────────────────────────────────

    async def _query_semantic(
        self, query: str, top_k: int = 20
    ) -> list[GraphResult]:
        """Embedding similarity search."""
        results = await self._embeddings.search(query, top_k=top_k)
        return [
            GraphResult(
                graph="semantic",
                id=r.id,
                text=r.text,
                score=r.similarity,
                metadata=r.metadata,
            )
            for r in results
        ]

    async def _query_temporal(
        self, query: str, hours: int = 168, limit: int = 20
    ) -> list[GraphResult]:
        """Recent events matching query keywords, scored by recency."""
        since = datetime.now(UTC) - timedelta(hours=hours)
        events = await self._event_log.replay(since=since, limit=limit * 5)

        # Simple keyword matching + recency scoring
        keywords = set(query.lower().split())
        results: list[GraphResult] = []
        now = datetime.now(UTC)

        for evt in events:
            payload_str = str(evt.get("payload", "")).lower()
            topic = evt.get("topic", "")
            text = f"[{topic}] {payload_str[:200]}"

            # Keyword overlap score
            words = set(payload_str.split())
            overlap = len(keywords & words)
            if overlap == 0:
                continue

            keyword_score = min(1.0, overlap / max(len(keywords), 1))

            # Recency boost: newer events score higher
            try:
                evt_time = datetime.fromisoformat(evt["timestamp"])
                age_hours = (now - evt_time).total_seconds() / 3600
                recency_score = max(0.0, 1.0 - (age_hours / hours))
            except (KeyError, ValueError):
                recency_score = 0.5

            score = 0.6 * keyword_score + 0.4 * recency_score

            results.append(
                GraphResult(
                    graph="temporal",
                    id=evt.get("envelope_id", ""),
                    text=text,
                    score=score,
                    metadata={
                        "topic": topic,
                        "timestamp": evt.get("timestamp"),
                        "source_service_id": evt.get("source_service_id"),
                    },
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def _query_entity(
        self, query: str, entity_ids: list[str] | None = None
    ) -> list[GraphResult]:
        """Entity-linked facts from claims and memory store."""
        results: list[GraphResult] = []

        # Search belief ledger for claims about queried entities
        if self._belief_ledger and entity_ids:
            for eid in entity_ids:
                claims = await self._belief_ledger.get_claims(
                    subject_entity_id=eid
                )
                for claim in claims:
                    text = (
                        f"{claim.subject_entity_id} "
                        f"{claim.predicate} "
                        f"{claim.object_value}"
                    )
                    results.append(
                        GraphResult(
                            graph="entity",
                            id=claim.claim_id,
                            text=text,
                            score=claim.confidence,
                            metadata={
                                "predicate": claim.predicate,
                                "entity_id": claim.subject_entity_id,
                                "source": claim.source_service_id,
                            },
                        )
                    )

        # Also search belief ledger via FTS if no entity_ids provided
        if self._belief_ledger and not entity_ids:
            fts_claims = await self._belief_ledger.search_full_text(query, limit=20)
            for claim in fts_claims:
                text = (
                    f"{claim.subject_entity_id} "
                    f"{claim.predicate} "
                    f"{claim.object_value}"
                )
                results.append(
                    GraphResult(
                        graph="entity",
                        id=claim.claim_id,
                        text=text,
                        score=claim.confidence,
                        metadata={
                            "predicate": claim.predicate,
                            "entity_id": claim.subject_entity_id,
                            "source": claim.source_service_id,
                        },
                    )
                )

        # Memory store entity memories
        if self._memory_store and entity_ids:
            for eid in entity_ids:
                memories = await self._memory_store.get_entity_memories(eid)
                for mem in memories:
                    results.append(
                        GraphResult(
                            graph="entity",
                            id=mem.memory_id,
                            text=f"{mem.key}: {mem.value}",
                            score=mem.confidence,
                            metadata={
                                "category": mem.category,
                                "entity_id": eid,
                            },
                        )
                    )

        return results

    async def _query_causal(
        self, query: str, hours: int = 168, limit: int = 20
    ) -> list[GraphResult]:
        """Trace causation chains via envelope causation_id links."""
        since = datetime.now(UTC) - timedelta(hours=hours)
        events = await self._event_log.replay(since=since, limit=limit * 10)

        # Build causation graph: envelope_id -> list of caused envelopes
        by_id: dict[str, dict] = {}
        children: dict[str, list[str]] = {}
        for evt in events:
            eid = evt.get("envelope_id", "")
            by_id[eid] = evt
            # Check payload for causation_id reference
            payload = evt.get("payload", {})
            causation = payload.get("causation_id") or payload.get("envelope_id")
            if causation and causation != eid:
                children.setdefault(causation, []).append(eid)

        # Find chains that match query keywords
        keywords = set(query.lower().split())
        results: list[GraphResult] = []

        for eid, evt in by_id.items():
            payload_str = str(evt.get("payload", "")).lower()
            words = set(payload_str.split())
            if not (keywords & words):
                continue

            # Compute chain depth (how many downstream effects)
            chain_depth = self._chain_depth(eid, children, max_depth=5)
            if chain_depth == 0:
                continue

            score = min(1.0, chain_depth / 5.0)
            child_topics = []
            for child_id in children.get(eid, []):
                child = by_id.get(child_id)
                if child:
                    child_topics.append(child.get("topic", ""))

            results.append(
                GraphResult(
                    graph="causal",
                    id=eid,
                    text=f"[{evt.get('topic', '')}] caused {chain_depth} downstream events",
                    score=score,
                    metadata={
                        "topic": evt.get("topic"),
                        "chain_depth": chain_depth,
                        "downstream_topics": child_topics[:5],
                        "timestamp": evt.get("timestamp"),
                    },
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    @staticmethod
    def _chain_depth(
        root: str,
        children: dict[str, list[str]],
        max_depth: int = 5,
    ) -> int:
        """BFS to count downstream causal chain depth."""
        visited: set[str] = set()
        queue = [root]
        depth = 0
        while queue and depth < max_depth:
            next_level: list[str] = []
            for node in queue:
                if node in visited:
                    continue
                visited.add(node)
                next_level.extend(children.get(node, []))
            queue = next_level
            if next_level:
                depth += 1
        return depth

    # ── Merge ────────────────────────────────────────────────────────

    @staticmethod
    def _merge(
        graph_results: dict[GraphName, list[GraphResult]],
        weights: dict[GraphName, float],
        top_k: int,
    ) -> list[MergedResult]:
        """Merge results across graphs into ranked MergedResults."""
        # Collect by id, accumulating scores from each graph
        by_id: dict[str, MergedResult] = {}

        for graph_name, results in graph_results.items():
            w = weights.get(graph_name, 0.25)
            for r in results:
                key = r.id
                if key not in by_id:
                    by_id[key] = MergedResult(
                        id=r.id,
                        text=r.text,
                        composite_score=0.0,
                        metadata=r.metadata,
                    )
                by_id[key].graph_scores[graph_name] = r.score
                by_id[key].composite_score += r.score * w
                # Merge metadata
                by_id[key].metadata.update(r.metadata)

        # Sort by composite score descending
        merged = sorted(
            by_id.values(), key=lambda m: m.composite_score, reverse=True
        )
        return merged[:top_k]

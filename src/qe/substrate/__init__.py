import logging
import math
import time
from pathlib import Path
from typing import Any

from qe.substrate.bayesian_belief import BayesianBeliefStore
from qe.substrate.belief_ledger import BeliefLedger
from qe.substrate.cold_storage import ColdStorage
from qe.substrate.embeddings import EmbeddingStore
from qe.substrate.entities import EntityResolver
from qe.substrate.magma import MultiGraphQuery

log = logging.getLogger(__name__)


def _bm25_rerank(
    claims: list[Any],
    query: str,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[Any]:
    """Re-rank claims using BM25 term-frequency scoring.

    Uses a simplified BM25 computation over claim text fields.
    """
    query_terms = set(query.lower().split())
    if not query_terms:
        return claims

    # Compute average document length
    def _claim_text(c: Any) -> str:
        subj = getattr(c, "subject_entity_id", "")
        pred = getattr(c, "predicate", "")
        obj = getattr(c, "object_value", "")
        return f"{subj} {pred} {obj}".lower()

    doc_texts = [_claim_text(c) for c in claims]
    doc_lengths = [len(t.split()) for t in doc_texts]
    avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
    n = len(claims)

    # IDF for each query term
    idf: dict[str, float] = {}
    for term in query_terms:
        df = sum(1 for t in doc_texts if term in t.split())
        idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0) if n > 0 else 0.0

    # Score each document
    scored: list[tuple[float, Any]] = []
    for i, claim in enumerate(claims):
        score = 0.0
        words = doc_texts[i].split()
        dl = doc_lengths[i]
        for term in query_terms:
            tf = words.count(term)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / avgdl))
            score += idf.get(term, 0.0) * (numerator / denominator) if denominator > 0 else 0.0
        scored.append((score, claim))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [claim for _, claim in scored]


class Substrate:
    def __init__(
        self,
        db_path: str = "data/belief_ledger.db",
        cold_path: str = "cold",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.belief_ledger = BeliefLedger(db_path, Path(__file__).parent / "migrations")
        # Keep .ledger as alias for backwards compatibility
        self.ledger = self.belief_ledger
        self.cold_storage = ColdStorage(cold_path)
        self.entity_resolver = EntityResolver(db_path)
        self.embeddings = EmbeddingStore(db_path, model=embedding_model)
        # MAGMA multi-graph query (event_log wired later via set_event_log)
        self._magma = MultiGraphQuery(
            embeddings=self.embeddings,
            belief_ledger=self.belief_ledger,
            entity_resolver=self.entity_resolver,
        )

    async def initialize(self) -> None:
        await self.ledger.initialize()

    async def commit_claim(self, claim):
        # Normalize entity name before committing
        claim.subject_entity_id = await self.entity_resolver.ensure_entity(
            claim.subject_entity_id
        )
        committed = await self.ledger.commit_claim(claim)

        # Best-effort semantic indexing for hybrid retrieval.
        # Fail-open so claim commits are never blocked by embeddings.
        try:
            await self.embeddings.store(
                id=f"claim:{committed.claim_id}",
                text=(
                    f"{committed.subject_entity_id} "
                    f"{committed.predicate} "
                    f"{committed.object_value}"
                ),
                metadata={
                    "kind": "claim",
                    "claim_id": committed.claim_id,
                    "subject_entity_id": committed.subject_entity_id,
                    "predicate": committed.predicate,
                },
            )
        except Exception:
            log.warning(
                "claim.embedding_index_failed claim_id=%s",
                committed.claim_id,
                exc_info=True,
            )

        return committed

    async def get_claims(self, **kwargs):
        return await self.ledger.get_claims(**kwargs)

    async def get_claim_by_id(self, claim_id: str):
        return await self.ledger.get_claim_by_id(claim_id)

    async def count_claims(self, **kwargs):
        return await self.ledger.count_claims(**kwargs)

    async def retract_claim(self, claim_id: str) -> bool:
        return await self.ledger.retract_claim(claim_id)

    async def search_full_text(self, query: str, limit: int = 20):
        return await self.ledger.search_full_text(query, limit)

    async def search_semantic(self, query: str, top_k: int = 10):
        return await self.embeddings.search(query, top_k=top_k)

    async def hybrid_search(
        self,
        query: str,
        *,
        fts_top_k: int = 20,
        semantic_top_k: int = 20,
        semantic_min_similarity: float = 0.3,
        fts_weight: float = 0.6,
        semantic_weight: float = 0.4,
        rrf_k: int = 60,
        use_bm25: bool = False,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ) -> list[Any]:
        """Hybrid claim retrieval using reciprocal-rank fusion (RRF).

        Combines:
        - lexical ranking from FTS5 over claims (optionally BM25-scored)
        - semantic ranking from embedding nearest-neighbors

        When ``use_bm25=True``, FTS results are re-ranked using BM25 term
        frequency scoring with configurable k1 and b parameters.
        """
        from qe.runtime.metrics import get_metrics

        metrics = get_metrics()
        metrics.counter("retrieval_hybrid_calls_total").inc()
        fts_claims = await self.search_full_text(query, limit=fts_top_k)
        if fts_claims:
            metrics.counter("retrieval_hybrid_fts_nonempty_total").inc()

        # Optional BM25 re-ranking of FTS results
        if use_bm25 and fts_claims:
            fts_claims = _bm25_rerank(fts_claims, query, k1=bm25_k1, b=bm25_b)

        semantic_claims: list[Any] = []
        try:
            if await self.embeddings.count() > 0:
                start = time.monotonic()
                semantic_hits = await self.embeddings.search(
                    query,
                    top_k=semantic_top_k,
                    min_similarity=semantic_min_similarity,
                )
                metrics.histogram("vector_query_latency_ms").observe(
                    (time.monotonic() - start) * 1000
                )
                if semantic_hits:
                    metrics.counter("retrieval_hybrid_semantic_nonempty_total").inc()
                for hit in semantic_hits:
                    claim_id = hit.metadata.get("claim_id")
                    if not claim_id and hit.id.startswith("claim:"):
                        claim_id = hit.id.split("claim:", 1)[1]
                    if not claim_id:
                        continue
                    claim = await self.get_claim_by_id(claim_id)
                    if claim is not None:
                        semantic_claims.append(claim)
        except Exception:
            log.warning("hybrid.semantic_search_failed", exc_info=True)

        if not fts_claims and not semantic_claims:
            return []

        fused: dict[str, float] = {}
        by_id: dict[str, Any] = {}

        for rank, claim in enumerate(fts_claims, start=1):
            by_id[claim.claim_id] = claim
            fused[claim.claim_id] = fused.get(claim.claim_id, 0.0) + (
                fts_weight * (1.0 / (rrf_k + rank))
            )

        for rank, claim in enumerate(semantic_claims, start=1):
            by_id[claim.claim_id] = claim
            fused[claim.claim_id] = fused.get(claim.claim_id, 0.0) + (
                semantic_weight * (1.0 / (rrf_k + rank))
            )

        ranked_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)
        limit = max(fts_top_k, semantic_top_k)
        return [by_id[cid] for cid in ranked_ids[:limit]]

    def set_event_log(self, event_log) -> None:
        """Wire the EventLog into MAGMA for temporal + causal queries."""
        self._magma = MultiGraphQuery(
            embeddings=self.embeddings,
            event_log=event_log,
            belief_ledger=self.belief_ledger,
            entity_resolver=self.entity_resolver,
        )

    def set_memory_store(self, memory_store) -> None:
        """Wire MemoryStore into MAGMA for entity memory queries."""
        self._magma = MultiGraphQuery(
            embeddings=self.embeddings,
            event_log=self._magma._event_log,
            belief_ledger=self.belief_ledger,
            entity_resolver=self.entity_resolver,
            memory_store=memory_store,
        )

    @property
    def bayesian_belief(self) -> BayesianBeliefStore:
        if not hasattr(self, "_bayesian_belief"):
            self._bayesian_belief = BayesianBeliefStore(db_path=self.belief_ledger._db_path)
        return self._bayesian_belief

    async def multi_query(self, query: str, **kwargs):
        """MAGMA multi-graph query: fan out across semantic, temporal, entity, causal."""
        return await self._magma.multi_query(query, **kwargs)

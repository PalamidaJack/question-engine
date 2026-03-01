"""Bayesian Belief Store — Tier 2 semantic memory with evidence accumulation.

Wraps the existing BeliefLedger with Bayesian updating semantics.
Instead of overwriting confidence when a new claim arrives, evidence
is accumulated and posterior probability is computed via Bayes' rule.
"""

import json
import logging
import math
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

import aiosqlite
from pydantic import BaseModel, Field

from qe.models.claim import Claim

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EvidenceRecord(BaseModel):
    """A single piece of evidence for or against a claim."""

    evidence_id: str = Field(default_factory=lambda: f"evi_{uuid.uuid4().hex[:12]}")
    claim_id: str = ""
    source: str  # tool call, claim, investigation result, user
    supports: bool  # True = confirms, False = contradicts
    strength: float = Field(ge=0.0, le=1.0)  # how strong this evidence is
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class BeliefContext(BaseModel):
    """Enriched view of a claim with its full evidence chain."""

    claim: Claim
    prior: float
    posterior: float
    evidence_count: int
    supporting_evidence: list[EvidenceRecord]
    contradicting_evidence: list[EvidenceRecord]
    likelihood_ratio: float
    confidence_interval: tuple[float, float]  # (lower, upper) 95% CI


class GraphEdge(BaseModel):
    """A directed edge in the knowledge graph."""

    edge_id: str = Field(default_factory=lambda: f"edg_{uuid.uuid4().hex[:12]}")
    source_entity: str
    target_entity: str
    relation: str
    confidence: float
    source_claim_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Hypothesis(BaseModel):
    """A testable hypothesis generated from contradictions."""

    hypothesis_id: str = Field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:12]}")
    statement: str
    prior_probability: float = 0.5
    current_probability: float = 0.5
    falsification_criteria: list[str] = Field(default_factory=list)
    experiments: list[str] = Field(default_factory=list)  # inquiry_ids
    status: Literal["active", "confirmed", "falsified", "abandoned"] = "active"
    source_contradiction_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class BayesianBeliefStore:
    """Tier 2 semantic memory: Bayesian belief updating over the BeliefLedger.

    Key difference from BeliefLedger.commit_claim():
    - Old: ``if new_confidence > old_confidence: supersede``
    - New: accumulate evidence, compute ``P(H|E) = P(E|H) * P(H) / P(E)``

    The store wraps an existing BeliefLedger and its database. New columns
    (prior, posterior, evidence_count, likelihood_ratio, updated_at) are added
    via migration 0012.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    # -------------------------------------------------------------------
    # Bayesian updating
    # -------------------------------------------------------------------

    async def update_belief(
        self,
        claim: Claim,
        evidence: EvidenceRecord,
    ) -> Claim:
        """Apply Bayesian update to a claim with new evidence.

        If the claim doesn't exist yet, it is created with the given confidence
        as the prior. If it exists, the posterior is updated using the evidence
        strength and direction.

        Returns the updated claim with new posterior as its confidence.
        """
        evidence.claim_id = claim.claim_id
        now = datetime.now(UTC)

        async with aiosqlite.connect(self._db_path) as db:
            # Check for existing claim with same (subject, predicate)
            cursor = await db.execute(
                """
                SELECT claim_id, confidence, prior, posterior,
                       evidence_count, likelihood_ratio
                FROM claims
                WHERE subject_entity_id = ? AND predicate = ?
                  AND superseded_by IS NULL
                ORDER BY created_at DESC LIMIT 1
                """,
                (claim.subject_entity_id, claim.predicate),
            )
            existing = await cursor.fetchone()

            if existing:
                existing_id = existing[0]
                prior = existing[3] if existing[3] is not None else existing[1]
                evidence_count = (existing[4] or 0) + 1
                old_lr = existing[5] if existing[5] is not None else 1.0

                # Compute likelihood ratio for this evidence
                lr = self._compute_likelihood_ratio(evidence)
                cumulative_lr = old_lr * lr

                # Bayesian update: posterior = prior * LR / (prior * LR + (1-prior))
                posterior = self._bayes_update(prior, cumulative_lr)

                # Update existing claim in place
                await db.execute(
                    """
                    UPDATE claims
                    SET confidence = ?, posterior = ?, evidence_count = ?,
                        likelihood_ratio = ?, updated_at = ?
                    WHERE claim_id = ?
                    """,
                    (posterior, posterior, evidence_count, cumulative_lr,
                     now.isoformat(), existing_id),
                )

                # Store evidence record
                await self._store_evidence(db, evidence)
                await db.commit()

                # Return updated claim
                updated = claim.model_copy(
                    update={
                        "claim_id": existing_id,
                        "confidence": posterior,
                    }
                )
                return updated
            else:
                # New claim — set prior = confidence, posterior = updated with evidence
                prior = claim.confidence
                lr = self._compute_likelihood_ratio(evidence)
                posterior = self._bayes_update(prior, lr)

                await db.execute(
                    """
                    INSERT INTO claims (
                        claim_id, schema_version, subject_entity_id, predicate,
                        object_value, confidence, source_service_id,
                        source_envelope_ids, created_at, valid_until,
                        superseded_by, tags, metadata,
                        prior, posterior, evidence_count, likelihood_ratio,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim.claim_id,
                        claim.schema_version,
                        claim.subject_entity_id,
                        claim.predicate,
                        claim.object_value,
                        posterior,
                        claim.source_service_id,
                        json.dumps(claim.source_envelope_ids),
                        claim.created_at.isoformat(),
                        claim.valid_until.isoformat() if claim.valid_until else None,
                        claim.superseded_by,
                        json.dumps(claim.tags),
                        json.dumps(claim.metadata),
                        prior,
                        posterior,
                        1,  # evidence_count
                        lr,
                        now.isoformat(),
                    ),
                )

                await self._store_evidence(db, evidence)
                await db.commit()

                return claim.model_copy(update={"confidence": posterior})

    async def add_evidence(
        self,
        claim_id: str,
        evidence: EvidenceRecord,
    ) -> float:
        """Add evidence to an existing claim and return the new posterior."""
        evidence.claim_id = claim_id
        now = datetime.now(UTC)

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT confidence, prior, posterior, evidence_count, likelihood_ratio
                FROM claims WHERE claim_id = ?
                """,
                (claim_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise ValueError(f"Claim {claim_id} not found")

            prior = row[1] if row[1] is not None else row[0]
            evidence_count = (row[3] or 0) + 1
            old_lr = row[4] if row[4] is not None else 1.0

            lr = self._compute_likelihood_ratio(evidence)
            cumulative_lr = old_lr * lr
            posterior = self._bayes_update(prior, cumulative_lr)

            await db.execute(
                """
                UPDATE claims
                SET confidence = ?, posterior = ?, evidence_count = ?,
                    likelihood_ratio = ?, updated_at = ?
                WHERE claim_id = ?
                """,
                (posterior, posterior, evidence_count, cumulative_lr,
                 now.isoformat(), claim_id),
            )

            await self._store_evidence(db, evidence)
            await db.commit()

        return posterior

    # -------------------------------------------------------------------
    # Enriched retrieval
    # -------------------------------------------------------------------

    async def get_belief_with_context(
        self,
        subject: str,
        predicate: str,
    ) -> BeliefContext | None:
        """Returns a claim with its full evidence chain and confidence interval.

        This replaces passive get_claims() with enriched context for LLM consumption.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT * FROM claims
                WHERE subject_entity_id = ? AND predicate = ?
                  AND superseded_by IS NULL
                ORDER BY created_at DESC LIMIT 1
                """,
                (subject, predicate),
            )
            row = await cursor.fetchone()
            if row is None:
                return None

            claim = self._row_to_claim(row)
            claim_id = claim.claim_id

            # Get Bayesian fields (may be NULL for pre-migration claims)
            prior = row[13] if len(row) > 13 and row[13] is not None else claim.confidence
            posterior = row[14] if len(row) > 14 and row[14] is not None else claim.confidence
            evidence_count = row[15] if len(row) > 15 and row[15] is not None else 0
            likelihood_ratio = row[16] if len(row) > 16 and row[16] is not None else 1.0

            # Get evidence records
            cursor = await db.execute(
                "SELECT * FROM evidence WHERE claim_id = ? ORDER BY timestamp",
                (claim_id,),
            )
            evidence_rows = await cursor.fetchall()

        supporting = []
        contradicting = []
        for erow in evidence_rows:
            record = EvidenceRecord(
                evidence_id=erow[0],
                claim_id=erow[1],
                source=erow[2],
                supports=bool(erow[3]),
                strength=erow[4],
                timestamp=datetime.fromisoformat(erow[5]),
                metadata=json.loads(erow[6]) if erow[6] else {},
            )
            if record.supports:
                supporting.append(record)
            else:
                contradicting.append(record)

        ci = self._compute_confidence_interval(posterior, evidence_count)

        return BeliefContext(
            claim=claim,
            prior=prior,
            posterior=posterior,
            evidence_count=evidence_count,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            likelihood_ratio=likelihood_ratio,
            confidence_interval=ci,
        )

    async def get_beliefs_for_entity(
        self,
        entity_id: str,
        min_posterior: float = 0.0,
    ) -> list[BeliefContext]:
        """Get all active beliefs about an entity with full context."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                """
                SELECT DISTINCT predicate FROM claims
                WHERE subject_entity_id = ? AND superseded_by IS NULL
                  AND confidence >= ?
                """,
                (entity_id, min_posterior),
            )
            predicates = [row[0] for row in await cursor.fetchall()]

        results = []
        for pred in predicates:
            ctx = await self.get_belief_with_context(entity_id, pred)
            if ctx is not None and ctx.posterior >= min_posterior:
                results.append(ctx)

        return results

    # -------------------------------------------------------------------
    # Knowledge graph
    # -------------------------------------------------------------------

    async def add_graph_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the knowledge graph."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO knowledge_graph_edges (
                    edge_id, source_entity, target_entity, relation,
                    confidence, source_claim_ids, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.edge_id,
                    edge.source_entity,
                    edge.target_entity,
                    edge.relation,
                    edge.confidence,
                    json.dumps(edge.source_claim_ids),
                    edge.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def get_graph_neighbors(
        self,
        entity_id: str,
        depth: int = 2,
    ) -> list[GraphEdge]:
        """Traverse knowledge graph edges for cross-chunk reasoning (GraphRAG)."""
        visited: set[str] = set()
        edges: list[GraphEdge] = []
        frontier = {entity_id}

        async with aiosqlite.connect(self._db_path) as db:
            for _ in range(depth):
                if not frontier:
                    break
                next_frontier: set[str] = set()
                for eid in frontier:
                    if eid in visited:
                        continue
                    visited.add(eid)

                    cursor = await db.execute(
                        """
                        SELECT * FROM knowledge_graph_edges
                        WHERE source_entity = ? OR target_entity = ?
                        """,
                        (eid, eid),
                    )
                    for row in await cursor.fetchall():
                        edge = GraphEdge(
                            edge_id=row[0],
                            source_entity=row[1],
                            target_entity=row[2],
                            relation=row[3],
                            confidence=row[4],
                            source_claim_ids=json.loads(row[5]),
                            created_at=datetime.fromisoformat(row[6]),
                        )
                        edges.append(edge)
                        # Expand frontier
                        other = row[2] if row[1] == eid else row[1]
                        if other not in visited:
                            next_frontier.add(other)
                frontier = next_frontier

        return edges

    # -------------------------------------------------------------------
    # Hypotheses
    # -------------------------------------------------------------------

    async def store_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        """Persist a hypothesis."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO hypotheses (
                    hypothesis_id, statement, prior_probability,
                    current_probability, falsification_criteria, experiments,
                    status, source_contradiction_ids, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis.hypothesis_id,
                    hypothesis.statement,
                    hypothesis.prior_probability,
                    hypothesis.current_probability,
                    json.dumps(hypothesis.falsification_criteria),
                    json.dumps(hypothesis.experiments),
                    hypothesis.status,
                    json.dumps(hypothesis.source_contradiction_ids),
                    hypothesis.created_at.isoformat(),
                    hypothesis.updated_at.isoformat(),
                ),
            )
            await db.commit()
        return hypothesis

    async def update_hypothesis(
        self,
        hypothesis_id: str,
        evidence: EvidenceRecord,
    ) -> Hypothesis:
        """Apply sequential Bayesian update to a hypothesis with new evidence.

        Uses current_probability as the basis for the update (sequential Bayes).
        Preserves the original prior_probability for auditability.
        """
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM hypotheses WHERE hypothesis_id = ?",
                (hypothesis_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                raise ValueError(f"Hypothesis {hypothesis_id} not found")

            original_prior = row[2]  # prior_probability — never overwritten
            current = row[3]  # current_probability — used as sequential prior
            lr = self._compute_likelihood_ratio(evidence)
            posterior = self._bayes_update(current, lr)
            now = datetime.now(UTC)

            # Determine status based on posterior
            status = row[6]
            if posterior >= 0.95:
                status = "confirmed"
            elif posterior <= 0.05:
                status = "falsified"

            # Only update current_probability; prior_probability is immutable
            await db.execute(
                """
                UPDATE hypotheses
                SET current_probability = ?,
                    status = ?, updated_at = ?
                WHERE hypothesis_id = ?
                """,
                (posterior, status, now.isoformat(), hypothesis_id),
            )
            await db.commit()

        return Hypothesis(
            hypothesis_id=row[0],
            statement=row[1],
            prior_probability=original_prior,
            current_probability=posterior,
            falsification_criteria=json.loads(row[4]),
            experiments=json.loads(row[5]),
            status=status,
            source_contradiction_ids=json.loads(row[7]),
            created_at=datetime.fromisoformat(row[8]),
            updated_at=now,
        )

    async def get_active_hypotheses(self) -> list[Hypothesis]:
        """Get all active (untested) hypotheses."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM hypotheses WHERE status = 'active' ORDER BY created_at"
            )
            rows = await cursor.fetchall()

        return [self._row_to_hypothesis(row) for row in rows]

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _compute_likelihood_ratio(evidence: EvidenceRecord) -> float:
        """Compute likelihood ratio from evidence.

        For supporting evidence: LR > 1 (increases belief)
        For contradicting evidence: LR < 1 (decreases belief)

        LR = P(evidence | H true) / P(evidence | H false)
        We approximate this from evidence strength:
          supporting:    LR = 1 + strength * 4   (range: 1.0 to 5.0)
          contradicting: LR = 1 / (1 + strength * 4)  (range: 0.2 to 1.0)
        """
        if evidence.supports:
            return 1.0 + evidence.strength * 4.0
        else:
            return 1.0 / (1.0 + evidence.strength * 4.0)

    @staticmethod
    def _bayes_update(prior: float, likelihood_ratio: float) -> float:
        """Apply Bayes' rule: posterior = prior * LR / (prior * LR + (1 - prior))

        Clamps to [0.01, 0.99] to prevent certainty lock.
        """
        prior = max(0.01, min(0.99, prior))
        numerator = prior * likelihood_ratio
        denominator = numerator + (1.0 - prior)
        if denominator == 0:
            return prior
        posterior = numerator / denominator
        return max(0.01, min(0.99, posterior))

    @staticmethod
    def _compute_confidence_interval(
        posterior: float,
        evidence_count: int,
    ) -> tuple[float, float]:
        """Approximate 95% CI using a Beta distribution approximation.

        With n evidence items and observed proportion p (posterior),
        the CI is approximately p +/- 1.96 * sqrt(p*(1-p)/n).
        With no evidence, we use a wide interval.
        """
        if evidence_count <= 0:
            return (0.0, 1.0)

        p = posterior
        se = math.sqrt(p * (1.0 - p) / evidence_count)
        margin = 1.96 * se
        lower = max(0.0, p - margin)
        upper = min(1.0, p + margin)
        return (round(lower, 4), round(upper, 4))

    async def _store_evidence(
        self,
        db: aiosqlite.Connection,
        evidence: EvidenceRecord,
    ) -> None:
        """Insert an evidence record."""
        await db.execute(
            """
            INSERT INTO evidence (
                evidence_id, claim_id, source, supports,
                strength, timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence.evidence_id,
                evidence.claim_id,
                evidence.source,
                evidence.supports,
                evidence.strength,
                evidence.timestamp.isoformat(),
                json.dumps(evidence.metadata),
            ),
        )

    @staticmethod
    def _row_to_claim(row: tuple) -> Claim:
        """Convert a database row to a Claim (backward compatible)."""
        return Claim(
            claim_id=row[0],
            schema_version=row[1],
            subject_entity_id=row[2],
            predicate=row[3],
            object_value=row[4],
            confidence=row[5],
            source_service_id=row[6],
            source_envelope_ids=json.loads(row[7]),
            created_at=datetime.fromisoformat(row[8]),
            valid_until=datetime.fromisoformat(row[9]) if row[9] else None,
            superseded_by=row[10],
            tags=json.loads(row[11]),
            metadata=json.loads(row[12]),
        )

    @staticmethod
    def _row_to_hypothesis(row: tuple) -> Hypothesis:
        return Hypothesis(
            hypothesis_id=row[0],
            statement=row[1],
            prior_probability=row[2],
            current_probability=row[3],
            falsification_criteria=json.loads(row[4]),
            experiments=json.loads(row[5]),
            status=row[6],
            source_contradiction_ids=json.loads(row[7]),
            created_at=datetime.fromisoformat(row[8]),
            updated_at=datetime.fromisoformat(row[9]),
        )

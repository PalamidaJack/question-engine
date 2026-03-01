"""Tests for BayesianBeliefStore — Tier 2 semantic memory."""

import json
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import pytest

from qe.models.claim import Claim
from qe.substrate.bayesian_belief import (
    BayesianBeliefStore,
    BeliefContext,
    EvidenceRecord,
    GraphEdge,
    Hypothesis,
)


@pytest.fixture
async def db_path(tmp_path):
    """Create a fresh SQLite database with the required schema."""
    path = str(tmp_path / "test_bayesian.db")
    async with aiosqlite.connect(path) as db:
        # Create claims table (from migration 0001)
        await db.execute("""
            CREATE TABLE claims (
                claim_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                subject_entity_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_value TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_service_id TEXT NOT NULL,
                source_envelope_ids TEXT NOT NULL,
                created_at TEXT NOT NULL,
                valid_until TEXT,
                superseded_by TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                prior REAL DEFAULT 0.5,
                posterior REAL,
                evidence_count INTEGER DEFAULT 0,
                likelihood_ratio REAL DEFAULT 1.0,
                updated_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE evidence (
                evidence_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL,
                source TEXT NOT NULL,
                supports BOOLEAN NOT NULL,
                strength REAL NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (claim_id) REFERENCES claims(claim_id)
            )
        """)
        await db.execute("""
            CREATE TABLE knowledge_graph_edges (
                edge_id TEXT PRIMARY KEY,
                source_entity TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                relation TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_claim_ids TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                prior_probability REAL NOT NULL DEFAULT 0.5,
                current_probability REAL NOT NULL DEFAULT 0.5,
                falsification_criteria TEXT NOT NULL DEFAULT '[]',
                experiments TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'active',
                source_contradiction_ids TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await db.commit()
    return path


@pytest.fixture
def store(db_path):
    return BayesianBeliefStore(db_path)


def make_claim(**overrides) -> Claim:
    defaults = {
        "subject_entity_id": "company_x",
        "predicate": "revenue_growth",
        "object_value": "20% YoY",
        "confidence": 0.5,
        "source_service_id": "researcher",
        "source_envelope_ids": ["env_001"],
    }
    defaults.update(overrides)
    return Claim(**defaults)


# ── Bayesian Updating ──────────────────────────────────────────────────────


class TestBayesianUpdating:
    async def test_new_claim_with_supporting_evidence(self, store):
        claim = make_claim(confidence=0.5)
        evidence = EvidenceRecord(source="web_search", supports=True, strength=0.8)

        result = await store.update_belief(claim, evidence)

        # Supporting evidence should increase confidence above 0.5
        assert result.confidence > 0.5
        assert result.confidence < 1.0

    async def test_new_claim_with_contradicting_evidence(self, store):
        claim = make_claim(confidence=0.5)
        evidence = EvidenceRecord(source="fact_check", supports=False, strength=0.8)

        result = await store.update_belief(claim, evidence)

        # Contradicting evidence should decrease confidence below 0.5
        assert result.confidence < 0.5
        assert result.confidence > 0.0

    async def test_evidence_accumulation(self, store):
        claim = make_claim(confidence=0.5)

        # First evidence: supporting
        e1 = EvidenceRecord(source="source_1", supports=True, strength=0.7)
        r1 = await store.update_belief(claim, e1)

        # Second evidence: also supporting
        e2 = EvidenceRecord(source="source_2", supports=True, strength=0.6)
        posterior = await store.add_evidence(r1.claim_id, e2)

        # Two supporting pieces should push confidence higher
        assert posterior > r1.confidence

    async def test_conflicting_evidence(self, store):
        claim = make_claim(confidence=0.5)

        # Strong supporting evidence
        e1 = EvidenceRecord(source="source_1", supports=True, strength=0.9)
        r1 = await store.update_belief(claim, e1)
        high_conf = r1.confidence

        # Strong contradicting evidence
        e2 = EvidenceRecord(source="source_2", supports=False, strength=0.9)
        posterior = await store.add_evidence(r1.claim_id, e2)

        # Conflicting evidence should pull back toward center
        assert posterior < high_conf

    async def test_posterior_clamped(self, store):
        """Posterior should never reach 0.0 or 1.0 (prevents certainty lock)."""
        claim = make_claim(confidence=0.99)
        e = EvidenceRecord(source="s", supports=True, strength=1.0)
        result = await store.update_belief(claim, e)

        assert result.confidence <= 0.99
        assert result.confidence >= 0.01

    async def test_existing_claim_updated_not_duplicated(self, store):
        claim = make_claim(confidence=0.5)
        e1 = EvidenceRecord(source="s1", supports=True, strength=0.5)
        await store.update_belief(claim, e1)

        # Second update should update same claim, not create new
        claim2 = make_claim(confidence=0.6)  # same subject+predicate
        e2 = EvidenceRecord(source="s2", supports=True, strength=0.5)
        await store.update_belief(claim2, e2)

        # Should still be one active claim
        ctx = await store.get_belief_with_context("company_x", "revenue_growth")
        assert ctx is not None
        assert ctx.evidence_count == 2


class TestBayesMath:
    def test_likelihood_ratio_supporting(self):
        e = EvidenceRecord(source="s", supports=True, strength=0.5)
        lr = BayesianBeliefStore._compute_likelihood_ratio(e)
        assert lr > 1.0  # supporting evidence increases belief

    def test_likelihood_ratio_contradicting(self):
        e = EvidenceRecord(source="s", supports=False, strength=0.5)
        lr = BayesianBeliefStore._compute_likelihood_ratio(e)
        assert lr < 1.0  # contradicting evidence decreases belief

    def test_likelihood_ratio_weak_evidence(self):
        e = EvidenceRecord(source="s", supports=True, strength=0.0)
        lr = BayesianBeliefStore._compute_likelihood_ratio(e)
        assert lr == pytest.approx(1.0)  # zero-strength evidence is neutral

    def test_bayes_update_neutral(self):
        # LR = 1.0 should not change the prior
        posterior = BayesianBeliefStore._bayes_update(0.5, 1.0)
        assert posterior == pytest.approx(0.5)

    def test_bayes_update_supporting(self):
        posterior = BayesianBeliefStore._bayes_update(0.5, 3.0)
        assert posterior > 0.5

    def test_bayes_update_contradicting(self):
        posterior = BayesianBeliefStore._bayes_update(0.5, 0.3)
        assert posterior < 0.5

    def test_confidence_interval_no_evidence(self):
        ci = BayesianBeliefStore._compute_confidence_interval(0.5, 0)
        assert ci == (0.0, 1.0)

    def test_confidence_interval_narrows_with_evidence(self):
        ci_few = BayesianBeliefStore._compute_confidence_interval(0.7, 5)
        ci_many = BayesianBeliefStore._compute_confidence_interval(0.7, 50)
        # More evidence = narrower interval
        assert (ci_many[1] - ci_many[0]) < (ci_few[1] - ci_few[0])


# ── Enriched Retrieval ──────────────────────────────────────────────────────


class TestBeliefContext:
    async def test_get_belief_with_context(self, store):
        claim = make_claim(confidence=0.6)
        e1 = EvidenceRecord(source="web", supports=True, strength=0.7)
        e2 = EvidenceRecord(source="paper", supports=False, strength=0.3)

        await store.update_belief(claim, e1)
        await store.add_evidence(claim.claim_id, e2)

        ctx = await store.get_belief_with_context("company_x", "revenue_growth")

        assert ctx is not None
        assert isinstance(ctx, BeliefContext)
        assert ctx.evidence_count == 2
        assert len(ctx.supporting_evidence) == 1
        assert len(ctx.contradicting_evidence) == 1
        assert ctx.confidence_interval[0] < ctx.posterior
        assert ctx.confidence_interval[1] > ctx.posterior

    async def test_get_nonexistent_belief(self, store):
        ctx = await store.get_belief_with_context("nobody", "nothing")
        assert ctx is None

    async def test_get_beliefs_for_entity(self, store):
        # Create two claims for the same entity
        c1 = make_claim(predicate="revenue", confidence=0.7)
        c2 = make_claim(predicate="market_share", confidence=0.6)

        e1 = EvidenceRecord(source="s1", supports=True, strength=0.5)
        e2 = EvidenceRecord(source="s2", supports=True, strength=0.5)

        await store.update_belief(c1, e1)
        await store.update_belief(c2, e2)

        beliefs = await store.get_beliefs_for_entity("company_x")
        assert len(beliefs) == 2


# ── Knowledge Graph ──────────────────────────────────────────────────────


class TestKnowledgeGraph:
    async def test_add_and_retrieve_edges(self, store):
        edge = GraphEdge(
            source_entity="company_x",
            target_entity="sector_tech",
            relation="belongs_to",
            confidence=0.9,
            source_claim_ids=["clm_001"],
        )
        await store.add_graph_edge(edge)

        neighbors = await store.get_graph_neighbors("company_x", depth=1)
        assert len(neighbors) == 1
        assert neighbors[0].target_entity == "sector_tech"

    async def test_graph_traversal_depth_2(self, store):
        # A -> B -> C
        e1 = GraphEdge(
            source_entity="a", target_entity="b",
            relation="r1", confidence=0.9,
        )
        e2 = GraphEdge(
            source_entity="b", target_entity="c",
            relation="r2", confidence=0.8,
        )
        await store.add_graph_edge(e1)
        await store.add_graph_edge(e2)

        # Depth 1: only A-B
        neighbors_1 = await store.get_graph_neighbors("a", depth=1)
        entities_1 = {n.source_entity for n in neighbors_1} | {n.target_entity for n in neighbors_1}
        assert "b" in entities_1
        assert "c" not in entities_1

        # Depth 2: A-B and B-C (may include duplicate edge traversals)
        neighbors_2 = await store.get_graph_neighbors("a", depth=2)
        entities_2 = {n.source_entity for n in neighbors_2} | {n.target_entity for n in neighbors_2}
        assert "b" in entities_2
        assert "c" in entities_2

    async def test_no_neighbors(self, store):
        neighbors = await store.get_graph_neighbors("isolated_entity")
        assert neighbors == []


# ── Hypotheses ──────────────────────────────────────────────────────────


class TestHypotheses:
    async def test_store_and_retrieve(self, store):
        h = Hypothesis(
            statement="Tech sector is overvalued relative to earnings",
            prior_probability=0.6,
            current_probability=0.6,
            falsification_criteria=["Find P/E ratios below historical average"],
        )
        await store.store_hypothesis(h)

        active = await store.get_active_hypotheses()
        assert len(active) == 1
        assert active[0].statement == h.statement

    async def test_update_with_supporting_evidence(self, store):
        h = Hypothesis(
            statement="Renewable infrastructure is mispriced",
            prior_probability=0.5,
            current_probability=0.5,
        )
        await store.store_hypothesis(h)

        evidence = EvidenceRecord(source="analysis", supports=True, strength=0.8)
        updated = await store.update_hypothesis(h.hypothesis_id, evidence)

        assert updated.current_probability > 0.5

    async def test_hypothesis_auto_confirmation(self, store):
        h = Hypothesis(
            statement="Test hypothesis",
            prior_probability=0.9,
            current_probability=0.9,
        )
        await store.store_hypothesis(h)

        evidence = EvidenceRecord(source="test", supports=True, strength=1.0)
        updated = await store.update_hypothesis(h.hypothesis_id, evidence)

        assert updated.status == "confirmed"

    async def test_hypothesis_auto_falsification(self, store):
        h = Hypothesis(
            statement="Test hypothesis",
            prior_probability=0.1,
            current_probability=0.1,
        )
        await store.store_hypothesis(h)

        evidence = EvidenceRecord(source="test", supports=False, strength=1.0)
        updated = await store.update_hypothesis(h.hypothesis_id, evidence)

        assert updated.status == "falsified"

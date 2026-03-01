"""Tests for KnowledgeLoop wiring — bus topics, schemas, feature flag."""

from __future__ import annotations

from qe.bus.protocol import TOPICS
from qe.bus.schemas import (
    TOPIC_SCHEMAS,
    KnowledgeBeliefPromotedPayload,
    KnowledgeConsolidationCompletedPayload,
    KnowledgeHypothesisUpdatedPayload,
    validate_payload,
)

# ── Bus Topics ───────────────────────────────────────────────────────────


class TestKnowledgeBusTopics:
    def test_consolidation_completed_in_topics(self):
        assert "knowledge.consolidation_completed" in TOPICS

    def test_belief_promoted_in_topics(self):
        assert "knowledge.belief_promoted" in TOPICS

    def test_hypothesis_updated_in_topics(self):
        assert "knowledge.hypothesis_updated" in TOPICS


# ── Bus Schemas ──────────────────────────────────────────────────────────


class TestKnowledgeBusSchemas:
    def test_consolidation_completed_schema_registered(self):
        assert "knowledge.consolidation_completed" in TOPIC_SCHEMAS

    def test_belief_promoted_schema_registered(self):
        assert "knowledge.belief_promoted" in TOPIC_SCHEMAS

    def test_hypothesis_updated_schema_registered(self):
        assert "knowledge.hypothesis_updated" in TOPIC_SCHEMAS

    def test_consolidation_completed_payload_validates(self):
        payload = validate_payload("knowledge.consolidation_completed", {
            "episodes_scanned": 42,
            "patterns_detected": 5,
            "beliefs_promoted": 2,
            "contradictions_found": 1,
            "hypotheses_reviewed": 3,
        })
        assert isinstance(payload, KnowledgeConsolidationCompletedPayload)
        assert payload.episodes_scanned == 42
        assert payload.beliefs_promoted == 2

    def test_belief_promoted_payload_validates(self):
        payload = validate_payload("knowledge.belief_promoted", {
            "subject_entity_id": "company_x",
            "predicate": "revenue",
            "object_value": "$10M",
            "confidence": 0.85,
            "evidence_count": 5,
        })
        assert isinstance(payload, KnowledgeBeliefPromotedPayload)
        assert payload.subject_entity_id == "company_x"
        assert payload.confidence == 0.85

    def test_hypothesis_updated_payload_validates(self):
        payload = validate_payload("knowledge.hypothesis_updated", {
            "hypothesis_id": "hyp_123",
            "old_status": "active",
            "new_status": "confirmed",
            "probability": 0.96,
        })
        assert isinstance(payload, KnowledgeHypothesisUpdatedPayload)
        assert payload.hypothesis_id == "hyp_123"
        assert payload.new_status == "confirmed"


# ── Schema Defaults ──────────────────────────────────────────────────────


class TestSchemaDefaults:
    def test_consolidation_completed_defaults(self):
        p = KnowledgeConsolidationCompletedPayload()
        assert p.episodes_scanned == 0
        assert p.patterns_detected == 0
        assert p.beliefs_promoted == 0
        assert p.contradictions_found == 0
        assert p.hypotheses_reviewed == 0

    def test_belief_promoted_defaults(self):
        p = KnowledgeBeliefPromotedPayload(
            subject_entity_id="x",
            predicate="p",
            object_value="v",
        )
        assert p.confidence == 0.5
        assert p.evidence_count == 0

    def test_hypothesis_updated_defaults(self):
        p = KnowledgeHypothesisUpdatedPayload(hypothesis_id="hyp_1")
        assert p.old_status == "active"
        assert p.new_status == "active"
        assert p.probability == 0.5

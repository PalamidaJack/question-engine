"""Tests for scout bus topics, schemas, config, and API endpoint wiring."""

from __future__ import annotations

import pytest

from qe.bus.protocol import TOPICS
from qe.bus.schemas import (
    TOPIC_SCHEMAS,
    ScoutCycleCompletedPayload,
    ScoutCycleStartedPayload,
    ScoutFindingDiscoveredPayload,
    ScoutIdeaAnalyzedPayload,
    ScoutLearningRecordedPayload,
    ScoutProposalAppliedPayload,
    ScoutProposalCreatedPayload,
    ScoutProposalTestedPayload,
    validate_payload,
)
from qe.config import QEConfig, ScoutConfig, load_config

# ── Bus Topics ─────────────────────────────────────────────────────────────

SCOUT_TOPICS = {
    "scout.cycle_started",
    "scout.cycle_completed",
    "scout.finding_discovered",
    "scout.idea_analyzed",
    "scout.proposal_created",
    "scout.proposal_tested",
    "scout.proposal_applied",
    "scout.learning_recorded",
}


def test_all_scout_topics_registered():
    """All 8 scout topics are in TOPICS set."""
    for topic in SCOUT_TOPICS:
        assert topic in TOPICS, f"Missing topic: {topic}"


def test_total_topic_count():
    """Total topics should include the 8 new scout topics."""
    # Should include the 8 new scout topics
    assert len(TOPICS) >= 134


# ── Bus Schemas ────────────────────────────────────────────────────────────


def test_all_scout_schemas_registered():
    """All 8 scout schemas are registered in TOPIC_SCHEMAS."""
    for topic in SCOUT_TOPICS:
        assert topic in TOPIC_SCHEMAS, f"Missing schema for topic: {topic}"


def test_scout_cycle_started_payload():
    p = ScoutCycleStartedPayload(cycle_id="cyc_test")
    assert p.cycle_id == "cyc_test"


def test_scout_cycle_completed_payload():
    p = ScoutCycleCompletedPayload(
        cycle_id="cyc_test",
        findings_count=5,
        ideas_count=3,
        proposals_count=1,
        duration_s=12.5,
    )
    assert p.findings_count == 5
    assert p.proposals_count == 1
    assert pytest.approx(p.duration_s) == 12.5


def test_scout_finding_discovered_payload():
    p = ScoutFindingDiscoveredPayload(
        finding_id="fnd_test",
        url="https://example.com",
        source_type="github",
        relevance_score=0.8,
    )
    assert p.finding_id == "fnd_test"


def test_scout_idea_analyzed_payload():
    p = ScoutIdeaAnalyzedPayload(
        idea_id="idea_test",
        finding_id="fnd_test",
        category="performance",
        composite_score=0.7,
    )
    assert p.composite_score == pytest.approx(0.7)


def test_scout_proposal_created_payload():
    p = ScoutProposalCreatedPayload(
        proposal_id="prop_test",
        title="Add retry",
        category="performance",
        branch_name="scout/prop_test",
    )
    assert p.branch_name == "scout/prop_test"


def test_scout_proposal_tested_payload():
    p = ScoutProposalTestedPayload(
        proposal_id="prop_test",
        passed=True,
        total_tests=100,
        passed_tests=100,
    )
    assert p.passed is True


def test_scout_proposal_applied_payload():
    p = ScoutProposalAppliedPayload(
        proposal_id="prop_test",
        title="Add retry",
        decision="applied",
    )
    assert p.decision == "applied"


def test_scout_learning_recorded_payload():
    p = ScoutLearningRecordedPayload(
        record_id="sfb_test",
        proposal_id="prop_test",
        decision="approved",
        category="performance",
    )
    assert p.decision == "approved"


def test_validate_payload_scout():
    """validate_payload works for scout topics."""
    result = validate_payload("scout.cycle_started", {"cycle_id": "test"})
    assert result is not None
    assert result.cycle_id == "test"


# ── Config ─────────────────────────────────────────────────────────────────


def test_scout_config_defaults():
    """ScoutConfig has sensible defaults."""
    c = ScoutConfig()
    assert c.enabled is False
    assert c.poll_interval_seconds == 3600
    assert c.max_findings_per_cycle == 20
    assert c.max_proposals_per_cycle == 3
    assert c.min_composite_score == pytest.approx(0.5)
    assert c.budget_limit_per_cycle_usd == pytest.approx(1.0)
    assert c.hil_timeout_seconds == 86400
    assert c.max_pending_proposals == 10
    assert len(c.search_topics) == 5


def test_scout_config_on_qeconfig():
    """QEConfig includes scout section."""
    config = QEConfig()
    assert hasattr(config, "scout")
    assert isinstance(config.scout, ScoutConfig)


def test_scout_config_from_dict():
    """ScoutConfig loads from dict (as from TOML)."""
    data = {
        "enabled": True,
        "poll_interval_seconds": 1800,
        "min_composite_score": 0.7,
        "search_topics": ["test topic"],
    }
    c = ScoutConfig.model_validate(data)
    assert c.enabled is True
    assert c.poll_interval_seconds == 1800
    assert c.min_composite_score == pytest.approx(0.7)
    assert c.search_topics == ["test topic"]


def test_load_config_includes_scout(tmp_path):
    """load_config() includes scout defaults."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("[scout]\nenabled = true\n")
    config = load_config(config_file)
    assert config.scout.enabled is True

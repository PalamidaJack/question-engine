"""Tests for scout Pydantic models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from qe.models.scout import (
    CodeChange,
    ImprovementIdea,
    ImprovementProposal,
    ScoutFeedbackRecord,
    ScoutFinding,
    TestResult,
)


class TestScoutFinding:
    def test_construction_with_defaults(self):
        f = ScoutFinding(
            url="https://example.com",
            title="Example",
            snippet="A snippet",
            source_type="github",
        )
        assert f.finding_id.startswith("fnd_")
        assert f.url == "https://example.com"
        assert f.title == "Example"
        assert f.snippet == "A snippet"
        assert f.full_content == ""
        assert f.source_type == "github"
        assert f.relevance_score == 0.0
        assert isinstance(f.discovered_at, datetime)
        assert f.tags == []

    def test_id_prefix(self):
        f = ScoutFinding(
            url="https://x.com", title="T", snippet="S", source_type="blog"
        )
        assert f.finding_id.startswith("fnd_")
        assert len(f.finding_id) == 16  # "fnd_" + 12 hex chars

    def test_serialization_roundtrip(self):
        f = ScoutFinding(
            url="https://example.com",
            title="Title",
            snippet="Snippet",
            source_type="arxiv",
            relevance_score=0.75,
            tags=["ml", "performance"],
        )
        data = f.model_dump(mode="json")
        f2 = ScoutFinding.model_validate(data)
        assert f2.finding_id == f.finding_id
        assert f2.url == f.url
        assert f2.relevance_score == 0.75
        assert f2.tags == ["ml", "performance"]
        assert f2.source_type == "arxiv"

    def test_relevance_score_bounds(self):
        # Valid boundary values
        f_low = ScoutFinding(
            url="u", title="t", snippet="s", source_type="github", relevance_score=0.0
        )
        assert f_low.relevance_score == 0.0

        f_high = ScoutFinding(
            url="u", title="t", snippet="s", source_type="github", relevance_score=1.0
        )
        assert f_high.relevance_score == 1.0

        # Out of bounds
        with pytest.raises(ValidationError):
            ScoutFinding(
                url="u", title="t", snippet="s", source_type="github",
                relevance_score=1.5,
            )

        with pytest.raises(ValidationError):
            ScoutFinding(
                url="u", title="t", snippet="s", source_type="github",
                relevance_score=-0.1,
            )

    def test_invalid_source_type(self):
        with pytest.raises(ValidationError):
            ScoutFinding(
                url="u", title="t", snippet="s", source_type="twitter"
            )

    def test_datetime_auto_generated(self):
        before = datetime.now(UTC)
        f = ScoutFinding(
            url="u", title="t", snippet="s", source_type="reddit"
        )
        after = datetime.now(UTC)
        assert before <= f.discovered_at <= after


class TestImprovementIdea:
    def test_construction_with_defaults(self):
        idea = ImprovementIdea(
            finding_id="fnd_abc123",
            title="Add caching",
            description="LRU cache for hot path",
            category="performance",
        )
        assert idea.idea_id.startswith("idea_")
        assert idea.finding_id == "fnd_abc123"
        assert idea.relevance_score == 0.0
        assert idea.feasibility_score == 0.0
        assert idea.impact_score == 0.0
        assert idea.composite_score == 0.0
        assert idea.source_url == ""
        assert idea.rationale == ""
        assert idea.affected_files == []

    def test_id_prefix(self):
        idea = ImprovementIdea(
            finding_id="fnd_x", title="T", description="D", category="refactor"
        )
        assert idea.idea_id.startswith("idea_")
        assert len(idea.idea_id) == 17  # "idea_" + 12 hex chars

    def test_all_score_bounds(self):
        # All four scores at maximum
        idea = ImprovementIdea(
            finding_id="fnd_x",
            title="T",
            description="D",
            category="security",
            relevance_score=1.0,
            feasibility_score=1.0,
            impact_score=1.0,
            composite_score=1.0,
        )
        assert idea.relevance_score == 1.0
        assert idea.feasibility_score == 1.0
        assert idea.impact_score == 1.0
        assert idea.composite_score == 1.0

        # Each score rejects out-of-range
        for field in ("relevance_score", "feasibility_score", "impact_score", "composite_score"):
            with pytest.raises(ValidationError):
                ImprovementIdea(
                    finding_id="fnd_x", title="T", description="D",
                    category="other", **{field: 1.1},
                )
            with pytest.raises(ValidationError):
                ImprovementIdea(
                    finding_id="fnd_x", title="T", description="D",
                    category="other", **{field: -0.01},
                )

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            ImprovementIdea(
                finding_id="fnd_x",
                title="T",
                description="D",
                category="invalid_category",
            )

    def test_valid_categories(self):
        valid = [
            "performance", "feature", "refactor", "testing",
            "security", "dependency", "pattern", "model", "other",
        ]
        for cat in valid:
            idea = ImprovementIdea(
                finding_id="fnd_x", title="T", description="D", category=cat
            )
            assert idea.category == cat


class TestCodeChange:
    def test_construction(self):
        c = CodeChange(file_path="src/main.py", change_type="modify", diff="+ new line")
        assert c.file_path == "src/main.py"
        assert c.change_type == "modify"
        assert c.diff == "+ new line"

    def test_diff_defaults_empty(self):
        c = CodeChange(file_path="f.py", change_type="delete")
        assert c.diff == ""

    def test_invalid_change_type(self):
        with pytest.raises(ValidationError):
            CodeChange(file_path="f.py", change_type="rename")

    def test_valid_change_types(self):
        for ct in ("create", "modify", "delete"):
            c = CodeChange(file_path="f.py", change_type=ct)
            assert c.change_type == ct


class TestTestResult:
    def test_construction_with_defaults(self):
        r = TestResult(passed=True)
        assert r.passed is True
        assert r.total_tests == 0
        assert r.passed_tests == 0
        assert r.failed_tests == 0
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.duration_s == 0.0

    def test_full_construction(self):
        r = TestResult(
            passed=False,
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            stdout="output",
            stderr="error msg",
            duration_s=3.14,
        )
        assert r.passed is False
        assert r.total_tests == 10
        assert r.passed_tests == 8
        assert r.failed_tests == 2
        assert r.duration_s == pytest.approx(3.14)

    def test_serialization_roundtrip(self):
        r = TestResult(passed=True, total_tests=5, passed_tests=5, duration_s=1.23)
        data = r.model_dump(mode="json")
        r2 = TestResult.model_validate(data)
        assert r2.passed is True
        assert r2.total_tests == 5
        assert r2.duration_s == pytest.approx(1.23)


class TestImprovementProposal:
    @pytest.fixture()
    def sample_idea(self):
        return ImprovementIdea(
            finding_id="fnd_abc123",
            title="Cache responses",
            description="Add LRU cache",
            category="performance",
            relevance_score=0.8,
        )

    def test_construction_with_defaults(self, sample_idea):
        p = ImprovementProposal(idea=sample_idea)
        assert p.proposal_id.startswith("prop_")
        assert p.status == "draft"
        assert p.changes == []
        assert p.test_result is None
        assert p.impact_assessment == ""
        assert p.risk_assessment == ""
        assert p.rollback_plan == ""
        assert p.branch_name == ""
        assert p.worktree_path == ""
        assert p.hil_envelope_id is None
        assert p.reviewer_feedback == ""
        assert isinstance(p.created_at, datetime)
        assert p.decided_at is None
        assert p.applied_at is None

    def test_id_prefix(self, sample_idea):
        p = ImprovementProposal(idea=sample_idea)
        assert p.proposal_id.startswith("prop_")
        assert len(p.proposal_id) == 17  # "prop_" + 12 hex chars

    def test_all_valid_statuses(self, sample_idea):
        valid_statuses = [
            "draft", "testing", "test_passed", "test_failed",
            "pending_review", "approved", "rejected", "applied", "reverted",
        ]
        for status in valid_statuses:
            p = ImprovementProposal(idea=sample_idea, status=status)
            assert p.status == status

    def test_invalid_status(self, sample_idea):
        with pytest.raises(ValidationError):
            ImprovementProposal(idea=sample_idea, status="cancelled")

    def test_nested_serialization_roundtrip(self, sample_idea):
        change = CodeChange(file_path="src/cache.py", change_type="create", diff="+cache")
        test_res = TestResult(passed=True, total_tests=3, passed_tests=3)
        p = ImprovementProposal(
            idea=sample_idea,
            status="test_passed",
            changes=[change],
            test_result=test_res,
            impact_assessment="Reduces latency by 50%",
            branch_name="scout/cache-improvement",
        )
        data = p.model_dump(mode="json")
        p2 = ImprovementProposal.model_validate(data)
        assert p2.proposal_id == p.proposal_id
        assert p2.idea.title == "Cache responses"
        assert p2.idea.relevance_score == 0.8
        assert p2.status == "test_passed"
        assert len(p2.changes) == 1
        assert p2.changes[0].file_path == "src/cache.py"
        assert p2.test_result is not None
        assert p2.test_result.passed is True
        assert p2.impact_assessment == "Reduces latency by 50%"
        assert p2.branch_name == "scout/cache-improvement"

    def test_datetime_auto_generated(self, sample_idea):
        before = datetime.now(UTC)
        p = ImprovementProposal(idea=sample_idea)
        after = datetime.now(UTC)
        assert before <= p.created_at <= after


class TestScoutFeedbackRecord:
    def test_construction_with_defaults(self):
        r = ScoutFeedbackRecord(
            proposal_id="prop_abc123",
            decision="approved",
        )
        assert r.record_id.startswith("sfb_")
        assert r.proposal_id == "prop_abc123"
        assert r.decision == "approved"
        assert r.feedback == ""
        assert r.category == ""
        assert r.source_type == ""

    def test_id_prefix(self):
        r = ScoutFeedbackRecord(proposal_id="prop_x", decision="rejected")
        assert r.record_id.startswith("sfb_")
        assert len(r.record_id) == 16  # "sfb_" + 12 hex chars

    def test_invalid_decision(self):
        with pytest.raises(ValidationError):
            ScoutFeedbackRecord(
                proposal_id="prop_x",
                decision="deferred",
            )

    def test_full_construction(self):
        r = ScoutFeedbackRecord(
            proposal_id="prop_abc123",
            decision="rejected",
            feedback="Too risky for production",
            category="security",
            source_type="github",
        )
        assert r.decision == "rejected"
        assert r.feedback == "Too risky for production"
        assert r.category == "security"
        assert r.source_type == "github"

    def test_serialization_roundtrip(self):
        r = ScoutFeedbackRecord(
            proposal_id="prop_abc123",
            decision="approved",
            feedback="LGTM",
        )
        data = r.model_dump(mode="json")
        r2 = ScoutFeedbackRecord.model_validate(data)
        assert r2.record_id == r.record_id
        assert r2.proposal_id == "prop_abc123"
        assert r2.decision == "approved"
        assert r2.feedback == "LGTM"

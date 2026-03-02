"""End-to-end integration tests for the Innovation Scout."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from qe.models.scout import (
    CodeChange,
    ImprovementIdea,
    ImprovementProposal,
    ScoutFeedbackRecord,
    ScoutFinding,
    TestResult,
)
from qe.substrate.scout_store import ScoutStore

# ── Store Integration Tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_finding_roundtrip(tmp_path):
    """Save and retrieve a finding from SQLite."""
    store = ScoutStore(db_path=str(tmp_path / "test.db"))
    await store.initialize()

    finding = ScoutFinding(
        url="https://github.com/user/repo",
        title="Cool Library",
        snippet="A useful lib",
        source_type="github",
        relevance_score=0.7,
        tags=["python", "async"],
    )
    await store.save_finding(finding)

    # Retrieve
    result = await store.get_finding(finding.finding_id)
    assert result is not None
    assert result.url == "https://github.com/user/repo"
    assert result.title == "Cool Library"
    assert result.relevance_score == pytest.approx(0.7)
    assert result.tags == ["python", "async"]


@pytest.mark.asyncio
async def test_store_finding_url_dedup(tmp_path):
    """has_finding_url returns True for saved URLs."""
    store = ScoutStore(db_path=str(tmp_path / "test.db"))
    await store.initialize()

    assert not await store.has_finding_url("https://example.com")

    finding = ScoutFinding(
        url="https://example.com",
        title="Test",
        snippet="snip",
        source_type="blog",
    )
    await store.save_finding(finding)

    assert await store.has_finding_url("https://example.com")


@pytest.mark.asyncio
async def test_store_proposal_roundtrip(tmp_path):
    """Save and retrieve a full proposal with nested objects."""
    store = ScoutStore(db_path=str(tmp_path / "test.db"))
    await store.initialize()

    idea = ImprovementIdea(
        finding_id="fnd_test",
        title="Add retry backoff",
        description="Exponential backoff for web_fetch",
        category="performance",
        relevance_score=0.8,
        feasibility_score=0.7,
        impact_score=0.6,
        composite_score=0.71,
        source_url="https://example.com",
        rationale="Reduces failures",
        affected_files=["src/qe/tools/web_fetch.py"],
    )

    test_result = TestResult(
        passed=True,
        total_tests=1950,
        passed_tests=1950,
        failed_tests=0,
        stdout="1950 passed in 23.4s",
        duration_s=23.4,
    )

    proposal = ImprovementProposal(
        idea=idea,
        status="pending_review",
        changes=[
            CodeChange(
                file_path="src/qe/tools/web_fetch.py",
                change_type="modify",
                diff="@@ -1 +1 @@\n-old\n+new",
            ),
        ],
        test_result=test_result,
        impact_assessment="Reduces transient failures by ~40%",
        risk_assessment="Low — only affects retry timing",
        rollback_plan="Revert the commit",
        branch_name="scout/prop_test_add-retry",
        worktree_path="data/scout_worktrees/prop_test",
        hil_envelope_id="env_123",
    )

    await store.save_proposal(proposal)

    # Retrieve
    result = await store.get_proposal(proposal.proposal_id)
    assert result is not None
    assert result.status == "pending_review"
    assert result.idea.title == "Add retry backoff"
    assert result.idea.category == "performance"
    assert result.test_result is not None
    assert result.test_result.passed is True
    assert result.test_result.total_tests == 1950
    assert len(result.changes) == 1
    assert result.changes[0].diff.startswith("@@")
    assert result.branch_name == "scout/prop_test_add-retry"


@pytest.mark.asyncio
async def test_store_list_proposals_by_status(tmp_path):
    """List proposals with status filter."""
    store = ScoutStore(db_path=str(tmp_path / "test.db"))
    await store.initialize()

    idea = ImprovementIdea(
        finding_id="fnd_1",
        title="T",
        description="D",
        category="other",
        composite_score=0.5,
    )

    for i, status in enumerate(["pending_review", "pending_review", "applied", "rejected"]):
        p = ImprovementProposal(
            proposal_id=f"prop_{i}",
            idea=idea,
            status=status,
        )
        await store.save_proposal(p)

    all_props = await store.list_proposals()
    assert len(all_props) == 4

    pending = await store.list_proposals(status="pending_review")
    assert len(pending) == 2

    applied = await store.list_proposals(status="applied")
    assert len(applied) == 1


@pytest.mark.asyncio
async def test_store_update_proposal_status(tmp_path):
    """Update proposal status and feedback."""
    store = ScoutStore(db_path=str(tmp_path / "test.db"))
    await store.initialize()

    idea = ImprovementIdea(
        finding_id="fnd_1", title="T", description="D", category="other", composite_score=0.5,
    )
    proposal = ImprovementProposal(idea=idea, status="pending_review")
    await store.save_proposal(proposal)

    now = datetime.now(UTC)
    await store.update_proposal_status(
        proposal.proposal_id, "rejected",
        reviewer_feedback="Not useful",
        decided_at=now,
    )

    result = await store.get_proposal(proposal.proposal_id)
    assert result.status == "rejected"
    assert result.reviewer_feedback == "Not useful"
    assert result.decided_at is not None


@pytest.mark.asyncio
async def test_store_feedback_stats(tmp_path):
    """Feedback stats track approval rate by category."""
    store = ScoutStore(db_path=str(tmp_path / "test.db"))
    await store.initialize()

    # Add some feedback
    for i in range(3):
        await store.save_feedback(ScoutFeedbackRecord(
            proposal_id=f"prop_{i}",
            decision="approved",
            category="performance",
            source_type="github",
        ))
    for i in range(2):
        await store.save_feedback(ScoutFeedbackRecord(
            proposal_id=f"prop_r{i}",
            decision="rejected",
            category="testing",
            source_type="reddit",
        ))

    stats = await store.get_feedback_stats()
    assert stats["total"] == 5
    assert stats["approved"] == 3
    assert stats["rejected"] == 2
    assert stats["approval_rate"] == pytest.approx(0.6)
    assert "performance" in stats["by_category"]
    assert stats["by_category"]["performance"]["approved"] == 3
    assert "testing" in stats["by_category"]
    assert stats["by_category"]["testing"]["rejected"] == 2


@pytest.mark.asyncio
async def test_store_rejected_patterns(tmp_path):
    """Rejected patterns detected from feedback."""
    store = ScoutStore(db_path=str(tmp_path / "test.db"))
    await store.initialize()

    # 4 rejections in "testing" category — should be flagged
    for i in range(4):
        await store.save_feedback(ScoutFeedbackRecord(
            proposal_id=f"prop_{i}",
            decision="rejected",
            category="testing",
            source_type="reddit",
        ))
    # 1 approval in "testing" — still >70% rejected
    await store.save_feedback(ScoutFeedbackRecord(
        proposal_id="prop_ok",
        decision="approved",
        category="testing",
        source_type="github",
    ))

    patterns = await store.get_rejected_patterns()
    assert "testing" in patterns["rejected_categories"]
    # reddit has 4 rejections out of 4 (100%)
    assert "reddit" in patterns["rejected_sources"]


@pytest.mark.asyncio
async def test_store_count_proposals(tmp_path):
    """Count proposals by status."""
    store = ScoutStore(db_path=str(tmp_path / "test.db"))
    await store.initialize()

    idea = ImprovementIdea(
        finding_id="fnd_1", title="T", description="D", category="other", composite_score=0.5,
    )
    for i in range(3):
        p = ImprovementProposal(
            proposal_id=f"prop_{i}", idea=idea, status="pending_review",
        )
        await store.save_proposal(p)

    assert await store.count_proposals() == 3
    assert await store.count_proposals(status="pending_review") == 3
    assert await store.count_proposals(status="applied") == 0

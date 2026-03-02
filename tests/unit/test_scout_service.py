"""Tests for InnovationScoutService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.config import ScoutConfig
from qe.models.envelope import Envelope
from qe.models.scout import ImprovementIdea, ImprovementProposal
from qe.services.scout.service import InnovationScoutService


def _make_service(**overrides):
    defaults = {
        "bus": MagicMock(),
        "scout_store": AsyncMock(),
        "config": ScoutConfig(poll_interval_seconds=1),
    }
    defaults.update(overrides)
    svc = InnovationScoutService(**defaults)
    # Replace pipeline with mock to avoid real LLM calls
    svc._pipeline = AsyncMock()
    svc._pipeline.run_cycle = AsyncMock(return_value={
        "cycle_id": "cyc_test",
        "findings_count": 2,
        "ideas_count": 1,
        "proposals_count": 1,
    })
    return svc


def _make_proposal(**overrides):
    idea = ImprovementIdea(
        finding_id="fnd_test",
        title="Test Improvement",
        description="Desc",
        category="performance",
        relevance_score=0.8,
        feasibility_score=0.7,
        impact_score=0.6,
        composite_score=0.7,
        source_url="https://example.com",
        rationale="Good",
    )
    defaults = {
        "proposal_id": "prop_test1",
        "idea": idea,
        "status": "pending_review",
        "branch_name": "scout/prop_test1_test",
        "worktree_path": "/tmp/wt",
        "hil_envelope_id": "env_123",
    }
    defaults.update(overrides)
    return ImprovementProposal(**defaults)


@pytest.mark.asyncio
async def test_service_start_stop():
    """Service starts and stops cleanly."""
    svc = _make_service()
    await svc.start()
    assert svc._running is True
    assert svc._poll_task is not None

    await svc.stop()
    assert svc._running is False


@pytest.mark.asyncio
async def test_service_status():
    """Status returns expected fields."""
    svc = _make_service()
    status = svc.status()
    assert "running" in status
    assert "cycles_completed" in status
    assert "poll_interval_seconds" in status
    assert status["running"] is False


@pytest.mark.asyncio
@patch("qe.services.scout.service.get_flag_store")
async def test_run_cycle_when_enabled(mock_flags):
    """Cycle runs when innovation_scout flag is enabled."""
    mock_flags.return_value.is_enabled.return_value = True
    store = AsyncMock()
    store.count_proposals = AsyncMock(return_value=0)
    store.get_rejected_patterns = AsyncMock(return_value={
        "rejected_categories": [],
        "rejected_sources": [],
    })
    store.get_feedback_stats = AsyncMock(return_value={"total": 0})

    svc = _make_service(scout_store=store)
    await svc._run_cycle()

    svc._pipeline.run_cycle.assert_called_once()
    assert svc._cycles_completed == 1


@pytest.mark.asyncio
@patch("qe.services.scout.service.get_flag_store")
async def test_run_cycle_skipped_when_disabled(mock_flags):
    """Cycle is skipped when flag is disabled."""
    mock_flags.return_value.is_enabled.return_value = False

    svc = _make_service()
    await svc._run_cycle()

    svc._pipeline.run_cycle.assert_not_called()
    assert svc._cycles_completed == 0


@pytest.mark.asyncio
@patch("qe.services.scout.service.get_flag_store")
async def test_run_cycle_skipped_too_many_pending(mock_flags):
    """Cycle is skipped when too many proposals are pending."""
    mock_flags.return_value.is_enabled.return_value = True
    store = AsyncMock()
    store.count_proposals = AsyncMock(return_value=100)  # Over limit

    svc = _make_service(scout_store=store, config=ScoutConfig(max_pending_proposals=5))
    await svc._run_cycle()

    svc._pipeline.run_cycle.assert_not_called()


@pytest.mark.asyncio
async def test_on_hil_approved_merges_branch():
    """Approved proposal triggers merge and cleanup."""
    proposal = _make_proposal()
    store = AsyncMock()
    store.get_proposal = AsyncMock(return_value=proposal)
    store.update_proposal_status = AsyncMock()
    store.save_feedback = AsyncMock()

    svc = _make_service(scout_store=store)
    svc._sandbox.merge_branch = AsyncMock(return_value=True)
    svc._sandbox.cleanup_worktree = AsyncMock()

    envelope = Envelope(
        topic="hil.approved",
        source_service_id="hil",
        correlation_id="prop_test1",
        payload={"decision": "approved"},
    )

    await svc._on_hil_approved(envelope)

    svc._sandbox.merge_branch.assert_called_once()
    svc._sandbox.cleanup_worktree.assert_called_once()
    store.update_proposal_status.assert_called_once()
    store.save_feedback.assert_called_once()

    # Verify feedback decision
    feedback = store.save_feedback.call_args[0][0]
    assert feedback.decision == "approved"


@pytest.mark.asyncio
async def test_on_hil_rejected_cleans_up():
    """Rejected proposal cleans up branch and records feedback."""
    proposal = _make_proposal()
    store = AsyncMock()
    store.get_proposal = AsyncMock(return_value=proposal)
    store.update_proposal_status = AsyncMock()
    store.save_feedback = AsyncMock()

    svc = _make_service(scout_store=store)
    svc._sandbox.cleanup_worktree = AsyncMock()

    envelope = Envelope(
        topic="hil.rejected",
        source_service_id="hil",
        correlation_id="prop_test1",
        payload={"decision": "rejected", "reason": "not useful"},
    )

    await svc._on_hil_rejected(envelope)

    svc._sandbox.cleanup_worktree.assert_called_once()
    store.update_proposal_status.assert_called_once()
    status_call = store.update_proposal_status.call_args
    assert status_call[0][1] == "rejected"
    assert status_call[1]["reviewer_feedback"] == "not useful"


@pytest.mark.asyncio
async def test_on_hil_approved_ignores_non_pending():
    """Approved event is ignored if proposal is not pending."""
    proposal = _make_proposal(status="applied")
    store = AsyncMock()
    store.get_proposal = AsyncMock(return_value=proposal)

    svc = _make_service(scout_store=store)
    svc._sandbox.merge_branch = AsyncMock()

    envelope = Envelope(
        topic="hil.approved",
        source_service_id="hil",
        correlation_id="prop_test1",
        payload={"decision": "approved"},
    )

    await svc._on_hil_approved(envelope)
    svc._sandbox.merge_branch.assert_not_called()


@pytest.mark.asyncio
async def test_on_hil_approved_no_correlation_id():
    """Approved event without correlation_id is ignored."""
    svc = _make_service()
    envelope = Envelope(
        topic="hil.approved",
        source_service_id="hil",
        payload={"decision": "approved"},
    )
    await svc._on_hil_approved(envelope)
    # No crash, no action


@pytest.mark.asyncio
@patch("qe.services.scout.service.get_flag_store")
async def test_apply_learning_adjusts_threshold(mock_flags):
    """Dynamic threshold adjustment based on feedback."""
    mock_flags.return_value.is_enabled.return_value = True
    store = AsyncMock()
    store.count_proposals = AsyncMock(return_value=0)
    store.get_rejected_patterns = AsyncMock(return_value={
        "rejected_categories": ["testing"],
        "rejected_sources": ["reddit"],
    })
    store.get_feedback_stats = AsyncMock(return_value={
        "total": 20,
        "approval_rate": 0.1,  # Very low → raise threshold
    })

    config = ScoutConfig(min_composite_score=0.5)
    svc = _make_service(scout_store=store, config=config)
    await svc._apply_learning()

    # Threshold should have been raised
    assert svc._analyzer._min_composite_score > 0.5


@pytest.mark.asyncio
@patch("qe.services.scout.service.get_flag_store")
async def test_apply_learning_lowers_threshold_on_high_approval(mock_flags):
    """High approval rate lowers the threshold."""
    mock_flags.return_value.is_enabled.return_value = True
    store = AsyncMock()
    store.count_proposals = AsyncMock(return_value=0)
    store.get_rejected_patterns = AsyncMock(return_value={
        "rejected_categories": [],
        "rejected_sources": [],
    })
    store.get_feedback_stats = AsyncMock(return_value={
        "total": 20,
        "approval_rate": 0.8,  # High → lower threshold
    })

    config = ScoutConfig(min_composite_score=0.5)
    svc = _make_service(scout_store=store, config=config)
    await svc._apply_learning()

    assert svc._analyzer._min_composite_score < 0.5


@pytest.mark.asyncio
async def test_on_hil_approved_publishes_events():
    """Approved proposal publishes scout.proposal_applied and scout.learning_recorded."""
    proposal = _make_proposal()
    store = AsyncMock()
    store.get_proposal = AsyncMock(return_value=proposal)
    store.update_proposal_status = AsyncMock()
    store.save_feedback = AsyncMock()
    bus = MagicMock()

    svc = _make_service(scout_store=store, bus=bus)
    svc._sandbox.merge_branch = AsyncMock(return_value=True)
    svc._sandbox.cleanup_worktree = AsyncMock()

    envelope = Envelope(
        topic="hil.approved",
        source_service_id="hil",
        correlation_id="prop_test1",
        payload={"decision": "approved"},
    )

    await svc._on_hil_approved(envelope)

    topics = [c.args[0].topic for c in bus.publish.call_args_list]
    assert "scout.proposal_applied" in topics
    assert "scout.learning_recorded" in topics

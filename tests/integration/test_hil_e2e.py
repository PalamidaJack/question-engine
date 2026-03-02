"""End-to-end integration tests for the HIL (Human-in-the-Loop) service.

Uses real HILService with mocked bus and tmp_path directories to verify
the full proposal → decision → publish lifecycle.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qe.models.envelope import Envelope
from qe.services.hil.service import HILService

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_blueprint() -> MagicMock:
    bp = MagicMock()
    bp.service_id = "hil_test"
    bp.capabilities.bus_topics_subscribe = ["hil.approval_required"]
    return bp


def _make_bus() -> MagicMock:
    bus = MagicMock()
    bus.subscribe = MagicMock()
    bus.unsubscribe = MagicMock()
    bus.publish = MagicMock()
    return bus


def _make_hil(tmp_path: Path) -> tuple[HILService, MagicMock]:
    bp = _make_blueprint()
    bus = _make_bus()
    hil = HILService(blueprint=bp, bus=bus, substrate=None)
    hil.pending_dir = tmp_path / "pending"
    hil.completed_dir = tmp_path / "completed"
    hil.pending_dir.mkdir(parents=True, exist_ok=True)
    hil.completed_dir.mkdir(parents=True, exist_ok=True)
    hil._running = True
    return hil, bus


def _hil_envelope(
    envelope_id: str = "env_001",
    timeout_seconds: int = 3600,
    reason: str = "approval_required",
    summary: str = "Test proposal",
) -> Envelope:
    return Envelope(
        envelope_id=envelope_id,
        topic="hil.approval_required",
        source_service_id="test_service",
        payload={
            "reason": reason,
            "proposal_summary": summary,
            "timeout_seconds": timeout_seconds,
        },
    )


# ── Tests ────────────────────────────────────────────────────────────────


class TestHILProposal:
    @pytest.mark.asyncio
    async def test_proposal_creates_pending_file(self, tmp_path: Path):
        hil, bus = _make_hil(tmp_path)
        env = _hil_envelope(envelope_id="env_pending")

        # Start the request handler but cancel the poll task immediately
        await hil._handle_hil_request(env)
        # Cancel background poll so test finishes
        for task in hil._poll_tasks.values():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        pending_file = hil.pending_dir / "env_pending.json"
        assert pending_file.exists()
        data = json.loads(pending_file.read_text(encoding="utf-8"))
        assert data["envelope_id"] == "env_pending"
        assert data["proposal_summary"] == "Test proposal"
        assert data["reason"] == "approval_required"

    @pytest.mark.asyncio
    async def test_proposal_contains_expiry(self, tmp_path: Path):
        hil, bus = _make_hil(tmp_path)
        env = _hil_envelope(envelope_id="env_expiry", timeout_seconds=60)

        await hil._handle_hil_request(env)
        for task in hil._poll_tasks.values():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        data = json.loads(
            (hil.pending_dir / "env_expiry.json").read_text(encoding="utf-8")
        )
        assert data["timeout_seconds"] == 60
        assert "expires_at" in data


class TestHILApproval:
    @pytest.mark.asyncio
    async def test_approved_publishes_hil_approved(self, tmp_path: Path):
        hil, bus = _make_hil(tmp_path)
        env = _hil_envelope(envelope_id="env_approve")

        # Pre-create the decision file so the poll picks it up immediately
        decision = {
            "decision": "approved",
            "decided_at": datetime.now(UTC).isoformat(),
        }
        completed_file = hil.completed_dir / "env_approve.json"
        completed_file.write_text(json.dumps(decision), encoding="utf-8")

        await hil._handle_hil_request(env)

        # Wait for poll task to complete
        for task in hil._poll_tasks.values():
            await asyncio.wait_for(task, timeout=5)

        # Verify hil.approved was published
        published = [
            c.args[0]
            for c in bus.publish.call_args_list
            if c.args[0].topic == "hil.approved"
        ]
        assert len(published) == 1
        assert published[0].payload["decision"] == "approved"
        assert published[0].correlation_id == "env_approve"

    @pytest.mark.asyncio
    async def test_rejected_publishes_hil_rejected(self, tmp_path: Path):
        hil, bus = _make_hil(tmp_path)
        env = _hil_envelope(envelope_id="env_reject")

        decision = {
            "decision": "rejected",
            "reason": "not_needed",
            "decided_at": datetime.now(UTC).isoformat(),
        }
        completed_file = hil.completed_dir / "env_reject.json"
        completed_file.write_text(json.dumps(decision), encoding="utf-8")

        await hil._handle_hil_request(env)
        for task in hil._poll_tasks.values():
            await asyncio.wait_for(task, timeout=5)

        published = [
            c.args[0]
            for c in bus.publish.call_args_list
            if c.args[0].topic == "hil.rejected"
        ]
        assert len(published) == 1
        assert published[0].payload["decision"] == "rejected"
        assert published[0].payload["reason"] == "not_needed"


class TestHILTimeout:
    @pytest.mark.asyncio
    async def test_timeout_auto_rejects(self, tmp_path: Path):
        hil, bus = _make_hil(tmp_path)
        # Use timeout_seconds=0 to force immediate expiry
        env = _hil_envelope(envelope_id="env_timeout", timeout_seconds=0)

        await hil._handle_hil_request(env)
        for task in hil._poll_tasks.values():
            await asyncio.wait_for(task, timeout=5)

        published = [
            c.args[0]
            for c in bus.publish.call_args_list
            if c.args[0].topic == "hil.rejected"
        ]
        assert len(published) == 1
        assert published[0].payload["reason"] == "timeout"


class TestHILCleanup:
    @pytest.mark.asyncio
    async def test_pending_file_cleaned_up_on_approval(self, tmp_path: Path):
        hil, bus = _make_hil(tmp_path)
        env = _hil_envelope(envelope_id="env_cleanup")

        decision = {
            "decision": "approved",
            "decided_at": datetime.now(UTC).isoformat(),
        }
        completed_file = hil.completed_dir / "env_cleanup.json"
        completed_file.write_text(json.dumps(decision), encoding="utf-8")

        await hil._handle_hil_request(env)
        for task in hil._poll_tasks.values():
            await asyncio.wait_for(task, timeout=5)

        pending_file = hil.pending_dir / "env_cleanup.json"
        assert not pending_file.exists()

    @pytest.mark.asyncio
    async def test_pending_file_cleaned_up_on_timeout(self, tmp_path: Path):
        hil, bus = _make_hil(tmp_path)
        env = _hil_envelope(envelope_id="env_timeout_clean", timeout_seconds=0)

        await hil._handle_hil_request(env)
        for task in hil._poll_tasks.values():
            await asyncio.wait_for(task, timeout=5)

        pending_file = hil.pending_dir / "env_timeout_clean.json"
        assert not pending_file.exists()


class TestHILConcurrent:
    @pytest.mark.asyncio
    async def test_multiple_concurrent_proposals(self, tmp_path: Path):
        hil, bus = _make_hil(tmp_path)

        # Create 3 proposals with different decisions
        decisions_map = {
            "env_c1": {"decision": "approved", "decided_at": datetime.now(UTC).isoformat()},
            "env_c2": {
                "decision": "rejected", "reason": "bad",
                "decided_at": datetime.now(UTC).isoformat(),
            },
            "env_c3": {"decision": "approved", "decided_at": datetime.now(UTC).isoformat()},
        }

        # Pre-create decision files
        for eid, dec in decisions_map.items():
            (hil.completed_dir / f"{eid}.json").write_text(
                json.dumps(dec), encoding="utf-8"
            )

        # Fire all requests
        for eid in decisions_map:
            env = _hil_envelope(envelope_id=eid)
            await hil._handle_hil_request(env)

        # Wait for all polls
        for task in hil._poll_tasks.values():
            await asyncio.wait_for(task, timeout=5)

        approved = [
            c.args[0]
            for c in bus.publish.call_args_list
            if c.args[0].topic == "hil.approved"
        ]
        rejected = [
            c.args[0]
            for c in bus.publish.call_args_list
            if c.args[0].topic == "hil.rejected"
        ]
        assert len(approved) == 2
        assert len(rejected) == 1

        # All pending files should be cleaned up
        remaining = list(hil.pending_dir.iterdir())
        assert len(remaining) == 0

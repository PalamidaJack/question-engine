"""Tests for Phase 4: VerificationGate, CheckpointManager, and Recovery Execution."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.envelope import Envelope
from qe.models.goal import GoalState, SubtaskResult
from qe.services.checkpoint.manager import CheckpointManager
from qe.services.recovery.service import RecoveryOrchestrator
from qe.services.verification.gate import VerificationGate
from qe.services.verification.service import CheckResult, VerificationService
from qe.substrate.failure_kb import FailureKnowledgeBase
from qe.substrate.goal_store import GoalStore

# ── Helpers ───────────────────────────────────────────────────────────


def _make_dispatch_envelope(
    goal_id: str = "goal_1",
    subtask_id: str = "sub_1",
    task_type: str = "research",
    description: str = "Investigate topic",
    model_tier: str = "balanced",
    contract: dict | None = None,
) -> Envelope:
    return Envelope(
        topic="tasks.dispatched",
        source_service_id="dispatcher",
        correlation_id=goal_id,
        payload={
            "goal_id": goal_id,
            "subtask_id": subtask_id,
            "task_type": task_type,
            "description": description,
            "model_tier": model_tier,
            "contract": contract,
            "dependency_context": None,
            "assigned_agent_id": "executor_default",
        },
    )


def _make_completed_envelope(
    goal_id: str = "goal_1",
    subtask_id: str = "sub_1",
    output: dict | None = None,
) -> Envelope:
    result = SubtaskResult(
        subtask_id=subtask_id,
        goal_id=goal_id,
        status="completed",
        output={"result": "some valid content", "confidence": 0.9} if output is None else output,
        model_used="test-model",
        latency_ms=100,
    )
    return Envelope(
        topic="tasks.completed",
        source_service_id="executor",
        correlation_id=goal_id,
        payload=result.model_dump(mode="json"),
    )


def _make_failed_envelope(
    goal_id: str = "goal_1",
    subtask_id: str = "sub_1",
    error: str = "Connection timeout",
) -> Envelope:
    result = SubtaskResult(
        subtask_id=subtask_id,
        goal_id=goal_id,
        status="failed",
        output={"error": error},
        model_used="test-model",
        latency_ms=50,
    )
    return Envelope(
        topic="tasks.failed",
        source_service_id="executor",
        correlation_id=goal_id,
        payload=result.model_dump(mode="json"),
    )


class _CollectingBus:
    """Minimal test bus that records published envelopes."""

    def __init__(self) -> None:
        self.published: list[Envelope] = []
        self._handlers: dict[str, list] = {}

    def subscribe(self, topic: str, handler) -> None:
        self._handlers.setdefault(topic, []).append(handler)

    def unsubscribe(self, topic: str, handler) -> None:
        if topic in self._handlers:
            self._handlers[topic] = [
                h for h in self._handlers[topic] if h is not handler
            ]

    def publish(self, envelope: Envelope) -> list:
        self.published.append(envelope)
        return []

    def published_topics(self) -> list[str]:
        return [e.topic for e in self.published]

    def published_by_topic(self, topic: str) -> list[Envelope]:
        return [e for e in self.published if e.topic == topic]


# ── TestVerificationGate ──────────────────────────────────────────────


class TestVerificationGate:
    def setup_method(self):
        self.bus = _CollectingBus()
        self.failure_kb = AsyncMock(spec=FailureKnowledgeBase)
        self.failure_kb.record = AsyncMock(return_value="fail_123")
        self.failure_kb.lookup = AsyncMock(return_value=[])
        self.verification_svc = VerificationService()
        self.recovery = RecoveryOrchestrator(
            failure_kb=self.failure_kb, bus=self.bus
        )
        self.gate = VerificationGate(
            bus=self.bus,
            verification_service=self.verification_svc,
            recovery_orchestrator=self.recovery,
            failure_kb=self.failure_kb,
            max_recovery_attempts=3,
        )

    async def _setup_dispatch_and_complete(
        self, output: dict | None = None, contract: dict | None = None
    ) -> None:
        """Cache a dispatch context then process a completed task."""
        dispatch = _make_dispatch_envelope(contract=contract)
        await self.gate._cache_dispatch_context(dispatch)
        completed = _make_completed_envelope(output=output)
        await self.gate._on_task_completed(completed)

    @pytest.mark.asyncio
    async def test_pass_publishes_verified(self):
        await self._setup_dispatch_and_complete(
            output={"result": "valid content", "confidence": 0.9}
        )
        assert "tasks.verified" in self.bus.published_topics()

    @pytest.mark.asyncio
    async def test_fail_publishes_verification_failed(self):
        await self._setup_dispatch_and_complete(output={})
        assert "tasks.verification_failed" in self.bus.published_topics()

    @pytest.mark.asyncio
    async def test_warn_still_publishes_verified(self):
        # WARN outputs are still forwarded as verified
        svc = AsyncMock(spec=VerificationService)
        from qe.services.verification.service import VerificationReport

        warn_report = VerificationReport(
            subtask_id="sub_1",
            goal_id="goal_1",
            overall=CheckResult.WARN,
            structural=CheckResult.PASS,
            anomaly=CheckResult.WARN,
        )
        svc.verify = AsyncMock(return_value=warn_report)
        gate = VerificationGate(
            self.bus, svc, self.recovery, self.failure_kb
        )
        dispatch = _make_dispatch_envelope()
        await gate._cache_dispatch_context(dispatch)
        completed = _make_completed_envelope()
        await gate._on_task_completed(completed)

        assert "tasks.verified" in self.bus.published_topics()

    @pytest.mark.asyncio
    async def test_contract_pass(self):
        contract = {"postconditions": ["score >= 0.5"]}
        output = {"result": "ok", "score": 0.8}
        await self._setup_dispatch_and_complete(
            output=output, contract=contract
        )
        assert "tasks.verified" in self.bus.published_topics()

    @pytest.mark.asyncio
    async def test_contract_fail_triggers_recovery(self):
        contract = {"postconditions": ["score >= 0.9"]}
        output = {"result": "ok", "score": 0.3}
        await self._setup_dispatch_and_complete(
            output=output, contract=contract
        )
        assert "tasks.verification_failed" in self.bus.published_topics()

    @pytest.mark.asyncio
    async def test_failed_task_triggers_recovery(self):
        dispatch = _make_dispatch_envelope()
        await self.gate._cache_dispatch_context(dispatch)
        failed = _make_failed_envelope(error="Connection timeout")
        await self.gate._on_task_failed(failed)

        # Recovery should produce a redispatch (transient -> retry_with_backoff)
        topics = self.bus.published_topics()
        assert "tasks.dispatched" in topics or "hil.approval_required" in topics

    @pytest.mark.asyncio
    async def test_retry_with_backoff_redispatches(self):
        dispatch = _make_dispatch_envelope()
        await self.gate._cache_dispatch_context(dispatch)
        failed = _make_failed_envelope(error="Connection timeout")

        with patch("qe.services.recovery.service.asyncio.sleep", new_callable=AsyncMock):
            await self.gate._on_task_failed(failed)

        dispatched = self.bus.published_by_topic("tasks.dispatched")
        assert len(dispatched) >= 1
        assert dispatched[0].payload["subtask_id"] == "sub_1"

    @pytest.mark.asyncio
    async def test_model_escalation_changes_tier(self):
        dispatch = _make_dispatch_envelope(model_tier="fast")
        await self.gate._cache_dispatch_context(dispatch)
        failed = _make_failed_envelope(error="Invalid JSON: parse error")

        await self.gate._on_task_failed(failed)

        dispatched = self.bus.published_by_topic("tasks.dispatched")
        assert len(dispatched) >= 1
        assert dispatched[0].payload["model_tier"] == "balanced"

    @pytest.mark.asyncio
    async def test_hil_escalation(self):
        dispatch = _make_dispatch_envelope(model_tier="powerful")
        await self.gate._cache_dispatch_context(dispatch)
        failed = _make_failed_envelope(
            error="Invalid JSON: parse error at line 5"
        )

        await self.gate._on_task_failed(failed)

        hil = self.bus.published_by_topic("hil.approval_required")
        assert len(hil) >= 1
        assert hil[0].payload["goal_id"] == "goal_1"

    @pytest.mark.asyncio
    async def test_retry_count_increments(self):
        dispatch = _make_dispatch_envelope()
        await self.gate._cache_dispatch_context(dispatch)

        key = ("goal_1", "sub_1")
        assert self.gate._retry_counts.get(key) is None

        with patch("qe.services.recovery.service.asyncio.sleep", new_callable=AsyncMock):
            failed = _make_failed_envelope(error="Connection timeout")
            await self.gate._on_task_failed(failed)
        assert self.gate._retry_counts[key] == 1

        with patch("qe.services.recovery.service.asyncio.sleep", new_callable=AsyncMock):
            await self.gate._on_task_failed(failed)
        assert self.gate._retry_counts[key] == 2

    @pytest.mark.asyncio
    async def test_max_retries_triggers_hil(self):
        dispatch = _make_dispatch_envelope()
        await self.gate._cache_dispatch_context(dispatch)

        self.gate._retry_counts[("goal_1", "sub_1")] = 3  # already at max

        with patch("qe.services.recovery.service.asyncio.sleep", new_callable=AsyncMock):
            failed = _make_failed_envelope(error="Connection timeout")
            await self.gate._on_task_failed(failed)

        hil = self.bus.published_by_topic("hil.approval_required")
        assert len(hil) >= 1
        self.failure_kb.record.assert_called()

    @pytest.mark.asyncio
    async def test_failure_kb_populated(self):
        dispatch = _make_dispatch_envelope()
        await self.gate._cache_dispatch_context(dispatch)

        with patch("qe.services.recovery.service.asyncio.sleep", new_callable=AsyncMock):
            failed = _make_failed_envelope(error="Connection timeout")
            await self.gate._on_task_failed(failed)

        self.failure_kb.record.assert_called_once()
        call_kwargs = self.failure_kb.record.call_args
        assert call_kwargs[1]["goal_id"] == "goal_1" or call_kwargs[0] is not None

    @pytest.mark.asyncio
    async def test_dispatch_context_cached(self):
        dispatch = _make_dispatch_envelope(
            goal_id="g99", subtask_id="s42"
        )
        await self.gate._cache_dispatch_context(dispatch)

        key = ("g99", "s42")
        assert key in self.gate._dispatch_contexts
        assert self.gate._dispatch_contexts[key]["task_type"] == "research"

    @pytest.mark.asyncio
    async def test_verification_result_attached(self):
        dispatch = _make_dispatch_envelope()
        await self.gate._cache_dispatch_context(dispatch)

        completed = _make_completed_envelope(
            output={"result": "valid", "confidence": 0.9}
        )
        await self.gate._on_task_completed(completed)

        verified = self.bus.published_by_topic("tasks.verified")
        assert len(verified) == 1
        payload = verified[0].payload
        assert payload["verification_result"] is not None
        assert payload["verification_result"]["overall"] == "pass"


# ── TestCheckpointManager ────────────────────────────────────────────


class TestCheckpointManager:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmp) / "test.db")

    async def _init_db(self) -> GoalStore:
        """Create tables and return a GoalStore."""
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    description TEXT,
                    status TEXT,
                    decomposition TEXT,
                    subtask_states TEXT,
                    subtask_results TEXT,
                    created_at TEXT,
                    completed_at TEXT,
                    project_id TEXT,
                    started_at TEXT,
                    due_at TEXT,
                    tags TEXT DEFAULT '[]',
                    metadata JSON DEFAULT '{}'
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    goal_id TEXT,
                    subtask_states TEXT,
                    subtask_results TEXT,
                    created_at TEXT
                )
            """)
            await db.commit()
        return GoalStore(self.db_path)

    @pytest.mark.asyncio
    async def test_find_rollback_point(self):
        store = await self._init_db()
        state = GoalState(
            goal_id="goal_1",
            description="test",
            status="executing",
            subtask_states={"sub_1": "pending", "sub_2": "pending"},
        )
        await store.save_goal(state)
        ckpt1 = await store.save_checkpoint("goal_1", state)
        state.checkpoints.append(ckpt1)

        state.subtask_states["sub_1"] = "completed"
        await store.save_goal(state)
        ckpt2 = await store.save_checkpoint("goal_1", state)
        state.checkpoints.append(ckpt2)

        state.subtask_states["sub_2"] = "failed"
        await store.save_goal(state)

        mgr = CheckpointManager(store)
        rollback = await mgr.find_rollback_point("goal_1", "sub_2")
        assert rollback == ckpt2  # sub_2 was still pending at ckpt2

    @pytest.mark.asyncio
    async def test_no_checkpoints_returns_none(self):
        store = await self._init_db()
        state = GoalState(
            goal_id="goal_1", description="test", status="executing"
        )
        await store.save_goal(state)

        mgr = CheckpointManager(store)
        result = await mgr.find_rollback_point("goal_1", "sub_1")
        assert result is None

    @pytest.mark.asyncio
    async def test_rollback_restores_state(self):
        store = await self._init_db()
        state = GoalState(
            goal_id="goal_1",
            description="test",
            status="executing",
            subtask_states={"sub_1": "pending", "sub_2": "pending"},
        )
        await store.save_goal(state)
        ckpt_id = await store.save_checkpoint("goal_1", state)
        state.checkpoints.append(ckpt_id)

        # Advance state
        state.subtask_states = {"sub_1": "completed", "sub_2": "failed"}
        state.subtask_results["sub_1"] = SubtaskResult(
            subtask_id="sub_1", goal_id="goal_1", status="completed",
            output={"result": "ok"},
        )
        await store.save_goal(state)

        mgr = CheckpointManager(store)
        result = await mgr.rollback_to("goal_1", ckpt_id)

        assert result is not None
        restored = await store.load_goal("goal_1")
        assert restored.subtask_states["sub_1"] == "pending"
        assert restored.subtask_states["sub_2"] == "pending"

    @pytest.mark.asyncio
    async def test_rollback_preserves_completed(self):
        store = await self._init_db()
        result_1 = SubtaskResult(
            subtask_id="sub_1", goal_id="goal_1", status="completed",
            output={"result": "first"},
        )
        state = GoalState(
            goal_id="goal_1",
            description="test",
            status="executing",
            subtask_states={"sub_1": "completed", "sub_2": "pending"},
            subtask_results={"sub_1": result_1},
        )
        await store.save_goal(state)
        ckpt_id = await store.save_checkpoint("goal_1", state)
        state.checkpoints.append(ckpt_id)

        # sub_2 completes then fails verification
        state.subtask_states["sub_2"] = "completed"
        state.subtask_results["sub_2"] = SubtaskResult(
            subtask_id="sub_2", goal_id="goal_1", status="completed",
            output={"result": "bad"},
        )
        await store.save_goal(state)

        mgr = CheckpointManager(store)
        await mgr.rollback_to("goal_1", ckpt_id)

        restored = await store.load_goal("goal_1")
        assert "sub_1" in restored.subtask_results
        assert "sub_2" not in restored.subtask_results
        assert restored.subtask_states["sub_1"] == "completed"
        assert restored.subtask_states["sub_2"] == "pending"


# ── TestRecoveryExecution ─────────────────────────────────────────────


class TestRecoveryExecution:
    def setup_method(self):
        self.failure_kb = AsyncMock(spec=FailureKnowledgeBase)
        self.failure_kb.lookup = AsyncMock(return_value=[])
        self.recovery = RecoveryOrchestrator(failure_kb=self.failure_kb)

    def test_build_redispatch_preserves_context(self):
        payload = {
            "goal_id": "g1",
            "subtask_id": "s1",
            "task_type": "research",
            "description": "Test desc",
            "contract": {"postconditions": ["score >= 0.5"]},
            "model_tier": "balanced",
            "dependency_context": {"dep_1": "result"},
            "assigned_agent_id": "executor_default",
        }
        envelope = self.recovery._build_redispatch(payload)
        assert envelope.topic == "tasks.dispatched"
        assert envelope.payload["goal_id"] == "g1"
        assert envelope.payload["subtask_id"] == "s1"
        assert envelope.payload["task_type"] == "research"
        assert envelope.payload["contract"] == {"postconditions": ["score >= 0.5"]}
        assert envelope.payload["model_tier"] == "balanced"

    def test_build_redispatch_with_tier_override(self):
        payload = {
            "goal_id": "g1",
            "subtask_id": "s1",
            "task_type": "research",
            "description": "Test",
            "model_tier": "fast",
        }
        envelope = self.recovery._build_redispatch(payload, model_tier="powerful")
        assert envelope.payload["model_tier"] == "powerful"

    def test_simplify_description(self):
        desc = "A" * 600
        simplified = self.recovery._simplify_description(desc)
        assert simplified.startswith("Simplified: ")
        assert len(simplified) <= 512 + len("Simplified: ")

    def test_build_hil_envelope(self):
        payload = {
            "goal_id": "g1",
            "subtask_id": "s1",
            "task_type": "research",
            "description": "Test desc",
        }
        envelope = self.recovery._build_hil_envelope(payload)
        assert envelope.topic == "hil.approval_required"
        assert envelope.payload["goal_id"] == "g1"
        assert envelope.payload["subtask_id"] == "s1"
        assert "reason" in envelope.payload


# ── TestEndToEndVerificationFlow ──────────────────────────────────────


class TestEndToEndVerificationFlow:
    """Integration-style tests using a real MemoryBus."""

    def setup_method(self):
        self.bus = MemoryBus()
        self.tmp = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmp) / "test.db")
        self.failure_kb = FailureKnowledgeBase(self.db_path)
        self.verification_svc = VerificationService()
        self.recovery = RecoveryOrchestrator(
            failure_kb=self.failure_kb, bus=self.bus
        )
        self.gate = VerificationGate(
            self.bus, self.verification_svc, self.recovery, self.failure_kb
        )

    @pytest.mark.asyncio
    async def test_full_flow_pass(self):
        """dispatched → completed → verified → handler receives."""
        received = []

        async def on_verified(envelope: Envelope) -> None:
            received.append(envelope)

        await self.gate.start()
        self.bus.subscribe("tasks.verified", on_verified)

        # 1. Dispatch
        dispatch_env = _make_dispatch_envelope()
        await self.bus.publish_and_wait(dispatch_env)

        # 2. Complete
        completed_env = _make_completed_envelope(
            output={"result": "good content", "confidence": 0.85}
        )
        await self.bus.publish_and_wait(completed_env)

        assert len(received) == 1
        assert received[0].payload["subtask_id"] == "sub_1"
        assert received[0].payload["verification_result"]["overall"] == "pass"

        await self.gate.stop()

    @pytest.mark.asyncio
    async def test_full_flow_fail_and_recover(self):
        """dispatched → failed → recovery redispatch → completed → verified."""
        verified = []
        redispatched = []

        async def on_verified(envelope: Envelope) -> None:
            verified.append(envelope)

        async def on_dispatched(envelope: Envelope) -> None:
            # Only collect recovery redispatches (from recovery service)
            if envelope.source_service_id == "recovery":
                redispatched.append(envelope)

        await self.gate.start()
        self.bus.subscribe("tasks.verified", on_verified)
        self.bus.subscribe("tasks.dispatched", on_dispatched)

        # 1. Original dispatch
        dispatch_env = _make_dispatch_envelope()
        await self.bus.publish_and_wait(dispatch_env)

        # 2. Task fails with transient error
        with patch("qe.services.recovery.service.asyncio.sleep", new_callable=AsyncMock):
            failed_env = _make_failed_envelope(error="Connection timeout")
            await self.bus.publish_and_wait(failed_env)

        # Should have redispatched
        assert len(redispatched) >= 1

        # 3. Second attempt succeeds — simulate completed after redispatch
        completed_env = _make_completed_envelope(
            output={"result": "recovered content", "confidence": 0.9}
        )
        await self.bus.publish_and_wait(completed_env)

        assert len(verified) == 1
        assert verified[0].payload["status"] == "completed"

        await self.gate.stop()

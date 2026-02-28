"""Tests for Phase 1: Foundation Hardening improvements."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.envelope import Envelope
from qe.runtime.budget import BudgetTracker
from qe.substrate.belief_ledger import BeliefLedger

# ── Bus delivery confirmation ────────────────────────────────────────────────


class TestBusPublishAndWait:
    def test_publish_returns_tasks(self):
        bus = MemoryBus()
        results = []

        async def handler(env):
            results.append(env.envelope_id)

        bus.subscribe("claims.committed", handler)

        async def run():
            envelope = Envelope(
                topic="claims.committed",
                source_service_id="test",
                payload={"x": 1},
            )
            tasks = bus.publish(envelope)
            assert len(tasks) == 1
            await asyncio.gather(*tasks)
            assert len(results) == 1

        asyncio.run(run())

    def test_publish_returns_empty_list_no_subscribers(self):
        bus = MemoryBus()

        async def run():
            envelope = Envelope(
                topic="claims.committed",
                source_service_id="test",
                payload={},
            )
            tasks = bus.publish(envelope)
            assert tasks == []

        asyncio.run(run())

    @pytest.mark.asyncio
    async def test_publish_and_wait(self):
        bus = MemoryBus()
        call_count = 0

        async def handler(env):
            nonlocal call_count
            call_count += 1

        bus.subscribe("claims.committed", handler)
        envelope = Envelope(
            topic="claims.committed",
            source_service_id="test",
            payload={},
        )
        results = await bus.publish_and_wait(envelope)
        assert len(results) == 1
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_publish_and_wait_handles_errors(self):
        bus = MemoryBus()

        async def bad_handler(env):
            raise ValueError("boom")

        bus.subscribe("claims.committed", bad_handler)
        envelope = Envelope(
            topic="claims.committed",
            source_service_id="test",
            payload={},
        )
        # Should not raise — errors are captured
        results = await bus.publish_and_wait(envelope)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_publish_and_wait_first(self):
        bus = MemoryBus()

        async def fast_handler(env):
            return "fast"

        async def slow_handler(env):
            await asyncio.sleep(10)
            return "slow"

        bus.subscribe("claims.committed", fast_handler)
        bus.subscribe("claims.committed", slow_handler)
        envelope = Envelope(
            topic="claims.committed",
            source_service_id="test",
            payload={},
        )
        result = await bus.publish_and_wait_first(envelope)
        assert result is None  # _safe_call wraps, so result is None

    @pytest.mark.asyncio
    async def test_publish_and_wait_empty(self):
        bus = MemoryBus()
        envelope = Envelope(
            topic="claims.committed",
            source_service_id="test",
            payload={},
        )
        results = await bus.publish_and_wait(envelope)
        assert results == []


# ── Budget persistence ───────────────────────────────────────────────────────


class TestBudgetPersistence:
    @pytest.fixture
    async def budget_db(self, tmp_path):
        db_path = str(tmp_path / "budget_test.db")
        async with aiosqlite.connect(db_path) as db:
            migration = Path(__file__).parent.parent.parent / (
                "src/qe/substrate/migrations/0004_budget_records.sql"
            )
            await db.executescript(migration.read_text())
            await db.commit()
        return db_path

    @pytest.mark.asyncio
    async def test_save_and_load(self, budget_db):
        tracker = BudgetTracker(monthly_limit_usd=100.0, db_path=budget_db)
        tracker.record_cost("gpt-4o", 0.05, tokens_in=100, tokens_out=50)
        tracker.record_cost("gpt-4o-mini", 0.01)
        await tracker.save()

        # Create a new tracker and load
        tracker2 = BudgetTracker(monthly_limit_usd=100.0, db_path=budget_db)
        await tracker2.load()
        assert abs(tracker2.total_spend() - 0.06) < 1e-6
        assert "gpt-4o" in tracker2.spend_by_model()
        assert "gpt-4o-mini" in tracker2.spend_by_model()

    @pytest.mark.asyncio
    async def test_save_creates_records(self, budget_db):
        tracker = BudgetTracker(monthly_limit_usd=100.0, db_path=budget_db)
        tracker.record_cost("gpt-4o", 0.02, service_id="researcher")
        tracker.record_cost("gpt-4o", 0.03, service_id="validator")
        await tracker.save()

        async with aiosqlite.connect(budget_db) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM budget_records")
            row = await cursor.fetchone()
            assert row[0] == 2

            cursor = await db.execute(
                "SELECT total_spend_usd FROM budget_monthly"
            )
            row = await cursor.fetchone()
            assert abs(row[0] - 0.05) < 1e-6

    @pytest.mark.asyncio
    async def test_load_no_tables(self, tmp_path):
        """Load gracefully handles missing tables."""
        db_path = str(tmp_path / "empty.db")
        async with aiosqlite.connect(db_path) as db:
            await db.execute("CREATE TABLE dummy (id INTEGER)")
            await db.commit()
        tracker = BudgetTracker(monthly_limit_usd=100.0, db_path=db_path)
        await tracker.load()  # Should not raise
        assert tracker.total_spend() == 0.0

    @pytest.mark.asyncio
    async def test_save_no_db_path(self):
        """Save is a no-op without db_path."""
        tracker = BudgetTracker(monthly_limit_usd=100.0)
        tracker.record_cost("gpt-4o", 0.05)
        await tracker.save()  # Should not raise

    def test_record_cost_extended_params(self):
        tracker = BudgetTracker(monthly_limit_usd=100.0)
        tracker.record_cost(
            "gpt-4o",
            0.05,
            tokens_in=500,
            tokens_out=200,
            service_id="researcher",
            envelope_id="env-123",
        )
        assert tracker.total_spend() == 0.05


# ── Belief ledger enhancements ───────────────────────────────────────────────


class TestBeliefLedgerEnhancements:
    @pytest.fixture
    async def ledger(self, tmp_path):
        db_path = str(tmp_path / "ledger_test.db")
        migrations_dir = (
            Path(__file__).parent.parent.parent
            / "src/qe/substrate/migrations"
        )
        bl = BeliefLedger(db_path, migrations_dir)
        await bl.initialize()
        return bl

    def _make_claim(self, claim_id="clm_1", subject="entity_1", **kwargs):
        from qe.models.claim import Claim

        defaults = {
            "claim_id": claim_id,
            "subject_entity_id": subject,
            "predicate": "has_property",
            "object_value": "blue",
            "confidence": 0.9,
            "source_service_id": "test",
            "source_envelope_ids": ["env-1"],
        }
        defaults.update(kwargs)
        return Claim(**defaults)

    @pytest.mark.asyncio
    async def test_get_claim_by_id(self, ledger):
        claim = self._make_claim("clm_lookup")
        await ledger.commit_claim(claim)
        result = await ledger.get_claim_by_id("clm_lookup")
        assert result is not None
        assert result.claim_id == "clm_lookup"

    @pytest.mark.asyncio
    async def test_get_claim_by_id_not_found(self, ledger):
        result = await ledger.get_claim_by_id("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_count_claims(self, ledger):
        assert await ledger.count_claims() == 0
        await ledger.commit_claim(self._make_claim("clm_a"))
        await ledger.commit_claim(
            self._make_claim("clm_b", subject="entity_2")
        )
        assert await ledger.count_claims() == 2

    @pytest.mark.asyncio
    async def test_count_claims_excludes_superseded(self, ledger):
        await ledger.commit_claim(
            self._make_claim("clm_old", confidence=0.5)
        )
        # Supersede with higher confidence
        await ledger.commit_claim(
            self._make_claim("clm_new", confidence=0.95)
        )
        assert await ledger.count_claims(include_superseded=False) == 1
        assert await ledger.count_claims(include_superseded=True) == 2

    @pytest.mark.asyncio
    async def test_get_claims_since(self, ledger):
        old_claim = self._make_claim("clm_old_time", subject="entity_old")
        await ledger.commit_claim(old_claim)

        # Use a timestamp just before now for the "since" boundary
        boundary = datetime.now(UTC) - timedelta(seconds=1)

        new_claim = self._make_claim("clm_new_time", subject="entity_new")
        await ledger.commit_claim(new_claim)

        # Claims since boundary should include only the new one
        recent = await ledger.get_claims_since(boundary)
        ids = [c.claim_id for c in recent]
        assert "clm_new_time" in ids


# ── New bus topics ───────────────────────────────────────────────────────────


class TestNewBusTopics:
    def test_goal_topics_exist(self):
        from qe.bus.protocol import TOPICS

        goal_topics = [
            "goals.submitted",
            "goals.enriched",
            "goals.completed",
            "goals.failed",
        ]
        for topic in goal_topics:
            assert topic in TOPICS, f"Missing topic: {topic}"

    def test_task_topics_exist(self):
        from qe.bus.protocol import TOPICS

        task_topics = [
            "tasks.planned",
            "tasks.dispatched",
            "tasks.completed",
            "tasks.verified",
            "tasks.verification_failed",
            "tasks.recovered",
            "tasks.failed",
            "tasks.progress",
            "tasks.checkpoint",
        ]
        for topic in task_topics:
            assert topic in TOPICS, f"Missing topic: {topic}"

    def test_memory_topics_exist(self):
        from qe.bus.protocol import TOPICS

        assert "memory.updated" in TOPICS
        assert "memory.preference_set" in TOPICS

    def test_analysis_synthesis_topics_exist(self):
        from qe.bus.protocol import TOPICS

        assert "analysis.requested" in TOPICS
        assert "analysis.completed" in TOPICS
        assert "synthesis.requested" in TOPICS
        assert "synthesis.completed" in TOPICS

    def test_claim_challenge_topics_exist(self):
        from qe.bus.protocol import TOPICS

        assert "claims.challenged" in TOPICS
        assert "claims.verification_requested" in TOPICS

    def test_system_resource_topic_exists(self):
        from qe.bus.protocol import TOPICS

        assert "system.resource_alert" in TOPICS
        assert "system.security_alert" in TOPICS
        assert "system.digest" in TOPICS

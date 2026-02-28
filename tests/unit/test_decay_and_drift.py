"""Tests for confidence decay (feature 5) and anti-drift gates (feature 6)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.claim import Claim
from qe.models.envelope import Envelope
from qe.models.goal import GoalDecomposition, GoalState, Subtask, SubtaskResult
from qe.services.dispatcher.service import Dispatcher, _text_similarity
from qe.substrate.belief_ledger import BeliefLedger

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_claim(
    claim_id: str = "clm_1",
    subject: str = "entity_1",
    confidence: float = 0.9,
    created_at: datetime | None = None,
    valid_until: datetime | None = None,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        subject_entity_id=subject,
        predicate="has_property",
        object_value="blue",
        confidence=confidence,
        source_service_id="test",
        source_envelope_ids=["env-1"],
        created_at=created_at or datetime.now(UTC),
        valid_until=valid_until,
    )


# ── Feature 5: Confidence Decay Tests ───────────────────────────────────────


class TestApplyDecay:
    """BeliefLedger._apply_decay computes age-adjusted confidence."""

    @pytest.fixture
    async def ledger(self, tmp_path):
        db_path = str(tmp_path / "decay_test.db")
        migrations_dir = (
            Path(__file__).parent.parent.parent
            / "src/qe/substrate/migrations"
        )
        bl = BeliefLedger(db_path, migrations_dir)
        await bl.initialize()
        return bl

    def test_fresh_claim_no_decay(self, ledger):
        """Claim created now retains ~100% confidence."""
        now = datetime.now(UTC)
        claim = _make_claim(confidence=0.9, created_at=now)
        result = ledger._apply_decay(claim, now)
        assert result.confidence == pytest.approx(0.9, abs=0.001)

    def test_aged_claim_decays(self, ledger):
        """Claim 30 days old has ~50% of original with 720h half-life."""
        now = datetime.now(UTC)
        created = now - timedelta(hours=720)  # exactly one half-life
        claim = _make_claim(confidence=0.9, created_at=created)
        result = ledger._apply_decay(claim, now)
        assert result.confidence == pytest.approx(0.45, abs=0.01)

    def test_expired_claim_zero_confidence(self, ledger):
        """Claim with valid_until in the past gets confidence 0.0."""
        now = datetime.now(UTC)
        expired = now - timedelta(hours=1)
        claim = _make_claim(
            confidence=0.9,
            created_at=now - timedelta(hours=24),
            valid_until=expired,
        )
        result = ledger._apply_decay(claim, now)
        assert result.confidence == 0.0

    def test_valid_until_in_future_still_decays(self, ledger):
        """Claim with valid_until in the future still gets age decay."""
        now = datetime.now(UTC)
        created = now - timedelta(hours=720)
        future = now + timedelta(hours=100)
        claim = _make_claim(
            confidence=0.9, created_at=created, valid_until=future,
        )
        result = ledger._apply_decay(claim, now)
        # Should be ~0.45 (half-life decay), NOT 0.0
        assert result.confidence == pytest.approx(0.45, abs=0.01)

    def test_custom_half_life(self):
        """Different half-life produces expected decay."""
        bl = BeliefLedger.__new__(BeliefLedger)
        bl._decay_half_life_hours = 24.0  # 1 day half-life

        now = datetime.now(UTC)
        created = now - timedelta(hours=24)
        claim = _make_claim(confidence=1.0, created_at=created)
        result = bl._apply_decay(claim, now)
        assert result.confidence == pytest.approx(0.5, abs=0.01)


class TestDecayInQueries:
    """get_claims applies decay and re-filters by min_confidence."""

    @pytest.fixture
    async def ledger(self, tmp_path):
        db_path = str(tmp_path / "decay_query_test.db")
        migrations_dir = (
            Path(__file__).parent.parent.parent
            / "src/qe/substrate/migrations"
        )
        bl = BeliefLedger(db_path, migrations_dir)
        await bl.initialize()
        return bl

    @pytest.mark.asyncio
    async def test_decayed_claim_filtered_by_min_confidence(self, ledger):
        """Old claims that decay below min_confidence are excluded."""
        old_claim = _make_claim(
            claim_id="clm_old",
            confidence=0.6,
            created_at=datetime.now(UTC) - timedelta(days=60),
        )
        fresh_claim = _make_claim(
            claim_id="clm_fresh",
            subject="entity_2",
            confidence=0.6,
            created_at=datetime.now(UTC),
        )
        await ledger.commit_claim(old_claim)
        await ledger.commit_claim(fresh_claim)

        # With min_confidence=0.5, old claim (decayed ~60 days) should be excluded
        # 0.6 * 0.5^(1440/720) = 0.6 * 0.25 = 0.15 → below 0.5
        results = await ledger.get_claims(min_confidence=0.5)
        ids = [c.claim_id for c in results]
        assert "clm_fresh" in ids
        assert "clm_old" not in ids

    @pytest.mark.asyncio
    async def test_decayed_claims_returned_with_low_threshold(self, ledger):
        """Old claims still appear when min_confidence is low."""
        old_claim = _make_claim(
            claim_id="clm_old",
            confidence=0.6,
            created_at=datetime.now(UTC) - timedelta(days=60),
        )
        await ledger.commit_claim(old_claim)

        results = await ledger.get_claims(min_confidence=0.0)
        assert len(results) == 1
        # Confidence should be decayed (< 0.6)
        assert results[0].confidence < 0.6

    @pytest.mark.asyncio
    async def test_decay_in_search_full_text(self, ledger):
        """FTS results also get decayed confidence."""
        claim = _make_claim(
            claim_id="clm_fts",
            confidence=0.9,
            created_at=datetime.now(UTC) - timedelta(days=30),
        )
        await ledger.commit_claim(claim)

        results = await ledger.search_full_text("blue", limit=10)
        assert len(results) == 1
        # Should be decayed: 0.9 * 0.5^(720/720) = ~0.45
        assert results[0].confidence < 0.9
        assert results[0].confidence == pytest.approx(0.45, abs=0.05)


# ── Feature 6: Anti-Drift Gate Tests ────────────────────────────────────────


class TestTextSimilarity:
    """_text_similarity computes Jaccard word overlap."""

    def test_identical_text(self):
        assert _text_similarity("hello world", "hello world") == 1.0

    def test_disjoint_text(self):
        assert _text_similarity("hello world", "foo bar") == 0.0

    def test_partial_overlap(self):
        sim = _text_similarity(
            "research quantum computing applications",
            "quantum computing is fascinating research",
        )
        # Words: {research, quantum, computing, applications}
        #   vs   {quantum, computing, is, fascinating, research}
        # Intersection: {research, quantum, computing} = 3
        # Union: {research, quantum, computing, applications, is, fascinating} = 6
        # Jaccard = 3/6 = 0.5
        assert sim == pytest.approx(0.5, abs=0.01)

    def test_empty_string(self):
        assert _text_similarity("", "hello") == 0.0
        assert _text_similarity("hello", "") == 0.0
        assert _text_similarity("", "") == 0.0


class TestCheckDrift:
    """Dispatcher._check_drift publishes events on low similarity."""

    def _make_state(
        self, goal_id: str = "goal_1", description: str = "Analyse quarterly earnings",
    ) -> GoalState:
        return GoalState(
            goal_id=goal_id,
            description=description,
            status="executing",
            decomposition=GoalDecomposition(
                goal_id=goal_id,
                original_description=description,
                strategy="test",
                subtasks=[
                    Subtask(
                        subtask_id="sub_1",
                        description="Research topic",
                        task_type="research",
                    ),
                ],
            ),
            subtask_states={"sub_1": "dispatched"},
        )

    def test_drift_detected_publishes_event(self):
        """Low similarity between goal and output triggers drift event."""
        bus = MemoryBus()
        published: list[Envelope] = []
        bus.add_publish_listener(lambda env: published.append(env))

        mock_store = MagicMock()
        dispatcher = Dispatcher(
            bus=bus, goal_store=mock_store, drift_threshold=0.1,
        )

        state = self._make_state(
            description="Analyse quarterly earnings for ACME Corp",
        )
        result = SubtaskResult(
            subtask_id="sub_1",
            goal_id="goal_1",
            status="completed",
            output={
                "content": "The weather in Tokyo is sunny today with mild temperatures.",
            },
        )

        dispatcher._check_drift(state, result)

        drift_events = [
            e for e in published if e.topic == "goals.drift_detected"
        ]
        assert len(drift_events) == 1
        assert drift_events[0].payload["goal_id"] == "goal_1"
        assert drift_events[0].payload["subtask_id"] == "sub_1"
        assert drift_events[0].payload["similarity"] < 0.1

    def test_no_drift_on_aligned_output(self):
        """High similarity between goal and output produces no event."""
        bus = MemoryBus()
        published: list[Envelope] = []
        bus.add_publish_listener(lambda env: published.append(env))

        mock_store = MagicMock()
        dispatcher = Dispatcher(
            bus=bus, goal_store=mock_store, drift_threshold=0.05,
        )

        state = self._make_state(
            description="Analyse quarterly earnings for ACME Corp",
        )
        result = SubtaskResult(
            subtask_id="sub_1",
            goal_id="goal_1",
            status="completed",
            output={
                "content": (
                    "ACME Corp quarterly earnings analysis shows strong "
                    "revenue growth in Q4. Earnings per share exceeded analyst "
                    "expectations."
                ),
            },
        )

        dispatcher._check_drift(state, result)

        drift_events = [
            e for e in published if e.topic == "goals.drift_detected"
        ]
        assert len(drift_events) == 0

    @pytest.mark.asyncio
    async def test_drift_check_skipped_on_failure(self):
        """Failed subtask results don't trigger drift check."""
        bus = MemoryBus()
        published: list[Envelope] = []
        bus.add_publish_listener(lambda env: published.append(env))

        mock_store = MagicMock()
        mock_store.save_checkpoint = AsyncMock(return_value="ckpt_1")
        mock_store.save_goal = AsyncMock()

        dispatcher = Dispatcher(
            bus=bus, goal_store=mock_store, drift_threshold=0.5,
        )

        state = self._make_state()
        dispatcher._active_goals["goal_1"] = state

        # Failed result — completely unrelated content
        result = SubtaskResult(
            subtask_id="sub_1",
            goal_id="goal_1",
            status="failed",
            output={"error": "LLM timeout"},
        )

        await dispatcher.handle_subtask_completed("goal_1", result)

        drift_events = [
            e for e in published if e.topic == "goals.drift_detected"
        ]
        assert len(drift_events) == 0

    def test_drift_check_skipped_on_empty_output(self):
        """Empty output content doesn't trigger drift check."""
        bus = MemoryBus()
        published: list[Envelope] = []
        bus.add_publish_listener(lambda env: published.append(env))

        mock_store = MagicMock()
        dispatcher = Dispatcher(
            bus=bus, goal_store=mock_store, drift_threshold=0.5,
        )

        state = self._make_state()
        result = SubtaskResult(
            subtask_id="sub_1",
            goal_id="goal_1",
            status="completed",
            output={"content": ""},
        )

        dispatcher._check_drift(state, result)

        drift_events = [
            e for e in published if e.topic == "goals.drift_detected"
        ]
        assert len(drift_events) == 0

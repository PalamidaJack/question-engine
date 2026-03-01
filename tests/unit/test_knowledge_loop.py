"""Tests for KnowledgeLoop — background consolidation service."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from qe.runtime.episodic_memory import Episode, EpisodicMemory
from qe.runtime.knowledge_loop import (
    ConsolidationResult,
    ExtractedClaim,
    KnowledgeLoop,
)
from qe.runtime.procedural_memory import (
    ProceduralMemory,
    QuestionTemplate,
    ToolSequence,
)
from qe.substrate.bayesian_belief import BayesianBeliefStore, Hypothesis

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_episode(
    episode_type: str = "synthesis",
    summary: str = "test observation",
    **kwargs,
) -> Episode:
    return Episode(
        episode_type=episode_type,
        summary=summary,
        timestamp=datetime.now(UTC),
        **kwargs,
    )


def _make_loop(**overrides) -> KnowledgeLoop:
    """Build a KnowledgeLoop with mocked dependencies."""
    episodic = AsyncMock(spec=EpisodicMemory)
    episodic.recall = AsyncMock(return_value=[])
    belief = AsyncMock(spec=BayesianBeliefStore)
    belief.get_active_hypotheses = AsyncMock(return_value=[])
    belief.update_belief = AsyncMock()
    procedural = AsyncMock(spec=ProceduralMemory)
    procedural.get_best_templates = AsyncMock(return_value=[])
    procedural.get_best_sequences = AsyncMock(return_value=[])

    defaults = {
        "episodic_memory": episodic,
        "belief_store": belief,
        "procedural_memory": procedural,
        "bus": MagicMock(),
        "model": "gpt-4o-mini",
        "consolidation_interval": 0.01,
    }
    defaults.update(overrides)
    return KnowledgeLoop(**defaults)


# ── Model Tests ───────────────────────────────────────────────────────────


class TestConsolidationResultModel:
    def test_defaults(self):
        r = ConsolidationResult()
        assert r.episodes_scanned == 0
        assert r.patterns_detected == 0
        assert r.beliefs_promoted == 0
        assert r.contradictions_found == 0
        assert r.hypotheses_reviewed == 0
        assert r.hypotheses_confirmed == 0
        assert r.hypotheses_falsified == 0
        assert r.templates_retired == 0
        assert r.cycle_duration_s == 0.0

    def test_custom_values(self):
        r = ConsolidationResult(
            episodes_scanned=10,
            beliefs_promoted=3,
            cycle_duration_s=1.5,
        )
        assert r.episodes_scanned == 10
        assert r.beliefs_promoted == 3
        assert r.cycle_duration_s == 1.5


class TestExtractedClaimModel:
    def test_fields(self):
        c = ExtractedClaim(
            subject_entity_id="entity_1",
            predicate="has_property",
            object_value="blue",
            confidence=0.8,
            reasoning="observed multiple times",
        )
        assert c.subject_entity_id == "entity_1"
        assert c.predicate == "has_property"
        assert c.object_value == "blue"
        assert c.confidence == 0.8
        assert c.reasoning == "observed multiple times"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ExtractedClaim(
                subject_entity_id="x",
                predicate="p",
                object_value="v",
                confidence=1.5,
            )
        with pytest.raises(ValidationError):
            ExtractedClaim(
                subject_entity_id="x",
                predicate="p",
                object_value="v",
                confidence=-0.1,
            )

    def test_default_reasoning(self):
        c = ExtractedClaim(
            subject_entity_id="x",
            predicate="p",
            object_value="v",
            confidence=0.5,
        )
        assert c.reasoning == ""


# ── Init Tests ────────────────────────────────────────────────────────────


class TestKnowledgeLoopInit:
    def test_defaults(self):
        loop = _make_loop()
        assert loop._consolidation_interval == 0.01
        assert loop._episode_lookback_hours == 1.0
        assert loop._promotion_confidence == 0.7
        assert loop._retirement_threshold == 0.2
        assert loop._min_evidence_count == 3

    def test_custom_params(self):
        loop = _make_loop(
            consolidation_interval=600.0,
            episode_lookback_hours=2.0,
            promotion_confidence=0.9,
            retirement_threshold=0.1,
            min_evidence_count=5,
        )
        assert loop._consolidation_interval == 600.0
        assert loop._episode_lookback_hours == 2.0
        assert loop._promotion_confidence == 0.9
        assert loop._retirement_threshold == 0.1
        assert loop._min_evidence_count == 5

    def test_status_before_start(self):
        loop = _make_loop()
        s = loop.status()
        assert s["running"] is False
        assert s["cycles_total"] == 0
        assert s["last_cycle_at"] is None
        assert s["last_cycle_result"] is None


# ── Lifecycle Tests ───────────────────────────────────────────────────────


class TestKnowledgeLoopLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        loop = _make_loop()
        loop.start()
        assert loop._running is True
        assert loop._loop_task is not None
        await loop.stop()
        assert loop._running is False
        assert loop._loop_task is None

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self):
        loop = _make_loop()
        loop.start()
        task1 = loop._loop_task
        loop.start()
        task2 = loop._loop_task
        assert task1 is task2
        await loop.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        loop = _make_loop()
        await loop.stop()
        assert loop._running is False


# ── Feature Flag Gating ──────────────────────────────────────────────────


class TestFeatureFlagGating:
    @pytest.mark.asyncio
    async def test_consolidation_skipped_when_flag_disabled(self):
        loop = _make_loop()
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = False

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        # No episodes should be scanned
        loop._episodic.recall.assert_not_called()

    @pytest.mark.asyncio
    async def test_consolidation_runs_when_flag_enabled(self):
        loop = _make_loop()
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        loop._episodic.recall.assert_called_once()


# ── Episode Scan ─────────────────────────────────────────────────────────


class TestEpisodeScan:
    @pytest.mark.asyncio
    async def test_scans_recent_episodes(self):
        loop = _make_loop()
        episodes = [_make_episode() for _ in range(5)]
        loop._episodic.recall = AsyncMock(return_value=episodes)
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        loop._episodic.recall.assert_called_once_with(
            query="",
            top_k=200,
            time_window_hours=1.0,
        )
        assert loop._last_cycle_result is not None
        assert loop._last_cycle_result.episodes_scanned == 5

    @pytest.mark.asyncio
    async def test_groups_by_type(self):
        episodes = [
            _make_episode(episode_type="synthesis"),
            _make_episode(episode_type="synthesis"),
            _make_episode(episode_type="claim_committed"),
            _make_episode(episode_type="tool_call"),  # not relevant
        ]
        groups = KnowledgeLoop._group_episodes(episodes)
        assert "synthesis" in groups
        assert len(groups["synthesis"]) == 2
        assert "claim_committed" in groups
        assert len(groups["claim_committed"]) == 1
        assert "tool_call" not in groups

    @pytest.mark.asyncio
    async def test_handles_empty_episodes(self):
        loop = _make_loop()
        loop._episodic.recall = AsyncMock(return_value=[])
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        assert loop._last_cycle_result.episodes_scanned == 0
        assert loop._last_cycle_result.patterns_detected == 0


# ── Pattern Detection ────────────────────────────────────────────────────


class TestPatternDetection:
    @pytest.mark.asyncio
    async def test_detects_repeated_patterns(self):
        """When enough episodes of same type exist, count as a pattern."""
        loop = _make_loop(min_evidence_count=2)
        episodes = [_make_episode(episode_type="synthesis") for _ in range(3)]
        loop._episodic.recall = AsyncMock(return_value=episodes)
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        # Mock LLM to return a claim
        mock_claim = ExtractedClaim(
            subject_entity_id="e1",
            predicate="is",
            object_value="blue",
            confidence=0.8,
        )
        with (
            patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag),
            patch.object(
                loop, "_extract_claim_from_episodes",
                new_callable=AsyncMock, return_value=mock_claim,
            ),
        ):
            await loop._consolidate()

        assert loop._last_cycle_result.patterns_detected == 1
        assert loop._last_cycle_result.beliefs_promoted == 1

    @pytest.mark.asyncio
    async def test_skips_low_count_patterns(self):
        """Patterns with fewer than min_evidence_count episodes are skipped."""
        loop = _make_loop(min_evidence_count=5)
        episodes = [_make_episode(episode_type="synthesis") for _ in range(3)]
        loop._episodic.recall = AsyncMock(return_value=episodes)
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        assert loop._last_cycle_result.patterns_detected == 0

    @pytest.mark.asyncio
    async def test_calls_llm_for_extraction(self):
        """Verify _extract_claim_from_episodes calls instructor."""
        loop = _make_loop()
        episodes = [_make_episode(summary=f"obs {i}") for i in range(3)]

        mock_claim = ExtractedClaim(
            subject_entity_id="test",
            predicate="is",
            object_value="good",
            confidence=0.9,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_claim)

        with patch("qe.runtime.knowledge_loop.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await loop._extract_claim_from_episodes(episodes)

        assert result is not None
        assert result.subject_entity_id == "test"
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_llm_error(self):
        loop = _make_loop()
        episodes = [_make_episode() for _ in range(3)]

        with patch("qe.runtime.knowledge_loop.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(side_effect=Exception("LLM failed"))
            mock_inst.from_litellm.return_value = mock_client
            result = await loop._extract_claim_from_episodes(episodes)

        assert result is None

    @pytest.mark.asyncio
    async def test_low_confidence_extraction_returns_none(self):
        loop = _make_loop()
        episodes = [_make_episode() for _ in range(3)]

        mock_claim = ExtractedClaim(
            subject_entity_id="x",
            predicate="p",
            object_value="v",
            confidence=0.2,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_claim)

        with patch("qe.runtime.knowledge_loop.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await loop._extract_claim_from_episodes(episodes)

        assert result is None


# ── Belief Promotion ─────────────────────────────────────────────────────


class TestBeliefPromotion:
    @pytest.mark.asyncio
    async def test_promotes_high_confidence_to_belief_store(self):
        loop = _make_loop(min_evidence_count=2, promotion_confidence=0.6)
        episodes = [_make_episode(episode_type="synthesis") for _ in range(3)]
        loop._episodic.recall = AsyncMock(return_value=episodes)
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        mock_claim = ExtractedClaim(
            subject_entity_id="e1",
            predicate="revenue",
            object_value="$10M",
            confidence=0.85,
        )
        with (
            patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag),
            patch.object(
                loop, "_extract_claim_from_episodes",
                new_callable=AsyncMock, return_value=mock_claim,
            ),
        ):
            await loop._consolidate()

        loop._beliefs.update_belief.assert_called_once()
        assert loop._beliefs_promoted_total == 1

    @pytest.mark.asyncio
    async def test_skips_low_confidence_claims(self):
        loop = _make_loop(min_evidence_count=2, promotion_confidence=0.7)
        episodes = [_make_episode(episode_type="synthesis") for _ in range(3)]
        loop._episodic.recall = AsyncMock(return_value=episodes)
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        mock_claim = ExtractedClaim(
            subject_entity_id="e1",
            predicate="revenue",
            object_value="$10M",
            confidence=0.5,
        )
        with (
            patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag),
            patch.object(
                loop, "_extract_claim_from_episodes",
                new_callable=AsyncMock, return_value=mock_claim,
            ),
        ):
            await loop._consolidate()

        loop._beliefs.update_belief.assert_not_called()

    @pytest.mark.asyncio
    async def test_publishes_belief_promoted_event(self):
        loop = _make_loop(min_evidence_count=2, promotion_confidence=0.6)
        episodes = [_make_episode(episode_type="synthesis") for _ in range(3)]
        loop._episodic.recall = AsyncMock(return_value=episodes)
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        mock_claim = ExtractedClaim(
            subject_entity_id="e1",
            predicate="has",
            object_value="value",
            confidence=0.8,
        )
        with (
            patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag),
            patch.object(
                loop, "_extract_claim_from_episodes",
                new_callable=AsyncMock, return_value=mock_claim,
            ),
        ):
            await loop._consolidate()

        # Check bus publish was called with belief_promoted topic
        publish_calls = loop._bus.publish.call_args_list
        belief_calls = [
            c for c in publish_calls
            if c.args[0].topic == "knowledge.belief_promoted"
        ]
        assert len(belief_calls) >= 1


# ── Hypothesis Review ────────────────────────────────────────────────────


class TestHypothesisReview:
    @pytest.mark.asyncio
    async def test_reviews_active_hypotheses(self):
        loop = _make_loop()
        hyp = Hypothesis(
            hypothesis_id="hyp_1",
            statement="test",
            current_probability=0.5,
        )
        loop._beliefs.get_active_hypotheses = AsyncMock(return_value=[hyp])
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        assert loop._last_cycle_result.hypotheses_reviewed == 1
        assert loop._last_cycle_result.hypotheses_confirmed == 0
        assert loop._last_cycle_result.hypotheses_falsified == 0

    @pytest.mark.asyncio
    async def test_detects_confirmed_hypothesis(self):
        loop = _make_loop()
        hyp = Hypothesis(
            hypothesis_id="hyp_1",
            statement="test",
            current_probability=0.96,
        )
        loop._beliefs.get_active_hypotheses = AsyncMock(return_value=[hyp])
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        assert loop._last_cycle_result.hypotheses_confirmed == 1

    @pytest.mark.asyncio
    async def test_detects_falsified_hypothesis(self):
        loop = _make_loop()
        hyp = Hypothesis(
            hypothesis_id="hyp_1",
            statement="test",
            current_probability=0.03,
        )
        loop._beliefs.get_active_hypotheses = AsyncMock(return_value=[hyp])
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        assert loop._last_cycle_result.hypotheses_falsified == 1

    @pytest.mark.asyncio
    async def test_publishes_hypothesis_updated_events(self):
        loop = _make_loop()
        hyp_confirmed = Hypothesis(
            hypothesis_id="hyp_c",
            statement="confirmed",
            current_probability=0.96,
        )
        hyp_falsified = Hypothesis(
            hypothesis_id="hyp_f",
            statement="falsified",
            current_probability=0.03,
        )
        loop._beliefs.get_active_hypotheses = AsyncMock(
            return_value=[hyp_confirmed, hyp_falsified],
        )
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        publish_calls = loop._bus.publish.call_args_list
        hyp_calls = [
            c for c in publish_calls
            if c.args[0].topic == "knowledge.hypothesis_updated"
        ]
        assert len(hyp_calls) == 2

    @pytest.mark.asyncio
    async def test_handles_no_hypotheses(self):
        loop = _make_loop()
        loop._beliefs.get_active_hypotheses = AsyncMock(return_value=[])
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        assert loop._last_cycle_result.hypotheses_reviewed == 0


# ── Procedural Retirement ────────────────────────────────────────────────


class TestProceduralRetirement:
    @pytest.mark.asyncio
    async def test_flags_low_success_templates(self):
        loop = _make_loop(retirement_threshold=0.2)
        tmpl = QuestionTemplate(
            template_id="qt_bad",
            pattern="bad pattern",
            success_count=1,
            failure_count=15,  # 6.25% success rate
        )
        loop._procedural.get_best_templates = AsyncMock(return_value=[tmpl])
        loop._procedural.get_best_sequences = AsyncMock(return_value=[])
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        assert loop._last_cycle_result.templates_retired == 1
        assert "qt_bad" in loop._retired_template_ids

    @pytest.mark.asyncio
    async def test_respects_min_count(self):
        """Templates with < 10 total observations should not be retired."""
        loop = _make_loop(retirement_threshold=0.2)
        tmpl = QuestionTemplate(
            template_id="qt_new",
            pattern="new pattern",
            success_count=0,
            failure_count=5,  # 0% but only 5 observations
        )
        loop._procedural.get_best_templates = AsyncMock(return_value=[tmpl])
        loop._procedural.get_best_sequences = AsyncMock(return_value=[])
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        assert loop._last_cycle_result.templates_retired == 0

    @pytest.mark.asyncio
    async def test_handles_empty_procedural(self):
        loop = _make_loop()
        loop._procedural.get_best_templates = AsyncMock(return_value=[])
        loop._procedural.get_best_sequences = AsyncMock(return_value=[])
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        assert loop._last_cycle_result.templates_retired == 0

    @pytest.mark.asyncio
    async def test_retirement_of_sequences(self):
        loop = _make_loop(retirement_threshold=0.2)
        seq = ToolSequence(
            sequence_id="ts_bad",
            tool_names=["web_search"],
            success_count=1,
            failure_count=20,
        )
        loop._procedural.get_best_templates = AsyncMock(return_value=[])
        loop._procedural.get_best_sequences = AsyncMock(return_value=[seq])
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        assert loop._last_cycle_result.templates_retired == 1
        assert "ts_bad" in loop._retired_sequence_ids


# ── Bus Events ───────────────────────────────────────────────────────────


class TestBusEvents:
    @pytest.mark.asyncio
    async def test_consolidation_completed_published(self):
        loop = _make_loop()
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        publish_calls = loop._bus.publish.call_args_list
        completed_calls = [
            c for c in publish_calls
            if c.args[0].topic == "knowledge.consolidation_completed"
        ]
        assert len(completed_calls) == 1

    @pytest.mark.asyncio
    async def test_publish_with_no_bus(self):
        """Publishing should silently succeed when bus is None."""
        loop = _make_loop(bus=None)
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag):
            await loop._consolidate()

        # Should not raise


# ── Integration ──────────────────────────────────────────────────────────


class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_cycle_with_real_episodic(self):
        """Full cycle with real EpisodicMemory (in-memory) + mocked LLM."""
        episodic = EpisodicMemory()  # in-memory, no db
        await episodic.initialize()

        # Store some episodes
        for i in range(5):
            await episodic.store(_make_episode(
                episode_type="synthesis",
                summary=f"Company X revenue is ${i}M",
            ))

        belief_store = AsyncMock(spec=BayesianBeliefStore)
        belief_store.get_active_hypotheses = AsyncMock(return_value=[])
        belief_store.update_belief = AsyncMock()

        procedural = ProceduralMemory()  # in-memory
        await procedural.initialize()

        bus = MagicMock()

        loop = KnowledgeLoop(
            episodic_memory=episodic,
            belief_store=belief_store,
            procedural_memory=procedural,
            bus=bus,
            consolidation_interval=0.01,
            min_evidence_count=3,
        )

        mock_claim = ExtractedClaim(
            subject_entity_id="company_x",
            predicate="revenue",
            object_value="growing",
            confidence=0.85,
        )
        mock_flag = MagicMock()
        mock_flag.is_enabled.return_value = True

        with (
            patch("qe.runtime.knowledge_loop.get_flag_store", return_value=mock_flag),
            patch.object(
                loop, "_extract_claim_from_episodes",
                new_callable=AsyncMock, return_value=mock_claim,
            ),
        ):
            await loop._consolidate()

        assert loop._cycles_total == 1
        assert loop._last_cycle_result is not None
        assert loop._last_cycle_result.episodes_scanned == 5
        assert loop._last_cycle_result.beliefs_promoted == 1
        belief_store.update_belief.assert_called_once()

        # Check consolidation_completed event was published
        publish_calls = bus.publish.call_args_list
        completed = [
            c for c in publish_calls
            if c.args[0].topic == "knowledge.consolidation_completed"
        ]
        assert len(completed) == 1

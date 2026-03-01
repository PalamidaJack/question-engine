"""Tests for PersistenceEngine — determination, reframing, root cause."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.cognition import (
    ReframingResult,
    RootCauseAnalysis,
    RootCauseLink,
)
from qe.runtime.persistence_engine import (
    REFRAMING_STRATEGIES,
    PersistenceEngine,
)


@pytest.fixture
def engine():
    return PersistenceEngine(model="test-model")


# ── Reframing Strategy Management ────────────────────────────────────────


class TestReframingStrategies:
    def test_all_strategies_defined(self):
        assert len(REFRAMING_STRATEGIES) == 7
        assert "inversion" in REFRAMING_STRATEGIES
        assert "temporal_shift" in REFRAMING_STRATEGIES

    def test_strategies_remaining_all(self, engine):
        remaining = engine.reframing_strategies_remaining("g1")
        assert len(remaining) == 7

    def test_strategies_remaining_after_use(self, engine):
        engine._reframe_history["g1"] = ["inversion", "proxy"]
        remaining = engine.reframing_strategies_remaining("g1")
        assert len(remaining) == 5
        assert "inversion" not in remaining
        assert "proxy" not in remaining


# ── Root Cause Analysis (Mocked LLM) ────────────────────────────────────


class TestRootCauseAnalysis:
    async def test_root_cause_chain(self, engine):
        mock_rca = RootCauseAnalysis(
            failure_summary="Search returned empty",
            chain=[
                RootCauseLink(level=1, question="Why empty?", answer="Wrong terms"),
                RootCauseLink(level=2, question="Why wrong?", answer="No context"),
                RootCauseLink(
                    level=3, question="Why no context?",
                    answer="Domain not specified", actionable=True,
                ),
            ],
            root_cause="Domain not specified in search",
            lesson_learned="Always specify domain in searches",
            prevention_strategy="Add domain to all search prompts",
        )

        mock_create = AsyncMock(return_value=mock_rca)
        with patch("qe.runtime.persistence_engine.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await engine.analyze_root_cause(
                "g1", "Search returned empty"
            )

        assert len(result.chain) == 3
        assert result.root_cause == "Domain not specified in search"
        assert len(engine._lessons) == 1
        assert engine._lessons[0]["lesson"] == "Always specify domain in searches"

    async def test_no_lesson_when_empty(self, engine):
        mock_rca = RootCauseAnalysis(
            failure_summary="Unknown error",
            chain=[RootCauseLink(level=1, question="Why?", answer="Unknown")],
            root_cause="Unknown",
            lesson_learned="",
        )

        mock_create = AsyncMock(return_value=mock_rca)
        with patch("qe.runtime.persistence_engine.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            await engine.analyze_root_cause("g1", "error")

        assert len(engine._lessons) == 0


# ── Reframe (Mocked LLM) ────────────────────────────────────────────────


class TestReframe:
    async def test_reframe_picks_first_strategy(self, engine):
        mock_result = ReframingResult(
            original_framing="Find X",
            reframing_strategy="inversion",
            reframed_question="What would imply X?",
            reasoning="Direct search failed",
            estimated_tractability=0.6,
        )

        mock_create = AsyncMock(return_value=mock_result)
        with patch("qe.runtime.persistence_engine.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await engine.reframe("g1", "Find X")

        assert result.reframing_strategy == "inversion"
        assert "inversion" in engine._reframe_history["g1"]

    async def test_reframe_cycles_strategies(self, engine):
        engine._reframe_history["g1"] = ["inversion"]

        mock_result = ReframingResult(
            original_framing="Find X",
            reframing_strategy="proxy",
            reframed_question="What proxy measures X?",
            reasoning="Inversion tried",
        )

        mock_create = AsyncMock(return_value=mock_result)
        with patch("qe.runtime.persistence_engine.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            await engine.reframe("g1", "Find X")

        assert "proxy" in engine._reframe_history["g1"]

    async def test_reframe_explicit_strategy(self, engine):
        mock_result = ReframingResult(
            original_framing="Find X",
            reframing_strategy="temporal_shift",
            reframed_question="When was X different?",
            reasoning="Explicit strategy",
        )

        mock_create = AsyncMock(return_value=mock_result)
        with patch("qe.runtime.persistence_engine.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            result = await engine.reframe(
                "g1", "Find X", strategy="temporal_shift"
            )

        assert result.reframing_strategy == "temporal_shift"


# ── Reframe Cascade ──────────────────────────────────────────────────────


class TestReframeCascade:
    async def test_cascade_stops_on_high_tractability(self, engine):
        mock_result = ReframingResult(
            original_framing="Find X",
            reframing_strategy="inversion",
            reframed_question="What implies X?",
            reasoning="Good reframe",
            estimated_tractability=0.8,
        )

        mock_create = AsyncMock(return_value=mock_result)
        with patch("qe.runtime.persistence_engine.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            results = await engine.reframe_cascade(
                "g1", "Find X", max_reframes=3
            )

        # Should stop after 1 because tractability > 0.7
        assert len(results) == 1

    async def test_cascade_tries_multiple(self, engine):
        mock_result = ReframingResult(
            original_framing="Find X",
            reframing_strategy="inversion",
            reframed_question="What implies X?",
            reasoning="Low tractability",
            estimated_tractability=0.3,
        )

        mock_create = AsyncMock(return_value=mock_result)
        with patch("qe.runtime.persistence_engine.instructor") as mock_inst:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_inst.from_litellm.return_value = mock_client

            results = await engine.reframe_cascade(
                "g1", "Find X", max_reframes=3
            )

        assert len(results) == 3


# ── Lesson Management ────────────────────────────────────────────────────


class TestLessonManagement:
    def test_get_relevant_lessons_empty(self, engine):
        assert engine.get_relevant_lessons("any context") == []

    def test_get_relevant_lessons(self, engine):
        engine._lessons = [
            {
                "failure": "search failed",
                "root_cause": "wrong terms",
                "lesson": "specify domain",
            },
            {
                "failure": "API timeout",
                "root_cause": "rate limit",
                "lesson": "add retry logic",
            },
        ]

        results = engine.get_relevant_lessons("search query failed")
        assert len(results) >= 1
        assert results[0]["failure"] == "search failed"

    def test_get_relevant_lessons_no_match(self, engine):
        engine._lessons = [
            {"failure": "xyz", "root_cause": "abc", "lesson": "123"},
        ]
        results = engine.get_relevant_lessons("completely unrelated words")
        assert len(results) == 0


# ── Cleanup & Status ─────────────────────────────────────────────────────


class TestCleanup:
    def test_clear_goal(self, engine):
        engine._reframe_history["g1"] = ["inversion"]
        engine.clear_goal("g1")
        assert "g1" not in engine._reframe_history

    def test_clear_nonexistent(self, engine):
        engine.clear_goal("nonexistent")

    def test_status(self, engine):
        engine._lessons.append({"lesson": "test"})
        engine._reframe_history["g1"] = ["inversion"]
        s = engine.status()
        assert s["total_lessons"] == 1
        assert s["active_reframe_histories"] == 1

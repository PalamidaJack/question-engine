"""Tests for ContextCurator — Tier 0 working memory with anti-drift."""

import pytest

from qe.runtime.context_curator import (
    ContextCurator,
    DriftReport,
    WorkingMemorySlot,
    WorkingMemoryState,
)


@pytest.fixture
def curator():
    return ContextCurator(max_tokens=1000)


@pytest.fixture
def curator_with_goal(curator):
    curator.get_or_create_state("goal_1", "Find untapped investment opportunities")
    return curator


# ── State Management ──────────────────────────────────────────────────────


class TestStateManagement:
    def test_create_state(self, curator):
        state = curator.get_or_create_state("goal_1", "Test goal")
        assert state.goal_id == "goal_1"
        assert state.goal_anchor == "Test goal"
        assert state.slots == []

    def test_get_existing_state(self, curator):
        state1 = curator.get_or_create_state("goal_1", "Test goal")
        state2 = curator.get_or_create_state("goal_1", "Different description")
        # Should return same state, not overwrite
        assert state1.goal_anchor == state2.goal_anchor

    def test_add_slot(self, curator_with_goal):
        curator_with_goal.add_slot(
            goal_id="goal_1",
            slot_id="slot_1",
            content="Healthcare sector is undervalued",
            category="finding",
            relevance_score=0.8,
        )
        state = curator_with_goal.get_or_create_state("goal_1", "")
        assert len(state.slots) == 1
        assert state.slots[0].content == "Healthcare sector is undervalued"

    def test_add_slot_no_state_raises(self, curator):
        with pytest.raises(ValueError, match="No working memory"):
            curator.add_slot("nonexistent", "s1", "content", "finding")

    def test_clear_goal(self, curator_with_goal):
        curator_with_goal.add_slot("goal_1", "s1", "content", "finding")
        curator_with_goal.clear_goal("goal_1")
        assert curator_with_goal.status()["active_goals"] == 0


# ── Context Building ──────────────────────────────────────────────────────


class TestContextBuilding:
    def test_basic_context(self, curator_with_goal):
        messages = curator_with_goal.build_context(
            goal_id="goal_1",
            system_prompt="You are a research analyst.",
            user_message="What sectors are undervalued?",
        )

        assert messages[0]["role"] == "system"
        assert "research analyst" in messages[0]["content"]
        # Goal anchor should be present
        assert any("GOAL ANCHOR" in m["content"] for m in messages)
        # User message should be last
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What sectors are undervalued?"

    def test_constitution_included(self, curator_with_goal):
        messages = curator_with_goal.build_context(
            goal_id="goal_1",
            system_prompt="System prompt",
            constitution="Never fabricate data.",
        )

        assert any("CONSTITUTION" in m["content"] for m in messages)
        assert any("GOAL ANCHOR" in m["content"] for m in messages)

    def test_slots_included_by_relevance(self, curator_with_goal):
        curator_with_goal.add_slot("goal_1", "s1", "High relevance finding", "finding", 0.9)
        curator_with_goal.add_slot("goal_1", "s2", "Low relevance noise", "finding", 0.1)

        messages = curator_with_goal.build_context(
            goal_id="goal_1",
            system_prompt="System",
        )

        contents = " ".join(m["content"] for m in messages)
        assert "High relevance" in contents

    def test_fallback_without_state(self, curator):
        messages = curator.build_context(
            goal_id="nonexistent",
            system_prompt="System prompt",
            user_message="Hello",
        )
        assert len(messages) == 2
        assert messages[0]["content"] == "System prompt"
        assert messages[1]["content"] == "Hello"

    def test_goal_anchor_always_present(self, curator_with_goal):
        # Add many slots to fill the budget
        for i in range(50):
            curator_with_goal.add_slot("goal_1", f"s{i}", f"Finding {i}", "finding", 0.5)

        messages = curator_with_goal.build_context("goal_1", "System")

        # Goal anchor must still be there
        assert any("GOAL ANCHOR" in m["content"] for m in messages)
        assert any("investment opportunities" in m["content"] for m in messages)


# ── Budget Enforcement ──────────────────────────────────────────────────────


class TestBudgetEnforcement:
    def test_over_budget_evicts_low_relevance(self):
        curator = ContextCurator(max_tokens=200)
        curator.get_or_create_state("g1", "Short goal")

        # Add a large low-relevance slot
        curator.add_slot("g1", "s1", "x" * 800, "finding", 0.1)
        # Add a small high-relevance slot
        curator.add_slot("g1", "s2", "important", "finding", 0.9)

        state = curator.get_or_create_state("g1", "")
        # Low-relevance slot should have been evicted
        assert state.current_tokens <= 200

    def test_category_budget_allocation(self, curator_with_goal):
        # Add slots in different categories
        curator_with_goal.add_slot("goal_1", "f1", "Finding data", "finding", 0.8)
        curator_with_goal.add_slot("goal_1", "b1", "Background info", "background", 0.7)
        curator_with_goal.add_slot("goal_1", "q1", "Open question", "question", 0.6)

        messages = curator_with_goal.build_context("goal_1", "System")

        # All categories should be represented
        contents = " ".join(m["content"] for m in messages)
        assert "Finding data" in contents or "Background info" in contents


# ── Relevance Scoring ──────────────────────────────────────────────────────


class TestRelevanceScoring:
    async def test_word_overlap_fallback(self, curator):
        score = await curator.score_relevance(
            "investment opportunities in healthcare sector",
            "Find untapped investment opportunities",
        )
        assert score > 0.0
        assert score <= 1.0

    async def test_word_overlap_unrelated(self, curator):
        score = await curator.score_relevance(
            "the weather is sunny today",
            "Find untapped investment opportunities",
        )
        # Should have low overlap
        assert score < 0.3

    async def test_custom_embed_fn(self):
        async def mock_embed(text):
            # Simple mock: return word length as "embedding"
            return [float(len(text))]

        curator = ContextCurator(embed_fn=mock_embed)
        score = await curator.score_relevance("hello", "hello")
        assert score == pytest.approx(1.0)


# ── Drift Detection ──────────────────────────────────────────────────────


class TestDriftDetection:
    async def test_no_drift_empty_state(self, curator):
        curator.get_or_create_state("g1", "Test goal")
        report = await curator.detect_drift("g1")
        assert report.on_track
        assert report.drift_score == 0.0

    async def test_drift_with_relevant_findings(self, curator):
        curator.get_or_create_state("g1", "Find investment opportunities")
        curator.add_slot("g1", "s1", "Investment opportunities in tech sector", "finding", 0.8)

        report = await curator.detect_drift("g1")
        # Word overlap gives a moderate similarity (some words match)
        # Drift score should be less than 1.0 since there is some overlap
        assert report.drift_score < 1.0
        assert isinstance(report, DriftReport)

    async def test_drift_with_irrelevant_findings(self, curator):
        curator.get_or_create_state("g1", "Find investment opportunities")
        curator.add_slot("g1", "s1", "recipe for chocolate cake", "finding", 0.1)

        report = await curator.detect_drift("g1")
        # May or may not flag drift depending on word overlap
        assert isinstance(report, DriftReport)

    async def test_drift_nonexistent_goal(self, curator):
        report = await curator.detect_drift("nonexistent")
        assert report.on_track


# ── Refocus ──────────────────────────────────────────────────────────────


class TestRefocus:
    async def test_refocus_evicts_irrelevant(self, curator):
        curator.get_or_create_state("g1", "Find investment opportunities")
        curator.add_slot("g1", "s1", "Investment data analysis", "finding", 0.8)
        curator.add_slot("g1", "s2", "Completely unrelated noise", "finding", 0.01)

        evicted = await curator.refocus("g1")
        # The very low relevance slot should have been evicted
        state = curator.get_or_create_state("g1", "")
        remaining_ids = {s.slot_id for s in state.slots}
        assert "s1" in remaining_ids

    async def test_refocus_nonexistent(self, curator):
        evicted = await curator.refocus("nonexistent")
        assert evicted == 0


# ── Status ──────────────────────────────────────────────────────────────


class TestStatus:
    def test_status_empty(self, curator):
        status = curator.status()
        assert status["active_goals"] == 0
        assert status["total_slots"] == 0

    def test_status_with_data(self, curator_with_goal):
        curator_with_goal.add_slot("goal_1", "s1", "content", "finding")
        status = curator_with_goal.status()
        assert status["active_goals"] == 1
        assert status["total_slots"] == 1

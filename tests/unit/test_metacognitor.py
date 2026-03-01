"""Tests for Metacognitor — self-awareness and approach management."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.cognition import (
    ApproachAssessment,
    CapabilityProfile,
    ToolCombinationSuggestion,
)
from qe.runtime.metacognitor import Metacognitor


@pytest.fixture
def meta():
    return Metacognitor(model="test-model")


@pytest.fixture
def meta_with_caps(meta):
    meta.register_capability(CapabilityProfile(
        tool_name="web_search",
        description="Search the web",
        domains=["general"],
        limitations=["no auth sites"],
    ))
    meta.register_capability(CapabilityProfile(
        tool_name="code_exec",
        description="Execute Python code",
        domains=["computation"],
    ))
    return meta


# ── Capability Management ────────────────────────────────────────────────


class TestCapabilityManagement:
    def test_register_capability(self, meta):
        meta.register_capability(CapabilityProfile(
            tool_name="web_search", description="Search",
        ))
        assert "web_search" in meta._capabilities

    def test_register_bulk(self, meta):
        meta.register_capabilities_bulk(
            ["tool_a", "tool_b"],
            {"tool_a": "Does A", "tool_b": "Does B"},
        )
        assert len(meta._capabilities) == 2
        assert meta._capabilities["tool_a"].description == "Does A"

    def test_register_bulk_no_descriptions(self, meta):
        meta.register_capabilities_bulk(["tool_x"])
        assert "tool_x" in meta._capabilities
        assert "Tool: tool_x" in meta._capabilities["tool_x"].description

    def test_identify_gap(self, meta):
        gap = meta.identify_gap("No database access", workaround="Use APIs")
        assert gap.description == "No database access"
        assert gap.workaround == "Use APIs"
        assert len(meta._gaps) == 1

    def test_capabilities_summary(self, meta_with_caps):
        summary = meta_with_caps.get_capabilities_summary()
        assert "web_search" in summary
        assert "code_exec" in summary
        assert "no auth sites" in summary

    def test_capabilities_summary_empty(self, meta):
        assert "No tools registered" in meta.get_capabilities_summary()

    def test_gaps_summary(self, meta):
        meta.identify_gap("No DB access")
        summary = meta.get_gaps_summary()
        assert "No DB access" in summary

    def test_gaps_summary_empty(self, meta):
        assert "No known gaps" in meta.get_gaps_summary()


# ── Approach Tree ────────────────────────────────────────────────────────


class TestApproachTree:
    def test_init_tree(self, meta):
        root = meta.init_approach_tree("g1", "Try web search")
        assert root.approach_description == "Try web search"
        assert root.status == "in_progress"
        assert meta.approach_count("g1") == 1

    def test_record_success(self, meta):
        meta.init_approach_tree("g1", "approach A")
        meta.record_approach_outcome("g1", success=True)
        node = meta.get_current_node("g1")
        assert node.status == "succeeded"

    def test_record_failure(self, meta):
        meta.init_approach_tree("g1", "approach A")
        meta.record_approach_outcome("g1", success=False, failure_reason="timeout")
        node = meta.get_current_node("g1")
        assert node.status == "failed"
        assert node.failure_reason == "timeout"

    def test_add_child(self, meta):
        meta.init_approach_tree("g1", "parent")
        child = meta.add_child_approach("g1", "child approach")
        assert child.status == "in_progress"
        assert child.parent_id is not None
        assert meta.approach_count("g1") == 2
        assert meta.get_current_node("g1").approach_description == "child approach"

    def test_add_sibling(self, meta):
        root = meta.init_approach_tree("g1", "root")
        meta.add_child_approach("g1", "child A")
        meta.record_approach_outcome("g1", success=False, failure_reason="failed")
        sibling = meta.add_sibling_approach("g1", "child B")
        assert sibling.parent_id == root.node_id
        assert meta.approach_count("g1") == 3

    def test_add_sibling_at_root(self, meta):
        root = meta.init_approach_tree("g1", "root approach")
        meta.record_approach_outcome("g1", success=False)
        sibling = meta.add_sibling_approach("g1", "alternative root")
        # At root level, sibling's parent is root itself
        assert sibling.parent_id == root.node_id

    def test_has_untried_false(self, meta):
        meta.init_approach_tree("g1", "approach")
        # Root is in_progress, not untried
        assert not meta.has_untried_approaches("g1")

    def test_all_approaches_exhausted(self, meta):
        meta.init_approach_tree("g1", "A")
        meta.record_approach_outcome("g1", success=False)
        meta.add_sibling_approach("g1", "B")
        meta.record_approach_outcome("g1", success=False)
        assert meta.all_approaches_exhausted("g1")

    def test_not_exhausted_when_in_progress(self, meta):
        meta.init_approach_tree("g1", "A")
        assert not meta.all_approaches_exhausted("g1")

    def test_exhausted_empty_tree(self, meta):
        assert not meta.all_approaches_exhausted("nonexistent")

    def test_tried_approaches_summary(self, meta):
        meta.init_approach_tree("g1", "approach A")
        meta.record_approach_outcome("g1", success=False, failure_reason="timeout")
        summary = meta.get_tried_approaches_summary("g1")
        assert "[FAIL]" in summary
        assert "timeout" in summary

    def test_tried_approaches_empty(self, meta):
        summary = meta.get_tried_approaches_summary("nonexistent")
        assert "No approaches" in summary

    def test_get_current_node_none(self, meta):
        assert meta.get_current_node("nonexistent") is None

    def test_record_outcome_no_tree(self, meta):
        # Should not raise
        meta.record_approach_outcome("nonexistent", success=True)

    def test_clear_goal(self, meta):
        meta.init_approach_tree("g1", "A")
        meta.clear_goal("g1")
        assert meta.approach_count("g1") == 0


# ── LLM-Powered Reasoning (Mocked) ──────────────────────────────────────


class TestLLMReasoning:
    async def test_suggest_next_approach(self, meta_with_caps):
        meta_with_caps.init_approach_tree("g1", "web search")
        meta_with_caps.record_approach_outcome("g1", False, "empty results")

        mock_assessment = ApproachAssessment(
            recommended_approach="Use code execution to compute from public data",
            reasoning="Web search failed, try computation",
            tools_needed=["code_exec"],
            estimated_success_probability=0.7,
        )

        mock_create = AsyncMock(return_value=mock_assessment)
        with patch("qe.runtime.metacognitor.instructor") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_instructor.from_litellm.return_value = mock_client

            result = await meta_with_caps.suggest_next_approach(
                "g1", "Find investment opportunities", "empty results"
            )

        assert result.recommended_approach == "Use code execution to compute from public data"
        assert result.estimated_success_probability == 0.7

    async def test_suggest_tool_combinations(self, meta_with_caps):
        mock_suggestions = [
            ToolCombinationSuggestion(
                description="Search + compute",
                tool_sequence=["web_search", "code_exec"],
                reasoning="Combine sources",
                novelty_score=0.8,
            )
        ]

        class MockResult:
            suggestions = mock_suggestions

        mock_create = AsyncMock(return_value=MockResult())
        with patch("qe.runtime.metacognitor.instructor") as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create = mock_create
            mock_instructor.from_litellm.return_value = mock_client

            result = await meta_with_caps.suggest_tool_combinations(
                "Can't find revenue data"
            )

        assert len(result) == 1
        assert result[0].novelty_score == 0.8


# ── Status ───────────────────────────────────────────────────────────────


class TestStatus:
    def test_status(self, meta_with_caps):
        meta_with_caps.init_approach_tree("g1", "A")
        meta_with_caps.identify_gap("No DB")
        s = meta_with_caps.status()
        assert s["registered_capabilities"] == 2
        assert s["known_gaps"] == 1
        assert s["active_approach_trees"] == 1

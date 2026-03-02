"""Tests for ScoutAnalyzer — LLM-powered relevance and feasibility analysis."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.scout import ImprovementIdea, ScoutFinding
from qe.services.scout.analyzer import ScoutAnalyzer, _RelevanceAssessment

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_finding(
    *,
    finding_id: str = "fnd_test001",
    title: str = "Cool library",
    url: str = "https://example.com/article",
    snippet: str = "A useful snippet about caching",
    full_content: str = "Full article about advanced caching strategies for Python.",
    source_type: str = "blog",
) -> ScoutFinding:
    return ScoutFinding(
        finding_id=finding_id,
        title=title,
        url=url,
        snippet=snippet,
        full_content=full_content,
        source_type=source_type,
    )


def _make_assessment(
    *,
    relevance_score: float = 0.8,
    feasibility_score: float = 0.7,
    impact_score: float = 0.6,
    title: str = "Add caching layer",
    description: str = "Implement a caching layer for LLM responses",
    category: str = "performance",
    rationale: str = "Reduces latency and cost",
    affected_files: list[str] | None = None,
) -> _RelevanceAssessment:
    return _RelevanceAssessment(
        title=title,
        description=description,
        category=category,
        relevance_score=relevance_score,
        feasibility_score=feasibility_score,
        impact_score=impact_score,
        rationale=rationale,
        affected_files=affected_files or ["src/qe/runtime/engram_cache.py"],
    )


def _patch_instructor(assessment: _RelevanceAssessment):
    """Return a patch context manager that mocks the instructor client."""
    mock_inst = MagicMock()
    mock_client = AsyncMock()
    mock_inst.from_litellm.return_value = mock_client
    mock_client.chat.completions.create = AsyncMock(return_value=assessment)
    return patch("qe.services.scout.analyzer.instructor", mock_inst), mock_client


def _patch_flags(enabled: bool = True):
    """Return a patch context manager for the feature flag store."""
    mock_flags = MagicMock()
    mock_flags.return_value.is_enabled.return_value = enabled
    return patch("qe.services.scout.analyzer.get_flag_store", mock_flags), mock_flags


# ── Tests ────────────────────────────────────────────────────────────────


class TestAnalyzeReturnsIdeasAboveThreshold:
    """test_analyze_returns_ideas_above_threshold"""

    @pytest.mark.asyncio
    async def test_ideas_above_threshold_are_returned(self):
        """Findings with composite score >= threshold should appear in results."""
        # relevance=0.8, feasibility=0.7, impact=0.6
        # composite = 0.8*0.4 + 0.7*0.3 + 0.6*0.3 = 0.32 + 0.21 + 0.18 = 0.71
        assessment = _make_assessment()
        inst_patch, mock_client = _patch_instructor(assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.5)
            findings = [_make_finding()]
            ideas = await analyzer.analyze(findings)

        assert len(ideas) == 1
        assert isinstance(ideas[0], ImprovementIdea)
        assert ideas[0].title == "Add caching layer"
        assert ideas[0].composite_score >= 0.5

    @pytest.mark.asyncio
    async def test_multiple_findings_all_above(self):
        """All findings above threshold should be returned."""
        assessment = _make_assessment(
            relevance_score=0.9, feasibility_score=0.9, impact_score=0.9,
        )
        inst_patch, _ = _patch_instructor(assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.5)
            findings = [
                _make_finding(finding_id="fnd_a"),
                _make_finding(finding_id="fnd_b"),
            ]
            ideas = await analyzer.analyze(findings)

        assert len(ideas) == 2


class TestAnalyzeFiltersBelowThreshold:
    """test_analyze_filters_below_threshold"""

    @pytest.mark.asyncio
    async def test_ideas_below_threshold_are_filtered_out(self):
        """Findings with composite score < threshold should be excluded."""
        # relevance=0.1, feasibility=0.1, impact=0.1
        # composite = 0.1*0.4 + 0.1*0.3 + 0.1*0.3 = 0.04 + 0.03 + 0.03 = 0.10
        assessment = _make_assessment(
            relevance_score=0.1, feasibility_score=0.1, impact_score=0.1,
        )
        inst_patch, _ = _patch_instructor(assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.5)
            findings = [_make_finding()]
            ideas = await analyzer.analyze(findings)

        assert len(ideas) == 0

    @pytest.mark.asyncio
    async def test_mixed_scores_filters_correctly(self):
        """Only above-threshold findings should survive."""
        high = _make_assessment(
            relevance_score=0.9, feasibility_score=0.9, impact_score=0.9,
        )
        low = _make_assessment(
            relevance_score=0.1, feasibility_score=0.1, impact_score=0.1,
        )

        call_count = 0
        assessments = [high, low]

        mock_inst = MagicMock()
        mock_client = AsyncMock()
        mock_inst.from_litellm.return_value = mock_client

        async def _side_effect(**kwargs):
            nonlocal call_count
            result = assessments[call_count]
            call_count += 1
            return result

        mock_client.chat.completions.create = AsyncMock(side_effect=_side_effect)
        flag_patch, _ = _patch_flags(enabled=True)

        with patch("qe.services.scout.analyzer.instructor", mock_inst), flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.5)
            findings = [
                _make_finding(finding_id="fnd_high"),
                _make_finding(finding_id="fnd_low"),
            ]
            ideas = await analyzer.analyze(findings)

        assert len(ideas) == 1
        assert ideas[0].composite_score >= 0.5


class TestAnalyzeDisabledFlagReturnsEmpty:
    """test_analyze_disabled_flag_returns_empty"""

    @pytest.mark.asyncio
    async def test_disabled_feature_flag_returns_empty_list(self):
        """When innovation_scout flag is disabled, analyze() returns []."""
        flag_patch, mock_flags = _patch_flags(enabled=False)

        with flag_patch:
            analyzer = ScoutAnalyzer(model="test-model")
            findings = [_make_finding()]
            ideas = await analyzer.analyze(findings)

        assert ideas == []
        mock_flags.return_value.is_enabled.assert_called_with("innovation_scout")

    @pytest.mark.asyncio
    async def test_disabled_flag_does_not_call_llm(self):
        """LLM should not be invoked when the flag is off."""
        inst_patch, mock_client = _patch_instructor(_make_assessment())
        flag_patch, _ = _patch_flags(enabled=False)

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model")
            await analyzer.analyze([_make_finding()])

        mock_client.chat.completions.create.assert_not_awaited()


class TestCompositeScoreCalculation:
    """test_composite_score_calculation — verify 0.4/0.3/0.3 weights."""

    @pytest.mark.asyncio
    async def test_composite_score_weights(self):
        """composite = relevance*0.4 + feasibility*0.3 + impact*0.3"""
        assessment = _make_assessment(
            relevance_score=1.0, feasibility_score=0.0, impact_score=0.0,
        )
        inst_patch, _ = _patch_instructor(assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            ideas = await analyzer.analyze([_make_finding()])

        # 1.0*0.4 + 0.0*0.3 + 0.0*0.3 = 0.4
        assert len(ideas) == 1
        assert ideas[0].composite_score == pytest.approx(0.4, abs=1e-3)

    @pytest.mark.asyncio
    async def test_composite_score_feasibility_only(self):
        """When only feasibility is 1.0, composite = 0.3."""
        assessment = _make_assessment(
            relevance_score=0.0, feasibility_score=1.0, impact_score=0.0,
        )
        inst_patch, _ = _patch_instructor(assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            ideas = await analyzer.analyze([_make_finding()])

        assert len(ideas) == 1
        assert ideas[0].composite_score == pytest.approx(0.3, abs=1e-3)

    @pytest.mark.asyncio
    async def test_composite_score_impact_only(self):
        """When only impact is 1.0, composite = 0.3."""
        assessment = _make_assessment(
            relevance_score=0.0, feasibility_score=0.0, impact_score=1.0,
        )
        inst_patch, _ = _patch_instructor(assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            ideas = await analyzer.analyze([_make_finding()])

        assert len(ideas) == 1
        assert ideas[0].composite_score == pytest.approx(0.3, abs=1e-3)

    @pytest.mark.asyncio
    async def test_composite_score_all_ones(self):
        """When all scores are 1.0, composite = 1.0."""
        assessment = _make_assessment(
            relevance_score=1.0, feasibility_score=1.0, impact_score=1.0,
        )
        inst_patch, _ = _patch_instructor(assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            ideas = await analyzer.analyze([_make_finding()])

        assert len(ideas) == 1
        assert ideas[0].composite_score == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.asyncio
    async def test_composite_score_mixed(self):
        """Specific mixed values: 0.5*0.4 + 0.8*0.3 + 0.3*0.3 = 0.2+0.24+0.09 = 0.53."""
        assessment = _make_assessment(
            relevance_score=0.5, feasibility_score=0.8, impact_score=0.3,
        )
        inst_patch, _ = _patch_instructor(assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            ideas = await analyzer.analyze([_make_finding()])

        assert len(ideas) == 1
        assert ideas[0].composite_score == pytest.approx(0.53, abs=1e-3)


class TestCategoryValidationDefaultsToOther:
    """test_category_validation_defaults_to_other"""

    @pytest.mark.asyncio
    async def test_invalid_category_defaults_to_other(self):
        """Category not in the valid set should be replaced with 'other'."""
        assessment = _make_assessment(category="banana")
        inst_patch, _ = _patch_instructor(assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            ideas = await analyzer.analyze([_make_finding()])

        assert len(ideas) == 1
        assert ideas[0].category == "other"

    @pytest.mark.asyncio
    async def test_valid_category_preserved(self):
        """All valid categories should be kept as-is."""
        valid_categories = [
            "performance", "feature", "refactor", "testing",
            "security", "dependency", "pattern", "model", "other",
        ]
        for cat in valid_categories:
            assessment = _make_assessment(category=cat)
            inst_patch, _ = _patch_instructor(assessment)
            flag_patch, _ = _patch_flags(enabled=True)

            with inst_patch, flag_patch:
                analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
                ideas = await analyzer.analyze([_make_finding()])

            assert len(ideas) == 1, f"Expected 1 idea for category {cat}"
            assert ideas[0].category == cat


class TestUpdateThresholdClamps:
    """test_update_threshold_clamps — boundary values."""

    def test_update_threshold_within_range(self):
        analyzer = ScoutAnalyzer(model="test-model")
        analyzer.update_threshold(0.7)
        assert analyzer._min_composite_score == pytest.approx(0.7)

    def test_update_threshold_clamps_below_minimum(self):
        """Values below 0.1 should be clamped to 0.1."""
        analyzer = ScoutAnalyzer(model="test-model")
        analyzer.update_threshold(0.0)
        assert analyzer._min_composite_score == pytest.approx(0.1)

        analyzer.update_threshold(-5.0)
        assert analyzer._min_composite_score == pytest.approx(0.1)

    def test_update_threshold_clamps_above_maximum(self):
        """Values above 0.9 should be clamped to 0.9."""
        analyzer = ScoutAnalyzer(model="test-model")
        analyzer.update_threshold(1.0)
        assert analyzer._min_composite_score == pytest.approx(0.9)

        analyzer.update_threshold(99.0)
        assert analyzer._min_composite_score == pytest.approx(0.9)

    def test_update_threshold_boundary_exact(self):
        """Exact boundary values 0.1 and 0.9 should be accepted."""
        analyzer = ScoutAnalyzer(model="test-model")

        analyzer.update_threshold(0.1)
        assert analyzer._min_composite_score == pytest.approx(0.1)

        analyzer.update_threshold(0.9)
        assert analyzer._min_composite_score == pytest.approx(0.9)


class TestSetApprovedPatternsStored:
    """test_set_approved_patterns_stored"""

    def test_patterns_stored_on_instance(self):
        analyzer = ScoutAnalyzer(model="test-model")
        patterns = ["caching", "async", "testing"]
        analyzer.set_approved_patterns(patterns)
        assert analyzer._approved_patterns == ["caching", "async", "testing"]

    def test_patterns_initially_empty(self):
        analyzer = ScoutAnalyzer(model="test-model")
        assert analyzer._approved_patterns == []

    def test_set_patterns_overwrites_previous(self):
        analyzer = ScoutAnalyzer(model="test-model")
        analyzer.set_approved_patterns(["old_pattern"])
        analyzer.set_approved_patterns(["new_pattern"])
        assert analyzer._approved_patterns == ["new_pattern"]

    @pytest.mark.asyncio
    async def test_approved_patterns_included_in_prompt(self):
        """When patterns are set, they should appear in the LLM system prompt."""
        assessment = _make_assessment()
        mock_inst = MagicMock()
        mock_client = AsyncMock()
        mock_inst.from_litellm.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with patch("qe.services.scout.analyzer.instructor", mock_inst), flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            analyzer.set_approved_patterns(["caching", "async patterns"])
            await analyzer.analyze([_make_finding()])

        # Verify system message includes approved patterns
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        system_content = messages[0]["content"]
        assert "caching" in system_content
        assert "async patterns" in system_content
        assert "previously approved" in system_content


class TestAnalyzeEmptyContentSkipped:
    """test_analyze_empty_content_skipped"""

    @pytest.mark.asyncio
    async def test_empty_full_content_and_snippet_skipped(self):
        """Findings with no content (empty strings) should return None from _analyze_one."""
        flag_patch, _ = _patch_flags(enabled=True)
        inst_patch, mock_client = _patch_instructor(_make_assessment())

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            finding = _make_finding(snippet="", full_content="")
            ideas = await analyzer.analyze([finding])

        assert ideas == []
        mock_client.chat.completions.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_only_content_skipped(self):
        """Content that is only whitespace should also be skipped."""
        flag_patch, _ = _patch_flags(enabled=True)
        inst_patch, mock_client = _patch_instructor(_make_assessment())

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            finding = _make_finding(snippet="   \n\t  ", full_content="")
            ideas = await analyzer.analyze([finding])

        assert ideas == []
        mock_client.chat.completions.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_full_content_used_over_snippet(self):
        """When full_content is available, it should be preferred over snippet."""
        assessment = _make_assessment()
        mock_inst = MagicMock()
        mock_client = AsyncMock()
        mock_inst.from_litellm.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with patch("qe.services.scout.analyzer.instructor", mock_inst), flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            finding = _make_finding(
                snippet="Short snippet",
                full_content="Full detailed article about caching",
            )
            await analyzer.analyze([finding])

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_content = messages[1]["content"]
        assert "Full detailed article about caching" in user_content

    @pytest.mark.asyncio
    async def test_snippet_used_when_full_content_empty(self):
        """When full_content is empty, snippet should be used."""
        assessment = _make_assessment()
        mock_inst = MagicMock()
        mock_client = AsyncMock()
        mock_inst.from_litellm.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with patch("qe.services.scout.analyzer.instructor", mock_inst), flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            finding = _make_finding(
                snippet="Short snippet fallback",
                full_content="",
            )
            await analyzer.analyze([finding])

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        user_content = messages[1]["content"]
        assert "Short snippet fallback" in user_content


class TestAnalyzeOneExceptionHandling:
    """Additional edge case: LLM failures are caught gracefully."""

    @pytest.mark.asyncio
    async def test_llm_exception_logs_and_continues(self):
        """If _analyze_one raises, the finding is skipped (not propagated)."""
        mock_inst = MagicMock()
        mock_client = AsyncMock()
        mock_inst.from_litellm.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("LLM unavailable"),
        )
        flag_patch, _ = _patch_flags(enabled=True)

        with patch("qe.services.scout.analyzer.instructor", mock_inst), flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            findings = [_make_finding()]
            ideas = await analyzer.analyze(findings)

        assert ideas == []

    @pytest.mark.asyncio
    async def test_idea_fields_populated_correctly(self):
        """Verify all fields on the returned ImprovementIdea are populated."""
        assessment = _make_assessment(
            title="My Title",
            description="My Description",
            category="security",
            relevance_score=0.9,
            feasibility_score=0.8,
            impact_score=0.7,
            rationale="Because reasons",
            affected_files=["src/a.py", "src/b.py"],
        )
        inst_patch, _ = _patch_instructor(assessment)
        flag_patch, _ = _patch_flags(enabled=True)

        with inst_patch, flag_patch:
            analyzer = ScoutAnalyzer(model="test-model", min_composite_score=0.1)
            finding = _make_finding(
                finding_id="fnd_xyz",
                url="https://example.com/security",
            )
            ideas = await analyzer.analyze([finding])

        assert len(ideas) == 1
        idea = ideas[0]
        assert idea.finding_id == "fnd_xyz"
        assert idea.title == "My Title"
        assert idea.description == "My Description"
        assert idea.category == "security"
        assert idea.relevance_score == pytest.approx(0.9)
        assert idea.feasibility_score == pytest.approx(0.8)
        assert idea.impact_score == pytest.approx(0.7)
        # composite = 0.9*0.4 + 0.8*0.3 + 0.7*0.3 = 0.36 + 0.24 + 0.21 = 0.81
        assert idea.composite_score == pytest.approx(0.81, abs=1e-3)
        assert idea.source_url == "https://example.com/security"
        assert idea.rationale == "Because reasons"
        assert idea.affected_files == ["src/a.py", "src/b.py"]
        assert idea.idea_id.startswith("idea_")

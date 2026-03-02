"""Tests for scout pipeline orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.models.scout import (
    CodeChange,
    ImprovementIdea,
    ScoutFinding,
    TestResult,
)
from qe.services.scout.pipeline import ScoutPipeline


def _make_finding(**overrides):
    defaults = {
        "finding_id": "fnd_test1",
        "url": "https://example.com/article",
        "title": "Test Article",
        "snippet": "Some snippet",
        "full_content": "Full content here",
        "source_type": "blog",
    }
    defaults.update(overrides)
    return ScoutFinding(**defaults)


def _make_idea(**overrides):
    defaults = {
        "idea_id": "idea_test1",
        "finding_id": "fnd_test1",
        "title": "Add retry logic",
        "description": "Add exponential backoff",
        "category": "performance",
        "relevance_score": 0.8,
        "feasibility_score": 0.7,
        "impact_score": 0.6,
        "composite_score": 0.71,
        "source_url": "https://example.com",
        "rationale": "Improves reliability",
    }
    defaults.update(overrides)
    return ImprovementIdea(**defaults)


def _make_pipeline(**overrides):
    defaults = {
        "source_manager": AsyncMock(),
        "analyzer": AsyncMock(),
        "codegen": AsyncMock(),
        "sandbox": AsyncMock(),
        "scout_store": AsyncMock(),
        "bus": MagicMock(),
    }
    defaults.update(overrides)
    return ScoutPipeline(**defaults)


@pytest.mark.asyncio
async def test_run_cycle_basic_flow():
    """Full cycle: sources → analysis → codegen → sandbox → submit."""
    finding = _make_finding()
    idea = _make_idea()
    test_result = TestResult(passed=True, total_tests=100, passed_tests=100)

    pipeline = _make_pipeline()
    pipeline._sources.generate_queries = AsyncMock(return_value=["query1"])
    pipeline._sources.search = AsyncMock(return_value=[finding])
    pipeline._sources.fetch_content = AsyncMock(return_value=[finding])
    pipeline._store.has_finding_url = AsyncMock(return_value=False)
    pipeline._analyzer.analyze = AsyncMock(return_value=[idea])
    pipeline._codegen.read_affected_files = AsyncMock(return_value={})
    pipeline._codegen.generate = AsyncMock(
        return_value=(
            [CodeChange(file_path="src/test.py", change_type="modify")],
            "impact",
            "risk",
            "rollback",
        )
    )
    pipeline._sandbox.create_worktree = AsyncMock(
        return_value=("/tmp/wt", "scout/prop_test")
    )
    pipeline._sandbox.apply_changes = AsyncMock()
    pipeline._sandbox.run_tests = AsyncMock(return_value=test_result)
    pipeline._sandbox.capture_diffs = AsyncMock(return_value=[
        CodeChange(file_path="src/test.py", change_type="modify", diff="@@ -1 +1 @@"),
    ])

    summary = await pipeline.run_cycle()

    assert summary["findings_count"] == 1
    assert summary["ideas_count"] == 1
    assert summary["proposals_count"] == 1
    assert summary["duration_s"] >= 0


@pytest.mark.asyncio
async def test_run_cycle_deduplicates_findings():
    """Findings with URLs already in store are skipped."""
    finding = _make_finding()

    pipeline = _make_pipeline()
    pipeline._sources.generate_queries = AsyncMock(return_value=["q"])
    pipeline._sources.search = AsyncMock(return_value=[finding])
    pipeline._sources.fetch_content = AsyncMock(return_value=[])
    pipeline._store.has_finding_url = AsyncMock(return_value=True)  # Already seen
    pipeline._analyzer.analyze = AsyncMock(return_value=[])

    summary = await pipeline.run_cycle()
    assert summary["findings_count"] == 0


@pytest.mark.asyncio
async def test_run_cycle_test_failure_skips_proposal():
    """If tests fail, proposal is not submitted."""
    finding = _make_finding()
    idea = _make_idea()
    test_result = TestResult(passed=False, total_tests=100, failed_tests=5, passed_tests=95)

    pipeline = _make_pipeline()
    pipeline._sources.generate_queries = AsyncMock(return_value=["q"])
    pipeline._sources.search = AsyncMock(return_value=[finding])
    pipeline._sources.fetch_content = AsyncMock(return_value=[finding])
    pipeline._store.has_finding_url = AsyncMock(return_value=False)
    pipeline._analyzer.analyze = AsyncMock(return_value=[idea])
    pipeline._codegen.read_affected_files = AsyncMock(return_value={})
    pipeline._codegen.generate = AsyncMock(
        return_value=(
            [CodeChange(file_path="src/test.py", change_type="modify")],
            "impact", "risk", "rollback",
        )
    )
    pipeline._sandbox.create_worktree = AsyncMock(
        return_value=("/tmp/wt", "scout/prop_test")
    )
    pipeline._sandbox.apply_changes = AsyncMock()
    pipeline._sandbox.run_tests = AsyncMock(return_value=test_result)
    pipeline._sandbox.cleanup_worktree = AsyncMock()

    summary = await pipeline.run_cycle()
    assert summary["proposals_count"] == 0
    pipeline._sandbox.cleanup_worktree.assert_called_once()


@pytest.mark.asyncio
async def test_run_cycle_no_changes_generated():
    """If codegen returns no changes, idea is skipped."""
    finding = _make_finding()
    idea = _make_idea()

    pipeline = _make_pipeline()
    pipeline._sources.generate_queries = AsyncMock(return_value=["q"])
    pipeline._sources.search = AsyncMock(return_value=[finding])
    pipeline._sources.fetch_content = AsyncMock(return_value=[finding])
    pipeline._store.has_finding_url = AsyncMock(return_value=False)
    pipeline._analyzer.analyze = AsyncMock(return_value=[idea])
    pipeline._codegen.read_affected_files = AsyncMock(return_value={})
    pipeline._codegen.generate = AsyncMock(return_value=([], "", "", ""))

    summary = await pipeline.run_cycle()
    assert summary["proposals_count"] == 0


@pytest.mark.asyncio
async def test_run_cycle_publishes_bus_events():
    """Bus events are published during the cycle."""
    pipeline = _make_pipeline()
    pipeline._sources.generate_queries = AsyncMock(return_value=[])
    pipeline._sources.search = AsyncMock(return_value=[])
    pipeline._sources.fetch_content = AsyncMock(return_value=[])
    pipeline._store.has_finding_url = AsyncMock(return_value=False)
    pipeline._analyzer.analyze = AsyncMock(return_value=[])

    await pipeline.run_cycle()

    # At minimum: cycle_started + cycle_completed
    assert pipeline._bus.publish.call_count >= 2
    topics = [c.args[0].topic for c in pipeline._bus.publish.call_args_list]
    assert "scout.cycle_started" in topics
    assert "scout.cycle_completed" in topics


@pytest.mark.asyncio
async def test_run_cycle_limits_findings():
    """Only max_findings_per_cycle findings are processed."""
    findings = [
        _make_finding(finding_id=f"fnd_{i}", url=f"https://example.com/{i}")
        for i in range(30)
    ]

    pipeline = _make_pipeline(max_findings_per_cycle=5)
    pipeline._sources.generate_queries = AsyncMock(return_value=["q"])
    pipeline._sources.search = AsyncMock(return_value=findings)
    pipeline._sources.fetch_content = AsyncMock(return_value=findings[:5])
    pipeline._store.has_finding_url = AsyncMock(return_value=False)
    pipeline._analyzer.analyze = AsyncMock(return_value=[])

    summary = await pipeline.run_cycle()
    assert summary["findings_count"] <= 5


@pytest.mark.asyncio
async def test_run_cycle_limits_proposals():
    """Only top max_proposals_per_cycle ideas become proposals."""
    finding = _make_finding()
    ideas = [_make_idea(idea_id=f"idea_{i}") for i in range(10)]

    pipeline = _make_pipeline(max_proposals_per_cycle=2)
    pipeline._sources.generate_queries = AsyncMock(return_value=["q"])
    pipeline._sources.search = AsyncMock(return_value=[finding])
    pipeline._sources.fetch_content = AsyncMock(return_value=[finding])
    pipeline._store.has_finding_url = AsyncMock(return_value=False)
    pipeline._analyzer.analyze = AsyncMock(return_value=ideas)
    pipeline._codegen.read_affected_files = AsyncMock(return_value={})
    pipeline._codegen.generate = AsyncMock(return_value=([], "", "", ""))

    await pipeline.run_cycle()
    # codegen called at most 2 times
    assert pipeline._codegen.generate.call_count <= 2


@pytest.mark.asyncio
async def test_process_idea_exception_handled():
    """Exceptions in idea processing don't crash the cycle."""
    finding = _make_finding()
    idea = _make_idea()

    pipeline = _make_pipeline()
    pipeline._sources.generate_queries = AsyncMock(return_value=["q"])
    pipeline._sources.search = AsyncMock(return_value=[finding])
    pipeline._sources.fetch_content = AsyncMock(return_value=[finding])
    pipeline._store.has_finding_url = AsyncMock(return_value=False)
    pipeline._analyzer.analyze = AsyncMock(return_value=[idea])
    pipeline._codegen.read_affected_files = AsyncMock(side_effect=RuntimeError("boom"))

    summary = await pipeline.run_cycle()
    assert summary["proposals_count"] == 0  # No crash


@pytest.mark.asyncio
async def test_hil_envelope_published_for_passing_proposal():
    """A hil.approval_required envelope is published for test-passing proposals."""
    finding = _make_finding()
    idea = _make_idea()
    test_result = TestResult(passed=True, total_tests=50, passed_tests=50)

    pipeline = _make_pipeline()
    pipeline._sources.generate_queries = AsyncMock(return_value=["q"])
    pipeline._sources.search = AsyncMock(return_value=[finding])
    pipeline._sources.fetch_content = AsyncMock(return_value=[finding])
    pipeline._store.has_finding_url = AsyncMock(return_value=False)
    pipeline._analyzer.analyze = AsyncMock(return_value=[idea])
    pipeline._codegen.read_affected_files = AsyncMock(return_value={})
    pipeline._codegen.generate = AsyncMock(
        return_value=(
            [CodeChange(file_path="src/x.py", change_type="modify")],
            "impact", "risk", "rollback",
        )
    )
    pipeline._sandbox.create_worktree = AsyncMock(return_value=("/tmp/wt", "scout/test"))
    pipeline._sandbox.apply_changes = AsyncMock()
    pipeline._sandbox.run_tests = AsyncMock(return_value=test_result)
    pipeline._sandbox.capture_diffs = AsyncMock(return_value=[])

    await pipeline.run_cycle()

    # Find hil.approval_required in published events
    topics = [c.args[0].topic for c in pipeline._bus.publish.call_args_list]
    assert "hil.approval_required" in topics
    assert "scout.proposal_created" in topics

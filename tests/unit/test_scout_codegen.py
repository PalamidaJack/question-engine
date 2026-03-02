"""Tests for ScoutCodeGenerator — LLM-powered code change generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.scout import CodeChange, ImprovementIdea
from qe.services.scout.codegen import (
    ScoutCodeGenerator,
    _CodeGenResult,
    _FileChange,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_idea(**overrides) -> ImprovementIdea:
    """Build a minimal ImprovementIdea for testing."""
    defaults = {
        "finding_id": "fnd_test",
        "title": "Add retry logic",
        "description": "Add exponential backoff to HTTP calls",
        "category": "performance",
        "relevance_score": 0.8,
        "feasibility_score": 0.7,
        "impact_score": 0.6,
        "composite_score": 0.7,
        "source_url": "https://example.com/article",
        "rationale": "Retries reduce transient failure rates by 90%",
        "affected_files": ["src/qe/services/executor/service.py"],
    }
    defaults.update(overrides)
    return ImprovementIdea(**defaults)


def _make_codegen_result(
    changes: list[_FileChange] | None = None,
    impact: str = "Reduces transient failures",
    risk: str = "Low risk",
    rollback: str = "Revert the commit",
) -> _CodeGenResult:
    """Build a _CodeGenResult for mocking LLM output."""
    if changes is None:
        changes = [
            _FileChange(
                file_path="src/qe/services/executor/service.py",
                change_type="modify",
                new_content="# modified content\n",
                rationale="Add retry wrapper",
            ),
        ]
    return _CodeGenResult(
        changes=changes,
        impact_assessment=impact,
        risk_assessment=risk,
        rollback_plan=rollback,
    )


def _mock_flag_store(enabled: bool = True):
    """Return a mock get_flag_store whose store.is_enabled returns *enabled*."""
    mock_store = MagicMock()
    mock_store.is_enabled.return_value = enabled
    mock_flags = MagicMock(return_value=mock_store)
    return mock_flags


def _mock_instructor(result: _CodeGenResult):
    """Return (mock_instructor_module, mock_client) wired to return *result*."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = result

    mock_inst = MagicMock()
    mock_inst.from_litellm.return_value = mock_client
    return mock_inst, mock_client


# ── Tests ────────────────────────────────────────────────────────────────


class TestGenerate:
    """Tests for ScoutCodeGenerator.generate()."""

    @pytest.mark.asyncio
    async def test_generate_returns_changes_and_assessments(self):
        """Happy path: LLM returns a valid result with changes and assessments."""
        idea = _make_idea()
        result = _make_codegen_result()
        mock_inst, mock_client = _mock_instructor(result)

        with (
            patch("qe.services.scout.codegen.get_flag_store", _mock_flag_store(True)),
            patch("qe.services.scout.codegen.instructor", mock_inst),
        ):
            gen = ScoutCodeGenerator(model="gpt-4o")
            changes, impact, risk, rollback = await gen.generate(idea)

        # Verify changes
        assert len(changes) == 1
        assert isinstance(changes[0], CodeChange)
        assert changes[0].file_path == "src/qe/services/executor/service.py"
        assert changes[0].change_type == "modify"
        assert changes[0].diff == ""  # Diff populated later after worktree apply

        # Verify assessments
        assert impact == "Reduces transient failures"
        assert risk == "Low risk"
        assert rollback == "Revert the commit"

        # Verify LLM was called with correct model and messages
        mock_client.chat.completions.create.assert_awaited_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["response_model"] is _CodeGenResult
        assert len(call_kwargs["messages"]) == 2
        assert "Add retry logic" in call_kwargs["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_generate_disabled_flag_returns_empty(self):
        """When innovation_scout flag is disabled, returns empty results immediately."""
        idea = _make_idea()

        with patch("qe.services.scout.codegen.get_flag_store", _mock_flag_store(False)):
            gen = ScoutCodeGenerator()
            changes, impact, risk, rollback = await gen.generate(idea)

        assert changes == []
        assert impact == ""
        assert risk == ""
        assert rollback == ""

    @pytest.mark.asyncio
    async def test_generate_llm_failure_returns_empty(self):
        """When the LLM call raises an exception, returns empty results gracefully."""
        idea = _make_idea()

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("LLM timeout")

        mock_inst = MagicMock()
        mock_inst.from_litellm.return_value = mock_client

        with (
            patch("qe.services.scout.codegen.get_flag_store", _mock_flag_store(True)),
            patch("qe.services.scout.codegen.instructor", mock_inst),
        ):
            gen = ScoutCodeGenerator()
            changes, impact, risk, rollback = await gen.generate(idea)

        assert changes == []
        assert impact == ""
        assert risk == ""
        assert rollback == ""

    @pytest.mark.asyncio
    async def test_generate_with_file_contents(self):
        """When file_contents is provided, they are included in the LLM prompt."""
        idea = _make_idea()
        file_contents = {
            "src/qe/services/executor/service.py": "class ExecutorService:\n    pass\n",
            "src/qe/runtime/tools.py": "class ToolSpec:\n    pass\n",
        }
        result = _make_codegen_result()
        mock_inst, mock_client = _mock_instructor(result)

        with (
            patch("qe.services.scout.codegen.get_flag_store", _mock_flag_store(True)),
            patch("qe.services.scout.codegen.instructor", mock_inst),
        ):
            gen = ScoutCodeGenerator()
            changes, impact, risk, rollback = await gen.generate(idea, file_contents=file_contents)

        # Verify file contents appear in the user message
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        user_msg = call_kwargs["messages"][1]["content"]
        assert "src/qe/services/executor/service.py" in user_msg
        assert "class ExecutorService" in user_msg
        assert "src/qe/runtime/tools.py" in user_msg
        assert "class ToolSpec" in user_msg

        # Still returns valid changes
        assert len(changes) == 1

    @pytest.mark.asyncio
    async def test_change_type_validation(self):
        """Invalid change_type values are normalized to 'modify'."""
        idea = _make_idea()
        result = _make_codegen_result(
            changes=[
                _FileChange(
                    file_path="src/new_module.py",
                    change_type="create",
                    new_content="# new file",
                    rationale="New module",
                ),
                _FileChange(
                    file_path="src/old_module.py",
                    change_type="delete",
                    new_content="",
                    rationale="Remove deprecated module",
                ),
                _FileChange(
                    file_path="src/main.py",
                    change_type="replace",  # invalid — should become "modify"
                    new_content="# replaced",
                    rationale="Invalid type test",
                ),
                _FileChange(
                    file_path="src/utils.py",
                    change_type="modify",
                    new_content="# updated",
                    rationale="Normal modify",
                ),
            ],
        )
        mock_inst, mock_client = _mock_instructor(result)

        with (
            patch("qe.services.scout.codegen.get_flag_store", _mock_flag_store(True)),
            patch("qe.services.scout.codegen.instructor", mock_inst),
        ):
            gen = ScoutCodeGenerator()
            changes, _, _, _ = await gen.generate(idea)

        assert len(changes) == 4
        assert changes[0].change_type == "create"
        assert changes[1].change_type == "delete"
        assert changes[2].change_type == "modify"  # normalized from "replace"
        assert changes[3].change_type == "modify"


class TestReadAffectedFiles:
    """Tests for ScoutCodeGenerator.read_affected_files()."""

    @pytest.mark.asyncio
    async def test_read_affected_files_success(self):
        """Successfully reads multiple affected files."""
        mock_read = AsyncMock(
            side_effect=[
                {"content": "file_a content"},
                {"content": "file_b content"},
            ]
        )

        with patch("qe.tools.file_ops.file_read", mock_read):
            gen = ScoutCodeGenerator()
            contents = await gen.read_affected_files(
                ["src/module_a.py", "src/module_b.py"]
            )

        assert contents == {
            "src/module_a.py": "file_a content",
            "src/module_b.py": "file_b content",
        }
        assert mock_read.await_count == 2
        mock_read.assert_any_await("src/module_a.py")
        mock_read.assert_any_await("src/module_b.py")

    @pytest.mark.asyncio
    async def test_read_affected_files_missing_file(self):
        """When a file read fails, that file is skipped and others are still returned."""
        mock_read = AsyncMock(
            side_effect=[
                {"content": "good file content"},
                FileNotFoundError("File not found: missing.py"),
                {"content": "another good file"},
            ]
        )

        with patch("qe.tools.file_ops.file_read", mock_read):
            gen = ScoutCodeGenerator()
            contents = await gen.read_affected_files(
                ["src/good.py", "src/missing.py", "src/also_good.py"]
            )

        # Only the two successful reads are in the result
        assert len(contents) == 2
        assert contents["src/good.py"] == "good file content"
        assert contents["src/also_good.py"] == "another good file"
        assert "src/missing.py" not in contents

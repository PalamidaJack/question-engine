"""Comprehensive unit tests for ScoutSandbox (git worktree sandbox)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.scout import CodeChange, TestResult
from qe.services.scout.sandbox import ScoutSandbox, _parse_pytest_output

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_proc(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> AsyncMock:
    """Create a mock async subprocess with the given return code and output."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = MagicMock()
    return proc


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_worktree_success():
    """create_worktree returns (worktree_path, branch_name) on success."""
    proc = _make_proc(returncode=0, stdout=b"Preparing worktree")

    patch_target = "qe.services.scout.sandbox.asyncio.create_subprocess_exec"
    with patch(patch_target, return_value=proc) as mock_sub:
        sandbox = ScoutSandbox(repo_root="/tmp/fake-repo")
        wt_path, branch = await sandbox.create_worktree("prop_abc123", slug="fix-bug")

    assert "prop_abc123" in wt_path
    assert branch == "scout/prop_abc123_fix-bug"
    # Verify git worktree add was called with the correct arguments
    call_args = mock_sub.call_args
    assert call_args[0][0] == "git"
    assert call_args[0][1] == "worktree"
    assert call_args[0][2] == "add"
    assert "-b" in call_args[0]
    assert branch in call_args[0]


@pytest.mark.asyncio
async def test_create_worktree_failure():
    """create_worktree raises RuntimeError when git returns non-zero."""
    proc = _make_proc(returncode=1, stderr=b"fatal: branch already exists")

    with patch("qe.services.scout.sandbox.asyncio.create_subprocess_exec", return_value=proc):
        sandbox = ScoutSandbox(repo_root="/tmp/fake-repo")

        with pytest.raises(RuntimeError, match="Failed to create worktree"):
            await sandbox.create_worktree("prop_abc123", slug="fix-bug")


@pytest.mark.asyncio
async def test_create_worktree_slug_sanitization():
    """create_worktree sanitizes special characters in slugs."""
    proc = _make_proc(returncode=0)

    with patch("qe.services.scout.sandbox.asyncio.create_subprocess_exec", return_value=proc):
        sandbox = ScoutSandbox(repo_root="/tmp/fake-repo")
        _, branch = await sandbox.create_worktree("prop_x", slug="hello world/foo@bar")

    # Spaces and slashes and @ should be replaced with hyphens
    assert "hello-world-foo-bar" in branch
    assert " " not in branch
    assert "/" not in branch.split("scout/", 1)[1].split("_", 1)[1]


@pytest.mark.asyncio
async def test_apply_changes_writes_files(tmp_path: Path):
    """apply_changes writes file contents to the worktree directory."""
    # Set up a worktree directory
    wt = tmp_path / "worktree"
    wt.mkdir()

    changes = [
        CodeChange(file_path="src/new_module.py", change_type="create"),
        CodeChange(file_path="src/nested/deep/file.py", change_type="create"),
    ]
    file_contents = {
        "src/new_module.py": "print('hello')\n",
        "src/nested/deep/file.py": "x = 42\n",
    }

    proc = _make_proc(returncode=0)

    with patch("qe.services.scout.sandbox.asyncio.create_subprocess_exec", return_value=proc):
        sandbox = ScoutSandbox()
        await sandbox.apply_changes(str(wt), changes, file_contents)

    # Verify files were written with correct content
    assert (wt / "src" / "new_module.py").read_text() == "print('hello')\n"
    assert (wt / "src" / "nested" / "deep" / "file.py").read_text() == "x = 42\n"


@pytest.mark.asyncio
async def test_apply_changes_delete_file(tmp_path: Path):
    """apply_changes deletes files marked with change_type='delete'."""
    wt = tmp_path / "worktree"
    wt.mkdir()
    target = wt / "obsolete.py"
    target.write_text("old code")

    changes = [CodeChange(file_path="obsolete.py", change_type="delete")]
    proc = _make_proc(returncode=0)

    with patch("qe.services.scout.sandbox.asyncio.create_subprocess_exec", return_value=proc):
        sandbox = ScoutSandbox()
        await sandbox.apply_changes(str(wt), changes, {})

    assert not target.exists()


@pytest.mark.asyncio
async def test_apply_changes_runs_git_add_and_commit(tmp_path: Path):
    """apply_changes stages and commits changes via git subprocess."""
    wt = tmp_path / "worktree"
    wt.mkdir()

    changes = [CodeChange(file_path="f.py", change_type="create")]
    file_contents = {"f.py": "pass\n"}
    proc = _make_proc(returncode=0)

    patch_target = "qe.services.scout.sandbox.asyncio.create_subprocess_exec"
    with patch(patch_target, return_value=proc) as mock_sub:
        sandbox = ScoutSandbox()
        await sandbox.apply_changes(str(wt), changes, file_contents)

    # Should have been called twice: git add -A, then git commit
    assert mock_sub.call_count == 2
    add_call = mock_sub.call_args_list[0]
    commit_call = mock_sub.call_args_list[1]
    assert add_call[0][:3] == ("git", "add", "-A")
    assert commit_call[0][:2] == ("git", "commit")
    assert "--allow-empty" in commit_call[0]


@pytest.mark.asyncio
async def test_run_tests_success():
    """run_tests returns a passing TestResult when pytest exits 0."""
    pytest_output = b"50 passed in 12.34s"
    proc = _make_proc(returncode=0, stdout=pytest_output, stderr=b"")

    with patch("qe.services.scout.sandbox.asyncio.create_subprocess_exec", return_value=proc), \
         patch("qe.services.scout.sandbox.asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
        # wait_for should return the communicate result
        mock_wait.return_value = (pytest_output, b"")
        sandbox = ScoutSandbox()
        result = await sandbox.run_tests("/tmp/fake-worktree", timeout=120)

    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.passed_tests == 50
    assert result.failed_tests == 0
    assert result.total_tests == 50
    assert "50 passed" in result.stdout


@pytest.mark.asyncio
async def test_run_tests_failure():
    """run_tests returns a failing TestResult when pytest reports failures."""
    pytest_output = b"5 failed, 45 passed in 15.00s"
    proc = _make_proc(returncode=1, stdout=pytest_output, stderr=b"")

    with patch("qe.services.scout.sandbox.asyncio.create_subprocess_exec", return_value=proc), \
         patch("qe.services.scout.sandbox.asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
        mock_wait.return_value = (pytest_output, b"")
        sandbox = ScoutSandbox()
        result = await sandbox.run_tests("/tmp/fake-worktree")

    assert isinstance(result, TestResult)
    assert result.passed is False
    assert result.passed_tests == 45
    assert result.failed_tests == 5
    assert result.total_tests == 50


@pytest.mark.asyncio
async def test_run_tests_timeout():
    """run_tests handles TimeoutError by killing the process and returning a timed-out result."""
    proc = _make_proc(returncode=0)

    with patch("qe.services.scout.sandbox.asyncio.create_subprocess_exec", return_value=proc), \
         patch("qe.services.scout.sandbox.asyncio.wait_for", side_effect=asyncio.TimeoutError):
        sandbox = ScoutSandbox()
        result = await sandbox.run_tests("/tmp/fake-worktree", timeout=5)

    assert isinstance(result, TestResult)
    assert result.passed is False
    assert "timed out" in result.stdout.lower()
    assert result.duration_s > 0
    # Process should have been killed
    proc.kill.assert_called_once()


def test_parse_pytest_output_passed_only():
    """_parse_pytest_output extracts counts from a 'passed only' summary."""
    passed, failed, total = _parse_pytest_output("50 passed in 12.34s")
    assert passed == 50
    assert failed == 0
    assert total == 50


def test_parse_pytest_output_mixed():
    """_parse_pytest_output extracts counts from mixed pass/fail output."""
    passed, failed, total = _parse_pytest_output("5 failed, 45 passed, 2 warnings in 8.00s")
    assert passed == 45
    assert failed == 5
    assert total == 50


def test_parse_pytest_output_all_failed():
    """_parse_pytest_output handles all-failed output."""
    passed, failed, total = _parse_pytest_output("10 failed in 3.00s")
    assert passed == 0
    assert failed == 10
    assert total == 10


def test_parse_pytest_output_no_matches():
    """_parse_pytest_output returns zeros when no counts are found."""
    passed, failed, total = _parse_pytest_output("no tests ran")
    assert passed == 0
    assert failed == 0
    assert total == 0


@pytest.mark.asyncio
async def test_capture_diffs_parses_git_output():
    """capture_diffs parses multi-file git diff output into CodeChange objects."""
    diff_output = (
        "diff --git a/src/foo.py b/src/foo.py\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,3 +1,4 @@\n"
        " import os\n"
        "+import sys\n"
        " \n"
        " def main():\n"
        "diff --git a/src/bar.py b/src/bar.py\n"
        "--- a/src/bar.py\n"
        "+++ b/src/bar.py\n"
        "@@ -10,2 +10,3 @@\n"
        " x = 1\n"
        "+y = 2\n"
    )
    proc = _make_proc(returncode=0, stdout=diff_output.encode())

    with patch("qe.services.scout.sandbox.asyncio.create_subprocess_exec", return_value=proc):
        sandbox = ScoutSandbox()
        changes = await sandbox.capture_diffs("/tmp/fake-worktree")

    assert len(changes) == 2
    assert all(isinstance(c, CodeChange) for c in changes)
    assert changes[0].file_path == "src/foo.py"
    assert changes[0].change_type == "modify"
    assert "+import sys" in changes[0].diff
    assert changes[1].file_path == "src/bar.py"
    assert "+y = 2" in changes[1].diff


@pytest.mark.asyncio
async def test_capture_diffs_empty_output():
    """capture_diffs returns an empty list when git diff produces no output."""
    proc = _make_proc(returncode=0, stdout=b"")

    with patch("qe.services.scout.sandbox.asyncio.create_subprocess_exec", return_value=proc):
        sandbox = ScoutSandbox()
        changes = await sandbox.capture_diffs("/tmp/fake-worktree")

    assert changes == []


@pytest.mark.asyncio
async def test_cleanup_worktree_removes_and_deletes_branch():
    """cleanup_worktree calls git worktree remove and git branch -D."""
    proc = _make_proc(returncode=0)

    patch_target = "qe.services.scout.sandbox.asyncio.create_subprocess_exec"
    with patch(patch_target, return_value=proc) as mock_sub:
        sandbox = ScoutSandbox(repo_root="/tmp/fake-repo")
        await sandbox.cleanup_worktree(
            "/tmp/wt", branch_name="scout/prop_x_fix", delete_branch=True,
        )

    assert mock_sub.call_count == 2
    remove_call = mock_sub.call_args_list[0]
    branch_call = mock_sub.call_args_list[1]
    assert remove_call[0][:3] == ("git", "worktree", "remove")
    assert branch_call[0][:3] == ("git", "branch", "-D")
    assert "scout/prop_x_fix" in branch_call[0]


@pytest.mark.asyncio
async def test_cleanup_worktree_skip_branch_delete():
    """cleanup_worktree skips branch deletion when delete_branch is False."""
    proc = _make_proc(returncode=0)

    patch_target = "qe.services.scout.sandbox.asyncio.create_subprocess_exec"
    with patch(patch_target, return_value=proc) as mock_sub:
        sandbox = ScoutSandbox(repo_root="/tmp/fake-repo")
        await sandbox.cleanup_worktree("/tmp/wt", branch_name="scout/x", delete_branch=False)

    # Only worktree remove, no branch delete
    assert mock_sub.call_count == 1
    assert mock_sub.call_args[0][:3] == ("git", "worktree", "remove")


@pytest.mark.asyncio
async def test_merge_branch_success():
    """merge_branch returns True on successful merge."""
    proc = _make_proc(returncode=0, stdout=b"Merge made by the 'ort' strategy.")

    patch_target = "qe.services.scout.sandbox.asyncio.create_subprocess_exec"
    with patch(patch_target, return_value=proc) as mock_sub:
        sandbox = ScoutSandbox(repo_root="/tmp/fake-repo")
        result = await sandbox.merge_branch("scout/prop_abc_fix", title="Add feature X")

    assert result is True
    call_args = mock_sub.call_args[0]
    assert "git" == call_args[0]
    assert "merge" == call_args[1]
    assert "scout/prop_abc_fix" == call_args[2]
    assert "--no-ff" in call_args
    # Title should appear in the commit message
    assert any("scout: Add feature X" in str(a) for a in call_args)


@pytest.mark.asyncio
async def test_merge_branch_conflict():
    """merge_branch returns False when merge fails (e.g., conflict)."""
    proc = _make_proc(returncode=1, stderr=b"CONFLICT (content): Merge conflict in src/foo.py")

    with patch("qe.services.scout.sandbox.asyncio.create_subprocess_exec", return_value=proc):
        sandbox = ScoutSandbox(repo_root="/tmp/fake-repo")
        result = await sandbox.merge_branch("scout/prop_abc_fix")

    assert result is False

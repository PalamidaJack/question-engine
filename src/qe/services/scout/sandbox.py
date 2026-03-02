"""Git worktree sandbox for isolated testing of scout proposals."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from qe.models.scout import CodeChange, TestResult

log = logging.getLogger(__name__)

WORKTREE_BASE = Path("data/scout_worktrees")


class ScoutSandbox:
    """Create git worktrees, apply changes, run tests, capture diffs."""

    def __init__(self, repo_root: str = ".") -> None:
        self._repo_root = Path(repo_root).resolve()

    async def create_worktree(
        self,
        proposal_id: str,
        slug: str = "",
    ) -> tuple[str, str]:
        """Create a git worktree for the proposal.

        Returns (worktree_path, branch_name).
        """
        safe_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", slug)[:40] if slug else "change"
        branch_name = f"scout/{proposal_id}_{safe_slug}"
        worktree_path = str(WORKTREE_BASE / proposal_id)

        WORKTREE_BASE.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

        # Create worktree with a new branch based on current HEAD
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "add", worktree_path, "-b", branch_name,
            cwd=str(self._repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip()
            log.error("scout.worktree_create_failed error=%s", err)
            raise RuntimeError(f"Failed to create worktree: {err}")

        log.info(
            "scout.worktree_created path=%s branch=%s",
            worktree_path,
            branch_name,
        )
        return worktree_path, branch_name

    async def apply_changes(
        self,
        worktree_path: str,
        changes: list[CodeChange],
        file_contents: dict[str, str] | None = None,
    ) -> None:
        """Apply code changes to the worktree."""
        wt = Path(worktree_path)
        contents = file_contents or {}

        for change in changes:
            target = wt / change.file_path
            if change.change_type == "delete":
                if target.exists():
                    target.unlink()
            elif change.change_type in ("create", "modify"):
                content = contents.get(change.file_path, "")
                if content:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")

        # Stage all changes
        proc = await asyncio.create_subprocess_exec(
            "git", "add", "-A",
            cwd=worktree_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # Commit
        proc = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", "scout: apply proposed changes",
            "--allow-empty",
            cwd=worktree_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    async def run_tests(
        self,
        worktree_path: str,
        timeout: int = 120,  # noqa: ASYNC109
    ) -> TestResult:
        """Run pytest in the worktree and capture results."""
        import time

        start = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest", "tests/", "-m", "not slow",
                "--timeout=60", "-q", "--tb=short",
                cwd=worktree_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return TestResult(
                passed=False,
                stdout="Test execution timed out",
                stderr="",
                duration_s=time.monotonic() - start,
            )

        duration = time.monotonic() - start
        stdout = stdout_bytes.decode(errors="replace")[:5000]
        stderr = stderr_bytes.decode(errors="replace")[:2000]

        # Parse pytest output for counts
        passed_tests, failed_tests, total_tests = _parse_pytest_output(stdout)
        passed = proc.returncode == 0

        return TestResult(
            passed=passed,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            stdout=stdout,
            stderr=stderr,
            duration_s=round(duration, 2),
        )

    async def capture_diffs(
        self,
        worktree_path: str,
    ) -> list[CodeChange]:
        """Capture unified diffs from the worktree against main."""
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "main", "--unified=3",
            cwd=worktree_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        full_diff = stdout.decode(errors="replace")

        # Parse per-file diffs
        changes: list[CodeChange] = []
        current_file = ""
        current_diff_lines: list[str] = []

        for line in full_diff.split("\n"):
            if line.startswith("diff --git"):
                if current_file and current_diff_lines:
                    changes.append(
                        CodeChange(
                            file_path=current_file,
                            change_type="modify",
                            diff="\n".join(current_diff_lines),
                        )
                    )
                # Extract filename from "diff --git a/path b/path"
                parts = line.split(" b/")
                current_file = parts[-1] if len(parts) > 1 else ""
                current_diff_lines = [line]
            else:
                current_diff_lines.append(line)

        if current_file and current_diff_lines:
            changes.append(
                CodeChange(
                    file_path=current_file,
                    change_type="modify",
                    diff="\n".join(current_diff_lines),
                )
            )

        return changes

    async def cleanup_worktree(
        self,
        worktree_path: str,
        branch_name: str = "",
        delete_branch: bool = False,
    ) -> None:
        """Remove a worktree and optionally its branch."""
        proc = await asyncio.create_subprocess_exec(
            "git", "worktree", "remove", worktree_path, "--force",
            cwd=str(self._repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        if delete_branch and branch_name:
            proc = await asyncio.create_subprocess_exec(
                "git", "branch", "-D", branch_name,
                cwd=str(self._repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

        log.info("scout.worktree_cleaned path=%s", worktree_path)

    async def merge_branch(
        self,
        branch_name: str,
        title: str = "",
    ) -> bool:
        """Merge a scout branch into main."""
        message = f"scout: {title}" if title else f"scout: merge {branch_name}"

        proc = await asyncio.create_subprocess_exec(
            "git", "merge", branch_name, "--no-ff", "-m", message,
            cwd=str(self._repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip()
            log.error("scout.merge_failed branch=%s error=%s", branch_name, err)
            return False

        log.info("scout.branch_merged branch=%s", branch_name)
        return True


def _parse_pytest_output(output: str) -> tuple[int, int, int]:
    """Parse pytest -q output for pass/fail counts."""
    # Look for pattern like "1950 passed" or "5 failed, 1945 passed"
    passed = 0
    failed = 0

    passed_match = re.search(r"(\d+) passed", output)
    if passed_match:
        passed = int(passed_match.group(1))

    failed_match = re.search(r"(\d+) failed", output)
    if failed_match:
        failed = int(failed_match.group(1))

    total = passed + failed
    return passed, failed, total

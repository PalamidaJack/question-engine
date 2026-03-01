"""Per-goal sandboxed workspace management."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


class WorkspaceManager:
    """Manages per-goal sandboxed workspaces."""

    def __init__(self, base_dir: str | Path = "data/workspaces") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create(self, goal_id: str) -> Path:
        """Create workspace directory for a goal."""
        workspace = self.base_dir / goal_id
        workspace.mkdir(parents=True, exist_ok=True)
        log.info("workspace.created goal=%s path=%s", goal_id, workspace)
        return workspace

    def get(self, goal_id: str) -> Path:
        """Get the workspace path for a goal."""
        workspace = self.base_dir / goal_id
        if not workspace.exists():
            return self.create(goal_id)
        return workspace

    def cleanup(self, goal_id: str) -> None:
        """Remove workspace after goal completion."""
        workspace = self.base_dir / goal_id
        if workspace.exists():
            shutil.rmtree(workspace)
            log.info("workspace.cleaned goal=%s", goal_id)

    def sandbox_path(self, goal_id: str, relative_path: str) -> Path:
        """Resolve a path within the sandbox.

        Raises ValueError if the resolved path escapes the workspace.
        """
        workspace = self.get(goal_id)
        resolved = (workspace / relative_path).resolve()
        workspace_resolved = workspace.resolve()

        if not resolved.is_relative_to(workspace_resolved):
            raise ValueError(
                f"Path escapes workspace sandbox: {relative_path}"
            )

        return resolved

    def list_workspaces(self) -> list[str]:
        """List all active workspace goal IDs."""
        if not self.base_dir.exists():
            return []
        return [
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir()
        ]

"""Sandboxed file read/write tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qe.runtime.tools import ToolSpec

log = logging.getLogger(__name__)

file_read_spec = ToolSpec(
    name="file_read",
    description="Read a file from the goal workspace. Path is relative to the workspace directory.",
    requires_capability="file_read",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path within the workspace",
            },
        },
        "required": ["path"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "size_bytes": {"type": "integer"},
        },
    },
    timeout_seconds=10,
    category="filesystem",
)

file_write_spec = ToolSpec(
    name="file_write",
    description=(
        "Write content to a file in the goal workspace. "
        "Path is relative to the workspace directory."
    ),
    requires_capability="file_write",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path within the workspace",
            },
            "content": {
                "type": "string",
                "description": "Content to write",
            },
        },
        "required": ["path", "content"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "size_bytes": {"type": "integer"},
            "created": {"type": "boolean"},
        },
    },
    timeout_seconds=10,
    category="filesystem",
)

# Module-level workspace root; set by WorkspaceManager before use
_workspace_root: Path | None = None


def set_workspace_root(root: Path) -> None:
    """Set the workspace root directory for sandboxing."""
    global _workspace_root  # noqa: PLW0603
    _workspace_root = root


def _resolve_sandboxed(relative_path: str) -> Path:
    """Resolve a relative path within the workspace sandbox.

    Raises ValueError if the resolved path escapes the workspace.
    """
    if _workspace_root is None:
        raise RuntimeError("Workspace root not configured. Call set_workspace_root() first.")

    resolved = (_workspace_root / relative_path).resolve()
    workspace_resolved = _workspace_root.resolve()

    if not str(resolved).startswith(str(workspace_resolved)):
        raise ValueError(
            f"Path escapes workspace sandbox: {relative_path}"
        )

    return resolved


async def file_read(path: str) -> dict[str, Any]:
    """Read a file from the sandboxed workspace."""
    resolved = _resolve_sandboxed(path)

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not resolved.is_file():
        raise ValueError(f"Not a file: {path}")

    content = resolved.read_text(encoding="utf-8")
    return {
        "content": content,
        "size_bytes": len(content.encode("utf-8")),
    }


async def file_write(path: str, content: str) -> dict[str, Any]:
    """Write content to a file in the sandboxed workspace."""
    resolved = _resolve_sandboxed(path)

    created = not resolved.exists()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")

    return {
        "path": path,
        "size_bytes": len(content.encode("utf-8")),
        "created": created,
    }

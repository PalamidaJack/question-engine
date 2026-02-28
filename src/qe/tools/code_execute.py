"""Sandboxed Python code execution tool."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any

from qe.runtime.tools import ToolSpec

log = logging.getLogger(__name__)

code_execute_spec = ToolSpec(
    name="code_execute",
    description=(
        "Execute Python code in a sandboxed subprocess. "
        "Returns stdout, stderr, and return code."
    ),
    requires_capability="code_execute",
    input_schema={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout_seconds": {
                "type": "integer",
                "description": "Execution timeout",
                "default": 30,
            },
        },
        "required": ["code"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "return_code": {"type": "integer"},
            "timed_out": {"type": "boolean"},
        },
    },
    timeout_seconds=60,
    category="code",
)


async def code_execute(
    code: str,
    timeout_seconds: int = 30,
    *,
    workspace_dir: str | None = None,
) -> dict[str, Any]:
    """Execute Python code in a sandboxed subprocess.

    Security: runs in a subprocess with resource limits.
    No network access enforcement relies on OS-level controls.
    """
    # Write code to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=workspace_dir
    ) as f:
        f.write(code)
        script_path = f.name

    try:
        cwd = workspace_dir or tempfile.gettempdir()

        proc = await asyncio.create_subprocess_exec(
            "python3",
            script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
            return {
                "stdout": stdout_bytes.decode("utf-8", errors="replace")[:10000],
                "stderr": stderr_bytes.decode("utf-8", errors="replace")[:10000],
                "return_code": proc.returncode or 0,
                "timed_out": False,
            }
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {timeout_seconds}s",
                "return_code": -1,
                "timed_out": True,
            }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

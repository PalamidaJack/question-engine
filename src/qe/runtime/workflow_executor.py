"""Workflow executor -- walks workflow graphs and executes nodes."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qe.models.workflow import WorkflowDefinition, WorkflowNode
from qe.runtime.workflow_nodes import get_executor

log = logging.getLogger(__name__)


class WorkflowExecution:
    """State for a single workflow execution."""

    def __init__(
        self,
        workflow: WorkflowDefinition,
        input_data: dict | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.workflow = workflow
        self.input_data = input_data or {}
        self.node_outputs: dict[str, Any] = {}
        self.status: str = "pending"
        self.current_node: str | None = None
        self.error: str | None = None
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.execution_log: list[dict] = []
        self._tool_executor: Any = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workflow_id": self.workflow.id,
            "workflow_name": self.workflow.name,
            "status": self.status,
            "current_node": self.current_node,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_log": self.execution_log[-50:],
            "node_outputs": {
                k: _safe_summary(v)
                for k, v in self.node_outputs.items()
            },
        }


class WorkflowExecutor:
    """Executes workflow definitions by walking the node graph."""

    def __init__(
        self,
        bus: Any = None,
        tool_executor: Callable | None = None,
        workflows_dir: str | Path = "data/workflows",
    ):
        self._bus = bus
        self._tool_executor = tool_executor
        self._workflows_dir = Path(workflows_dir)
        self._workflows_dir.mkdir(parents=True, exist_ok=True)
        self._executions: dict[str, WorkflowExecution] = {}
        self._max_executions = 100

    async def execute(
        self,
        workflow: WorkflowDefinition,
        input_data: dict | None = None,
        progress_callback: Callable | None = None,
    ) -> WorkflowExecution:
        """Execute a workflow from start to finish."""
        execution = WorkflowExecution(workflow, input_data)
        execution._tool_executor = self._tool_executor
        self._executions[execution.id] = execution
        self._trim_executions()

        execution.status = "running"
        execution.started_at = time.time()

        entry = workflow.get_entry_node()
        if not entry:
            execution.status = "failed"
            execution.error = "No entry node found"
            return execution

        try:
            await self._execute_node(
                execution, entry, progress_callback,
            )
            if execution.status == "running":
                execution.status = "completed"
        except Exception as e:
            execution.status = "failed"
            execution.error = str(e)
            log.error(
                "workflow.failed id=%s error=%s",
                execution.id,
                e,
            )

        execution.completed_at = time.time()

        if self._bus:
            from qe.models.envelope import Envelope

            await self._bus.publish(
                "workflow.completed",
                Envelope(
                    topic="workflow.completed",
                    source_service_id="workflow_executor",
                    payload={
                        "execution_id": execution.id,
                        "status": execution.status,
                    },
                ),
            )

        return execution

    async def _execute_node(
        self,
        execution: WorkflowExecution,
        node: WorkflowNode,
        progress_callback: Callable | None = None,
    ) -> None:
        """Execute a single node and follow edges to next."""
        if execution.status not in ("running",):
            return

        execution.current_node = node.id
        node_start = time.time()

        # Build context for this node
        context: dict[str, Any] = {
            "input": execution.input_data,
            "last_output": self._get_last_output(
                execution, node,
            ),
            "node_type": node.type,
            "node_outputs": execution.node_outputs,
        }
        if execution._tool_executor:
            context["tool_executor"] = (
                execution._tool_executor
            )

        # Get and run executor
        executor = get_executor(node.type)
        if executor is None:
            execution.status = "failed"
            execution.error = (
                f"No executor for node type: {node.type}"
            )
            return

        result = await executor(node.config, context)
        execution.node_outputs[node.id] = result

        # Log execution
        log_entry = {
            "node_id": node.id,
            "node_type": node.type,
            "duration_ms": int(
                (time.time() - node_start) * 1000,
            ),
            "output_keys": (
                list(result.keys())
                if isinstance(result, dict)
                else []
            ),
        }
        execution.execution_log.append(log_entry)

        if progress_callback:
            await progress_callback(log_entry)

        # Check for human-in-the-loop pause
        if isinstance(result, dict) and result.get(
            "status",
        ) in ("waiting_approval", "waiting_input"):
            execution.status = "paused"
            return

        # Follow edges
        outgoing = execution.workflow.get_outgoing_edges(
            node.id,
        )
        if not outgoing:
            return  # Terminal node

        # Handle condition branching
        if node.type == "condition":
            branch = result.get("branch", False)
            for edge in outgoing:
                edge_cond = edge.condition or edge.label
                if edge_cond:
                    is_true = edge_cond.lower().startswith(
                        "true",
                    )
                    if (branch and is_true) or (
                        not branch and not is_true
                    ):
                        next_node = (
                            execution.workflow.get_node(
                                edge.to_node,
                            )
                        )
                        if next_node:
                            await self._execute_node(
                                execution,
                                next_node,
                                progress_callback,
                            )
                        return
            # Default: follow first edge
            if outgoing:
                next_node = execution.workflow.get_node(
                    outgoing[0].to_node,
                )
                if next_node:
                    await self._execute_node(
                        execution,
                        next_node,
                        progress_callback,
                    )
            return

        # Handle parallel fan-out
        if node.type == "parallel":
            tasks = []
            for edge in outgoing:
                next_node = execution.workflow.get_node(
                    edge.to_node,
                )
                if next_node and next_node.type != "merge":
                    tasks.append(
                        self._execute_parallel_branch(
                            execution,
                            next_node,
                            progress_callback,
                        )
                    )
            if tasks:
                results = await asyncio.gather(
                    *tasks, return_exceptions=True,
                )
                # Store parallel results for merge node
                execution.node_outputs[
                    "_parallel_results"
                ] = [
                    r
                    for r in results
                    if not isinstance(r, Exception)
                ]
            # Find and execute merge node
            for edge in outgoing:
                next_node = execution.workflow.get_node(
                    edge.to_node,
                )
                if (
                    next_node
                    and next_node.type == "merge"
                ):
                    context["parallel_results"] = (
                        execution.node_outputs.get(
                            "_parallel_results", [],
                        )
                    )
                    await self._execute_node(
                        execution,
                        next_node,
                        progress_callback,
                    )
            return

        # Handle loop
        if node.type == "loop":
            items = result.get(
                "items", result.get("results", []),
            )
            max_iter = node.config.get(
                "max_iterations", 100,
            )
            loop_results = []
            for _i, item in enumerate(items[:max_iter]):
                execution.node_outputs[
                    f"{node.id}_item"
                ] = item
                for edge in outgoing:
                    next_node = (
                        execution.workflow.get_node(
                            edge.to_node,
                        )
                    )
                    if next_node:
                        await self._execute_node(
                            execution,
                            next_node,
                            progress_callback,
                        )
                        loop_results.append(
                            execution.node_outputs.get(
                                next_node.id,
                            ),
                        )
            execution.node_outputs[node.id] = {
                "loop_results": loop_results,
            }
            return

        # Standard: follow all edges sequentially
        for edge in outgoing:
            next_node = execution.workflow.get_node(
                edge.to_node,
            )
            if next_node:
                await self._execute_node(
                    execution, next_node, progress_callback,
                )

    async def _execute_parallel_branch(
        self,
        execution: WorkflowExecution,
        node: WorkflowNode,
        progress_callback: Callable | None,
    ) -> dict:
        """Execute a branch for parallel node."""
        await self._execute_node(
            execution, node, progress_callback,
        )
        return execution.node_outputs.get(node.id, {})

    def _get_last_output(
        self,
        execution: WorkflowExecution,
        node: WorkflowNode,
    ) -> dict:
        """Get output from the previous node."""
        incoming = execution.workflow.get_incoming_edges(
            node.id,
        )
        if not incoming:
            return execution.input_data
        # Use the most recent incoming node's output
        for edge in reversed(incoming):
            output = execution.node_outputs.get(
                edge.from_node,
            )
            if output is not None:
                return (
                    output
                    if isinstance(output, dict)
                    else {"result": output}
                )
        return {}

    # ── Workflow CRUD ──

    def save_workflow(
        self, workflow: WorkflowDefinition,
    ) -> None:
        """Save workflow definition to disk."""
        path = self._workflows_dir / f"{workflow.id}.json"
        path.write_text(
            json.dumps(
                workflow.model_dump(by_alias=True),
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_workflow(
        self, workflow_id: str,
    ) -> WorkflowDefinition | None:
        """Load workflow from disk."""
        path = self._workflows_dir / f"{workflow_id}.json"
        if not path.exists():
            return None
        data = json.loads(
            path.read_text(encoding="utf-8"),
        )
        return WorkflowDefinition.model_validate(data)

    def list_workflows(self) -> list[dict]:
        """List all saved workflows."""
        result = []
        for p in sorted(self._workflows_dir.glob("*.json")):
            try:
                data = json.loads(
                    p.read_text(encoding="utf-8"),
                )
                result.append({
                    "id": data.get("id", p.stem),
                    "name": data.get("name", p.stem),
                    "description": data.get(
                        "description", "",
                    ),
                    "node_count": len(
                        data.get("nodes", []),
                    ),
                    "metadata": data.get("metadata", {}),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return result

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a saved workflow."""
        path = self._workflows_dir / f"{workflow_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def get_execution(
        self, execution_id: str,
    ) -> WorkflowExecution | None:
        return self._executions.get(execution_id)

    def list_executions(self) -> list[dict]:
        return [
            e.to_dict() for e in self._executions.values()
        ]

    # ── Checkpoint/Resume (#55) ──

    def checkpoint(self, execution: WorkflowExecution) -> dict:
        """Serialize execution state for persistence.

        Returns a JSON-serializable dict that can be saved to SQLite or file.
        """
        return {
            "id": execution.id,
            "workflow_id": execution.workflow.id,
            "workflow_name": execution.workflow.name,
            "status": execution.status,
            "current_node": execution.current_node,
            "error": execution.error,
            "started_at": execution.started_at,
            "completed_at": execution.completed_at,
            "input_data": execution.input_data,
            "node_outputs": {
                k: _safe_summary(v)
                for k, v in execution.node_outputs.items()
            },
            "execution_log": execution.execution_log,
        }

    def save_checkpoint(self, execution: WorkflowExecution) -> None:
        """Persist checkpoint to the checkpoints directory."""
        cp_dir = self._workflows_dir / "checkpoints"
        cp_dir.mkdir(parents=True, exist_ok=True)
        path = cp_dir / f"{execution.id}.json"
        path.write_text(
            json.dumps(self.checkpoint(execution), indent=2),
            encoding="utf-8",
        )
        log.debug("workflow.checkpoint_saved id=%s", execution.id)

    def load_checkpoint(self, execution_id: str) -> dict | None:
        """Load a checkpoint from disk."""
        cp_dir = self._workflows_dir / "checkpoints"
        path = cp_dir / f"{execution_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_checkpoints(self) -> list[dict]:
        """List all saved checkpoints."""
        cp_dir = self._workflows_dir / "checkpoints"
        if not cp_dir.exists():
            return []
        result = []
        for p in sorted(cp_dir.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                result.append({
                    "id": data.get("id"),
                    "workflow_id": data.get("workflow_id"),
                    "status": data.get("status"),
                    "current_node": data.get("current_node"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return result

    async def resume(
        self,
        execution_id: str,
        progress_callback: Callable | None = None,
    ) -> WorkflowExecution | None:
        """Resume a checkpointed execution from its last completed node.

        Returns the resumed WorkflowExecution, or None if not found.
        """
        cp = self.load_checkpoint(execution_id)
        if cp is None:
            return None

        workflow = self.load_workflow(cp["workflow_id"])
        if workflow is None:
            log.warning("workflow.resume_failed workflow_id=%s not_found", cp["workflow_id"])
            return None

        execution = WorkflowExecution(workflow, cp.get("input_data"))
        execution.id = cp["id"]
        execution.status = "running"
        execution.started_at = cp.get("started_at")
        execution.node_outputs = cp.get("node_outputs", {})
        execution.execution_log = cp.get("execution_log", [])
        execution._tool_executor = self._tool_executor
        self._executions[execution.id] = execution

        # Find the next unexecuted node after current_node
        current_node_id = cp.get("current_node")
        if current_node_id:
            outgoing = workflow.get_outgoing_edges(current_node_id)
            for edge in outgoing:
                next_node = workflow.get_node(edge.to_node)
                if next_node and next_node.id not in execution.node_outputs:
                    try:
                        await self._execute_node(execution, next_node, progress_callback)
                    except Exception as e:
                        execution.status = "failed"
                        execution.error = str(e)

        if execution.status == "running":
            execution.status = "completed"
        execution.completed_at = time.time()

        # Clean up checkpoint
        cp_path = self._workflows_dir / "checkpoints" / f"{execution_id}.json"
        if cp_path.exists():
            cp_path.unlink()

        return execution

    def delete_checkpoint(self, execution_id: str) -> bool:
        """Delete a checkpoint."""
        cp_dir = self._workflows_dir / "checkpoints"
        path = cp_dir / f"{execution_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def _trim_executions(self) -> None:
        if len(self._executions) > self._max_executions:
            oldest = sorted(
                self._executions.keys(),
                key=lambda k: (
                    self._executions[k].started_at or 0
                ),
            )
            for k in oldest[
                : len(oldest) - self._max_executions
            ]:
                del self._executions[k]


def _safe_summary(val: Any) -> Any:
    """Summarize a value for serialization."""
    if isinstance(val, dict):
        return {
            k: _safe_summary(v) for k, v in val.items()
        }
    if isinstance(val, str) and len(val) > 200:
        return val[:200] + "..."
    if isinstance(val, list) and len(val) > 10:
        return val[:10] + ["..."]
    return val

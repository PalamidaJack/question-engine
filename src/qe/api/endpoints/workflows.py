"""Workflow management and execution endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/workflows", tags=["workflows"])


def _get_workflow_executor():
    import qe.api.app as app_mod

    executor = getattr(app_mod, "_workflow_executor", None)
    if executor is None:
        raise HTTPException(503, "Workflow executor not initialized")
    return executor


# ── Request / Response Models ───────────────────────────────────────


class WorkflowDefinition(BaseModel):
    name: str
    description: str | None = None
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


class ExecuteWorkflowBody(BaseModel):
    input: dict[str, Any] = {}


# ── Endpoints ───────────────────────────────────────────────────────
# NOTE: /executions and /node-types are placed before /{workflow_id}
# to avoid route shadowing.


@router.get("/executions")
async def list_executions(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List recent workflow executions."""
    executor = _get_workflow_executor()
    executions = await executor.list_executions(
        limit=limit, offset=offset,
    )
    return {
        "executions": executions,
        "count": len(executions),
        "limit": limit,
        "offset": offset,
    }


@router.get("/executions/{execution_id}")
async def get_execution(execution_id: str):
    """Get execution status and log for a specific run."""
    executor = _get_workflow_executor()
    execution = await executor.get_execution(execution_id)
    if execution is None:
        raise HTTPException(
            404, f"Execution {execution_id} not found",
        )
    return execution


@router.get("/node-types")
async def list_node_types():
    """List available node types for the workflow builder."""
    executor = _get_workflow_executor()
    node_types = await executor.list_node_types()
    return {"node_types": node_types, "count": len(node_types)}


@router.get("/")
async def list_workflows():
    """List all saved workflows."""
    executor = _get_workflow_executor()
    workflows = await executor.list_workflows()
    return {"workflows": workflows, "count": len(workflows)}


@router.post("/")
async def create_workflow(body: WorkflowDefinition):
    """Create and save a new workflow definition."""
    executor = _get_workflow_executor()
    workflow = await executor.create_workflow(
        name=body.name,
        description=body.description,
        nodes=body.nodes,
        edges=body.edges,
        metadata=body.metadata,
    )
    return workflow


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get a workflow definition by ID."""
    executor = _get_workflow_executor()
    workflow = await executor.get_workflow(workflow_id)
    if workflow is None:
        raise HTTPException(
            404, f"Workflow {workflow_id} not found",
        )
    return workflow


@router.put("/{workflow_id}")
async def update_workflow(workflow_id: str, body: WorkflowDefinition):
    """Update an existing workflow definition."""
    executor = _get_workflow_executor()
    existing = await executor.get_workflow(workflow_id)
    if existing is None:
        raise HTTPException(
            404, f"Workflow {workflow_id} not found",
        )
    updated = await executor.update_workflow(
        workflow_id,
        name=body.name,
        description=body.description,
        nodes=body.nodes,
        edges=body.edges,
        metadata=body.metadata,
    )
    return updated


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow by ID."""
    executor = _get_workflow_executor()
    existing = await executor.get_workflow(workflow_id)
    if existing is None:
        raise HTTPException(
            404, f"Workflow {workflow_id} not found",
        )
    await executor.delete_workflow(workflow_id)
    return {"status": "deleted", "workflow_id": workflow_id}


@router.post("/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: str,
    body: ExecuteWorkflowBody,
):
    """Execute a workflow with the given input."""
    executor = _get_workflow_executor()
    existing = await executor.get_workflow(workflow_id)
    if existing is None:
        raise HTTPException(
            404, f"Workflow {workflow_id} not found",
        )
    result = await executor.execute(
        workflow_id, input_data=body.input,
    )
    return result

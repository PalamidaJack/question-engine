"""A2A HTTP routes (Phase 4): inbound server for external agents.

Minimal, test-friendly implementation that maps A2A tasks to internal QE goals
via the planner/dispatcher when available. Maintains an in-memory mapping
`_a2a_task_map` in the module for task_id -> goal_id.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request

from qe.api.a2a.models import A2AMessage, A2ATask, AgentCard

router = APIRouter()

# In-memory mapping for A2A task -> goal
_a2a_task_map: dict[str, str] = {}

# In-memory mapping for A2A task -> received permissions
_a2a_task_permissions: dict[str, dict[str, Any]] = {}


def register_a2a_routes(app: FastAPI) -> None:
    app.include_router(router)


@router.get("/.well-known/agent.json")
async def agent_card():
    card = AgentCard()
    return card.model_dump()


@router.post("/api/a2a/tasks")
async def create_task(payload: dict[str, Any]):
    task_id = payload.get("id") or f"a2a_{uuid.uuid4().hex[:12]}"
    _task = A2ATask(id=task_id)

    # Map to an internal goal if planner available
    try:
        from qe.api import app as _app_module  # type: ignore
        planner = getattr(_app_module, "_planner", None)
        dispatcher = getattr(_app_module, "_dispatcher", None)
        bus = getattr(_app_module, "_bus", None)
    except Exception:
        planner = None
        dispatcher = None
        bus = None

    # simple: submit as planner.decompose -> dispatcher.submit_goal
    goal_id = None
    if planner and dispatcher:
        description = payload.get("description", "A2A delegated task")
        try:
            state = await planner.decompose(description)
            await dispatcher.submit_goal(state)
            goal_id = state.goal_id
        except Exception:
            goal_id = None

    _a2a_task_map[task_id] = goal_id or ""

    # Store received permissions if present
    metadata = payload.get("metadata") or {}
    if "permissions" in metadata:
        _a2a_task_permissions[task_id] = metadata["permissions"]

    # publish bus event
    if bus is not None:
        try:
            bus.publish({
                "topic": "a2a.task_received",
                "payload": {"task_id": task_id, "goal_id": goal_id},
            })
        except Exception:
            pass

    return {"task_id": task_id, "goal_id": goal_id}


@router.get("/api/a2a/tasks/{task_id}")
async def get_task(task_id: str):
    if task_id not in _a2a_task_map:
        raise HTTPException(status_code=404, detail="task not found")
    return {"task_id": task_id, "goal_id": _a2a_task_map.get(task_id)}


@router.post("/api/a2a/tasks/{task_id}/messages")
async def post_message(task_id: str, request: Request):
    if task_id not in _a2a_task_map:
        raise HTTPException(status_code=404, detail="task not found")
    payload = await request.json()
    msg = A2AMessage(**payload)

    # forward to bus
    try:
        from qe.api import app as _app_module  # type: ignore
        bus = getattr(_app_module, "_bus", None)
        if bus is not None:
            bus.publish({
                "topic": "a2a.message_received",
                "payload": {"task_id": task_id, "message": msg.model_dump()},
            })
    except Exception:
        pass

    return {"status": "ok"}


@router.get("/api/a2a/tasks/{task_id}/messages")
async def get_messages(task_id: str):
    # Minimal: return empty list — bus-driven message history would be implemented later
    if task_id not in _a2a_task_map:
        raise HTTPException(status_code=404, detail="task not found")
    return {"messages": []}


@router.get("/api/a2a/tasks/{task_id}/stream")
async def stream_task(task_id: str):
    from fastapi.responses import StreamingResponse

    if task_id not in _a2a_task_map:
        raise HTTPException(status_code=404, detail="task not found")

    queue: asyncio.Queue | None = asyncio.Queue()

    async def generator():
        # Subscribe to bus events for this task_id
        try:
            from qe.bus import get_bus

            bus = get_bus()

            async def _on_event(envelope):
                try:
                    payload = envelope.payload
                    if payload.get("task_id") == task_id:
                        await queue.put(payload)
                except Exception:
                    pass

            bus.subscribe("a2a.task_completed", _on_event)
            bus.subscribe("a2a.task_failed", _on_event)
        except Exception:
            pass

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {item}\n\n"
        finally:
            try:
                bus.unsubscribe("a2a.task_completed", _on_event)
                bus.unsubscribe("a2a.task_failed", _on_event)
            except Exception:
                pass

    return StreamingResponse(generator(), media_type="text/event-stream")

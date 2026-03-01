"""Goals API endpoints."""

from __future__ import annotations

from typing import Any


def register_goal_routes(
    app: Any,
    planner: Any,
    dispatcher: Any,
    goal_store: Any,
) -> None:
    """Register goal-related API routes."""
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    _ = planner

    @app.get("/api/goals/{goal_id}/dag")
    async def get_goal_dag(goal_id: str):
        if not goal_store:
            raise HTTPException(
                503, "Goal store not initialized"
            )
        state = await goal_store.load_goal(goal_id)
        if not state:
            raise HTTPException(404, "Goal not found")
        if not state.decomposition:
            return {"nodes": [], "edges": []}

        nodes = []
        edges = []
        for st in state.decomposition.subtasks:
            nodes.append({
                "id": st.subtask_id,
                "description": st.description,
                "task_type": st.task_type,
                "model_tier": st.model_tier,
                "status": state.subtask_states.get(
                    st.subtask_id, "pending"
                ),
            })
            for dep in st.depends_on:
                edges.append({
                    "source": dep,
                    "target": st.subtask_id,
                })

        return {"nodes": nodes, "edges": edges}

    @app.get("/api/goals/{goal_id}/progress")
    async def get_goal_progress(goal_id: str):
        # Try in-memory dispatcher state first, fall back to persisted store
        state = dispatcher.get_goal_state(goal_id) if dispatcher else None
        if state is None and goal_store:
            state = await goal_store.load_goal(goal_id)
        if state is None:
            raise HTTPException(404, "Goal not found")

        subtasks_info = []
        completed = 0
        failed = 0
        pending = 0
        total = len(state.subtask_states)

        for sid, status in state.subtask_states.items():
            if status == "completed":
                completed += 1
            elif status == "failed":
                failed += 1
            else:
                pending += 1

            result = state.subtask_results.get(sid)
            subtask_desc = ""
            task_type = ""
            if state.decomposition:
                for st in state.decomposition.subtasks:
                    if st.subtask_id == sid:
                        subtask_desc = st.description
                        task_type = st.task_type
                        break

            subtasks_info.append({
                "subtask_id": sid,
                "description": subtask_desc,
                "task_type": task_type,
                "status": status,
                "latency_ms": result.latency_ms if result else 0,
                "cost_usd": result.cost_usd if result else 0.0,
                "model_used": result.model_used if result else "",
            })

        pct_complete = (completed / total * 100) if total > 0 else 0.0

        return {
            "goal_id": state.goal_id,
            "description": state.description,
            "status": state.status,
            "progress": {
                "total": total,
                "completed": completed,
                "failed": failed,
                "pending": pending,
                "pct_complete": round(pct_complete, 1),
            },
            "subtasks": subtasks_info,
        }

    @app.get("/api/goals/{goal_id}/result")
    async def get_goal_result(goal_id: str):
        if not goal_store:
            raise HTTPException(503, "Goal store not initialized")
        state = await goal_store.load_goal(goal_id)
        if state is None:
            raise HTTPException(404, "Goal not found")

        if state.status != "completed":
            return JSONResponse(
                status_code=202,
                content={
                    "goal_id": goal_id,
                    "status": state.status,
                    "message": "Goal not yet completed",
                },
            )

        goal_result = state.metadata.get("goal_result")
        if goal_result is None:
            return JSONResponse(
                status_code=202,
                content={"goal_id": goal_id, "status": "completed", "message": "Synthesis pending"},
            )

        return goal_result

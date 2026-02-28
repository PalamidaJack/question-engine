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

    _ = planner
    _ = dispatcher

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

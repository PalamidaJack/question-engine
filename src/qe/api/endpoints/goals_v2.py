"""Goals API v2 endpoints extracted from app.py."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from qe.bus import get_bus
from qe.models.envelope import Envelope
from qe.runtime.feature_flags import get_flag_store
from qe.runtime.readiness import get_readiness

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/goals", tags=["Goals"])


def get_app_globals():
    import qe.api.app as app_mod

    return app_mod

@router.post("")
async def submit_goal(request: Request, body: dict[str, Any]):
    """Submit a new goal for decomposition and execution.

    When the inquiry_mode feature flag is enabled, routes through the
    InquiryEngine (v2 7-phase loop) instead of the v1 pipeline.
    """
    description = body.get("description", "").strip()
    if not description:
        return JSONResponse(
            {"error": "description is required"}, status_code=400
        )

    # v2 Multi-agent path
    app_mod = get_app_globals()
    flag_store = get_flag_store()
    if flag_store.is_enabled("multi_agent_mode"):
        try:
            _cognitive_pool = app_mod._cognitive_pool
            if _cognitive_pool is not None:
                goal_id = f"goal_{uuid.uuid4().hex[:12]}"

                # Select strategy for this inquiry
                config = None
                _strategy_evolver = app_mod._strategy_evolver
                if _strategy_evolver is not None:
                    from qe.runtime.strategy_models import strategy_to_inquiry_config
                    strategy = _strategy_evolver.select_strategy()
                    config = strategy_to_inquiry_config(strategy)

                # Competitive arena path: tournament verification
                if (
                    flag_store.is_enabled("competitive_arena")
                    and app_mod._competitive_arena is not None
                ):
                    from qe.models.arena import ArenaResult

                    result = await _cognitive_pool.run_competitive_inquiry(
                        goal_id=goal_id,
                        goal_description=description,
                        config=config,
                    )
                    if isinstance(result, ArenaResult):
                        return {
                            "arena_id": result.arena_id,
                            "goal_id": result.goal_id,
                            "winner_id": result.winner_id,
                            "sycophancy_detected": result.sycophancy_detected,
                            "match_count": len(result.matches),
                            "total_cost_usd": result.total_cost_usd,
                            "mode": "competitive_arena",
                        }
                    # Fell through to InquiryResult (< 2 agents)
                    if result is not None:
                        return {
                            "goal_id": result.goal_id,
                            "inquiry_id": result.inquiry_id,
                            "status": result.status,
                            "findings_summary": result.findings_summary[:1000],
                            "mode": "multi_agent",
                        }

                # Standard multi-agent path: parallel + merge
                results = await _cognitive_pool.run_parallel_inquiry(
                    goal_id=goal_id,
                    goal_description=description,
                    config=config,
                )
                if results:
                    merged = await _cognitive_pool.merge_results(results)
                    return {
                        "goal_id": merged.goal_id,
                        "inquiry_id": merged.inquiry_id,
                        "status": merged.status,
                        "termination_reason": merged.termination_reason,
                        "iterations": merged.iterations_completed,
                        "questions_answered": merged.total_questions_answered,
                        "insights_count": len(merged.insights),
                        "findings_summary": merged.findings_summary[:1000],
                        "mode": "multi_agent",
                    }
        except Exception:
            log.debug("submit_goal.multi_agent_fallthrough")

    # v2 Inquiry path (single agent)
    _inquiry_engine = app_mod._inquiry_engine
    if flag_store.is_enabled("inquiry_mode") and _inquiry_engine is not None:
        goal_id = f"goal_{uuid.uuid4().hex[:12]}"

        # Select strategy via Thompson sampling
        config = None
        _strategy_evolver = app_mod._strategy_evolver
        if _strategy_evolver is not None:
            from qe.runtime.strategy_models import strategy_to_inquiry_config
            strategy = _strategy_evolver.select_strategy()
            config = strategy_to_inquiry_config(strategy)

        result = await _inquiry_engine.run_inquiry(
            goal_id=goal_id,
            goal_description=description,
            config=config,
        )
        app_mod._last_inquiry_profile = result.phase_timings
        app_mod._inquiry_profiling_store.record(result.phase_timings, result.duration_seconds)

        # Update readiness with inquiry status
        readiness = get_readiness()
        readiness.last_inquiry_status = result.status
        readiness.last_inquiry_at = time.monotonic()
        readiness.last_inquiry_duration_s = result.duration_seconds

        return {
            "goal_id": result.goal_id,
            "inquiry_id": result.inquiry_id,
            "status": result.status,
            "termination_reason": result.termination_reason,
            "iterations": result.iterations_completed,
            "questions_answered": result.total_questions_answered,
            "insights_count": len(result.insights),
            "findings_summary": result.findings_summary[:1000],
        }

    # v1 Pipeline path
    if not request.app.state.planner or not request.app.state.dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    state = await request.app.state.planner.decompose(description)
    await request.app.state.dispatcher.submit_goal(state)

    get_bus().publish(
        Envelope(
            topic="goals.submitted",
            source_service_id="api",
            correlation_id=state.goal_id,
            payload={
                "goal_id": state.goal_id,
                "description": description,
            },
        )
    )

    return {
        "goal_id": state.goal_id,
        "status": state.status,
        "subtask_count": len(state.subtask_states),
        "strategy": (
            state.decomposition.strategy if state.decomposition else ""
        ),
    }


@router.get("")
async def list_goals(request: Request, status: str | None = None):
    """List all goals with status."""
    if not request.app.state.goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    goals = await request.app.state.goal_store.list_goals(status=status)
    return {
        "goals": [
            {
                "goal_id": g.goal_id,
                "description": g.description,
                "status": g.status,
                "subtask_count": len(g.subtask_states),
                "created_at": g.created_at.isoformat(),
                "completed_at": (
                    g.completed_at.isoformat()
                    if g.completed_at
                    else None
                ),
            }
            for g in goals
        ],
        "count": len(goals),
    }


@router.get("/{goal_id}")
async def get_goal(request: Request, goal_id: str):
    """Get goal detail with DAG and subtask states."""
    if not request.app.state.dispatcher or not request.app.state.goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    # Try in-memory first, then store
    state = request.app.state.dispatcher.get_goal_state(goal_id)
    if not state:
        state = await request.app.state.goal_store.load_goal(goal_id)
    if not state:
        return JSONResponse({"error": "Goal not found"}, status_code=404)

    return {
        "goal_id": state.goal_id,
        "description": state.description,
        "status": state.status,
        "subtask_states": state.subtask_states,
        "created_at": state.created_at.isoformat(),
        "completed_at": (
            state.completed_at.isoformat()
            if state.completed_at
            else None
        ),
        "decomposition": (
            state.decomposition.model_dump(mode="json")
            if state.decomposition
            else None
        ),
    }


@router.post("/{goal_id}/pause")
async def pause_goal(request: Request, goal_id: str):
    """Pause execution of a goal."""
    if not request.app.state.dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)
    ok = await request.app.state.dispatcher.pause_goal(goal_id)
    if not ok:
        return JSONResponse(
            {"error": "Goal not found or not running"}, status_code=404
        )
    return {"status": "paused", "goal_id": goal_id}


@router.post("/{goal_id}/resume")
async def resume_goal(request: Request, goal_id: str):
    """Resume a paused goal."""
    if not request.app.state.dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)
    ok = await request.app.state.dispatcher.resume_goal(goal_id)
    if not ok:
        return JSONResponse(
            {"error": "Goal not found or not paused"}, status_code=404
        )
    return {"status": "resumed", "goal_id": goal_id}


@router.post("/{goal_id}/assign")
async def assign_goal_to_project(request: Request, goal_id: str, body: dict[str, Any]):
    """Assign a goal to a project."""
    if not request.app.state.goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    project_id = body.get("project_id")
    if not project_id:
        return JSONResponse(
            {"error": "project_id is required"}, status_code=400
        )

    state = await request.app.state.goal_store.load_goal(goal_id)
    if not state:
        return JSONResponse({"error": "Goal not found"}, status_code=404)

    state.project_id = project_id
    await request.app.state.goal_store.save_goal(state)
    return {"status": "assigned", "goal_id": goal_id, "project_id": project_id}


@router.post("/{goal_id}/cancel")
async def cancel_goal(request: Request, goal_id: str):
    """Cancel a running goal."""
    if not request.app.state.dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)
    ok = await request.app.state.dispatcher.cancel_goal(goal_id)
    if not ok:
        return JSONResponse(
            {"error": "Goal not found"}, status_code=404
        )
    return {"status": "cancelled", "goal_id": goal_id}


# ── Projects ─────────────────────────────────────────────────────────────────



"""Telemetry API endpoints extracted from app.py."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from qe.audit import get_audit_log
from qe.runtime.metrics import get_metrics
from qe.bus.bus_metrics import get_bus_metrics

router = APIRouter(prefix="/api", tags=["Telemetry"])




@router.get("/pool/status")
async def pool_status(request: Request):
    """Return cognitive agent pool status."""
    if _cognitive_pool is None:
        return JSONResponse({"error": "Cognitive pool not initialized"}, status_code=503)
    return _cognitive_pool.pool_status()


@router.get("/arena/status")
async def arena_status(request: Request):
    """Return competitive arena status and Elo rankings."""
    if _competitive_arena is None:
        return {"enabled": False, "rankings": []}
    return _competitive_arena.status()


@router.get("/strategy/snapshots")
async def strategy_snapshots(request: Request):
    """Return strategy evolver snapshots and current strategy."""
    if _strategy_evolver is None:
        return JSONResponse({"error": "Strategy evolver not initialized"}, status_code=503)
    return {
        "current_strategy": _strategy_evolver._current_strategy,
        "scaling_profile": (
            _elastic_scaler.current_profile_name() if _elastic_scaler else None
        ),
        "snapshots": [s.model_dump() for s in _strategy_evolver.get_snapshots()],
    }


@router.get("/flags")
async def list_flags(request: Request):
    """List all feature flags and stats."""
    store = get_flag_store()
    return {
        "flags": store.list_flags(),
        "stats": store.stats(),
    }


@router.get("/flags/evaluations")
async def flag_evaluations(request: Request, limit: int = 100):
    """Return recent flag evaluation log."""
    store = get_flag_store()
    evaluations = store.evaluation_log(limit=limit)
    return {
        "evaluations": evaluations,
        "count": len(evaluations),
    }


@router.get("/flags/{flag_name}")
async def get_flag(request: Request, flag_name: str):
    """Get a single feature flag state."""
    flag = get_flag_store().get(flag_name)
    if flag is None:
        return JSONResponse({"error": f"Flag '{flag_name}' not found"}, status_code=404)
    return flag.to_dict()


@router.post("/flags/{flag_name}/enable")
async def enable_flag(request: Request, flag_name: str):
    """Enable a feature flag at runtime."""
    if not get_flag_store().enable(flag_name):
        return JSONResponse({"error": f"Flag '{flag_name}' not found"}, status_code=404)
    get_audit_log().record("flag.enabled", resource=f"flag/{flag_name}")
    return {"status": "enabled", "flag_name": flag_name}


@router.post("/flags/{flag_name}/disable")
async def disable_flag(request: Request, flag_name: str):
    """Disable a feature flag at runtime."""
    if not get_flag_store().disable(flag_name):
        return JSONResponse({"error": f"Flag '{flag_name}' not found"}, status_code=404)
    get_audit_log().record("flag.disabled", resource=f"flag/{flag_name}")
    return {"status": "disabled", "flag_name": flag_name}


@router.get("/episodic/status")
async def episodic_status(request: Request):
    """Return episodic memory status overview."""
    if _episodic_memory is None:
        return JSONResponse({"error": "Episodic memory not initialized"}, status_code=503)
    status = _episodic_memory.status()
    warm = await _episodic_memory.warm_count()
    return {
        "hot_entries": status["hot_entries"],
        "max_hot": status["max_hot"],
        "warm_entries": warm,
    }


@router.get("/episodic/search")
async def episodic_search(request: Request, 
    query: str = "",
    top_k: int = 10,
    goal_id: str | None = None,
    episode_type: str | None = None,
    time_window_hours: float | None = None,
):
    """Search episodic memory by keyword + recency."""
    if _episodic_memory is None:
        return JSONResponse({"error": "Episodic memory not initialized"}, status_code=503)
    if not query:
        return JSONResponse({"error": "query parameter is required"}, status_code=400)
    episodes = await _episodic_memory.recall(
        query, top_k=top_k, time_window_hours=time_window_hours,
        goal_id=goal_id, episode_type=episode_type,
    )
    return {
        "episodes": [ep.model_dump(mode="json") for ep in episodes],
        "count": len(episodes),
    }


@router.get("/episodic/goal/{goal_id}")
async def episodic_goal(request: Request, goal_id: str, top_k: int = 20):
    """Return episodes for a specific goal."""
    if _episodic_memory is None:
        return JSONResponse({"error": "Episodic memory not initialized"}, status_code=503)
    episodes = await _episodic_memory.recall_for_goal(goal_id, top_k=top_k)
    return {
        "goal_id": goal_id,
        "episodes": [ep.model_dump(mode="json") for ep in episodes],
        "count": len(episodes),
    }


@router.get("/episodic/latest")
async def episodic_latest(request: Request, limit: int = 20):
    """Return most recent episodes from hot store."""
    if _episodic_memory is None:
        return JSONResponse({"error": "Episodic memory not initialized"}, status_code=503)
    episodes = _episodic_memory.get_latest(limit=limit)
    return {
        "episodes": [ep.model_dump(mode="json") for ep in episodes],
        "count": len(episodes),
    }


# ── Bus Stats ─────────────────────────────────────────────────────────────


@router.get("/bus/stats")
async def bus_stats(request: Request):
    """Return per-topic bus metrics: publish counts, latency, errors."""
    return get_bus_metrics().snapshot()



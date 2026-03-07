"""Model Intelligence API endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])


def _get_model_intelligence():
    import qe.api.app as app_mod

    svc = getattr(app_mod, "_model_intelligence", None)
    if svc is None:
        raise HTTPException(503, "Model Intelligence not initialized")
    return svc


# ── Request / Response Models ───────────────────────────────────────


class SetPreferenceBody(BaseModel):
    model_id: str
    task_type: str
    preference: float


class CompareBody(BaseModel):
    model_a: str
    model_b: str
    winner: str
    task_context: str = ""


# ── Endpoints ───────────────────────────────────────────────────────
# NOTE: /stats is placed before /{model_id} to avoid route shadowing.


@router.get("/stats")
async def model_stats():
    """Return model intelligence aggregate statistics."""
    svc = _get_model_intelligence()
    return await svc.stats()


@router.get("/")
async def list_models(
    provider: str | None = None,
    task_type: str | None = None,
    is_free: bool | None = None,
):
    """List all known models with summary info."""
    svc = _get_model_intelligence()
    filters: dict[str, Any] = {}
    if provider is not None:
        filters["provider"] = provider
    if task_type is not None:
        filters["task_type"] = task_type
    if is_free is not None:
        filters["is_free"] = is_free
    models = await svc.list_models(filters=filters or None)
    return {"models": models, "count": len(models)}


@router.get("/rankings")
async def get_rankings(
    task_type: str | None = Query(None),
):
    """Get ranked models, optionally filtered by task type."""
    svc = _get_model_intelligence()
    rankings = await svc.get_rankings(task_type=task_type)
    return {"rankings": rankings, "count": len(rankings)}


@router.get("/{model_id:path}/profile")
async def get_model_profile(model_id: str):
    """Get full model profile: scores and narrative."""
    svc = _get_model_intelligence()
    profile = await svc.get_model_profile(model_id)
    if profile is None:
        raise HTTPException(404, f"Model {model_id} not found")
    return profile


@router.get("/{model_id:path}/scores")
async def get_model_scores(model_id: str):
    """Get benchmark scores for a model."""
    svc = _get_model_intelligence()
    scores = await svc.get_model_scores(model_id)
    if scores is None:
        raise HTTPException(404, f"Model {model_id} not found")
    return scores


@router.get("/{model_id:path}/markdown")
async def get_model_markdown(model_id: str):
    """Get the narrative markdown profile for a model."""
    svc = _get_model_intelligence()
    md = await svc.get_markdown_profile(model_id)
    if md is None:
        raise HTTPException(404, f"Model {model_id} not found")
    return {"model_id": model_id, "markdown": md}


@router.post("/{model_id:path}/profile")
async def trigger_reprofiling(model_id: str):
    """Trigger re-profiling for a model."""
    svc = _get_model_intelligence()
    result = await svc.trigger_profile(model_id)
    return {
        "status": "reprofiling_started",
        "model_id": model_id,
        "detail": result,
    }


@router.post("/preference")
async def set_preference(body: SetPreferenceBody):
    """Set a user preference for a model on a task type."""
    svc = _get_model_intelligence()
    await svc.set_user_preference(
        model_id=body.model_id,
        task_type=body.task_type,
        preference=body.preference,
    )
    return {
        "status": "saved",
        "model_id": body.model_id,
        "task_type": body.task_type,
    }


@router.post("/compare")
async def record_comparison(body: CompareBody):
    """Record a head-to-head model comparison result."""
    svc = _get_model_intelligence()
    await svc.record_preference(
        model_a=body.model_a,
        model_b=body.model_b,
        winner=body.winner,
        task_context=body.task_context,
    )
    return {
        "status": "recorded",
        "model_a": body.model_a,
        "model_b": body.model_b,
        "winner": body.winner,
    }

"""API endpoints for tool metrics and tier configuration."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/tools", tags=["tools"])


def _get_app_globals():
    import qe.api.app as app_mod
    return app_mod


@router.get("/metrics")
async def get_tool_metrics(request: Request):
    """Return per-tool quality metrics."""
    app = _get_app_globals()
    tm = getattr(app, "_tool_metrics", None)
    if tm is None:
        return {"error": "tool_quality_metrics not initialized"}
    return {
        "summary": tm.summary(),
        "tools": tm.all_stats(),
    }


@router.get("/metrics/{tool_name}")
async def get_tool_metric(tool_name: str, request: Request):
    """Return metrics for a specific tool."""
    app = _get_app_globals()
    tm = getattr(app, "_tool_metrics", None)
    if tm is None:
        return {"error": "tool_quality_metrics not initialized"}
    stats = tm.get_stats(tool_name)
    if stats is None:
        return {"error": f"no metrics for tool: {tool_name}"}
    return stats


@router.get("/tiers")
async def get_tier_config(request: Request):
    """Return current tier configuration from the router."""
    # Try to get router from chat service or supervisor
    app = _get_app_globals()
    cs = getattr(app, "_chat_service", None)
    if cs and hasattr(cs, "_router") and cs._router:
        return cs._router.tier_config()
    return {"error": "router not available"}

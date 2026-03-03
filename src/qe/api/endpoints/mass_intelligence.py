"""Mass Intelligence API endpoints extracted from app.py."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/mass-intelligence", tags=["Mass Intelligence"])




@router.get("/status")
async def mass_intelligence_status(request: Request):
    """Return mass intelligence services status."""
    if request.app.state.mass_intelligence_market_agent is None:
        return {"running": False}

    stats = await request.app.state.mass_intelligence_market_agent.get_stats()
    agent_status = request.app.state.mass_intelligence_market_agent.status()

    return {
        "running": agent_status["running"],
        "poll_interval_seconds": agent_status["poll_interval_seconds"],
        "stats": stats,
    }


@router.get("/models")
async def mass_intelligence_models(request: Request):
    """Return list of available free models."""
    if request.app.state.mass_intelligence_store is None:
        return {"models": [], "error": "Service not initialized"}

    models = await request.app.state.mass_intelligence_store.get_available_models()
    return {"models": models, "count": len(models)}


@router.post("/query")
async def mass_intelligence_query(request: Request, prompt: str, system_message: str | None = None):
    """Execute a prompt across all available free models."""
    if request.app.state.mass_intelligence_executor is None:
        return {"error": "Service not initialized", "responses": []}

    result = await request.app.state.mass_intelligence_executor.execute(
        prompt=prompt,
        system_message=system_message,
    )

    return {
        "prompt": result.prompt,
        "total_models": result.total_models,
        "successful": result.successful,
        "failed": result.failed,
        "total_time_ms": result.total_time_ms,
        "responses": [
            {
                "provider": r.provider,
                "model_id": r.model_id,
                "model_name": r.model_name,
                "response": r.response,
                "latency_ms": r.latency_ms,
                "success": r.success,
                "error": r.error,
            }
            for r in result.responses
        ],
    }


@router.post("/quick")
async def mass_intelligence_quick(request: Request, prompt: str, max_models: int = 5):
    """Quick query with limited models for faster response."""
    if request.app.state.mass_intelligence_executor is None:
        return {"error": "Service not initialized", "responses": []}

    result = await request.app.state.mass_intelligence_executor.quick_query(
        prompt=prompt,
        max_models=max_models,
    )

    return {
        "prompt": result.prompt,
        "total_models": result.total_models,
        "successful": result.successful,
        "failed": result.failed,
        "total_time_ms": result.total_time_ms,
        "responses": [
            {
                "provider": r.provider,
                "model_id": r.model_id,
                "model_name": r.model_name,
                "response": r.response,
                "latency_ms": r.latency_ms,
                "success": r.success,
                "error": r.error,
            }
            for r in result.responses
        ],
    }


@router.post("/refresh")
async def mass_intelligence_refresh(request: Request):
    """Force refresh of model inventory from providers."""
    if request.app.state.mass_intelligence_market_agent is None:
        return {"error": "Service not initialized"}

    await request.app.state.mass_intelligence_market_agent._scrape_all_providers()
    models = await request.app.state.mass_intelligence_store.get_available_models()

    return {
        "success": True,
        "models_count": len(models),
    }
    if proposal is None:
        return JSONResponse({"error": "Proposal not found"}, status_code=404)
    return proposal.model_dump(mode="json")



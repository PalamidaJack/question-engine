"""API endpoints for Guardrails (Phase 2)."""
from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/guardrails", tags=["Guardrails"])

class GuardrailsConfigUpdate(BaseModel):
    enabled: bool | None = None
    content_filter_enabled: bool | None = None
    pii_detection_enabled: bool | None = None
    cost_guard_enabled: bool | None = None
    cost_guard_threshold_usd: float | None = None
    hallucination_guard_enabled: bool | None = None

@router.get("/status")
async def guardrails_status(request: Request) -> dict[str, Any]:
    pipeline = request.app.state.guardrails_pipeline
    # Note: access _guardrails_config from state too
    config = getattr(request.app.state, "guardrails_config", None)
    
    data = {"configured": pipeline is not None, "config": None, "rules": []}
    if config is not None:
        data["config"] = config.model_dump() if hasattr(config, "model_dump") else config
    if pipeline is not None:
        data["rules"] = [r.name for r in pipeline._rules]
    return data

@router.post("/configure")
async def guardrails_configure(request: Request, update: GuardrailsConfigUpdate):
    cfg = getattr(request.app.state, "guardrails_config", None)
    pipeline = request.app.state.guardrails_pipeline
    
    if cfg is None or pipeline is None:
        raise HTTPException(status_code=503, detail="guardrails not initialized")
    
    body = update.model_dump(exclude_none=True)
    for k, v in body.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
            
    from qe.runtime.guardrails import GuardrailsPipeline
    # Rebuild
    new_pipeline = GuardrailsPipeline.default_pipeline(
        config=cfg, 
        bus=request.app.state.bus
    )
    request.app.state.guardrails_pipeline = new_pipeline
    return {"status": "ok", "applied": body}

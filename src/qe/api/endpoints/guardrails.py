"""API endpoints for Guardrails pipeline (Phase 2).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

router = APIRouter()


class GuardrailsConfigUpdate(BaseModel):
    enabled: bool | None = None
    content_filter_enabled: bool | None = None
    pii_detection_enabled: bool | None = None
    cost_guard_enabled: bool | None = None
    cost_guard_threshold_usd: float | None = None
    hallucination_guard_enabled: bool | None = None


def register_guardrails_routes(app: FastAPI, pipeline: Any | None = None) -> None:
    app.include_router(router, prefix="/api/guardrails")


@router.get("/status")
async def guardrails_status() -> dict[str, Any]:
    from qe.api import app as _app_module  # type: ignore
    pipeline = getattr(_app_module, "_guardrails_pipeline", None)
    config = getattr(_app_module, "_guardrails_config", None)
    data = {"configured": pipeline is not None, "config": None, "rules": []}
    if config is not None:
        data["config"] = config.model_dump() if hasattr(config, "model_dump") else config
    if pipeline is not None:
        data["rules"] = [r.name for r in pipeline._rules]
    return data


@router.post("/configure")
async def guardrails_configure(update: GuardrailsConfigUpdate):
    from qe.api import app as _app_module  # type: ignore
    cfg = getattr(_app_module, "_guardrails_config", None)
    pipeline = getattr(_app_module, "_guardrails_pipeline", None)
    if cfg is None or pipeline is None:
        raise HTTPException(status_code=503, detail="guardrails not initialized")
    # Apply updates conservatively
    body = update.model_dump(exclude_none=True)
    for k, v in body.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    # rebuild pipeline
    from qe.runtime.guardrails import GuardrailsPipeline

    new_pipeline = GuardrailsPipeline.default_pipeline(
        config=cfg, bus=getattr(_app_module, "_bus", None),
    )
    _app_module._guardrails_pipeline = new_pipeline
    return {"status": "ok", "applied": body}

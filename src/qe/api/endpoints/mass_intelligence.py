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
async def mass_intelligence_query(
    request: Request,
    prompt: str,
    system_message: str | None = None,
    model_ids: str | None = None,
    providers: str | None = None,
    timeout_seconds: float | None = None,
):
    """Execute a prompt across available free models. Optionally filter by model_ids or providers (comma-separated)."""
    if request.app.state.mass_intelligence_executor is None:
        return {"error": "Service not initialized", "responses": []}

    result = await request.app.state.mass_intelligence_executor.execute(
        prompt=prompt,
        system_message=system_message,
        timeout_seconds=timeout_seconds,
        model_ids=model_ids.split(",") if model_ids else None,
        providers=providers.split(",") if providers else None,
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


@router.post("/consensus")
async def mass_intelligence_consensus(request: Request):
    """Generate a consensus synthesis from multiple model responses.

    Expects JSON body with:
      - prompt: the original question
      - responses: list of {model_name, response} objects from successful models
    """
    import litellm

    body = await request.json()
    original_prompt = body.get("prompt", "")
    responses = body.get("responses", [])

    if not responses:
        return JSONResponse({"error": "No responses to synthesize"}, status_code=400)

    # Build the synthesis prompt
    model_answers = "\n\n".join(
        f"--- {r.get('model_name', 'Unknown Model')} ---\n{r.get('response', '')}"
        for r in responses
        if r.get("response")
    )

    synthesis_prompt = (
        f"You are an expert analyst. Multiple AI models were asked the following question:\n\n"
        f'"{original_prompt}"\n\n'
        f"Here are their responses:\n\n{model_answers}\n\n"
        f"Please synthesize these responses into a single, comprehensive consensus answer. "
        f"Identify the key points that most models agree on, note any unique insights, "
        f"and highlight any contradictions. Structure your response clearly."
    )

    # Try multiple models with fallback
    import os

    kilo_key = os.environ.get("KILOCODE_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    candidates = []
    if kilo_key:
        candidates.extend([
            {"model": "openrouter/arcee-ai/trinity-large-preview:free", "api_base": "https://kilo.ai/api/openrouter", "api_key": kilo_key},
            {"model": "openrouter/google/gemma-3-27b-it:free", "api_base": "https://kilo.ai/api/openrouter", "api_key": kilo_key},
            {"model": "openrouter/meta-llama/llama-3.3-70b-instruct:free", "api_base": "https://kilo.ai/api/openrouter", "api_key": kilo_key},
        ])
    if openrouter_key:
        candidates.extend([
            {"model": "openrouter/google/gemma-3-27b-it:free", "api_key": openrouter_key},
            {"model": "openrouter/meta-llama/llama-3.3-70b-instruct:free", "api_key": openrouter_key},
        ])

    if not candidates:
        return JSONResponse({"error": "No API keys configured for consensus generation"}, status_code=503)

    messages = [{"role": "user", "content": synthesis_prompt}]
    last_error = None

    for candidate in candidates:
        kwargs = {"messages": messages, "max_tokens": 4096, **candidate}
        try:
            result = await litellm.acompletion(**kwargs)
            consensus = result.choices[0].message.content or ""
            return {
                "consensus": consensus,
                "model_used": candidate["model"],
                "source_count": len(responses),
            }
        except Exception as e:
            last_error = str(e)
            continue

    return JSONResponse({"error": f"All models failed. Last error: {last_error}"}, status_code=500)


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



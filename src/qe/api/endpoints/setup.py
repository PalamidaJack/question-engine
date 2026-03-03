"""Setup API endpoints extracted from app.py."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/setup", tags=["Setup"])

from qe.api.setup import (
    CHANNELS,
    PROVIDERS,
    get_configured_channels,
    get_configured_providers,
    get_current_tiers,
    is_setup_complete,
    save_setup,
)




@router.get("/status")
async def setup_status():
    """Return setup status: whether complete, configured providers, tier mapping, channels."""
    return {
        "complete": is_setup_complete(),
        "providers": get_configured_providers(),
        "tiers": get_current_tiers(),
        "channels": get_configured_channels(),
    }


@router.get("/providers")
async def setup_providers():
    """Return the static list of supported providers."""
    return {
        "providers": [
            {
                "name": p["name"],
                "env_var": p["env_var"],
                "example_models": p["example_models"],
                "tier_defaults": p["tier_defaults"],
            }
            for p in PROVIDERS
        ],
    }


@router.post("/save")
async def setup_save(body: dict[str, Any]):
    """Save provider API keys, tier assignments, and channel config.

    Expects:
        {
            "providers": {"OPENAI_API_KEY": "sk-...", ...},
            "tiers": {
                "fast": {"provider": "OpenAI", "model": "gpt-4o-mini"},
                "balanced": {"provider": "OpenAI", "model": "gpt-4o"},
                "powerful": {"provider": "Anthropic", "model": "claude-sonnet-4-20250514"}
            },
            "channels": {"TELEGRAM_BOT_TOKEN": "123:ABC...", ...}
        }
    """
    # Block setup changes after initial setup is complete
    if is_setup_complete():
        return JSONResponse(
            {
                "error": "Setup already complete. Use POST /api/setup/reconfigure to update.",
            },
            status_code=403,
        )

    providers = body.get("providers", {})
    tiers = body.get("tiers", {})
    channels = body.get("channels")

    if not providers and not tiers:
        return JSONResponse(
            {"error": "providers or tiers required"}, status_code=400
        )

    save_setup(providers=providers, tier_config=tiers, channels=channels)
    return {"status": "saved", "complete": is_setup_complete()}


@router.get("/channels")
async def setup_channels():
    """Return the static list of available communication channels."""
    return {
        "channels": [
            {
                "id": ch["id"],
                "name": ch["name"],
                "description": ch["description"],
                "always_on": ch.get("always_on", False),
                "env_vars": [
                    {"key": ev["key"], "label": ev["label"], "type": ev["type"]}
                    for ev in ch.get("env_vars", [])
                ],
            }
            for ch in CHANNELS
        ],
    }


@router.post("/reconfigure")
async def setup_reconfigure(body: dict[str, Any]):
    """Reconfigure providers, tiers, and channels after initial setup.

    Same payload shape as /api/setup/save but works after setup is complete.
    """
    providers = body.get("providers", {})
    tiers = body.get("tiers", {})
    channels = body.get("channels")

    if not providers and not tiers and not channels:
        return JSONResponse(
            {"error": "providers, tiers, or channels required"}, status_code=400
        )

    save_setup(providers=providers, tier_config=tiers, channels=channels)
    return {
        "status": "saved",
        "complete": is_setup_complete(),
        "note": "Restart required for channel changes to take effect.",
    }



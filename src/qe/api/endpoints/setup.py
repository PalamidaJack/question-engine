"""Setup API endpoints extracted from app.py."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from qe.api.setup import (
    CHANNELS,
    PROVIDERS,
    get_configured_channels,
    get_configured_providers,
    get_current_tiers,
    is_setup_complete,
    save_setup,
)

router = APIRouter(prefix="/api/setup", tags=["Setup"])


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
async def setup_reconfigure(request: Request, body: dict[str, Any]):
    """Reconfigure providers, tiers, and channels after initial setup.

    Same payload shape as /api/setup/save but works after setup is complete.
    API keys are set as environment variables immediately so services pick them up.
    """
    import os

    providers = body.get("providers", {})
    tiers = body.get("tiers", {})
    channels = body.get("channels")

    if not providers and not tiers and not channels:
        return JSONResponse(
            {"error": "providers, tiers, or channels required"}, status_code=400
        )

    # Set API keys as env vars so running services pick them up immediately
    provider_env_map = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
        "Google Gemini": "GEMINI_API_KEY",
        "Groq": "GROQ_API_KEY",
        "Mistral": "MISTRAL_API_KEY",
        "Cerebras": "CEREBRAS_API_KEY",
        "Together AI": "TOGETHERAI_API_KEY",
        "Kilo Code": "KILOCODE_API_KEY",
        "GitHub Models": "GITHUB_TOKEN",
        "Cloudflare Workers AI": "CLOUDFLARE_API_TOKEN",
        "NVIDIA NIM": "NVIDIA_NIM_API_KEY",
        "Fireworks AI": "FIREWORKS_AI_API_KEY",
        "Sambanova": "SAMBANOVA_API_KEY",
    }
    env_vars_set = []
    for provider_name, api_key in providers.items():
        if api_key and provider_name in provider_env_map:
            env_var = provider_env_map[provider_name]
            os.environ[env_var] = api_key
            env_vars_set.append(env_var)

    # Also update the mass intelligence executor's api_keys if running
    if env_vars_set and hasattr(request.app.state, "mass_intelligence_executor"):
        executor = request.app.state.mass_intelligence_executor
        if executor:
            key_map = {
                "OPENROUTER_API_KEY": "openrouter",
                "GROQ_API_KEY": "groq",
                "CEREBRAS_API_KEY": "cerebras",
                "MISTRAL_API_KEY": "mistral",
                "GEMINI_API_KEY": "google",
                "SAMBANOVA_API_KEY": "sambanova",
                "CLOUDFLARE_API_TOKEN": "cloudflare",
                "KILOCODE_API_KEY": "kilo",
            }
            for env_var in env_vars_set:
                if env_var in key_map:
                    executor.api_keys[key_map[env_var]] = os.environ[env_var]

    # Translate display names to env var names for save_setup (.env persistence)
    env_providers = {}
    for provider_name, api_key in providers.items():
        if api_key and provider_name in provider_env_map:
            env_providers[provider_env_map[provider_name]] = api_key
    save_setup(providers=env_providers, tier_config=tiers, channels=channels)
    return {
        "status": "saved",
        "complete": is_setup_complete(),
        "env_vars_updated": env_vars_set,
    }



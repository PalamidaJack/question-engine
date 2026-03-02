"""Per-provider model fetchers for free LLM discovery."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import httpx

from qe.runtime.discovery.schemas import DiscoveredModel

log = logging.getLogger(__name__)

_CLIENT_TIMEOUT = 15.0


# ── Shared helpers ────────────────────────────────────────────────────────


async def _http_get_json(
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> Any:
    """GET a JSON endpoint, returning parsed response."""
    async with httpx.AsyncClient(timeout=_CLIENT_TIMEOUT) as client:
        resp = await client.get(url, headers=headers or {}, params=params or {})
        resp.raise_for_status()
        return resp.json()


_POWERFUL_PATTERNS = re.compile(
    r"(70b|72b|405b|671b|opus|pro(?!mpt)|large|command-r-plus)", re.IGNORECASE
)
_FAST_PATTERNS = re.compile(
    r"(8b|7b|3b|1b|mini|flash|haiku|small|instant|nano|lite|micro|tiny)", re.IGNORECASE
)


def _infer_quality_tier(model_name: str, context_length: int = 4096) -> str:
    """Infer quality tier from model name and context length."""
    if _POWERFUL_PATTERNS.search(model_name):
        tier = "powerful"
    elif _FAST_PATTERNS.search(model_name):
        tier = "fast"
    else:
        tier = "balanced"

    # Context length > 100k bumps up one tier
    if context_length > 100_000:
        if tier == "fast":
            tier = "balanced"
        elif tier == "balanced":
            tier = "powerful"

    return tier


def _infer_capabilities(model_data: dict[str, Any]) -> dict[str, bool]:
    """Infer tool calling and JSON mode support from model metadata."""
    caps: dict[str, bool] = {
        "supports_tool_calling": False,
        "supports_json_mode": False,
        "supports_system_messages": True,
        "supports_streaming": True,
    }

    # OpenRouter-style architecture fields
    if "tool" in str(model_data).lower():
        caps["supports_tool_calling"] = True

    # Check for explicit capability markers
    name_lower = str(model_data.get("id", "")).lower()
    if any(k in name_lower for k in ("gpt-4", "claude", "gemini", "command-r", "llama-3")):
        caps["supports_tool_calling"] = True
        caps["supports_json_mode"] = True

    return caps


# ── Provider fetchers ─────────────────────────────────────────────────────


async def fetch_openrouter(api_key: str) -> list[DiscoveredModel]:
    """Fetch free models from OpenRouter."""
    now = datetime.now()
    data = await _http_get_json(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models: list[DiscoveredModel] = []
    for m in data.get("data", []):
        pricing = m.get("pricing", {})
        prompt_price = str(pricing.get("prompt", "1"))
        completion_price = str(pricing.get("completion", "1"))
        try:
            is_free = float(prompt_price) == 0 and float(completion_price) == 0
        except (ValueError, TypeError):
            is_free = False

        if not is_free:
            continue

        model_id = f"openrouter/{m['id']}"
        ctx = m.get("context_length", 4096) or 4096
        caps = _infer_capabilities(m)
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="openrouter",
                base_model_name=m["id"].rsplit("/", 1)[-1],
                is_free=True,
                context_length=ctx,
                quality_tier=_infer_quality_tier(m["id"], ctx),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=20,
                discovered_at=now,
                last_seen=now,
                **caps,
            )
        )
    log.info("discovery.openrouter found=%d free models", len(models))
    return models


async def fetch_groq(api_key: str) -> list[DiscoveredModel]:
    """Fetch models from Groq (all free, rate-limited)."""
    now = datetime.now()
    data = await _http_get_json(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models: list[DiscoveredModel] = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        model_id = f"groq/{mid}"
        ctx = m.get("context_window", 8192) or 8192
        caps = _infer_capabilities(m)
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="groq",
                base_model_name=mid,
                is_free=True,
                context_length=ctx,
                quality_tier=_infer_quality_tier(mid, ctx),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=30,
                discovered_at=now,
                last_seen=now,
                **caps,
            )
        )
    log.info("discovery.groq found=%d models", len(models))
    return models


async def fetch_cerebras(api_key: str) -> list[DiscoveredModel]:
    """Fetch models from Cerebras (all free, rate-limited)."""
    now = datetime.now()
    data = await _http_get_json(
        "https://api.cerebras.ai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models: list[DiscoveredModel] = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        model_id = f"cerebras/{mid}"
        ctx = m.get("context_length", 8192) or 8192
        caps = _infer_capabilities(m)
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="cerebras",
                base_model_name=mid,
                is_free=True,
                context_length=ctx,
                quality_tier=_infer_quality_tier(mid, ctx),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=30,
                discovered_at=now,
                last_seen=now,
                **caps,
            )
        )
    log.info("discovery.cerebras found=%d models", len(models))
    return models


async def fetch_gemini(api_key: str) -> list[DiscoveredModel]:
    """Fetch models from Google AI Studio (generous free tier)."""
    now = datetime.now()
    data = await _http_get_json(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": api_key},
    )
    models: list[DiscoveredModel] = []
    for m in data.get("models", []):
        name = m.get("name", "")  # e.g. "models/gemini-2.0-flash"
        short = name.removeprefix("models/")
        if not short:
            continue

        # Skip embedding / image models
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" not in methods:
            continue

        model_id = f"gemini/{short}"
        ctx = m.get("inputTokenLimit", 4096) or 4096
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="gemini",
                base_model_name=short,
                is_free=True,
                context_length=ctx,
                supports_tool_calling=True,
                supports_json_mode=True,
                quality_tier=_infer_quality_tier(short, ctx),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=15,
                discovered_at=now,
                last_seen=now,
            )
        )
    log.info("discovery.gemini found=%d models", len(models))
    return models


async def fetch_github_models(token: str) -> list[DiscoveredModel]:
    """Fetch models from GitHub Models (free preview)."""
    now = datetime.now()
    data = await _http_get_json(
        "https://models.inference.ai.azure.com/models",
        headers={"Authorization": f"Bearer {token}"},
    )
    models: list[DiscoveredModel] = []
    items = data if isinstance(data, list) else data.get("data", data.get("value", []))
    for m in items:
        mid = m.get("id", m.get("name", ""))
        if not mid:
            continue
        model_id = f"github/{mid}"
        ctx = m.get("context_length", m.get("model_limits", {}).get("max_context_length", 8192))
        caps = _infer_capabilities(m)
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="github",
                base_model_name=mid,
                is_free=True,
                context_length=ctx or 8192,
                quality_tier=_infer_quality_tier(mid, ctx or 8192),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=15,
                discovered_at=now,
                last_seen=now,
                **caps,
            )
        )
    log.info("discovery.github found=%d models", len(models))
    return models


async def fetch_together(api_key: str) -> list[DiscoveredModel]:
    """Fetch models from Together AI, filtering for free ones."""
    now = datetime.now()
    data = await _http_get_json(
        "https://api.together.xyz/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models: list[DiscoveredModel] = []
    items = data if isinstance(data, list) else data.get("data", [])
    for m in items:
        mid = m.get("id", "")
        pricing = m.get("pricing", {})
        try:
            input_cost = float(pricing.get("input", 1))
            output_cost = float(pricing.get("output", 1))
        except (ValueError, TypeError):
            continue

        if input_cost > 0 or output_cost > 0:
            continue

        model_id = f"together_ai/{mid}"
        ctx = m.get("context_length", 4096) or 4096
        caps = _infer_capabilities(m)
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="together_ai",
                base_model_name=mid.rsplit("/", 1)[-1],
                is_free=True,
                context_length=ctx,
                quality_tier=_infer_quality_tier(mid, ctx),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=60,
                discovered_at=now,
                last_seen=now,
                **caps,
            )
        )
    log.info("discovery.together found=%d free models", len(models))
    return models


async def fetch_mistral(api_key: str) -> list[DiscoveredModel]:
    """Fetch models from Mistral (free tier)."""
    now = datetime.now()
    data = await _http_get_json(
        "https://api.mistral.ai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models: list[DiscoveredModel] = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        model_id = f"mistral/{mid}"
        ctx = m.get("max_context_length", 32768) or 32768
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="mistral",
                base_model_name=mid,
                is_free=True,
                context_length=ctx,
                supports_tool_calling=True,
                supports_json_mode=True,
                quality_tier=_infer_quality_tier(mid, ctx),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=60,
                discovered_at=now,
                last_seen=now,
            )
        )
    log.info("discovery.mistral found=%d models", len(models))
    return models


async def fetch_kilo(api_key: str) -> list[DiscoveredModel]:
    """Fetch models from Kilo Code (all free via API gateway).

    Kilo's catalog endpoint returns a flat JSON list (not OpenAI-compatible).
    Each entry uses ``openrouterId`` as the model identifier and
    ``contextLength`` for context window.  All models are free through
    the Kilo gateway regardless of the listed pricing.
    """
    now = datetime.now()
    data = await _http_get_json(
        "https://kilo.ai/api/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    items = data if isinstance(data, list) else data.get("data", [])
    models: list[DiscoveredModel] = []
    for m in items:
        or_id = m.get("openrouterId", "")
        if not or_id:
            continue
        # Kilo routes via openai/ prefix + api_base
        model_id = f"openai/{or_id}"
        ctx = m.get("contextLength", 8192) or 8192
        base_name = or_id.rsplit("/", 1)[-1]
        caps = _infer_capabilities({"id": or_id})
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="kilo",
                base_model_name=base_name,
                is_free=True,
                context_length=ctx,
                quality_tier=_infer_quality_tier(or_id, ctx),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=60,
                discovered_at=now,
                last_seen=now,
                **caps,
            )
        )
    log.info("discovery.kilo found=%d models", len(models))
    return models


async def fetch_cloudflare(account_id: str, api_token: str) -> list[DiscoveredModel]:
    """Fetch models from Cloudflare Workers AI."""
    now = datetime.now()
    data = await _http_get_json(
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search",
        headers={"Authorization": f"Bearer {api_token}"},
    )
    models: list[DiscoveredModel] = []
    for m in data.get("result", []):
        mid = m.get("name", "")
        task = m.get("task", {}).get("name", "")
        if task != "Text Generation":
            continue

        model_id = f"cloudflare/{mid}"
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="cloudflare",
                base_model_name=mid.rsplit("/", 1)[-1],
                is_free=True,
                context_length=4096,
                quality_tier=_infer_quality_tier(mid, 4096),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=60,
                discovered_at=now,
                last_seen=now,
            )
        )
    log.info("discovery.cloudflare found=%d models", len(models))
    return models


async def fetch_nvidia_nim(api_key: str) -> list[DiscoveredModel]:
    """Fetch models from NVIDIA NIM (free preview)."""
    now = datetime.now()
    data = await _http_get_json(
        "https://integrate.api.nvidia.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models: list[DiscoveredModel] = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        model_id = f"nvidia_nim/{mid}"
        ctx = m.get("context_length", 8192) or 8192
        caps = _infer_capabilities(m)
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="nvidia_nim",
                base_model_name=mid.rsplit("/", 1)[-1],
                is_free=True,
                context_length=ctx,
                quality_tier=_infer_quality_tier(mid, ctx),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=30,
                discovered_at=now,
                last_seen=now,
                **caps,
            )
        )
    log.info("discovery.nvidia_nim found=%d models", len(models))
    return models


async def fetch_fireworks(api_key: str) -> list[DiscoveredModel]:
    """Fetch models from Fireworks AI."""
    now = datetime.now()
    data = await _http_get_json(
        "https://api.fireworks.ai/inference/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models: list[DiscoveredModel] = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        model_id = f"fireworks_ai/{mid}"
        ctx = m.get("context_length", 8192) or 8192
        caps = _infer_capabilities(m)
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="fireworks_ai",
                base_model_name=mid.rsplit("/", 1)[-1],
                is_free=True,
                context_length=ctx,
                quality_tier=_infer_quality_tier(mid, ctx),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=60,
                discovered_at=now,
                last_seen=now,
                **caps,
            )
        )
    log.info("discovery.fireworks found=%d models", len(models))
    return models


async def fetch_sambanova(api_key: str) -> list[DiscoveredModel]:
    """Fetch models from Sambanova (OpenAI-compatible)."""
    now = datetime.now()
    data = await _http_get_json(
        "https://api.sambanova.ai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models: list[DiscoveredModel] = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        model_id = f"sambanova/{mid}"
        ctx = m.get("context_length", 8192) or 8192
        caps = _infer_capabilities(m)
        models.append(
            DiscoveredModel(
                model_id=model_id,
                provider="sambanova",
                base_model_name=mid,
                is_free=True,
                context_length=ctx,
                quality_tier=_infer_quality_tier(mid, ctx),
                cost_per_m_input=0.0,
                cost_per_m_output=0.0,
                rate_limit_rpm=30,
                discovered_at=now,
                last_seen=now,
                **caps,
            )
        )
    log.info("discovery.sambanova found=%d models", len(models))
    return models


# ── Provider registry ─────────────────────────────────────────────────────

# Maps provider name → (env_var_name(s), fetcher_function)
# Used by ModelDiscoveryService to auto-detect available providers
PROVIDER_FETCHERS: dict[str, tuple[list[str], Any]] = {
    "openrouter": (["OPENROUTER_API_KEY"], fetch_openrouter),
    "groq": (["GROQ_API_KEY"], fetch_groq),
    "cerebras": (["CEREBRAS_API_KEY"], fetch_cerebras),
    "gemini": (["GEMINI_API_KEY"], fetch_gemini),
    "github": (["GITHUB_TOKEN"], fetch_github_models),
    "together_ai": (["TOGETHERAI_API_KEY"], fetch_together),
    "mistral": (["MISTRAL_API_KEY"], fetch_mistral),
    "kilo": (["KILOCODE_API_KEY"], fetch_kilo),
    "cloudflare": (["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"], fetch_cloudflare),
    "nvidia_nim": (["NVIDIA_NIM_API_KEY"], fetch_nvidia_nim),
    "fireworks_ai": (["FIREWORKS_AI_API_KEY"], fetch_fireworks),
    "sambanova": (["SAMBANOVA_API_KEY"], fetch_sambanova),
}

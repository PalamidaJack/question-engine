"""Model capability detection and caching."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

log = logging.getLogger(__name__)


class ModelProfile(BaseModel):
    """Detected capabilities of a model."""

    model: str = ""
    supports_json_mode: bool = False
    supports_tool_calling: bool = False
    supports_system_messages: bool = True
    max_context_tokens: int = 4096
    supports_grammar: bool = False
    estimated_quality_tier: str = "balanced"


# Known model profiles (avoid probing for well-known models)
_KNOWN_PROFILES: dict[str, dict[str, Any]] = {
    "gpt-4": {
        "supports_json_mode": True,
        "supports_tool_calling": True,
        "max_context_tokens": 128000,
        "estimated_quality_tier": "powerful",
    },
    "gpt-4o": {
        "supports_json_mode": True,
        "supports_tool_calling": True,
        "max_context_tokens": 128000,
        "estimated_quality_tier": "powerful",
    },
    "gpt-4o-mini": {
        "supports_json_mode": True,
        "supports_tool_calling": True,
        "max_context_tokens": 128000,
        "estimated_quality_tier": "fast",
    },
    "claude-3-5-sonnet": {
        "supports_json_mode": True,
        "supports_tool_calling": True,
        "max_context_tokens": 200000,
        "estimated_quality_tier": "powerful",
    },
    "claude-3-haiku": {
        "supports_json_mode": True,
        "supports_tool_calling": True,
        "max_context_tokens": 200000,
        "estimated_quality_tier": "fast",
    },
}


class ModelCapabilities:
    """Detect and cache model capabilities."""

    def __init__(self, discovery: Any | None = None) -> None:
        self._cache: dict[str, ModelProfile] = {}
        self._discovery = discovery

    def get_profile(self, model: str) -> ModelProfile:
        """Get model profile from discovery, cache, or known
        profiles."""
        if model in self._cache:
            return self._cache[model]

        # Try discovery first
        if self._discovery is not None:
            dm = self._discovery.get_model(model)
            if dm is not None:
                profile = ModelProfile(
                    model=model,
                    supports_json_mode=dm.supports_json_mode,
                    supports_tool_calling=dm.supports_tool_calling,
                    supports_system_messages=dm.supports_system_messages,
                    max_context_tokens=dm.context_length,
                    estimated_quality_tier=dm.quality_tier,
                )
                self._cache[model] = profile
                return profile

        profile = ModelProfile(model=model)

        # Check known profiles
        for prefix, known in _KNOWN_PROFILES.items():
            if model.startswith(prefix):
                profile = ModelProfile(
                    model=model, **known
                )
                break
        else:
            # Infer from model name
            lower = model.lower()
            if "llama" in lower or "mistral" in lower:
                profile.estimated_quality_tier = "local"
                profile.supports_grammar = True
                profile.max_context_tokens = 8192
            elif "qwen" in lower:
                profile.estimated_quality_tier = "local"
                profile.max_context_tokens = 32768

        self._cache[model] = profile
        return profile

    def is_capable(
        self, model: str, capability: str
    ) -> bool:
        """Check if model supports a specific
        capability."""
        profile = self.get_profile(model)
        return getattr(profile, capability, False)

    def clear_cache(self) -> None:
        self._cache.clear()

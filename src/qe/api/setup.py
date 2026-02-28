"""First-run setup: provider & API key configuration."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

# Supported LLM providers and their env var / default models per tier
PROVIDERS: list[dict] = [
    {
        "name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "example_models": ["gpt-4o", "gpt-4o-mini"],
        "tier_defaults": {"fast": "gpt-4o-mini", "balanced": "gpt-4o", "powerful": "o1-preview"},
    },
    {
        "name": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "example_models": ["claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"],
        "tier_defaults": {
            "fast": "claude-haiku-4-5-20251001",
            "balanced": "claude-sonnet-4-20250514",
            "powerful": "claude-sonnet-4-20250514",
        },
    },
    {
        "name": "OpenRouter",
        "env_var": "OPENROUTER_API_KEY",
        "example_models": ["openrouter/auto"],
        "tier_defaults": {
            "fast": "openrouter/google/gemini-2.0-flash-001",
            "balanced": "openrouter/anthropic/claude-sonnet-4",
            "powerful": "openrouter/anthropic/claude-sonnet-4",
        },
    },
    {
        "name": "Google Gemini",
        "env_var": "GEMINI_API_KEY",
        "example_models": ["gemini/gemini-2.0-flash", "gemini/gemini-2.5-pro-preview-06-05"],
        "tier_defaults": {
            "fast": "gemini/gemini-2.0-flash",
            "balanced": "gemini/gemini-2.5-pro-preview-06-05",
            "powerful": "gemini/gemini-2.5-pro-preview-06-05",
        },
    },
    {
        "name": "Groq",
        "env_var": "GROQ_API_KEY",
        "example_models": ["groq/llama-3.3-70b-versatile"],
        "tier_defaults": {
            "fast": "groq/llama-3.3-70b-versatile",
            "balanced": "groq/llama-3.3-70b-versatile",
            "powerful": "groq/llama-3.3-70b-versatile",
        },
    },
    {
        "name": "Mistral",
        "env_var": "MISTRAL_API_KEY",
        "example_models": ["mistral/mistral-large-latest"],
        "tier_defaults": {
            "fast": "mistral/mistral-small-latest",
            "balanced": "mistral/mistral-large-latest",
            "powerful": "mistral/mistral-large-latest",
        },
    },
    {
        "name": "Together AI",
        "env_var": "TOGETHERAI_API_KEY",
        "example_models": ["together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"],
        "tier_defaults": {
            "fast": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "balanced": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "powerful": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        },
    },
    {
        "name": "Ollama (local)",
        "env_var": None,
        "example_models": ["ollama/llama3.2", "ollama/qwen3"],
        "tier_defaults": {
            "fast": "ollama/llama3.2",
            "balanced": "ollama/qwen3",
            "powerful": "ollama/qwen3",
        },
    },
]

ENV_PATH = Path(".env")
CONFIG_PATH = Path("config.toml")

_API_KEY_RE = re.compile(r"^[A-Z_]+_API_KEY$")


def mask_key(key: str) -> str:
    """Return a masked version of an API key (first 3 + last 4 chars)."""
    if len(key) <= 8:
        return "****"
    return f"{key[:3]}...{key[-4:]}"


def _parse_env(path: Path | None = None) -> dict[str, str]:
    """Parse a .env file into a dict (simple KEY=VALUE lines)."""
    env_file = path or ENV_PATH
    if not env_file.exists():
        return {}
    result: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        result[key] = value
    return result


def is_setup_complete(env_path: Path | None = None) -> bool:
    """Return True if .env exists and contains at least one *_API_KEY or Ollama is configured."""
    env = _parse_env(env_path)
    # Check for any real API key
    for key, value in env.items():
        if _API_KEY_RE.match(key) and value and not value.startswith("sk-your-"):
            return True
    # Check if Ollama is explicitly enabled in config
    config = _load_config()
    models = config.get("models", {})
    for tier in ("fast", "balanced", "powerful"):
        model = models.get(tier, "")
        if model.startswith("ollama/"):
            return True
    return False


def _load_config() -> dict:
    """Load config.toml, returning empty dict if missing."""
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def _dump_toml(config: dict) -> str:
    """Serialize a simple nested dict to TOML format.

    Handles the config.toml structure: top-level scalars and one level of
    [section] tables with scalar values.
    """
    lines: list[str] = []
    # Top-level scalar keys first
    for key, value in config.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_toml_value(value)}")
    # Then sections
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"\n[{key}]")
            for k, v in value.items():
                lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines) + "\n"


def _toml_value(v: object) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return f'"{v}"'


def get_configured_providers(env_path: Path | None = None) -> list[dict]:
    """Return list of configured providers with masked keys."""
    env = _parse_env(env_path)
    configured = []
    for provider in PROVIDERS:
        env_var = provider["env_var"]
        if env_var is None:
            # Ollama — always available, no key needed
            configured.append({
                "name": provider["name"],
                "env_var": None,
                "configured": True,
                "masked_key": None,
            })
            continue
        value = env.get(env_var, "")
        if value and not value.startswith("sk-your-"):
            configured.append({
                "name": provider["name"],
                "env_var": env_var,
                "configured": True,
                "masked_key": mask_key(value),
            })
        else:
            configured.append({
                "name": provider["name"],
                "env_var": env_var,
                "configured": False,
                "masked_key": None,
            })
    return configured


def get_current_tiers() -> dict[str, str]:
    """Return current tier→model mapping from config.toml."""
    config = _load_config()
    models = config.get("models", {})
    return {
        "fast": models.get("fast", "gpt-4o-mini"),
        "balanced": models.get("balanced", "gpt-4o"),
        "powerful": models.get("powerful", "o1-preview"),
    }


def save_setup(
    providers: dict[str, str],
    tier_config: dict[str, dict],
    env_path: Path | None = None,
) -> None:
    """Write API keys to .env and tier model assignments to config.toml.

    Args:
        providers: Mapping of env_var_name → api_key value.
                   e.g. {"OPENAI_API_KEY": "sk-...", "ANTHROPIC_API_KEY": "sk-ant-..."}
        tier_config: Mapping of tier → {"provider": name, "model": model_string}.
                     e.g. {"fast": {"provider": "OpenAI", "model": "gpt-4o-mini"}, ...}
        env_path: Override .env path (for testing).
    """
    env_file = env_path or ENV_PATH

    # ── Write .env ──────────────────────────────────────────────────────
    existing = _parse_env(env_file)
    existing.update({k: v for k, v in providers.items() if v})
    lines = [f"{k}={v}" for k, v in sorted(existing.items()) if v]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── Update config.toml [models] ────────────────────────────────────
    config = _load_config()
    if "models" not in config:
        config["models"] = {}

    for tier, info in tier_config.items():
        if tier in ("fast", "balanced", "powerful") and "model" in info:
            config["models"][tier] = info["model"]

    CONFIG_PATH.write_text(_dump_toml(config), encoding="utf-8")


_SETTINGS_DEFAULTS: dict[str, dict] = {
    "budget": {"monthly_limit_usd": 50.0, "alert_at_pct": 0.80},
    "runtime": {"log_level": "INFO", "hil_timeout_seconds": 3600},
}


def get_settings() -> dict:
    """Return runtime settings from config.toml with defaults."""
    config = _load_config()
    result: dict[str, dict] = {}
    for section, defaults in _SETTINGS_DEFAULTS.items():
        values = config.get(section, {})
        result[section] = {k: values.get(k, v) for k, v in defaults.items()}
    return result


def save_settings(settings: dict) -> None:
    """Merge runtime settings into config.toml, preserving all other sections."""
    config = _load_config()
    for section in ("budget", "runtime"):
        incoming = settings.get(section)
        if not incoming or not isinstance(incoming, dict):
            continue
        if section not in config:
            config[section] = {}
        config[section].update(incoming)
    CONFIG_PATH.write_text(_dump_toml(config), encoding="utf-8")

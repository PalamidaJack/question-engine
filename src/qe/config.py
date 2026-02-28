"""Typed configuration models for Question Engine.

Provides Pydantic validation for config.toml, catching typos, wrong types,
and invalid values at startup rather than at runtime.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)


class BudgetConfig(BaseModel):
    monthly_limit_usd: float = Field(default=50.0, gt=0)
    alert_at_pct: float = Field(default=0.80, ge=0.0, le=1.0)


class RuntimeConfig(BaseModel):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False
    log_dir: str | None = None
    hil_timeout_seconds: int = Field(default=3600, gt=0)
    prefer_local_models: bool = False
    module_levels: dict[str, str] | None = None


class ModelsConfig(BaseModel):
    fast: str = "gpt-4o-mini"
    balanced: str = "gpt-4o"
    powerful: str = "o1-preview"
    local: str | None = None

    @field_validator("fast", "balanced", "powerful", "local", mode="before")
    @classmethod
    def strip_model_name(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v


class SubstrateConfig(BaseModel):
    db_path: str = "data/qe.db"
    cold_storage_path: str | None = None


class BusConfig(BaseModel):
    type: Literal["memory", "redis"] = "memory"
    redis_url: str | None = None


class SecurityConfig(BaseModel):
    api_key: str | None = None
    admin_api_key: str | None = None
    require_auth: bool = False


class QEConfig(BaseModel):
    """Root configuration model for config.toml."""

    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    substrate: SubstrateConfig = Field(default_factory=SubstrateConfig)
    bus: BusConfig = Field(default_factory=BusConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    model_config = {"extra": "allow"}


def load_config(path: Path | None = None) -> QEConfig:
    """Load and validate config.toml, returning typed QEConfig.

    Missing file or sections are filled with defaults.
    Raises pydantic.ValidationError on invalid values.
    """
    config_path = path or Path("config.toml")
    raw: dict[str, Any] = {}

    if config_path.exists():
        with config_path.open("rb") as f:
            raw = tomllib.load(f)

    config = QEConfig.model_validate(raw)
    log.debug(
        "config.loaded path=%s budget_limit=%.2f log_level=%s models=%s/%s/%s",
        config_path,
        config.budget.monthly_limit_usd,
        config.runtime.log_level,
        config.models.fast,
        config.models.balanced,
        config.models.powerful,
    )
    return config

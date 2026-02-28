from typing import Literal

from pydantic import BaseModel, Field


class ModelPreference(BaseModel):
    tier: Literal["fast", "balanced", "powerful", "local"]
    max_cost_per_call_usd: float = 0.10
    fallback_tier: str | None = "fast"


class CapabilityDeclaration(BaseModel):
    web_search: bool = False
    file_read: bool = False
    file_write: bool = False
    code_execute: bool = False
    browser_control: bool = False
    bus_topics_subscribe: list[str] = Field(default_factory=list)
    bus_topics_publish: list[str] = Field(default_factory=list)
    substrate_read: bool = False
    substrate_write: bool = False

    def to_capability_set(self) -> set[str]:
        """Convert boolean capability flags to a set of capability strings."""
        caps: set[str] = set()
        for field_name in (
            "web_search",
            "file_read",
            "file_write",
            "code_execute",
            "browser_control",
        ):
            if getattr(self, field_name):
                caps.add(field_name)
        return caps


class Blueprint(BaseModel):
    service_id: str
    display_name: str
    version: str
    system_prompt: str
    constitution: str = ""
    service_class: str | None = None  # e.g. "qe.services.validator:ClaimValidatorService"
    model_preference: ModelPreference
    capabilities: CapabilityDeclaration
    max_context_tokens: int = 8000
    context_compression_threshold: float = 0.75
    reinforcement_interval_turns: int = 10
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

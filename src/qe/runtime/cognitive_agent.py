"""CognitiveAgent model — agent identity for Phase 4 scaling.

Defines the Pydantic model for a cognitive agent with epistemic state,
core memory, and capabilities. Used by InquiryEngine for agent identity
now; will support multi-agent pools in Phase 4.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class CoreMemory(BaseModel):
    """Persistent knowledge a cognitive agent accumulates."""

    learned_patterns: list[str] = Field(default_factory=list)
    domain_expertise: list[str] = Field(default_factory=list)
    tool_preferences: dict[str, float] = Field(default_factory=dict)


class AgentEpistemicState(BaseModel):
    """What an agent knows about what it knows."""

    known_facts_count: int = 0
    known_unknowns_count: int = 0
    active_hypotheses_count: int = 0
    confidence_level: float = Field(default=0.5, ge=0.0, le=1.0)


class CognitiveAgent(BaseModel):
    """A cognitive agent with identity, specialization, and epistemic state."""

    agent_id: str = Field(default_factory=lambda: f"cag_{uuid.uuid4().hex[:12]}")
    persona: str = "general researcher"
    specialization: str = "general"
    epistemic_state: AgentEpistemicState = Field(default_factory=AgentEpistemicState)
    core_memory: CoreMemory = Field(default_factory=CoreMemory)
    capabilities: list[str] = Field(default_factory=list)
    model_preference: str = ""
    active_inquiry_id: str | None = None
    status: Literal["idle", "active", "paused", "retired"] = "idle"

    def update_epistemic_state(
        self,
        known_facts: int,
        known_unknowns: int,
        active_hypotheses: int,
    ) -> None:
        """Update the agent's epistemic state counters."""
        self.epistemic_state.known_facts_count = known_facts
        self.epistemic_state.known_unknowns_count = known_unknowns
        self.epistemic_state.active_hypotheses_count = active_hypotheses
        # Confidence rises with facts, drops with unknowns
        total = known_facts + known_unknowns
        if total > 0:
            self.epistemic_state.confidence_level = round(
                known_facts / total, 3
            )
        else:
            self.epistemic_state.confidence_level = 0.5

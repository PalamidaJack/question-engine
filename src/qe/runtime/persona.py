"""Cognitive tool personas — assign behavioral profiles to tool types.

Each persona defines how the agent should approach a specific category
of tool usage (research, analysis, code, etc.).  Gated behind the
``cognitive_personas`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CognitivePersona:
    """Behavioral profile for a tool category."""

    name: str
    description: str
    tool_categories: list[str] = field(default_factory=list)
    system_addendum: str = ""  # Extra instructions injected into system prompt
    temperature_hint: float | None = None
    style: str = "balanced"  # concise | balanced | thorough


# Built-in personas
BUILTIN_PERSONAS: dict[str, CognitivePersona] = {
    "researcher": CognitivePersona(
        name="researcher",
        description="Thorough web research with source validation",
        tool_categories=["web_search", "web_fetch", "browser"],
        system_addendum=(
            "When researching, always cross-reference multiple sources. "
            "Cite URLs and note confidence in findings."
        ),
        style="thorough",
    ),
    "analyst": CognitivePersona(
        name="analyst",
        description="Data analysis and pattern recognition",
        tool_categories=["code_execute", "file_read"],
        system_addendum=(
            "Approach analysis methodically. Present findings with "
            "supporting data. Note assumptions and limitations."
        ),
        style="thorough",
    ),
    "coder": CognitivePersona(
        name="coder",
        description="Code generation and debugging",
        tool_categories=["code_execute", "file_write", "file_read"],
        system_addendum=(
            "Write clean, well-tested code. Follow existing patterns. "
            "Explain non-obvious decisions."
        ),
        style="concise",
    ),
    "synthesizer": CognitivePersona(
        name="synthesizer",
        description="Combine findings into coherent summaries",
        tool_categories=[],
        system_addendum=(
            "Synthesize information from multiple sources into a "
            "coherent narrative. Highlight key insights and gaps."
        ),
        style="balanced",
    ),
}


class PersonaManager:
    """Manages cognitive personas for tool-category assignment."""

    def __init__(self) -> None:
        self._personas: dict[str, CognitivePersona] = dict(BUILTIN_PERSONAS)
        self._tool_to_persona: dict[str, str] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._tool_to_persona.clear()
        for name, persona in self._personas.items():
            for cat in persona.tool_categories:
                self._tool_to_persona[cat] = name

    def register(self, persona: CognitivePersona) -> None:
        """Register or update a persona."""
        self._personas[persona.name] = persona
        self._rebuild_index()

    def get_persona_for_tool(self, tool_category: str) -> CognitivePersona | None:
        """Return the persona assigned to a tool category."""
        name = self._tool_to_persona.get(tool_category)
        return self._personas.get(name) if name else None

    def get_persona(self, name: str) -> CognitivePersona | None:
        return self._personas.get(name)

    def list_personas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "description": p.description,
                "tool_categories": p.tool_categories,
                "style": p.style,
            }
            for p in self._personas.values()
        ]

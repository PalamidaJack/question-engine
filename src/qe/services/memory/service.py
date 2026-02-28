"""Memory service: enriches goals and infers memories."""

from __future__ import annotations

import logging
from typing import Any

from qe.substrate.memory_store import MemoryStore

log = logging.getLogger(__name__)


class MemoryService:
    """Manages the persistent memory system.

    Enriches LLM calls with relevant past knowledge and
    infers new memories from committed claims.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        bus: Any = None,
    ) -> None:
        self.store = memory_store
        self.bus = bus
        self._entity_mention_counts: dict[str, int] = {}

    async def get_enrichment_context(
        self,
        query: str = "",
        entity_ids: list[str] | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Build enrichment context for LLM calls.

        Collects preferences, relevant project context,
        and entity memories.
        """
        context: dict[str, Any] = {}

        # Always include user preferences
        preferences = await self.store.get_preferences()
        if preferences:
            context["preferences"] = [
                {"key": p.key, "value": p.value}
                for p in preferences
            ]

        # Include project context if specified
        if project_id:
            project_ctx = (
                await self.store.get_project_context(project_id)
            )
            if project_ctx:
                context["project_context"] = [
                    {
                        "key": m.key.split(":", 1)[-1],
                        "value": m.value,
                    }
                    for m in project_ctx
                ]

        # Include entity memories
        if entity_ids:
            entity_memories: dict[str, list[dict[str, Any]]] = {}
            for eid in entity_ids:
                memories = (
                    await self.store.get_entity_memories(eid)
                )
                if memories:
                    entity_memories[eid] = [
                        {
                            "key": m.key.split(":", 1)[-1],
                            "value": m.value,
                            "confidence": m.confidence,
                        }
                        for m in memories
                    ]
            if entity_memories:
                context["entity_memories"] = entity_memories

        return context

    async def infer_entity_memory(
        self, entity_id: str, attribute: str, value: str
    ) -> None:
        """Infer and store a memory from repeated patterns."""
        count_key = f"{entity_id}:{attribute}"
        self._entity_mention_counts[count_key] = (
            self._entity_mention_counts.get(count_key, 0) + 1
        )
        # Only store after seeing it multiple times
        if self._entity_mention_counts[count_key] >= 2:
            await self.store.set_entity_memory(
                entity_id, attribute, value
            )
            log.info(
                "memory.inferred entity=%s attr=%s",
                entity_id,
                attribute,
            )

    def format_context_for_prompt(
        self, context: dict[str, Any]
    ) -> str:
        """Format enrichment context for system prompt."""
        parts: list[str] = []

        if "preferences" in context:
            pref_lines = [
                f"- {p['key']}: {p['value']}"
                for p in context["preferences"]
            ]
            parts.append(
                "[USER PREFERENCES]\n"
                + "\n".join(pref_lines)
            )

        if "project_context" in context:
            ctx_lines = [
                f"- {c['key']}: {c['value']}"
                for c in context["project_context"]
            ]
            parts.append(
                "[PROJECT CONTEXT]\n"
                + "\n".join(ctx_lines)
            )

        if "entity_memories" in context:
            for eid, memories in (
                context["entity_memories"].items()
            ):
                mem_lines = [
                    f"- {m['key']}: {m['value']}"
                    f" (confidence: {m['confidence']:.2f})"
                    for m in memories
                ]
                parts.append(
                    f"[ENTITY: {eid}]\n"
                    + "\n".join(mem_lines)
                )

        return "\n\n".join(parts)

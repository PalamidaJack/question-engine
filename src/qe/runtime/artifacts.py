"""4-Tier Artifact System — Instructions/Prompts/Agents/Skills taxonomy.

Provides structured metadata (name, version, maturity, tags, deps) for
each artifact type.  Gated behind ``artifact_system`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Artifact:
    """A versioned artifact with metadata."""

    artifact_id: str
    name: str
    artifact_type: str  # instruction | prompt | agent | skill
    version: str = "0.1.0"
    maturity: str = "experimental"  # experimental | preview | stable
    description: str = ""
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "type": self.artifact_type,
            "version": self.version,
            "maturity": self.maturity,
            "description": self.description,
            "tags": self.tags,
            "dependencies": self.dependencies,
        }


ARTIFACT_TYPES = ("instruction", "prompt", "agent", "skill")


class ArtifactRegistry:
    """Registry for 4-tier artifacts."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}

    def register(self, artifact: Artifact) -> None:
        if artifact.artifact_type not in ARTIFACT_TYPES:
            raise ValueError(
                f"Invalid artifact type: {artifact.artifact_type}"
            )
        self._artifacts[artifact.artifact_id] = artifact
        log.debug(
            "artifact.registered id=%s type=%s",
            artifact.artifact_id, artifact.artifact_type,
        )

    def get(self, artifact_id: str) -> Artifact | None:
        return self._artifacts.get(artifact_id)

    def list_by_type(
        self, artifact_type: str,
    ) -> list[Artifact]:
        return [
            a for a in self._artifacts.values()
            if a.artifact_type == artifact_type
        ]

    def list_by_tag(self, tag: str) -> list[Artifact]:
        return [
            a for a in self._artifacts.values()
            if tag in a.tags
        ]

    def search(self, query: str) -> list[Artifact]:
        q = query.lower()
        return [
            a for a in self._artifacts.values()
            if q in a.name.lower() or q in a.description.lower()
        ]

    def resolve_dependencies(
        self, artifact_id: str,
    ) -> list[str]:
        """Resolve full dependency tree for an artifact."""
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return []
        resolved: list[str] = []
        visited: set[str] = set()
        self._resolve_deps(artifact_id, resolved, visited)
        return resolved

    def _resolve_deps(
        self,
        aid: str,
        resolved: list[str],
        visited: set[str],
    ) -> None:
        if aid in visited:
            return
        visited.add(aid)
        artifact = self._artifacts.get(aid)
        if not artifact:
            return
        for dep_id in artifact.dependencies:
            self._resolve_deps(dep_id, resolved, visited)
        resolved.append(aid)

    def list_all(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self._artifacts.values()]

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for a in self._artifacts.values():
            by_type[a.artifact_type] = (
                by_type.get(a.artifact_type, 0) + 1
            )
        return {
            "total": len(self._artifacts),
            "by_type": by_type,
        }

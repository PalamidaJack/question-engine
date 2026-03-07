"""Agent profile loader -- document-defined agent configuration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # PyYAML -- already in project deps

log = logging.getLogger(__name__)


class ProfileLoader:
    """Loads and caches agent profile files from a profile directory.

    Profile directory structure::

        profiles/{profile_name}/
            manifest.yaml
            identity.md
            personality.md
            strategies.md
            tool-policies.md
            knowledge-seeds/self-knowledge.md
            playbooks/*.md
    """

    def __init__(
        self,
        profiles_dir: str | Path = "profiles",
        active_profile: str = "default",
    ):
        self._profiles_dir = Path(profiles_dir)
        self._active_profile = active_profile
        self._cache: dict[str, ProfileFile] = {}
        self._cache_time: dict[str, float] = {}
        self._cache_ttl = 30.0  # seconds
        self._manifest: dict[str, Any] = {}
        self._load_manifest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _profile_path(self) -> Path:
        return self._profiles_dir / self._active_profile

    def _load_manifest(self) -> None:
        manifest_path = self._profile_path() / "manifest.yaml"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self._manifest = yaml.safe_load(f) or {}
            log.info("profile.loaded manifest=%s", self._active_profile)
        else:
            self._manifest = {}

    # ------------------------------------------------------------------
    # Core file access
    # ------------------------------------------------------------------

    def get(self, filename: str) -> str:
        """Get content of a profile file.

        Returns empty string if not found.
        """
        now = time.time()
        cached = self._cache.get(filename)
        cache_age = now - self._cache_time.get(filename, 0)

        if cached is not None and cache_age < self._cache_ttl:
            return cached.content

        path = self._profile_path() / filename
        if not path.exists():
            return ""

        content = path.read_text(encoding="utf-8")
        self._cache[filename] = ProfileFile(
            path=str(path), content=content,
        )
        self._cache_time[filename] = now
        return content

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def identity(self) -> str:
        return self.get("identity.md")

    @property
    def personality(self) -> str:
        return self.get("personality.md")

    @property
    def strategies(self) -> str:
        return self.get("strategies.md")

    @property
    def tool_policies(self) -> str:
        return self.get("tool-policies.md")

    @property
    def self_knowledge(self) -> str:
        return self.get("knowledge-seeds/self-knowledge.md")

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    @property
    def autonomy_mode(self) -> str:
        return self._manifest.get("autonomy", {}).get(
            "mode", "supervised",
        )

    # ------------------------------------------------------------------
    # Playbooks
    # ------------------------------------------------------------------

    def list_playbooks(self) -> list[dict[str, str]]:
        """List available playbooks with name and description."""
        playbooks_dir = self._profile_path() / "playbooks"
        if not playbooks_dir.exists():
            return []
        result = []
        for p in sorted(playbooks_dir.glob("*.md")):
            content = p.read_text(encoding="utf-8")
            first_line = content.split("\n", 1)[0].strip().lstrip("# ")
            result.append({
                "name": p.stem,
                "filename": p.name,
                "description": first_line,
            })
        return result

    def get_playbook(self, name: str) -> str:
        """Get a playbook by name (without .md extension)."""
        return self.get(f"playbooks/{name}.md")

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def list_profiles(self) -> list[str]:
        """List available profile directories."""
        if not self._profiles_dir.exists():
            return []
        return [
            d.name for d in self._profiles_dir.iterdir() if d.is_dir()
        ]

    def switch_profile(self, name: str) -> bool:
        """Switch active profile.

        Returns False if profile doesn't exist.
        """
        path = self._profiles_dir / name
        if not path.exists():
            return False
        self._active_profile = name
        self._cache.clear()
        self._cache_time.clear()
        self._load_manifest()
        return True

    # ------------------------------------------------------------------
    # File CRUD
    # ------------------------------------------------------------------

    def save_file(self, filename: str, content: str) -> bool:
        """Save content to a profile file.

        Creates directories as needed.
        """
        path = self._profile_path() / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        # Invalidate cache
        self._cache.pop(filename, None)
        self._cache_time.pop(filename, None)
        log.info("profile.saved file=%s", filename)
        return True

    def delete_file(self, filename: str) -> bool:
        """Delete a profile file."""
        path = self._profile_path() / filename
        if not path.exists():
            return False
        path.unlink()
        self._cache.pop(filename, None)
        self._cache_time.pop(filename, None)
        return True

    def list_files(self) -> list[dict[str, Any]]:
        """List all files in the active profile directory."""
        base = self._profile_path()
        if not base.exists():
            return []
        files = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rel = str(p.relative_to(base))
                files.append({
                    "path": rel,
                    "size": p.stat().st_size,
                    "modified": p.stat().st_mtime,
                })
        return files

    # ------------------------------------------------------------------
    # Import / export
    # ------------------------------------------------------------------

    def export_profile(self) -> dict[str, str]:
        """Export all profile files as {relative_path: content}."""
        base = self._profile_path()
        result: dict[str, str] = {}
        for p in base.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(base))
                try:
                    result[rel] = p.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    pass  # Skip binary files
        return result

    def import_profile(
        self, name: str, files: dict[str, str],
    ) -> None:
        """Import a profile from {relative_path: content}."""
        base = self._profiles_dir / name
        base.mkdir(parents=True, exist_ok=True)
        for rel_path, content in files.items():
            path = base / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        log.info(
            "profile.imported name=%s files=%d", name, len(files),
        )


@dataclass
class ProfileFile:
    path: str
    content: str


# ── Specialist Profiles (#51) ────────────────────────────────────────────

@dataclass
class SpecialistProfile:
    """A specialist agent profile with role-specific configuration."""

    name: str
    description: str
    system_prompt: str
    tool_whitelist: list[str]
    style: str = "balanced"  # concise | balanced | thorough
    temperature: float = 0.7
    max_tokens: int = 4096
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt": (
                self.system_prompt[:200] + "..."
                if len(self.system_prompt) > 200
                else self.system_prompt
            ),
            "tool_whitelist": self.tool_whitelist,
            "style": self.style,
            "temperature": self.temperature,
            "tags": self.tags or [],
        }


BUILTIN_SPECIALISTS: dict[str, SpecialistProfile] = {
    "analyst": SpecialistProfile(
        name="analyst",
        description="Data analysis and pattern recognition specialist",
        system_prompt=(
            "You are a data analyst. Approach problems methodically. "
            "Present findings with supporting data. Note assumptions and limitations."
        ),
        tool_whitelist=["query_beliefs", "list_entities", "get_entity_details", "reason_about"],
        style="thorough",
        tags=["analysis", "data"],
    ),
    "researcher": SpecialistProfile(
        name="researcher",
        description="Deep research and investigation specialist",
        system_prompt=(
            "You are a thorough researcher. Cross-reference multiple sources. "
            "Cite evidence and note confidence levels in findings."
        ),
        tool_whitelist=["deep_research", "swarm_research", "query_beliefs", "crystallize_insights"],
        style="thorough",
        temperature=0.5,
        tags=["research", "investigation"],
    ),
    "coder": SpecialistProfile(
        name="coder",
        description="Code generation and debugging specialist",
        system_prompt=(
            "You are a software engineer. Write clean, well-tested code. "
            "Follow existing patterns. Explain non-obvious decisions."
        ),
        tool_whitelist=["query_beliefs", "plan_and_execute"],
        style="concise",
        temperature=0.3,
        tags=["code", "development"],
    ),
    "strategist": SpecialistProfile(
        name="strategist",
        description="Strategic planning and goal decomposition specialist",
        system_prompt=(
            "You are a strategic planner. Break complex goals into actionable steps. "
            "Consider risks, dependencies, and alternative approaches."
        ),
        tool_whitelist=["plan_and_execute", "reason_about", "crystallize_insights"],
        style="balanced",
        tags=["strategy", "planning"],
    ),
    "critic": SpecialistProfile(
        name="critic",
        description="Critical analysis and dialectic reasoning specialist",
        system_prompt=(
            "You are a critical thinker. Challenge assumptions, identify weaknesses, "
            "and provide constructive counterarguments. Surface hidden risks."
        ),
        tool_whitelist=["reason_about", "query_beliefs", "crystallize_insights"],
        style="thorough",
        temperature=0.6,
        tags=["critique", "reasoning"],
    ),
    "synthesizer": SpecialistProfile(
        name="synthesizer",
        description="Information synthesis and summarization specialist",
        system_prompt=(
            "You synthesize information from multiple sources into coherent narratives. "
            "Highlight key insights, gaps, and connections across findings."
        ),
        tool_whitelist=["query_beliefs", "consolidate_knowledge", "crystallize_insights"],
        style="balanced",
        tags=["synthesis", "summary"],
    ),
    "factchecker": SpecialistProfile(
        name="factchecker",
        description="Fact verification and claim validation specialist",
        system_prompt=(
            "You verify claims against available evidence. Rate confidence levels. "
            "Flag unsupported assertions and suggest verification steps."
        ),
        tool_whitelist=["query_beliefs", "deep_research", "reason_about"],
        style="thorough",
        temperature=0.3,
        tags=["verification", "facts"],
    ),
    "creative": SpecialistProfile(
        name="creative",
        description="Creative ideation and brainstorming specialist",
        system_prompt=(
            "You generate creative ideas and novel perspectives. "
            "Think outside conventional boundaries. Explore unexpected connections."
        ),
        tool_whitelist=["query_beliefs", "crystallize_insights", "reason_about"],
        style="balanced",
        temperature=0.9,
        tags=["creative", "ideation"],
    ),
}


class SpecialistManager:
    """Manages specialist agent profiles."""

    def __init__(self) -> None:
        self._specialists: dict[str, SpecialistProfile] = dict(BUILTIN_SPECIALISTS)

    def register(self, profile: SpecialistProfile) -> None:
        self._specialists[profile.name] = profile

    def get(self, name: str) -> SpecialistProfile | None:
        return self._specialists.get(name)

    def list_specialists(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._specialists.values()]

    def get_by_tag(self, tag: str) -> list[SpecialistProfile]:
        return [s for s in self._specialists.values() if tag in (s.tags or [])]

    def names(self) -> list[str]:
        return list(self._specialists.keys())

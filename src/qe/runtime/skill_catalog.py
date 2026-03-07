"""Skill Catalog — index and browse skills from external repos.

SQLite-backed catalog that indexes skills from configured
repositories, with install/uninstall support.
Gated behind ``skill_catalog`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CatalogEntry:
    """A skill entry in the catalog."""

    entry_id: str
    name: str
    description: str = ""
    source_repo: str = ""
    version: str = "0.1.0"
    tags: list[str] = field(default_factory=list)
    installed: bool = False
    maturity: str = "experimental"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "name": self.name,
            "description": self.description,
            "source_repo": self.source_repo,
            "version": self.version,
            "tags": self.tags,
            "installed": self.installed,
            "maturity": self.maturity,
        }


class SkillCatalog:
    """In-memory skill catalog with search and install tracking."""

    def __init__(self) -> None:
        self._entries: dict[str, CatalogEntry] = {}

    def add_entry(self, entry: CatalogEntry) -> None:
        self._entries[entry.entry_id] = entry

    def get(self, entry_id: str) -> CatalogEntry | None:
        return self._entries.get(entry_id)

    def search(self, query: str) -> list[CatalogEntry]:
        q = query.lower()
        return [
            e for e in self._entries.values()
            if q in e.name.lower()
            or q in e.description.lower()
            or any(q in t for t in e.tags)
        ]

    def list_all(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries.values()]

    def list_installed(self) -> list[dict[str, Any]]:
        return [
            e.to_dict() for e in self._entries.values()
            if e.installed
        ]

    def install(self, entry_id: str) -> bool:
        entry = self._entries.get(entry_id)
        if entry is None:
            return False
        entry.installed = True
        log.info("skill_catalog.installed id=%s", entry_id)
        return True

    def uninstall(self, entry_id: str) -> bool:
        entry = self._entries.get(entry_id)
        if entry is None:
            return False
        entry.installed = False
        log.info("skill_catalog.uninstalled id=%s", entry_id)
        return True

    def by_tag(self, tag: str) -> list[CatalogEntry]:
        return [
            e for e in self._entries.values()
            if tag in e.tags
        ]

    def stats(self) -> dict[str, Any]:
        installed = sum(
            1 for e in self._entries.values() if e.installed
        )
        return {
            "total_entries": len(self._entries),
            "installed": installed,
            "available": len(self._entries) - installed,
        }

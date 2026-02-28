"""Admin audit trail for tracking who changed what and when.

Provides an append-only log of admin actions for compliance,
debugging, and accountability.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class AuditEntry(BaseModel):
    """A single audit log entry."""

    timestamp: float = Field(default_factory=time.time)
    action: str  # e.g. "settings.update", "hil.approve", "circuit.reset"
    resource: str = ""  # e.g. "budget.monthly_limit_usd", "service/researcher_v1"
    actor: str = ""  # key_id or "anonymous"
    detail: dict[str, Any] = Field(default_factory=dict)
    result: str = "success"  # "success", "denied", "error"


class AuditLog:
    """Append-only audit log for admin actions.

    Stores entries in memory with a configurable max size.
    Can be extended with SQLite persistence if needed.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._entries: deque[AuditEntry] = deque(maxlen=max_entries)

    def record(
        self,
        action: str,
        *,
        resource: str = "",
        actor: str = "",
        detail: dict[str, Any] | None = None,
        result: str = "success",
    ) -> AuditEntry:
        """Record an admin action."""
        entry = AuditEntry(
            action=action,
            resource=resource,
            actor=actor,
            detail=detail or {},
            result=result,
        )
        self._entries.append(entry)
        log.info(
            "audit.%s resource=%s actor=%s result=%s",
            action,
            resource,
            actor or "anonymous",
            result,
        )
        return entry

    def query(
        self,
        *,
        action: str | None = None,
        actor: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit entries with optional filters."""
        results = []
        for entry in reversed(self._entries):
            if action and entry.action != action:
                continue
            if actor and entry.actor != actor:
                continue
            results.append(entry.model_dump())
            if len(results) >= limit:
                break
        return results

    def count(self) -> int:
        return len(self._entries)


# ── Singleton ──────────────────────────────────────────────────────────────

_audit_log = AuditLog()


def get_audit_log() -> AuditLog:
    return _audit_log

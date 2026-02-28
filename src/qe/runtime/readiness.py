"""Readiness probe: tracks startup phases for container orchestration.

Reports whether the engine is fully initialized and ready to serve
traffic. Returns 503 during startup until all components are operational.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ReadinessState:
    """Tracks which startup phases have completed."""

    substrate_ready: bool = False
    event_log_ready: bool = False
    services_subscribed: bool = False
    supervisor_ready: bool = False
    started_at: float = field(default_factory=time.monotonic)
    ready_at: float | None = None

    @property
    def is_ready(self) -> bool:
        return (
            self.substrate_ready
            and self.event_log_ready
            and self.services_subscribed
            and self.supervisor_ready
        )

    def mark_ready(self, phase: str) -> None:
        """Mark a startup phase as complete."""
        if not hasattr(self, phase):
            log.warning("readiness.unknown_phase phase=%s", phase)
            return

        setattr(self, phase, True)
        log.info("readiness.phase_complete phase=%s", phase)

        if self.is_ready and self.ready_at is None:
            self.ready_at = time.monotonic()
            elapsed = self.ready_at - self.started_at
            log.info("readiness.fully_ready elapsed=%.2fs", elapsed)

    def to_dict(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started_at
        return {
            "ready": self.is_ready,
            "phases": {
                "substrate_ready": self.substrate_ready,
                "event_log_ready": self.event_log_ready,
                "services_subscribed": self.services_subscribed,
                "supervisor_ready": self.supervisor_ready,
            },
            "uptime_seconds": round(elapsed, 2),
            "startup_duration": (
                round(self.ready_at - self.started_at, 2)
                if self.ready_at is not None
                else None
            ),
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_readiness = ReadinessState()


def get_readiness() -> ReadinessState:
    return _readiness


def reset_readiness() -> ReadinessState:
    """Reset readiness state (for testing)."""
    global _readiness
    _readiness = ReadinessState()
    return _readiness

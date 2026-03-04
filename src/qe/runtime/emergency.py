"""Emergency stop (kill switch) for all LLM operations."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)


class EmergencyStop:
    """Kill switch that halts all LLM operations without killing the server.

    When activated:
    - All new LLM calls are blocked (check via is_active)
    - Feature flag 'llm_calls' is disabled
    - Bus stops accepting non-system envelopes
    """

    def __init__(self) -> None:
        self._active = False
        self._lock = threading.Lock()
        self._reason: str = ""
        self._activated_at: str | None = None
        self._bus: Any | None = None
        self._flag_store: Any | None = None

    def configure(
        self,
        bus: Any | None = None,
        flag_store: Any | None = None,
    ) -> None:
        """Wire optional dependencies after construction."""
        self._bus = bus
        self._flag_store = flag_store

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self, reason: str = "manual") -> None:
        """Activate the emergency stop."""
        with self._lock:
            if self._active:
                return
            self._active = True
            self._reason = reason
            self._activated_at = datetime.now(UTC).isoformat()
            log.critical("EMERGENCY STOP ACTIVATED: %s", reason)

            if self._flag_store:
                try:
                    self._flag_store.define(
                        "llm_calls", enabled=False,
                        description="LLM calls master switch",
                    )
                    self._flag_store.disable("llm_calls")
                except Exception:
                    pass

    def deactivate(self) -> None:
        """Resume normal operations."""
        with self._lock:
            if not self._active:
                return
            self._active = False
            log.info("Emergency stop deactivated (was: %s)", self._reason)
            self._reason = ""
            self._activated_at = None

            if self._flag_store:
                try:
                    self._flag_store.enable("llm_calls")
                except Exception:
                    pass

    def status(self) -> dict[str, Any]:
        return {
            "active": self._active,
            "reason": self._reason,
            "activated_at": self._activated_at,
        }

    def check_or_raise(self) -> None:
        """Raise RuntimeError if emergency stop is active.

        Call this before any LLM operation.
        """
        if self._active:
            raise RuntimeError(
                f"Emergency stop active: {self._reason}"
            )


# ── Singleton ──────────────────────────────────────────────────────────────

_emergency = EmergencyStop()


def get_emergency_stop() -> EmergencyStop:
    return _emergency

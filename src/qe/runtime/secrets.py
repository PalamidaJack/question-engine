"""Secrets rotation: hot-reload .env without restart.

Watches the .env file for changes and refreshes environment variables,
API keys, and auth provider configuration automatically.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

log = logging.getLogger(__name__)


class SecretsManager:
    """Manages API key lifecycle with hot-reload support.

    Watches .env for changes and notifies registered callbacks
    when secrets are rotated. Tracks rotation history for auditing.
    """

    def __init__(self, env_path: Path | None = None) -> None:
        self._env_path = env_path or Path(".env")
        self._last_mtime: float = 0.0
        self._callbacks: list[Callable[[dict[str, str | None]], None]] = []
        self._rotation_history: list[dict[str, Any]] = []
        self._current_values: dict[str, str | None] = {}
        self._load_initial()

    def _load_initial(self) -> None:
        """Load initial values from .env file."""
        if self._env_path.exists():
            self._current_values = dotenv_values(self._env_path)
            self._last_mtime = self._env_path.stat().st_mtime
        else:
            self._current_values = {}

    def on_rotate(self, callback: Callable[[dict[str, str | None]], None]) -> None:
        """Register a callback invoked when secrets change.

        Callback receives the dict of changed key -> new_value pairs.
        """
        self._callbacks.append(callback)

    def check_for_changes(self) -> dict[str, str | None] | None:
        """Check if .env has been modified. Returns changed keys or None.

        Call this periodically (e.g., from a background loop) to detect
        file modifications and trigger rotation callbacks.
        """
        if not self._env_path.exists():
            return None

        try:
            mtime = self._env_path.stat().st_mtime
        except OSError:
            return None

        if mtime <= self._last_mtime:
            return None

        self._last_mtime = mtime
        new_values = dotenv_values(self._env_path)

        # Find changed keys
        changed: dict[str, str | None] = {}
        all_keys = set(self._current_values.keys()) | set(new_values.keys())
        for key in all_keys:
            old = self._current_values.get(key)
            new = new_values.get(key)
            if old != new:
                changed[key] = new

        if not changed:
            return None

        # Apply changes to environment
        for key, value in changed.items():
            if value is not None:
                os.environ[key] = value
                log.info("secrets.rotated key=%s", key)
            else:
                os.environ.pop(key, None)
                log.info("secrets.removed key=%s", key)

        self._current_values = new_values

        # Record rotation event
        self._rotation_history.append({
            "timestamp": time.time(),
            "changed_keys": list(changed.keys()),
            "count": len(changed),
        })

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(changed)
            except Exception:
                log.exception("secrets.callback_error")

        return changed

    def force_reload(self) -> dict[str, str | None]:
        """Force reload all values from .env regardless of mtime."""
        self._last_mtime = 0.0
        return self.check_for_changes() or {}

    def rotation_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent rotation events for audit trail."""
        return list(reversed(self._rotation_history[-limit:]))

    def get(self, key: str) -> str | None:
        """Get current value of a secret."""
        return self._current_values.get(key)

    def masked_values(self) -> dict[str, str]:
        """Return all keys with masked values for debugging."""
        result = {}
        for key, value in self._current_values.items():
            if value:
                result[key] = value[:4] + "..." + value[-4:] if len(value) > 8 else "****"
            else:
                result[key] = "(empty)"
        return result


# ── Singleton ──────────────────────────────────────────────────────────────

_secrets_manager: SecretsManager | None = None


def get_secrets_manager() -> SecretsManager:
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager

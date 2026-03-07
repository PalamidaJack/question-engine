"""Filesystem-backed artifact store with TTL cleanup.

Replaces the in-memory ``_ArtifactStore`` in ChatService with durable
filesystem storage organized by session.  Gated behind the
``filesystem_artifacts`` feature flag.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class FilesystemArtifactStore:
    """Store large tool results on disk instead of in memory.

    Directory structure::

        {base_dir}/{session_id}/{artifact_id}.json

    Each artifact file contains:
    - content: the full text/data
    - metadata: creation time, size, content type
    - ttl: time-to-live in seconds (0 = no expiry)
    """

    def __init__(
        self,
        base_dir: str | Path = "data/artifacts",
        default_ttl_seconds: int = 3600,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._default_ttl = default_ttl_seconds

    def store(
        self,
        session_id: str,
        artifact_id: str,
        content: str,
        *,
        content_type: str = "text",
        ttl_seconds: int | None = None,
    ) -> str:
        """Store an artifact and return its handle string."""
        session_dir = self._base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        record = {
            "content": content,
            "content_type": content_type,
            "created_at": time.time(),
            "ttl_seconds": ttl,
            "size_bytes": len(content.encode("utf-8")),
        }

        path = session_dir / f"{artifact_id}.json"
        path.write_text(json.dumps(record), encoding="utf-8")

        handle = f"[artifact:{artifact_id}]"
        log.debug(
            "artifact.stored session=%s id=%s size=%d",
            session_id,
            artifact_id,
            record["size_bytes"],
        )
        return handle

    def retrieve(self, session_id: str, artifact_id: str) -> str | None:
        """Retrieve artifact content by session and ID."""
        path = self._base_dir / session_id / f"{artifact_id}.json"
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            # Check TTL
            ttl = record.get("ttl_seconds", 0)
            if ttl > 0:
                age = time.time() - record.get("created_at", 0)
                if age > ttl:
                    path.unlink(missing_ok=True)
                    return None
            return record.get("content")
        except (json.JSONDecodeError, KeyError):
            return None

    def list_artifacts(self, session_id: str) -> list[dict[str, Any]]:
        """List artifacts in a session."""
        session_dir = self._base_dir / session_id
        if not session_dir.exists():
            return []
        result = []
        for p in sorted(session_dir.glob("*.json")):
            try:
                record = json.loads(p.read_text(encoding="utf-8"))
                result.append({
                    "id": p.stem,
                    "content_type": record.get("content_type", "text"),
                    "size_bytes": record.get("size_bytes", 0),
                    "created_at": record.get("created_at", 0),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return result

    def delete(self, session_id: str, artifact_id: str) -> bool:
        """Delete a specific artifact."""
        path = self._base_dir / session_id / f"{artifact_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove all expired artifacts across all sessions.

        Returns the number of artifacts removed.
        """
        removed = 0
        if not self._base_dir.exists():
            return 0
        now = time.time()
        for session_dir in self._base_dir.iterdir():
            if not session_dir.is_dir():
                continue
            for p in session_dir.glob("*.json"):
                try:
                    record = json.loads(p.read_text(encoding="utf-8"))
                    ttl = record.get("ttl_seconds", 0)
                    if ttl > 0:
                        age = now - record.get("created_at", 0)
                        if age > ttl:
                            p.unlink()
                            removed += 1
                except (json.JSONDecodeError, KeyError, OSError):
                    continue
            # Remove empty session dirs
            if session_dir.exists() and not any(session_dir.iterdir()):
                session_dir.rmdir()
        if removed:
            log.info("artifact.cleanup removed=%d", removed)
        return removed

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics."""
        total = 0
        total_bytes = 0
        sessions = 0
        if self._base_dir.exists():
            for session_dir in self._base_dir.iterdir():
                if session_dir.is_dir():
                    sessions += 1
                    for p in session_dir.glob("*.json"):
                        total += 1
                        total_bytes += p.stat().st_size
        return {
            "total_artifacts": total,
            "total_bytes": total_bytes,
            "sessions": sessions,
            "base_dir": str(self._base_dir),
        }

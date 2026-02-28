"""Security monitor for genome integrity and behavioral
auditing."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Patterns that may indicate anomalous behavior
_ANOMALY_PATTERNS = [
    "unauthorized_access",
    "genome_modified",
    "excessive_retries",
    "budget_exceeded",
    "unknown_service",
]


class SecurityMonitor:
    """Monitors genome file integrity and audits system
    behavior for anomalies."""

    def __init__(
        self, genome_dir: str = "genomes"
    ) -> None:
        self.genome_dir = Path(genome_dir)
        self._known_hashes: dict[str, str] = {}

    async def integrity_check(self) -> list[dict]:
        """Check genome files against recorded baseline
        hashes.

        Returns a list of alerts for files that have changed,
        been added, or been removed since the baseline was
        recorded.
        """
        alerts: list[dict] = []

        if not self._known_hashes:
            log.warning(
                "No baseline recorded; "
                "run record_baseline() first"
            )
            alerts.append(
                {
                    "level": "warning",
                    "message": "No baseline recorded.",
                    "file": None,
                }
            )
            return alerts

        current = self._scan_hashes()

        # Check for modified or removed files
        for path, expected_hash in self._known_hashes.items():
            actual_hash = current.get(path)
            if actual_hash is None:
                alerts.append(
                    {
                        "level": "error",
                        "message": "File missing from disk.",
                        "file": path,
                    }
                )
            elif actual_hash != expected_hash:
                alerts.append(
                    {
                        "level": "error",
                        "message": "File hash mismatch.",
                        "file": path,
                    }
                )

        # Check for new files not in baseline
        for path in current:
            if path not in self._known_hashes:
                alerts.append(
                    {
                        "level": "warning",
                        "message": "New file not in baseline.",
                        "file": path,
                    }
                )

        if not alerts:
            log.info("Integrity check passed")
        else:
            log.warning(
                "Integrity check found %d issue(s)",
                len(alerts),
            )

        return alerts

    async def record_baseline(self) -> None:
        """Scan genome files and store their hashes as the
        trusted baseline."""
        self._known_hashes = self._scan_hashes()
        log.info(
            "Baseline recorded: %d file(s)",
            len(self._known_hashes),
        )

    async def behavioral_audit(
        self,
        recent_events: list[dict] | None = None,
    ) -> list[dict]:
        """Audit recent events for anomalous behavior
        patterns.

        Args:
            recent_events: List of event dicts to audit.

        Returns a list of alert dicts for detected anomalies.
        """
        if not recent_events:
            return []

        alerts: list[dict] = []

        for event in recent_events:
            event_type = event.get("type", "")
            for pattern in _ANOMALY_PATTERNS:
                if pattern in event_type:
                    alerts.append(
                        {
                            "level": "warning",
                            "pattern": pattern,
                            "event": event,
                            "message": (
                                f"Anomalous pattern: "
                                f"{pattern}"
                            ),
                        }
                    )

        if alerts:
            log.warning(
                "Behavioral audit found %d anomaly(ies)",
                len(alerts),
            )
        else:
            log.info("Behavioral audit clean")

        return alerts

    def _scan_hashes(self) -> dict[str, str]:
        """Compute SHA-256 hashes for all files in the genome
        directory."""
        hashes: dict[str, str] = {}

        if not self.genome_dir.is_dir():
            log.warning(
                "Genome directory not found: %s",
                self.genome_dir,
            )
            return hashes

        for file_path in sorted(self.genome_dir.rglob("*")):
            if file_path.is_file():
                digest = hashlib.sha256(
                    file_path.read_bytes()
                ).hexdigest()
                hashes[str(file_path)] = digest

        return hashes

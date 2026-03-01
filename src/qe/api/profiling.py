"""Inquiry profiling store — ring buffer for phase timing history.

Stores per-inquiry phase_timings + duration for the last N runs,
and computes percentile aggregates across runs.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class _ProfilingEntry:
    phase_timings: dict[str, dict[str, float]]
    duration_s: float


class InquiryProfilingStore:
    """Ring buffer (max N entries) of inquiry profiling data."""

    def __init__(self, max_entries: int = 50) -> None:
        self._max = max_entries
        self._buffer: deque[_ProfilingEntry] = deque(maxlen=max_entries)

    def record(self, phase_timings: dict[str, dict[str, float]], duration_s: float) -> None:
        """Record a completed inquiry's profiling data."""
        self._buffer.append(_ProfilingEntry(phase_timings=phase_timings, duration_s=duration_s))

    @property
    def last(self) -> dict[str, dict[str, float]] | None:
        """Return the most recent phase_timings, or None if empty."""
        if not self._buffer:
            return None
        return self._buffer[-1].phase_timings

    @property
    def count(self) -> int:
        return len(self._buffer)

    def percentiles(self) -> dict[str, dict[str, float]]:
        """Compute p50/p95/p99 across all recorded runs, per phase.

        Uses avg_s from each run's phase_timings.
        """
        if not self._buffer:
            return {}

        # Collect avg_s values per phase across all entries
        phase_values: dict[str, list[float]] = {}
        for entry in self._buffer:
            for phase, stats in entry.phase_timings.items():
                phase_values.setdefault(phase, []).append(stats.get("avg_s", 0.0))

        result: dict[str, dict[str, float]] = {}
        for phase, values in phase_values.items():
            result[phase] = _compute_percentiles(values)

        return result

    def to_dict(self) -> dict[str, Any]:
        """Full snapshot for the profiling endpoint."""
        return {
            "last_inquiry": self.last or {},
            "history_count": self.count,
            "percentiles": self.percentiles(),
        }


def _compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute p50, p95, p99 from a list of floats."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    if len(values) < 2:
        v = values[0]
        return {"p50": v, "p95": v, "p99": v}
    q = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "p50": q[49],
        "p95": q[94],
        "p99": q[98],
    }

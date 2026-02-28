"""Per-topic bus metrics: publish counts, handler latency, error counts.

Tracks throughput and performance per bus topic for dashboard
visualization and operational diagnostics.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class TopicStats:
    """Aggregated statistics for a single bus topic."""

    publish_count: int = 0
    handler_calls: int = 0
    handler_errors: int = 0
    dlq_count: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")

    @property
    def avg_latency_ms(self) -> float:
        if self.handler_calls == 0:
            return 0.0
        return self.total_latency_ms / self.handler_calls

    @property
    def error_rate(self) -> float:
        if self.handler_calls == 0:
            return 0.0
        return self.handler_errors / self.handler_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "publish_count": self.publish_count,
            "handler_calls": self.handler_calls,
            "handler_errors": self.handler_errors,
            "dlq_count": self.dlq_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "min_latency_ms": (
                round(self.min_latency_ms, 2)
                if self.min_latency_ms != float("inf")
                else 0.0
            ),
            "error_rate": round(self.error_rate, 4),
        }


class BusMetrics:
    """Collects per-topic metrics for the event bus.

    Usage:
        metrics = BusMetrics()
        metrics.record_publish("claims.proposed")
        metrics.record_handler_start("claims.proposed")
        metrics.record_handler_done("claims.proposed", duration_ms=42.5)
        metrics.record_handler_error("claims.proposed")
    """

    def __init__(self) -> None:
        self._topics: dict[str, TopicStats] = defaultdict(TopicStats)
        self._subscriber_counts: dict[str, int] = defaultdict(int)
        self._started_at = time.time()

    def record_publish(self, topic: str) -> None:
        """Record that an envelope was published to a topic."""
        self._topics[topic].publish_count += 1

    def record_handler_done(
        self, topic: str, duration_ms: float
    ) -> None:
        """Record a successful handler execution."""
        stats = self._topics[topic]
        stats.handler_calls += 1
        stats.total_latency_ms += duration_ms
        if duration_ms > stats.max_latency_ms:
            stats.max_latency_ms = duration_ms
        if duration_ms < stats.min_latency_ms:
            stats.min_latency_ms = duration_ms

    def record_handler_error(self, topic: str) -> None:
        """Record a handler execution error."""
        self._topics[topic].handler_errors += 1

    def record_dlq(self, topic: str) -> None:
        """Record an envelope routed to DLQ for this topic."""
        self._topics[topic].dlq_count += 1

    def set_subscriber_count(self, topic: str, count: int) -> None:
        """Update the subscriber count for a topic."""
        self._subscriber_counts[topic] = count

    def get_topic_stats(self, topic: str) -> dict[str, Any]:
        """Get stats for a single topic."""
        return self._topics[topic].to_dict()

    def snapshot(self) -> dict[str, Any]:
        """Return full bus metrics snapshot."""
        total_published = sum(s.publish_count for s in self._topics.values())
        total_errors = sum(s.handler_errors for s in self._topics.values())
        total_dlq = sum(s.dlq_count for s in self._topics.values())

        # Top topics by volume
        by_volume = sorted(
            self._topics.items(),
            key=lambda x: x[1].publish_count,
            reverse=True,
        )

        return {
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "total_published": total_published,
            "total_errors": total_errors,
            "total_dlq": total_dlq,
            "active_topics": len(self._topics),
            "topics": {
                topic: stats.to_dict()
                for topic, stats in self._topics.items()
            },
            "top_topics": [
                {"topic": t, "publish_count": s.publish_count}
                for t, s in by_volume[:10]
            ],
            "subscriber_counts": dict(self._subscriber_counts),
        }

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        self._topics.clear()
        self._subscriber_counts.clear()
        self._started_at = time.time()


# ── Singleton ──────────────────────────────────────────────────────────────

_bus_metrics = BusMetrics()


def get_bus_metrics() -> BusMetrics:
    return _bus_metrics

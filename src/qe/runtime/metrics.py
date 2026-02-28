"""Observability metrics for Question Engine.

Provides counters, histograms, and gauges for LLM calls, bus throughput,
errors, and latency. Designed to work without external dependencies
(Prometheus, StatsD) while being easy to export to them.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class Counter:
    """Monotonically increasing counter."""

    name: str
    value: int = 0
    labels: dict[str, str] = field(default_factory=dict)

    def inc(self, n: int = 1) -> None:
        self.value += n


@dataclass
class Histogram:
    """Tracks distribution of values in predefined buckets."""

    name: str
    buckets: list[float] = field(
        default_factory=lambda: [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
    )
    _counts: list[int] = field(default_factory=list)
    _sum: float = 0.0
    _count: int = 0

    def __post_init__(self) -> None:
        if not self._counts:
            self._counts = [0] * (len(self.buckets) + 1)  # +1 for +Inf

    def observe(self, value: float) -> None:
        self._sum += value
        self._count += 1
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self._counts[i] += 1
                return
        self._counts[-1] += 1  # +Inf bucket

    @property
    def p50(self) -> float:
        return self._percentile(0.50)

    @property
    def p95(self) -> float:
        return self._percentile(0.95)

    @property
    def p99(self) -> float:
        return self._percentile(0.99)

    @property
    def avg(self) -> float:
        return self._sum / self._count if self._count else 0.0

    def _percentile(self, pct: float) -> float:
        if self._count == 0:
            return 0.0
        target = pct * self._count
        cumulative = 0
        for i, count in enumerate(self._counts):
            cumulative += count
            if cumulative >= target:
                return self.buckets[i] if i < len(self.buckets) else self.buckets[-1]
        return self.buckets[-1] if self.buckets else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "count": self._count,
            "sum": round(self._sum, 2),
            "avg": round(self.avg, 2),
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
        }


@dataclass
class Gauge:
    """Point-in-time value (can go up or down)."""

    name: str
    value: float = 0.0

    def set(self, v: float) -> None:
        self.value = v

    def inc(self, n: float = 1.0) -> None:
        self.value += n

    def dec(self, n: float = 1.0) -> None:
        self.value -= n


# ── SLO Definitions ────────────────────────────────────────────────────────


@dataclass
class SLO:
    """Service Level Objective definition."""

    name: str
    metric: str  # metric name to evaluate
    target: float  # target value
    comparator: str = "lte"  # "lte", "gte", "lt", "gt"
    window_seconds: int = 3600  # evaluation window

    def evaluate(self, actual: float) -> bool:
        if self.comparator == "lte":
            return actual <= self.target
        if self.comparator == "gte":
            return actual >= self.target
        if self.comparator == "lt":
            return actual < self.target
        if self.comparator == "gt":
            return actual > self.target
        return False


_DEFAULT_SLOS = [
    SLO("llm_latency_p99", "llm_latency_ms.p99", target=5000.0, comparator="lte"),
    SLO("llm_error_rate", "llm_errors_total", target=0.05, comparator="lte"),
    SLO("bus_dlq_rate", "bus_dlq_total", target=0.01, comparator="lte"),
    SLO("handler_latency_p95", "handler_latency_ms.p95", target=2000.0, comparator="lte"),
    SLO("budget_remaining", "budget_remaining_pct", target=0.10, comparator="gte"),
]


# ── Metrics Collector ──────────────────────────────────────────────────────


class MetricsCollector:
    """Central metrics registry for the QE process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._gauges: dict[str, Gauge] = {}
        self._slos: list[SLO] = list(_DEFAULT_SLOS)
        self._started_at = time.time()

        # Pre-register core metrics
        self._register_defaults()

    def _register_defaults(self) -> None:
        # Counters
        self.counter("llm_calls_total")
        self.counter("llm_errors_total")
        self.counter("bus_published_total")
        self.counter("bus_delivered_total")
        self.counter("bus_dlq_total")
        self.counter("bus_dedup_total")
        self.counter("bus_retries_total")
        self.counter("api_requests_total")
        self.counter("api_errors_total")
        self.counter("goals_submitted_total")
        self.counter("goals_completed_total")
        self.counter("goals_failed_total")
        self.counter("retrieval_queries_total")
        self.counter("retrieval_hybrid_calls_total")
        self.counter("retrieval_hybrid_semantic_nonempty_total")
        self.counter("retrieval_hybrid_fts_nonempty_total")

        # Histograms
        self.histogram("llm_latency_ms")
        self.histogram("handler_latency_ms")
        self.histogram("api_latency_ms")
        self.histogram("vector_query_latency_ms")

        # Gauges
        self.gauge("active_services")
        self.gauge("active_goals")
        self.gauge("circuit_broken_services")
        self.gauge("dlq_size")
        self.gauge("budget_remaining_pct")
        self.gauge("bus_in_flight")
        self.gauge("vector_index_size")
        self.gauge("vector_hnsw_enabled")

    def counter(self, name: str) -> Counter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name=name)
            return self._counters[name]

    def histogram(self, name: str) -> Histogram:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name=name)
            return self._histograms[name]

    def gauge(self, name: str) -> Gauge:
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name=name)
            return self._gauges[name]

    def evaluate_slos(self) -> list[dict[str, Any]]:
        """Evaluate all SLOs against current metrics."""
        results = []
        for slo in self._slos:
            parts = slo.metric.split(".")
            metric_name = parts[0]
            sub_metric = parts[1] if len(parts) > 1 else None

            actual = 0.0
            if metric_name in self._histograms and sub_metric:
                snap = self._histograms[metric_name].snapshot()
                actual = snap.get(sub_metric, 0.0)
            elif metric_name in self._counters:
                total = self.counter("llm_calls_total").value
                actual = (
                    self._counters[metric_name].value / total
                    if total > 0
                    else 0.0
                )
            elif metric_name in self._gauges:
                actual = self._gauges[metric_name].value

            passing = slo.evaluate(actual)
            results.append({
                "name": slo.name,
                "target": slo.target,
                "actual": round(actual, 4),
                "passing": passing,
                "comparator": slo.comparator,
            })
        return results

    def snapshot(self) -> dict[str, Any]:
        """Return full metrics snapshot for /api/metrics."""
        return {
            "uptime_seconds": round(time.time() - self._started_at, 1),
            "counters": {
                name: c.value for name, c in self._counters.items()
            },
            "histograms": {
                name: h.snapshot() for name, h in self._histograms.items()
            },
            "gauges": {
                name: g.value for name, g in self._gauges.items()
            },
            "slos": self.evaluate_slos(),
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    return _metrics

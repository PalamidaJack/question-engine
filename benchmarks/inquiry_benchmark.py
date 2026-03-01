"""Benchmark harness for InquiryEngine on M1.

Runs the inquiry engine N times with mock LLMs and reports:
- Per-phase timing percentiles (p50, p95, p99)
- Memory usage (RSS before/after)
- Throughput (inquiries/sec)
- EngramCache stats

Usage:
    .venv/bin/python benchmarks/inquiry_benchmark.py [--iterations 20] [--engine-iters 3] [--json]
"""

from __future__ import annotations

import argparse
import json as _json
import resource
import statistics
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# Ensure src/ is importable when running standalone
sys.path.insert(0, "src")

from qe.models.cognition import (  # noqa: E402, I001
    ApproachAssessment,
    DialecticReport,
    UncertaintyAssessment,
)
from qe.services.inquiry.engine import InquiryEngine  # noqa: E402
from qe.services.inquiry.schemas import InquiryConfig, Question  # noqa: E402


# ---------------------------------------------------------------------------
# Mock factory (same pattern as test fixtures, inlined)
# ---------------------------------------------------------------------------


def _make_mock_engine(engine_iters: int = 3) -> InquiryEngine:
    """Build an InquiryEngine with fully mocked components."""
    em = MagicMock()
    em.recall_for_goal = AsyncMock(return_value=[])

    mc = MagicMock()
    mc.suggest_next_approach = AsyncMock(
        return_value=ApproachAssessment(
            recommended_approach="benchmark approach",
            reasoning="benchmark",
        )
    )

    qg = MagicMock()
    qg.generate = AsyncMock(
        return_value=[
            Question(
                text="Benchmark question",
                expected_info_gain=0.8,
                relevance_to_goal=0.9,
                novelty_score=0.7,
            ),
        ]
    )
    qg.prioritize = AsyncMock(
        side_effect=lambda goal, qs: sorted(
            qs, key=lambda q: q.expected_info_gain, reverse=True
        )
    )

    hm = MagicMock()
    hm.get_active_hypotheses = AsyncMock(return_value=[])
    hm.generate_hypotheses = AsyncMock(return_value=[])

    ep = MagicMock()
    ep.get_epistemic_state = MagicMock(return_value=None)
    ep.assess_uncertainty = AsyncMock(
        return_value=UncertaintyAssessment(finding_summary="bench")
    )
    ep.detect_surprise = AsyncMock(return_value=None)
    ep.get_blind_spot_warning = MagicMock(return_value="")

    de = MagicMock()
    de.full_dialectic = AsyncMock(
        return_value=DialecticReport(
            original_conclusion="bench", revised_confidence=0.6
        )
    )

    ic = MagicMock()
    ic.crystallize = AsyncMock(return_value=None)

    cc = MagicMock()
    cc.detect_drift = MagicMock(return_value=None)

    pe = MagicMock()

    pm = MagicMock()
    pm.get_best_templates = AsyncMock(return_value=[])

    return InquiryEngine(
        episodic_memory=em,
        context_curator=cc,
        metacognitor=mc,
        epistemic_reasoner=ep,
        dialectic_engine=de,
        persistence_engine=pe,
        insight_crystallizer=ic,
        question_generator=qg,
        hypothesis_manager=hm,
        procedural_memory=pm,
        tool_registry=None,
        budget_tracker=None,
        bus=MagicMock(),
        config=InquiryConfig(max_iterations=engine_iters, confidence_threshold=0.8),
    )


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


def compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute p50, p95, p99 from a list of values."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    if len(values) < 2:
        v = values[0]
        return {"p50": v, "p95": v, "p99": v}
    # statistics.quantiles needs at least 2 values
    q = statistics.quantiles(values, n=100, method="inclusive")
    return {
        "p50": q[49],
        "p95": q[94],
        "p99": q[98],
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    iterations: int = 20,
    engine_iters: int = 3,
) -> dict[str, Any]:
    """Run the inquiry engine N times and collect metrics."""
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    all_durations: list[float] = []
    phase_timings_all: dict[str, list[float]] = {}

    for i in range(iterations):
        engine = _make_mock_engine(engine_iters)
        t0 = time.monotonic()
        result = await engine.run_inquiry(f"bench_{i}", f"Benchmark goal {i}")
        elapsed = time.monotonic() - t0
        all_durations.append(elapsed)

        # Collect per-phase timings
        for phase, stats in result.phase_timings.items():
            phase_timings_all.setdefault(phase, []).append(stats.get("avg_s", 0.0))

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    total_time = sum(all_durations)

    # Cache stats
    cache_stats: dict[str, Any] = {}
    try:
        from qe.runtime.engram_cache import get_engram_cache

        cache_stats = get_engram_cache().stats()
    except Exception:
        pass

    # Build report
    phase_percentiles: dict[str, dict[str, float]] = {}
    for phase, values in phase_timings_all.items():
        phase_percentiles[phase] = compute_percentiles(values)

    return {
        "iterations": iterations,
        "engine_iters_per_run": engine_iters,
        "total_time_s": round(total_time, 4),
        "throughput_inquiries_per_sec": round(iterations / total_time, 2) if total_time > 0 else 0,
        "duration_percentiles": compute_percentiles(all_durations),
        "phase_percentiles": phase_percentiles,
        "memory": {
            "rss_before": rss_before,
            "rss_after": rss_after,
            "rss_delta": rss_after - rss_before,
        },
        "cache_stats": cache_stats,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_report(report: dict[str, Any]) -> str:
    """Format a benchmark report as human-readable text."""
    lines = [
        "=" * 60,
        "  InquiryEngine Benchmark Report",
        "=" * 60,
        f"  Iterations:      {report['iterations']}",
        f"  Engine iters:    {report['engine_iters_per_run']}",
        f"  Total time:      {report['total_time_s']:.4f}s",
        f"  Throughput:      {report['throughput_inquiries_per_sec']:.2f} inquiries/sec",
        "",
        "  Duration Percentiles:",
        f"    p50: {report['duration_percentiles']['p50']:.4f}s",
        f"    p95: {report['duration_percentiles']['p95']:.4f}s",
        f"    p99: {report['duration_percentiles']['p99']:.4f}s",
        "",
        "  Per-Phase Timing Percentiles (avg_s across runs):",
    ]
    for phase, pcts in sorted(report["phase_percentiles"].items()):
        lines.append(f"    {phase}:")
        lines.append(f"      p50={pcts['p50']:.6f}  p95={pcts['p95']:.6f}  p99={pcts['p99']:.6f}")

    lines.extend([
        "",
        "  Memory:",
        f"    RSS before: {report['memory']['rss_before']}",
        f"    RSS after:  {report['memory']['rss_after']}",
        f"    RSS delta:  {report['memory']['rss_delta']}",
    ])

    if report["cache_stats"]:
        lines.append("")
        lines.append("  Cache Stats:")
        for k, v in report["cache_stats"].items():
            lines.append(f"    {k}: {v}")

    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    import asyncio

    parser = argparse.ArgumentParser(description="InquiryEngine benchmark")
    parser.add_argument("--iterations", type=int, default=20, help="Number of inquiry runs")
    parser.add_argument("--engine-iters", type=int, default=3, help="Max iterations per engine run")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    report = asyncio.run(run_benchmark(args.iterations, args.engine_iters))

    if args.json:
        print(_json.dumps(report, indent=2))
    else:
        print(_format_report(report))


if __name__ == "__main__":
    main()

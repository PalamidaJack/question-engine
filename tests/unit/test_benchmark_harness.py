"""Tests for the benchmark harness."""

from __future__ import annotations

import pytest
from benchmarks.inquiry_benchmark import compute_percentiles, run_benchmark


class TestBenchmarkHarness:
    @pytest.mark.asyncio
    async def test_benchmark_report_structure(self):
        """Run 2 iterations and verify report keys."""
        report = await run_benchmark(iterations=2, engine_iters=2)

        assert "iterations" in report
        assert report["iterations"] == 2
        assert "engine_iters_per_run" in report
        assert "total_time_s" in report
        assert "throughput_inquiries_per_sec" in report
        assert "duration_percentiles" in report
        assert "phase_percentiles" in report
        assert "memory" in report
        assert "cache_stats" in report

        # Duration percentiles have expected keys
        dp = report["duration_percentiles"]
        assert "p50" in dp
        assert "p95" in dp
        assert "p99" in dp

        # Phase percentiles should have at least 'observe' phase
        assert len(report["phase_percentiles"]) > 0

    def test_benchmark_percentile_computation(self):
        """Unit test percentile helper."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = compute_percentiles(values)

        assert "p50" in result
        assert "p95" in result
        assert "p99" in result
        # p50 should be around 5.5
        assert 4.0 <= result["p50"] <= 6.0
        # p95 should be high
        assert result["p95"] > result["p50"]
        # p99 should be highest
        assert result["p99"] >= result["p95"]

        # Edge: empty list
        empty = compute_percentiles([])
        assert empty == {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        # Edge: single value
        single = compute_percentiles([42.0])
        assert single == {"p50": 42.0, "p95": 42.0, "p99": 42.0}

    @pytest.mark.asyncio
    async def test_benchmark_memory_tracking(self):
        """Verify RSS values are positive."""
        report = await run_benchmark(iterations=2, engine_iters=1)

        mem = report["memory"]
        assert mem["rss_before"] > 0
        assert mem["rss_after"] > 0

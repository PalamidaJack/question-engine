"""Tests for Phases 8-11: Learning, Optimization, Dashboard, Self-Management."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from qe.kernel.watchdog import Watchdog
from qe.runtime.assumption_monitor import (
    AssumptionMonitor,
    InvalidatedAssumption,
)
from qe.runtime.calibration import CalibrationTracker
from qe.runtime.model_capabilities import ModelCapabilities
from qe.runtime.output_enforcement import OutputEnforcer
from qe.runtime.routing_optimizer import ModelScore, RoutingOptimizer

# ── Phase 8: Calibration ──────────────────────────────────────────


class TestCalibrationTracker:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmp) / "cal.db")

    @pytest.mark.asyncio
    async def test_record_and_calibrate(self):
        tracker = CalibrationTracker()
        # Record enough data to calibrate
        for _ in range(10):
            await tracker.record("gpt-4", "research", 0.85, True)
        for _ in range(10):
            await tracker.record("gpt-4", "research", 0.85, False)
        # Calibrated confidence should be ~0.5
        cal = tracker.calibrated_confidence("gpt-4", "research", 0.85)
        assert 0.4 <= cal <= 0.6

    @pytest.mark.asyncio
    async def test_insufficient_data_returns_raw(self):
        tracker = CalibrationTracker()
        await tracker.record("gpt-4", "research", 0.9, True)
        cal = tracker.calibrated_confidence("gpt-4", "research", 0.9)
        assert cal == 0.9  # Not enough data, returns raw

    @pytest.mark.asyncio
    async def test_calibration_curve(self):
        tracker = CalibrationTracker()
        for i in range(20):
            conf = 0.85
            correct = i % 2 == 0
            await tracker.record("gpt-4", "test", conf, correct)
        curve = tracker.get_calibration_curve("gpt-4", "test")
        assert len(curve) >= 1
        assert all(
            0 <= reported <= 1 and 0 <= actual <= 1
            for reported, actual in curve
        )

    @pytest.mark.asyncio
    async def test_count(self):
        tracker = CalibrationTracker()
        await tracker.record("m1", "t1", 0.5, True)
        await tracker.record("m1", "t1", 0.5, False)
        assert await tracker.count() == 2

    @pytest.mark.asyncio
    async def test_persistence(self):
        tracker = CalibrationTracker(db_path=self.db_path)
        await tracker.record("m1", "t1", 0.7, True)
        # Verify it was written to the DB
        import aiosqlite

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM calibration_records"
            )
            row = await cursor.fetchone()
            assert row[0] == 1


# ── Phase 8: Routing Optimizer ────────────────────────────────────


class TestRoutingOptimizer:
    @pytest.mark.asyncio
    async def test_record_and_score(self):
        opt = RoutingOptimizer(exploration_rate=0.0)
        await opt.record_outcome("gpt-4", "research", True, 500, 0.01)
        await opt.record_outcome("gpt-4", "research", True, 400, 0.01)
        await opt.record_outcome("llama", "research", False, 200, 0.0)

        scores = opt.get_model_scores(
            "research", ["gpt-4", "llama"]
        )
        gpt_score = next(s for s in scores if s.model == "gpt-4")
        llama_score = next(s for s in scores if s.model == "llama")
        assert gpt_score.success_rate == 1.0
        assert llama_score.success_rate == 0.0

    @pytest.mark.asyncio
    async def test_select_model_exploitation(self):
        opt = RoutingOptimizer(exploration_rate=0.0)
        for _ in range(5):
            await opt.record_outcome("good", "t1", True)
            await opt.record_outcome("bad", "t1", False)

        selected = opt.select_model("t1", ["good", "bad"])
        assert selected == "good"

    @pytest.mark.asyncio
    async def test_select_model_empty_raises(self):
        opt = RoutingOptimizer()
        with pytest.raises(ValueError, match="No models"):
            opt.select_model("t1", [])

    def test_model_score_composite(self):
        score = ModelScore(
            model="test",
            success_rate=0.9,
            avg_latency_ms=500,
            avg_cost_usd=0.01,
            sample_count=10,
        )
        composite = score.compute_composite()
        assert composite > 0

    def test_model_score_budget_penalty(self):
        score = ModelScore(
            model="expensive",
            success_rate=0.9,
            avg_latency_ms=500,
            avg_cost_usd=5.0,
            sample_count=10,
        )
        # Low budget should penalize expensive models
        composite_low = score.compute_composite(budget_remaining=1.0)
        score2 = ModelScore(
            model="expensive",
            success_rate=0.9,
            avg_latency_ms=500,
            avg_cost_usd=5.0,
            sample_count=10,
        )
        composite_high = score2.compute_composite(
            budget_remaining=1000.0
        )
        assert composite_low < composite_high


# ── Phase 9: Model Capabilities ───────────────────────────────────


class TestModelCapabilities:
    def test_known_model_profile(self):
        caps = ModelCapabilities()
        profile = caps.get_profile("gpt-4o")
        assert profile.supports_json_mode is True
        assert profile.supports_tool_calling is True
        assert profile.estimated_quality_tier == "powerful"

    def test_local_model_inferred(self):
        caps = ModelCapabilities()
        profile = caps.get_profile("llama-3.1-8b")
        assert profile.estimated_quality_tier == "local"
        assert profile.supports_grammar is True

    def test_unknown_model_defaults(self):
        caps = ModelCapabilities()
        profile = caps.get_profile("completely-unknown-model")
        assert profile.estimated_quality_tier == "balanced"

    def test_is_capable(self):
        caps = ModelCapabilities()
        assert caps.is_capable("gpt-4o", "supports_json_mode") is True
        assert (
            caps.is_capable("gpt-4o", "supports_grammar") is False
        )

    def test_cache(self):
        caps = ModelCapabilities()
        p1 = caps.get_profile("gpt-4o")
        p2 = caps.get_profile("gpt-4o")
        assert p1 is p2
        caps.clear_cache()
        p3 = caps.get_profile("gpt-4o")
        assert p3 is not p1


# ── Phase 9: Output Enforcement ───────────────────────────────────


class TestOutputEnforcer:
    def setup_method(self):
        self.enforcer = OutputEnforcer()

    def test_parse_valid_json(self):
        result = self.enforcer.try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_from_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = self.enforcer.try_parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_embedded(self):
        text = 'Here is the result: {"key": "value"} done.'
        result = self.enforcer.try_parse_json(text)
        assert result == {"key": "value"}

    def test_parse_invalid_returns_none(self):
        result = self.enforcer.try_parse_json("no json here")
        assert result is None

    def test_extract_fields(self):
        text = "name: John Smith\nage: 30\ncity: NYC"
        result = self.enforcer.extract_fields(
            text, {"name": "Name", "age": "Age"}
        )
        assert result["name"] == "John Smith"
        assert result["age"] == "30"

    def test_repair_json_trailing_comma(self):
        bad = '{"a": 1, "b": 2, }'
        repaired = self.enforcer.repair_json(bad)
        assert repaired is not None
        import json

        parsed = json.loads(repaired)
        assert parsed["a"] == 1


# ── Phase 11: Assumption Monitor ──────────────────────────────────


class TestAssumptionMonitor:
    @pytest.mark.asyncio
    async def test_check_time_assumption(self):
        monitor = AssumptionMonitor()
        assumptions = ["Data available before deadline"]
        result = await monitor.check_assumptions(assumptions)
        assert len(result) == 1
        assert isinstance(result[0], InvalidatedAssumption)

    @pytest.mark.asyncio
    async def test_check_no_issues(self):
        monitor = AssumptionMonitor()
        assumptions = ["The model produces good output"]
        result = await monitor.check_assumptions(assumptions)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_stale_assumptions(self):
        monitor = AssumptionMonitor()
        assumptions = ["a"] * 10
        stale = await monitor.get_stale_assumptions(assumptions)
        assert len(stale) == 10

    @pytest.mark.asyncio
    async def test_few_assumptions_not_stale(self):
        monitor = AssumptionMonitor()
        assumptions = ["a", "b"]
        stale = await monitor.get_stale_assumptions(assumptions)
        assert len(stale) == 0


# ── Phase 11: Watchdog ────────────────────────────────────────────


class TestWatchdog:
    def test_init(self):
        wd = Watchdog(check_interval=5)
        assert wd.check_interval == 5
        assert wd.is_running is False

    @pytest.mark.asyncio
    async def test_stop(self):
        wd = Watchdog()
        await wd.stop()
        assert wd.is_running is False

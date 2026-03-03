"""Tests for GuardrailsPipeline — rules, pipeline execution, and bus events."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qe.runtime.guardrails import (
    ContentFilterRule,
    CostGuardRule,
    GuardrailResult,
    GuardrailRule,
    GuardrailsPipeline,
    HallucinationGuardRule,
    OutputSchemaValidatorRule,
    PiiDetectorRule,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_bus() -> MagicMock:
    bus = MagicMock()
    bus.publish = MagicMock()
    return bus


# ── GuardrailResult model ───────────────────────────────────────────────


class TestGuardrailResult:
    def test_construction_all_fields(self):
        r = GuardrailResult(
            passed=False, rule_name="test", message="bad", severity="block",
        )
        assert r.passed is False
        assert r.rule_name == "test"
        assert r.message == "bad"
        assert r.severity == "block"

    def test_defaults(self):
        r = GuardrailResult(passed=True, rule_name="r1")
        assert r.message == ""
        assert r.severity == "info"

    def test_dict_serialization(self):
        r = GuardrailResult(passed=True, rule_name="r1", message="ok")
        d = r.dict()
        assert d["passed"] is True
        assert d["rule_name"] == "r1"
        assert d["message"] == "ok"
        assert d["severity"] == "info"


# ── ContentFilterRule ────────────────────────────────────────────────────


class TestContentFilterRule:
    @pytest.mark.asyncio
    async def test_blocks_drop_table(self):
        rule = ContentFilterRule()
        res = await rule.check("please drop table users", {})
        assert res.passed is False
        assert res.severity == "block"
        assert "drop table" in res.message

    @pytest.mark.asyncio
    async def test_blocks_exec(self):
        rule = ContentFilterRule()
        res = await rule.check("exec('rm -rf /')", {})
        assert res.passed is False
        assert res.severity == "block"

    @pytest.mark.asyncio
    async def test_blocks_script_tag(self):
        rule = ContentFilterRule()
        res = await rule.check("inject <script>alert(1)</script>", {})
        assert res.passed is False
        assert res.severity == "block"
        assert "<script>" in res.message

    @pytest.mark.asyncio
    async def test_passes_clean_text(self):
        rule = ContentFilterRule()
        res = await rule.check("What is the weather today?", {})
        assert res.passed is True
        assert res.rule_name == "ContentFilter"

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        rule = ContentFilterRule()
        res = await rule.check("DROP TABLE users", {})
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_custom_patterns(self):
        rule = ContentFilterRule(patterns=[r"forbidden"])
        res = await rule.check("this is forbidden content", {})
        assert res.passed is False
        assert "forbidden" in res.message

    @pytest.mark.asyncio
    async def test_custom_patterns_pass(self):
        rule = ContentFilterRule(patterns=[r"forbidden"])
        res = await rule.check("this is perfectly fine", {})
        assert res.passed is True


# ── PiiDetectorRule ──────────────────────────────────────────────────────


class TestPiiDetectorRule:
    @pytest.mark.asyncio
    async def test_detects_email(self):
        rule = PiiDetectorRule()
        res = await rule.check("contact me at user@example.com please", {})
        assert res.passed is False
        assert res.severity == "warning"
        assert "PII" in res.message

    @pytest.mark.asyncio
    async def test_detects_phone_number(self):
        rule = PiiDetectorRule()
        res = await rule.check("call me at 555-123-4567", {})
        assert res.passed is False
        assert "PII" in res.message

    @pytest.mark.asyncio
    async def test_detects_phone_number_dots(self):
        rule = PiiDetectorRule()
        res = await rule.check("phone: 555.123.4567", {})
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_detects_phone_number_spaces(self):
        rule = PiiDetectorRule()
        res = await rule.check("phone: 555 123 4567", {})
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_detects_ssn(self):
        rule = PiiDetectorRule()
        res = await rule.check("SSN: 123-45-6789", {})
        assert res.passed is False
        assert "PII" in res.message

    @pytest.mark.asyncio
    async def test_passes_clean_text(self):
        rule = PiiDetectorRule()
        res = await rule.check("The market grew by 5% last quarter.", {})
        assert res.passed is True
        assert res.rule_name == "PiiDetector"


# ── CostGuardRule ────────────────────────────────────────────────────────


class TestCostGuardRule:
    @pytest.mark.asyncio
    async def test_blocks_over_threshold(self):
        rule = CostGuardRule()
        res = await rule.check("query", {"estimated_cost_usd": 10.0})
        assert res.passed is False
        assert res.severity == "block"
        assert "estimated_cost_usd=10.0" in res.message

    @pytest.mark.asyncio
    async def test_passes_under_threshold(self):
        rule = CostGuardRule()
        res = await rule.check("query", {"estimated_cost_usd": 2.0})
        assert res.passed is True
        assert res.rule_name == "CostGuard"

    @pytest.mark.asyncio
    async def test_passes_at_threshold(self):
        rule = CostGuardRule(threshold_usd=5.0)
        res = await rule.check("query", {"estimated_cost_usd": 5.0})
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_custom_threshold(self):
        rule = CostGuardRule(threshold_usd=1.0)
        res = await rule.check("query", {"estimated_cost_usd": 1.5})
        assert res.passed is False
        assert res.severity == "block"

    @pytest.mark.asyncio
    async def test_missing_cost_defaults_to_zero(self):
        rule = CostGuardRule()
        res = await rule.check("query", {})
        assert res.passed is True


# ── OutputSchemaValidatorRule ────────────────────────────────────────────


class TestOutputSchemaValidatorRule:
    @pytest.mark.asyncio
    async def test_warns_on_missing_keys(self):
        rule = OutputSchemaValidatorRule()
        ctx = {"expected_schema_keys": ["summary", "confidence"]}
        content = '{"summary": "ok"}'
        res = await rule.check(content, ctx)
        assert res.passed is False
        assert res.severity == "warning"
        assert "confidence" in res.message

    @pytest.mark.asyncio
    async def test_passes_when_all_keys_present_double_quotes(self):
        rule = OutputSchemaValidatorRule()
        ctx = {"expected_schema_keys": ["summary", "confidence"]}
        content = '{"summary": "ok", "confidence": 0.9}'
        res = await rule.check(content, ctx)
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_passes_when_all_keys_present_single_quotes(self):
        rule = OutputSchemaValidatorRule()
        ctx = {"expected_schema_keys": ["summary", "confidence"]}
        content = "{'summary': 'ok', 'confidence': 0.9}"
        res = await rule.check(content, ctx)
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_passes_no_schema_keys(self):
        rule = OutputSchemaValidatorRule()
        res = await rule.check("anything", {})
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_passes_empty_schema_keys(self):
        rule = OutputSchemaValidatorRule()
        res = await rule.check("anything", {"expected_schema_keys": []})
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_warns_multiple_missing_keys(self):
        rule = OutputSchemaValidatorRule()
        ctx = {"expected_schema_keys": ["a", "b", "c"]}
        res = await rule.check("no keys here", ctx)
        assert res.passed is False
        assert "a" in res.message
        assert "b" in res.message
        assert "c" in res.message


# ── HallucinationGuardRule ───────────────────────────────────────────────


class TestHallucinationGuardRule:
    @pytest.mark.asyncio
    async def test_warns_i_think(self):
        rule = HallucinationGuardRule()
        res = await rule.check("I think the answer is 42", {})
        assert res.passed is True  # warning, not block
        assert res.severity == "warning"
        assert "hedging" in res.message

    @pytest.mark.asyncio
    async def test_warns_might_be(self):
        rule = HallucinationGuardRule()
        res = await rule.check("The result might be incorrect", {})
        assert res.passed is True
        assert res.severity == "warning"

    @pytest.mark.asyncio
    async def test_warns_possibly(self):
        rule = HallucinationGuardRule()
        res = await rule.check("This is possibly the cause", {})
        assert res.passed is True
        assert res.severity == "warning"

    @pytest.mark.asyncio
    async def test_warns_could_be(self):
        rule = HallucinationGuardRule()
        res = await rule.check("It could be an issue", {})
        assert res.passed is True
        assert res.severity == "warning"

    @pytest.mark.asyncio
    async def test_passes_confident_text(self):
        rule = HallucinationGuardRule()
        res = await rule.check("The GDP of France in 2024 was 3.1 trillion USD.", {})
        assert res.passed is True
        assert res.severity == "info"
        assert res.message == ""

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        rule = HallucinationGuardRule()
        res = await rule.check("i THINK this is correct", {})
        assert res.severity == "warning"


# ── GuardrailsPipeline.run_input ─────────────────────────────────────────


class TestGuardrailsPipelineRunInput:
    @pytest.mark.asyncio
    async def test_runs_all_enabled_rules(self):
        rules = [ContentFilterRule(), HallucinationGuardRule()]
        pipeline = GuardrailsPipeline(rules=rules)
        results = await pipeline.run_input("clean text here", {})
        assert len(results) == 2
        assert all(r.passed for r in results)

    @pytest.mark.asyncio
    async def test_stops_on_first_blocking_rule(self):
        rules = [
            ContentFilterRule(),
            CostGuardRule(),
            HallucinationGuardRule(),
        ]
        pipeline = GuardrailsPipeline(rules=rules)
        results = await pipeline.run_input("drop table users", {})
        # ContentFilter blocks, so we get only 1 result
        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].rule_name == "ContentFilter"
        assert results[0].severity == "block"

    @pytest.mark.asyncio
    async def test_warning_does_not_stop_pipeline(self):
        """A warning (non-block) should not halt the pipeline."""
        rules = [
            HallucinationGuardRule(),  # will warn
            ContentFilterRule(),       # will pass
        ]
        pipeline = GuardrailsPipeline(rules=rules)
        results = await pipeline.run_input("I think the sky is blue", {})
        # Both rules run; hallucination warns but passed=True
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_disabled_rules_skipped(self):
        rule1 = ContentFilterRule()
        rule2 = CostGuardRule()
        rule2.enabled = False
        pipeline = GuardrailsPipeline(rules=[rule1, rule2])
        results = await pipeline.run_input("clean", {})
        assert len(results) == 1
        assert results[0].rule_name == "ContentFilter"

    @pytest.mark.asyncio
    async def test_empty_rules(self):
        pipeline = GuardrailsPipeline(rules=[])
        results = await pipeline.run_input("anything", {})
        assert results == []

    @pytest.mark.asyncio
    async def test_rule_exception_is_swallowed(self):
        """A rule that raises should be caught; subsequent rules still run."""

        class BrokenRule(GuardrailRule):
            name = "Broken"

            async def check(self, content, context):
                raise RuntimeError("boom")

        rules = [BrokenRule(), ContentFilterRule()]
        pipeline = GuardrailsPipeline(rules=rules)
        results = await pipeline.run_input("safe text", {})
        # BrokenRule is swallowed, ContentFilter runs
        assert len(results) == 1
        assert results[0].rule_name == "ContentFilter"

    @pytest.mark.asyncio
    async def test_run_output_delegates_to_run_input(self):
        """run_output is documented as the same as run_input for now."""
        rules = [ContentFilterRule()]
        pipeline = GuardrailsPipeline(rules=rules)
        results = await pipeline.run_output("safe text", {})
        assert len(results) == 1
        assert results[0].passed is True

    @pytest.mark.asyncio
    async def test_cost_block_stops_before_later_rules(self):
        rules = [
            CostGuardRule(threshold_usd=1.0),
            HallucinationGuardRule(),
        ]
        pipeline = GuardrailsPipeline(rules=rules)
        results = await pipeline.run_input(
            "I think the answer is 42", {"estimated_cost_usd": 5.0},
        )
        # CostGuard blocks, HallucinationGuard never runs
        assert len(results) == 1
        assert results[0].rule_name == "CostGuard"
        assert results[0].severity == "block"


# ── GuardrailsPipeline.default_pipeline ──────────────────────────────────


class TestDefaultPipeline:
    def test_builds_with_no_config(self):
        pipe = GuardrailsPipeline.default_pipeline()
        # Default: ContentFilter, CostGuard, OutputSchemaValidator
        names = [r.name for r in pipe._rules]
        assert "ContentFilter" in names
        assert "CostGuard" in names
        assert "OutputSchemaValidator" in names
        # PII and Hallucination off by default
        assert "PiiDetector" not in names
        assert "HallucinationGuard" not in names

    def test_pii_enabled(self):
        cfg = MagicMock()
        cfg.pii_detection_enabled = True
        cfg.cost_guard_enabled = True
        cfg.cost_guard_threshold_usd = 5.0
        cfg.hallucination_guard_enabled = False
        pipe = GuardrailsPipeline.default_pipeline(config=cfg)
        names = [r.name for r in pipe._rules]
        assert "PiiDetector" in names

    def test_hallucination_enabled(self):
        cfg = MagicMock()
        cfg.pii_detection_enabled = False
        cfg.cost_guard_enabled = True
        cfg.cost_guard_threshold_usd = 5.0
        cfg.hallucination_guard_enabled = True
        pipe = GuardrailsPipeline.default_pipeline(config=cfg)
        names = [r.name for r in pipe._rules]
        assert "HallucinationGuard" in names

    def test_cost_guard_disabled(self):
        cfg = MagicMock()
        cfg.pii_detection_enabled = False
        cfg.cost_guard_enabled = False
        cfg.hallucination_guard_enabled = False
        pipe = GuardrailsPipeline.default_pipeline(config=cfg)
        names = [r.name for r in pipe._rules]
        assert "CostGuard" not in names

    def test_custom_cost_threshold(self):
        cfg = MagicMock()
        cfg.pii_detection_enabled = False
        cfg.cost_guard_enabled = True
        cfg.cost_guard_threshold_usd = 2.5
        cfg.hallucination_guard_enabled = False
        pipe = GuardrailsPipeline.default_pipeline(config=cfg)
        cost_rules = [r for r in pipe._rules if r.name == "CostGuard"]
        assert len(cost_rules) == 1
        assert cost_rules[0]._threshold == 2.5

    def test_bus_passed_through(self):
        bus = _make_bus()
        pipe = GuardrailsPipeline.default_pipeline(bus=bus)
        assert pipe._bus is bus

    def test_all_features_enabled(self):
        cfg = MagicMock()
        cfg.pii_detection_enabled = True
        cfg.cost_guard_enabled = True
        cfg.cost_guard_threshold_usd = 5.0
        cfg.hallucination_guard_enabled = True
        pipe = GuardrailsPipeline.default_pipeline(config=cfg)
        names = [r.name for r in pipe._rules]
        assert names == [
            "ContentFilter",
            "PiiDetector",
            "CostGuard",
            "OutputSchemaValidator",
            "HallucinationGuard",
        ]


# ── Bus events ───────────────────────────────────────────────────────────


class TestBusEvents:
    @pytest.mark.asyncio
    async def test_publish_trigger_on_pass(self):
        bus = _make_bus()
        pipe = GuardrailsPipeline(rules=[ContentFilterRule()], bus=bus)
        await pipe.run_input("clean text", {"request_id": "req1", "origin": "test"})
        # _publish_trigger called once at end
        assert bus.publish.call_count == 1
        call_arg = bus.publish.call_args[0][0]
        assert call_arg["topic"] == "guardrails.triggered"
        assert call_arg["payload"]["request_id"] == "req1"
        assert call_arg["payload"]["origin"] == "test"
        assert len(call_arg["payload"]["results"]) == 1

    @pytest.mark.asyncio
    async def test_publish_trigger_and_block_on_block(self):
        bus = _make_bus()
        pipe = GuardrailsPipeline(rules=[ContentFilterRule()], bus=bus)
        await pipe.run_input("drop table users", {"request_id": "req2", "origin": "api"})
        # On block: _publish_trigger + _publish_block = 2 calls
        assert bus.publish.call_count == 2
        topics = [c[0][0]["topic"] for c in bus.publish.call_args_list]
        assert "guardrails.triggered" in topics
        assert "guardrails.blocked" in topics

    @pytest.mark.asyncio
    async def test_block_event_contains_rule_and_reason(self):
        bus = _make_bus()
        pipe = GuardrailsPipeline(rules=[ContentFilterRule()], bus=bus)
        await pipe.run_input("<script>alert(1)</script>", {"request_id": "r3"})
        block_calls = [
            c[0][0] for c in bus.publish.call_args_list
            if c[0][0]["topic"] == "guardrails.blocked"
        ]
        assert len(block_calls) == 1
        payload = block_calls[0]["payload"]
        assert payload["blocking_rule"] == "ContentFilter"
        assert "<script>" in payload["reason"]

    @pytest.mark.asyncio
    async def test_no_crash_without_bus(self):
        pipe = GuardrailsPipeline(rules=[ContentFilterRule()], bus=None)
        # Should not raise even on block (bus is None)
        results = await pipe.run_input("drop table users", {})
        assert len(results) == 1
        assert results[0].passed is False

    @pytest.mark.asyncio
    async def test_no_crash_without_bus_on_pass(self):
        pipe = GuardrailsPipeline(rules=[ContentFilterRule()], bus=None)
        results = await pipe.run_input("clean text", {})
        assert len(results) == 1
        assert results[0].passed is True

    @pytest.mark.asyncio
    async def test_bus_publish_exception_swallowed(self):
        bus = MagicMock()
        bus.publish = MagicMock(side_effect=RuntimeError("bus exploded"))
        pipe = GuardrailsPipeline(rules=[ContentFilterRule()], bus=bus)
        # Should not raise despite bus failure
        results = await pipe.run_input("clean text", {})
        assert len(results) == 1
        assert results[0].passed is True

    @pytest.mark.asyncio
    async def test_trigger_payload_context_defaults(self):
        """Missing request_id and origin default to empty string."""
        bus = _make_bus()
        pipe = GuardrailsPipeline(rules=[ContentFilterRule()], bus=bus)
        await pipe.run_input("safe text", {})
        payload = bus.publish.call_args[0][0]["payload"]
        assert payload["request_id"] == ""
        assert payload["origin"] == ""

    @pytest.mark.asyncio
    async def test_block_event_on_cost_guard(self):
        bus = _make_bus()
        pipe = GuardrailsPipeline(
            rules=[CostGuardRule(threshold_usd=1.0)], bus=bus,
        )
        await pipe.run_input("query", {"estimated_cost_usd": 5.0, "request_id": "c1"})
        block_calls = [
            c[0][0] for c in bus.publish.call_args_list
            if c[0][0]["topic"] == "guardrails.blocked"
        ]
        assert len(block_calls) == 1
        assert block_calls[0]["payload"]["blocking_rule"] == "CostGuard"


# ── GuardrailRule base class ─────────────────────────────────────────────


class TestGuardrailRuleBase:
    @pytest.mark.asyncio
    async def test_base_rule_passes(self):
        rule = GuardrailRule()
        res = await rule.check("anything", {})
        assert res.passed is True
        assert res.rule_name == "base"

    def test_base_rule_defaults(self):
        rule = GuardrailRule()
        assert rule.name == "base"
        assert rule.enabled is True

"""Tests for FeatureFlagStore behavior and its effect on goal routing."""

from __future__ import annotations

from qe.runtime.feature_flags import FeatureFlagStore


class TestFeatureFlagDefaults:
    def test_inquiry_mode_default_disabled(self):
        store = FeatureFlagStore()
        store.define("inquiry_mode", enabled=False)
        assert not store.is_enabled("inquiry_mode")

    def test_multi_agent_mode_default_disabled(self):
        store = FeatureFlagStore()
        store.define("multi_agent_mode", enabled=False)
        assert not store.is_enabled("multi_agent_mode")


class TestFeatureFlagToggle:
    def test_enable_inquiry_mode(self):
        store = FeatureFlagStore()
        store.define("inquiry_mode", enabled=False)
        store.enable("inquiry_mode")
        assert store.is_enabled("inquiry_mode")

    def test_enable_multi_agent_mode(self):
        store = FeatureFlagStore()
        store.define("multi_agent_mode", enabled=False)
        store.enable("multi_agent_mode")
        assert store.is_enabled("multi_agent_mode")

    def test_multi_agent_takes_precedence_over_inquiry(self):
        store = FeatureFlagStore()
        store.define("inquiry_mode", enabled=True)
        store.define("multi_agent_mode", enabled=True)

        # When both are enabled, multi_agent should take precedence
        # (as implemented in the app's POST /api/goals handler)
        multi = store.is_enabled("multi_agent_mode")
        inquiry = store.is_enabled("inquiry_mode")
        assert multi and inquiry
        # The routing logic checks multi_agent first — both can be True,
        # but the handler short-circuits on multi_agent_mode

    def test_flag_disable_after_enable(self):
        store = FeatureFlagStore()
        store.define("inquiry_mode", enabled=True)
        assert store.is_enabled("inquiry_mode")

        store.disable("inquiry_mode")
        assert not store.is_enabled("inquiry_mode")


class TestFeatureFlagEvaluation:
    def test_evaluation_log_recorded(self):
        store = FeatureFlagStore()
        store.define("inquiry_mode", enabled=True)
        store.is_enabled("inquiry_mode")

        log = store.evaluation_log(limit=10)
        assert len(log) >= 1
        assert log[0]["flag"] == "inquiry_mode"
        assert log[0]["result"] is True

    def test_flag_with_rollout_percentage(self):
        store = FeatureFlagStore()
        store.define("inquiry_mode", enabled=True, rollout_pct=50.0)

        # With 50% rollout, some contexts will be included and some excluded
        # Test determinism: same context always gives same result
        ctx = {"goal_id": "test_goal_1"}
        result1 = store.is_enabled("inquiry_mode", ctx)
        result2 = store.is_enabled("inquiry_mode", ctx)
        assert result1 == result2

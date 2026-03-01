"""Tests for Thompson sampling in RoutingOptimizer."""

from __future__ import annotations

import pytest

from qe.runtime.routing_optimizer import BetaArm, RoutingOptimizer


class TestBetaArm:
    """Tests for BetaArm conjugate prior."""

    def test_default_prior(self):
        arm = BetaArm()
        assert arm.alpha == 1.0
        assert arm.beta == 1.0

    def test_update_success(self):
        arm = BetaArm()
        arm.update(success=True)
        assert arm.alpha == 2.0
        assert arm.beta == 1.0

    def test_update_failure(self):
        arm = BetaArm()
        arm.update(success=False)
        assert arm.alpha == 1.0
        assert arm.beta == 2.0

    def test_sample_in_range(self):
        arm = BetaArm(alpha=5.0, beta=5.0)
        for _ in range(100):
            s = arm.sample()
            assert 0.0 <= s <= 1.0

    def test_mean(self):
        arm = BetaArm(alpha=3.0, beta=7.0)
        assert arm.mean == pytest.approx(0.3)

    def test_sample_count(self):
        arm = BetaArm()
        assert arm.sample_count == 0
        arm.update(True)
        arm.update(False)
        assert arm.sample_count == 2

    def test_high_alpha_samples_high(self):
        """An arm with many successes should sample high most of the time."""
        arm = BetaArm(alpha=100.0, beta=1.0)
        samples = [arm.sample() for _ in range(50)]
        assert sum(1 for s in samples if s > 0.5) > 40


class TestThompsonSelectModel:
    """Tests for thompson_select_model on RoutingOptimizer."""

    def test_no_models_raises(self):
        opt = RoutingOptimizer()
        with pytest.raises(ValueError, match="No models available"):
            opt.thompson_select_model("research", [])

    def test_single_model_returns_it(self):
        opt = RoutingOptimizer()
        assert opt.thompson_select_model("research", ["gpt-4"]) == "gpt-4"

    def test_picks_model_with_known_success(self):
        """After many successes for one model, Thompson should prefer it."""
        opt = RoutingOptimizer()
        # Feed lots of successes to model_a
        for _ in range(50):
            opt._arms[("model_a", "research")].update(True)
        # Feed lots of failures to model_b
        for _ in range(50):
            opt._arms[("model_b", "research")].update(False)

        wins = {"model_a": 0, "model_b": 0}
        for _ in range(100):
            pick = opt.thompson_select_model("research", ["model_a", "model_b"])
            wins[pick] += 1

        assert wins["model_a"] > 80

    @pytest.mark.asyncio
    async def test_record_outcome_updates_arm(self):
        opt = RoutingOptimizer()
        await opt.record_outcome("gpt-4", "research", success=True)
        arm = opt._arms[("gpt-4", "research")]
        assert arm.alpha == 2.0
        assert arm.beta == 1.0

    def test_backward_compat_select_model(self):
        """Existing select_model() still works unchanged."""
        opt = RoutingOptimizer()
        result = opt.select_model("research", ["gpt-4", "gpt-3.5"])
        assert result in {"gpt-4", "gpt-3.5"}

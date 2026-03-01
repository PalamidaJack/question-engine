"""Tests for PromptMutator — LLM-powered prompt variant generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.optimization.prompt_mutator import (
    MUTATION_STRATEGIES,
    MutatedPrompt,
    PromptMutator,
    _build_mutation_user_prompt,
    _extract_format_keys,
    _validate_mutation,
)
from qe.optimization.prompt_registry import PromptRegistry

# ── Test Helpers ──────────────────────────────────────────────────────────


def _make_registry(enabled: bool = True) -> PromptRegistry:
    """Create an in-memory PromptRegistry with a test baseline."""
    reg = PromptRegistry(bus=MagicMock(), enabled=enabled)
    reg.register_baseline("test.slot.system", "You are a {role}. Do {task}.")
    return reg


def _mock_instructor_result(content: str = "You are a {role}. Execute {task}."):
    """Create a mock instructor client that returns a MutatedPrompt."""
    mock_client = MagicMock()
    mock_completions = MagicMock()
    mock_completions.create = AsyncMock(
        return_value=MutatedPrompt(
            mutated_content=content,
            mutation_rationale="Rephrased for clarity",
            preserved_format_keys=["role", "task"],
        )
    )
    mock_client.chat.completions = mock_completions
    return mock_client


# ── TestMutationStrategies ────────────────────────────────────────────────


class TestMutationStrategies:
    def test_four_strategies_defined(self):
        assert len(MUTATION_STRATEGIES) == 4
        assert set(MUTATION_STRATEGIES.keys()) == {
            "rephrase",
            "elaborate",
            "simplify",
            "restructure",
        }

    def test_rephrase_instruction_content(self):
        assert "Rephrase" in MUTATION_STRATEGIES["rephrase"]

    def test_elaborate_instruction_content(self):
        assert "Elaborate" in MUTATION_STRATEGIES["elaborate"]

    def test_simplify_instruction_content(self):
        assert "Simplify" in MUTATION_STRATEGIES["simplify"]

    def test_restructure_instruction_content(self):
        assert "Restructure" in MUTATION_STRATEGIES["restructure"]

    def test_all_strategies_mention_format_keys(self):
        for name, instruction in MUTATION_STRATEGIES.items():
            assert "{format_keys}" in instruction, f"{name} missing format_keys"


# ── TestMutatedPromptModel ────────────────────────────────────────────────


class TestMutatedPromptModel:
    def test_fields(self):
        mp = MutatedPrompt(
            mutated_content="hello {name}",
            mutation_rationale="simplified",
            preserved_format_keys=["name"],
        )
        assert mp.mutated_content == "hello {name}"
        assert mp.mutation_rationale == "simplified"
        assert mp.preserved_format_keys == ["name"]

    def test_validation_requires_fields(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MutatedPrompt()


# ── TestFormatKeyExtraction ───────────────────────────────────────────────


class TestFormatKeyExtraction:
    def test_extract_keys(self):
        keys = _extract_format_keys("Hello {name}, you are {role}.")
        assert keys == {"name", "role"}

    def test_no_keys(self):
        keys = _extract_format_keys("Hello world, no placeholders.")
        assert keys == set()

    def test_duplicate_keys(self):
        keys = _extract_format_keys("{a} {b} {a}")
        assert keys == {"a", "b"}

    def test_validate_success(self):
        assert _validate_mutation(
            "Do {task} as {role}",
            "Execute {task} in the capacity of {role}",
        )

    def test_validate_failure_missing_key(self):
        assert not _validate_mutation(
            "Do {task} as {role}",
            "Execute {task} completely",
        )

    def test_validate_extra_keys_ok(self):
        assert _validate_mutation(
            "Do {task}",
            "Do {task} with {extra}",
        )

    def test_build_user_prompt(self):
        prompt = _build_mutation_user_prompt(
            original="Hello {name}",
            strategy="rephrase",
            strategy_instruction="Rephrase it",
            slot_key="test.slot",
        )
        assert "test.slot" in prompt
        assert "rephrase" in prompt
        assert "Hello {name}" in prompt


# ── TestPromptMutatorInit ─────────────────────────────────────────────────


class TestPromptMutatorInit:
    def test_defaults(self):
        reg = _make_registry()
        m = PromptMutator(registry=reg)
        assert m._eval_interval == 300.0
        assert m._min_samples == 20
        assert m._rollback_threshold == 0.3
        assert m._promote_threshold == 0.6
        assert m._max_variants_per_slot == 5
        assert m._max_mutations_per_cycle == 3
        assert m._initial_rollout_pct == 10.0
        assert m._promoted_rollout_pct == 50.0

    def test_custom_params(self):
        reg = _make_registry()
        m = PromptMutator(
            registry=reg,
            eval_interval=60.0,
            min_samples=5,
            rollback_threshold=0.2,
            promote_threshold=0.7,
            max_variants_per_slot=3,
            max_mutations_per_cycle=1,
        )
        assert m._eval_interval == 60.0
        assert m._min_samples == 5
        assert m._rollback_threshold == 0.2
        assert m._promote_threshold == 0.7

    def test_status_before_start(self):
        reg = _make_registry()
        m = PromptMutator(registry=reg)
        s = m.status()
        assert s["running"] is False
        assert s["mutations_total"] == 0
        assert s["rollbacks_total"] == 0
        assert s["promotions_total"] == 0
        assert s["last_cycle_at"] is None


# ── TestMutatorLifecycle ──────────────────────────────────────────────────


class TestMutatorLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        reg = _make_registry()
        m = PromptMutator(registry=reg, eval_interval=1000.0)
        m.start()
        assert m._running is True
        assert m._loop_task is not None
        await m.stop()
        assert m._running is False
        assert m._loop_task is None

    @pytest.mark.asyncio
    async def test_double_start_noop(self):
        reg = _make_registry()
        m = PromptMutator(registry=reg, eval_interval=1000.0)
        m.start()
        first_task = m._loop_task
        m.start()
        assert m._loop_task is first_task
        await m.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        reg = _make_registry()
        m = PromptMutator(registry=reg)
        await m.stop()  # should not raise
        assert m._running is False


# ── TestEvaluationLogic ───────────────────────────────────────────────────


class TestEvaluationLogic:
    @pytest.mark.asyncio
    async def test_rollback_low_performer(self):
        reg = _make_registry()
        variant = reg.add_variant("test.slot.system", "Bad prompt {role} {task}.")
        # Simulate low performance: many failures
        arm = reg._arms[variant.variant_id]
        arm.alpha = 3.0  # ~3 successes
        arm.beta = 20.0  # ~18 failures → mean ≈ 0.13

        m = PromptMutator(registry=reg, min_samples=5, max_mutations_per_cycle=0)
        with patch("qe.optimization.prompt_mutator.get_flag_store") as mock_flags:
            mock_flags.return_value.is_enabled.return_value = True
            await m._evaluate()

        assert variant.active is False
        assert m._rollbacks_total == 1

    @pytest.mark.asyncio
    async def test_no_rollback_insufficient_samples(self):
        reg = _make_registry()
        variant = reg.add_variant("test.slot.system", "Low sample {role} {task}.")
        arm = reg._arms[variant.variant_id]
        arm.alpha = 1.5  # mean ≈ 0.23 (low)
        arm.beta = 5.0

        m = PromptMutator(registry=reg, min_samples=50, max_mutations_per_cycle=0)
        with patch("qe.optimization.prompt_mutator.get_flag_store") as mock_flags:
            mock_flags.return_value.is_enabled.return_value = True
            await m._evaluate()

        assert variant.active is True
        assert m._rollbacks_total == 0

    @pytest.mark.asyncio
    async def test_promote_high_performer(self):
        reg = _make_registry()
        variant = reg.add_variant(
            "test.slot.system", "Great prompt {role} {task}.", rollout_pct=10.0
        )
        arm = reg._arms[variant.variant_id]
        arm.alpha = 20.0  # ~19 successes
        arm.beta = 3.0  # ~2 failures → mean ≈ 0.87

        m = PromptMutator(
            registry=reg,
            bus=MagicMock(),
            min_samples=5,
            max_mutations_per_cycle=0,
        )
        with patch("qe.optimization.prompt_mutator.get_flag_store") as mock_flags:
            mock_flags.return_value.is_enabled.return_value = True
            await m._evaluate()

        assert variant.rollout_pct == 50.0
        assert m._promotions_total == 1

    @pytest.mark.asyncio
    async def test_no_promote_already_promoted(self):
        reg = _make_registry()
        variant = reg.add_variant(
            "test.slot.system", "Good prompt {role} {task}.", rollout_pct=50.0
        )
        arm = reg._arms[variant.variant_id]
        arm.alpha = 20.0
        arm.beta = 3.0

        m = PromptMutator(registry=reg, min_samples=5, max_mutations_per_cycle=0)
        with patch("qe.optimization.prompt_mutator.get_flag_store") as mock_flags:
            mock_flags.return_value.is_enabled.return_value = True
            await m._evaluate()

        assert variant.rollout_pct == 50.0
        assert m._promotions_total == 0

    @pytest.mark.asyncio
    async def test_feature_flag_gates(self):
        reg = _make_registry()
        m = PromptMutator(registry=reg)
        with patch("qe.optimization.prompt_mutator.get_flag_store") as mock_flags:
            mock_flags.return_value.is_enabled.return_value = False
            await m._evaluate()

        assert m._mutations_total == 0
        assert m._rollbacks_total == 0

    @pytest.mark.asyncio
    async def test_max_variants_respected(self):
        reg = _make_registry()
        # Add variants up to max (5 = 1 baseline + 4 added)
        for i in range(4):
            reg.add_variant(
                "test.slot.system", f"Variant {i} {{role}} {{task}}."
            )

        m = PromptMutator(registry=reg, max_variants_per_slot=5)
        with patch("qe.optimization.prompt_mutator.get_flag_store") as mock_flags:
            mock_flags.return_value.is_enabled.return_value = True
            await m._evaluate()

        # No mutations should happen — slot is full
        assert m._mutations_total == 0

    @pytest.mark.asyncio
    async def test_max_mutations_per_cycle_respected(self):
        reg = PromptRegistry(bus=MagicMock(), enabled=True)
        # Create 3 slots with room for mutations
        for i in range(5):
            reg.register_baseline(f"slot{i}.system", f"Baseline {i} {{name}}.")

        mock_client = _mock_instructor_result("Mutated {name}.")

        m = PromptMutator(
            registry=reg,
            max_mutations_per_cycle=2,
            max_variants_per_slot=5,
        )
        with (
            patch("qe.optimization.prompt_mutator.get_flag_store") as mock_flags,
            patch("qe.optimization.prompt_mutator.instructor") as mock_inst,
        ):
            mock_flags.return_value.is_enabled.return_value = True
            mock_inst.from_litellm.return_value = mock_client
            await m._evaluate()

        assert m._mutations_total <= 2


# ── TestMutateVariant ─────────────────────────────────────────────────────


class TestMutateVariant:
    @pytest.mark.asyncio
    async def test_calls_llm_and_adds_to_registry(self):
        reg = _make_registry()
        baseline = reg._variants["test.slot.system"][0]
        mock_client = _mock_instructor_result("You are a {role}. Execute {task}.")

        m = PromptMutator(registry=reg)
        with patch("qe.optimization.prompt_mutator.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await m._mutate_variant(
                "test.slot.system", baseline, "rephrase"
            )

        assert result is not None
        assert result.content == "You are a {role}. Execute {task}."
        # Should be added to registry
        assert len(reg._variants["test.slot.system"]) == 2

    @pytest.mark.asyncio
    async def test_passes_strategy(self):
        reg = _make_registry()
        baseline = reg._variants["test.slot.system"][0]
        mock_client = _mock_instructor_result("You are a {role}. Execute {task}.")

        m = PromptMutator(registry=reg)
        with patch("qe.optimization.prompt_mutator.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            await m._mutate_variant(
                "test.slot.system", baseline, "elaborate"
            )

        # Verify the user message contains the strategy
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "elaborate" in user_msg

    @pytest.mark.asyncio
    async def test_validates_format_keys(self):
        reg = _make_registry()
        baseline = reg._variants["test.slot.system"][0]
        # Return mutation with missing keys
        mock_client = _mock_instructor_result("Bad mutation with no placeholders")

        m = PromptMutator(registry=reg)
        with patch("qe.optimization.prompt_mutator.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await m._mutate_variant(
                "test.slot.system", baseline, "rephrase"
            )

        assert result is None
        # Should not be added to registry
        assert len(reg._variants["test.slot.system"]) == 1

    @pytest.mark.asyncio
    async def test_handles_llm_error(self):
        reg = _make_registry()
        baseline = reg._variants["test.slot.system"][0]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("LLM failed")
        )

        m = PromptMutator(registry=reg)
        with patch("qe.optimization.prompt_mutator.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await m._mutate_variant(
                "test.slot.system", baseline, "simplify"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_strategy_passed_to_add_variant(self):
        reg = _make_registry()
        reg._bus = MagicMock()
        baseline = reg._variants["test.slot.system"][0]
        mock_client = _mock_instructor_result("You are a {role}. Execute {task}.")

        m = PromptMutator(registry=reg)
        with patch("qe.optimization.prompt_mutator.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            await m._mutate_variant(
                "test.slot.system", baseline, "restructure"
            )

        # Check bus event had the right strategy
        calls = reg._bus.publish_sync.call_args_list
        created_call = [c for c in calls if c.args[0] == "prompt.variant_created"]
        assert len(created_call) == 1
        assert created_call[0].args[1]["strategy"] == "restructure"


# ── TestMutatorBusEvents ──────────────────────────────────────────────────


class TestMutatorBusEvents:
    @pytest.mark.asyncio
    async def test_cycle_completed_published(self):
        reg = _make_registry()
        bus = MagicMock()

        m = PromptMutator(registry=reg, bus=bus, max_mutations_per_cycle=0)
        with patch("qe.optimization.prompt_mutator.get_flag_store") as mock_flags:
            mock_flags.return_value.is_enabled.return_value = True
            await m._evaluate()

        # Should publish cycle completed
        calls = bus.publish_sync.call_args_list
        cycle_calls = [
            c for c in calls if c.args[0] == "prompt.mutation_cycle_completed"
        ]
        assert len(cycle_calls) == 1
        payload = cycle_calls[0].args[1]
        assert payload["slots_evaluated"] >= 1

    @pytest.mark.asyncio
    async def test_variant_promoted_published(self):
        reg = _make_registry()
        variant = reg.add_variant(
            "test.slot.system", "Great prompt {role} {task}.", rollout_pct=10.0
        )
        arm = reg._arms[variant.variant_id]
        arm.alpha = 20.0
        arm.beta = 3.0

        bus = MagicMock()
        m = PromptMutator(
            registry=reg, bus=bus, min_samples=5, max_mutations_per_cycle=0
        )
        with patch("qe.optimization.prompt_mutator.get_flag_store") as mock_flags:
            mock_flags.return_value.is_enabled.return_value = True
            await m._evaluate()

        calls = bus.publish_sync.call_args_list
        promoted_calls = [
            c for c in calls if c.args[0] == "prompt.variant_promoted"
        ]
        assert len(promoted_calls) == 1
        payload = promoted_calls[0].args[1]
        assert payload["slot_key"] == "test.slot.system"
        assert payload["old_rollout_pct"] == 10.0
        assert payload["new_rollout_pct"] == 50.0


# ── TestMutatorIntegration ────────────────────────────────────────────────


class TestMutatorIntegration:
    @pytest.mark.asyncio
    async def test_full_cycle_with_real_registry_and_mock_llm(self):
        """Full cycle: register baseline, add variant, simulate outcomes,
        run evaluate, verify rollback/promotion/mutation."""
        reg = PromptRegistry(bus=MagicMock(), enabled=True)
        reg.register_baseline("test.slot", "Analyze {topic} for {audience}.")

        # Add a high performer
        good = reg.add_variant(
            "test.slot", "Carefully analyze {topic} for {audience}.", rollout_pct=10.0
        )
        arm_good = reg._arms[good.variant_id]
        arm_good.alpha = 25.0
        arm_good.beta = 3.0

        # Add a low performer
        bad = reg.add_variant(
            "test.slot", "Do {topic} for {audience}.", rollout_pct=10.0
        )
        arm_bad = reg._arms[bad.variant_id]
        arm_bad.alpha = 2.0
        arm_bad.beta = 22.0

        mock_client = _mock_instructor_result(
            "Deeply analyze {topic} tailored for {audience}."
        )

        bus = MagicMock()
        m = PromptMutator(
            registry=reg,
            bus=bus,
            min_samples=5,
            max_mutations_per_cycle=1,
            max_variants_per_slot=10,
        )

        with (
            patch("qe.optimization.prompt_mutator.get_flag_store") as mock_flags,
            patch("qe.optimization.prompt_mutator.instructor") as mock_inst,
        ):
            mock_flags.return_value.is_enabled.return_value = True
            mock_inst.from_litellm.return_value = mock_client
            await m._evaluate()

        # Bad variant should be rolled back
        assert bad.active is False

        # Good variant should be promoted
        assert good.rollout_pct == 50.0

        # A new mutation should have been created
        assert m._mutations_total == 1
        assert m._rollbacks_total == 1
        assert m._promotions_total == 1

        # Verify cycle event published
        calls = bus.publish_sync.call_args_list
        cycle_calls = [
            c for c in calls if c.args[0] == "prompt.mutation_cycle_completed"
        ]
        assert len(cycle_calls) == 1

    @pytest.mark.asyncio
    async def test_strategy_rotation(self):
        """Verify strategies rotate across mutations."""
        reg = _make_registry()
        m = PromptMutator(registry=reg)

        strategies_seen = []
        for _ in range(8):
            strategies_seen.append(m._next_strategy())

        # Should cycle through all 4 strategies twice
        assert strategies_seen[:4] == ["rephrase", "elaborate", "simplify", "restructure"]
        assert strategies_seen[4:8] == ["rephrase", "elaborate", "simplify", "restructure"]

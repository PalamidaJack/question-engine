"""Tests for PromptMutator wiring — bus topics, schemas, registry extensions, API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

from qe.bus.protocol import TOPICS
from qe.bus.schemas import (
    TOPIC_SCHEMAS,
    PromptMutationCyclePayload,
    PromptVariantPromotedPayload,
    validate_payload,
)
from qe.optimization.prompt_registry import PromptRegistry

# ── Bus Topics ────────────────────────────────────────────────────────────


class TestNewBusTopics:
    def test_mutation_cycle_completed_in_topics(self):
        assert "prompt.mutation_cycle_completed" in TOPICS

    def test_variant_promoted_in_topics(self):
        assert "prompt.variant_promoted" in TOPICS


# ── Bus Schemas ───────────────────────────────────────────────────────────


class TestNewBusSchemas:
    def test_mutation_cycle_schema_registered(self):
        assert "prompt.mutation_cycle_completed" in TOPIC_SCHEMAS

    def test_variant_promoted_schema_registered(self):
        assert "prompt.variant_promoted" in TOPIC_SCHEMAS

    def test_mutation_cycle_payload_validates(self):
        payload = validate_payload("prompt.mutation_cycle_completed", {
            "slots_evaluated": 10,
            "variants_created": 2,
            "variants_rolled_back": 1,
            "variants_promoted": 1,
        })
        assert isinstance(payload, PromptMutationCyclePayload)
        assert payload.slots_evaluated == 10
        assert payload.variants_created == 2

    def test_variant_promoted_payload_validates(self):
        payload = validate_payload("prompt.variant_promoted", {
            "slot_key": "dialectic.challenge.user",
            "variant_id": "v1",
            "old_rollout_pct": 10.0,
            "new_rollout_pct": 50.0,
        })
        assert isinstance(payload, PromptVariantPromotedPayload)
        assert payload.slot_key == "dialectic.challenge.user"
        assert payload.new_rollout_pct == 50.0

    def test_mutation_cycle_defaults(self):
        p = PromptMutationCyclePayload()
        assert p.slots_evaluated == 0
        assert p.variants_created == 0
        assert p.variants_rolled_back == 0
        assert p.variants_promoted == 0

    def test_variant_promoted_defaults(self):
        p = PromptVariantPromotedPayload(
            slot_key="test", variant_id="v1"
        )
        assert p.old_rollout_pct == 10.0
        assert p.new_rollout_pct == 50.0


# ── Registry Extensions ──────────────────────────────────────────────────


class TestRegistryExtensions:
    def test_add_variant_strategy_param_passes_through(self):
        bus = MagicMock()
        reg = PromptRegistry(bus=bus, enabled=True)
        reg.register_baseline("test.slot", "Hello {name}.")
        reg.add_variant("test.slot", "Hi {name}.", strategy="rephrase")

        calls = bus.publish_sync.call_args_list
        created_calls = [
            c for c in calls if c.args[0] == "prompt.variant_created"
        ]
        assert len(created_calls) == 1
        assert created_calls[0].args[1]["strategy"] == "rephrase"

    def test_add_variant_default_strategy_is_manual(self):
        bus = MagicMock()
        reg = PromptRegistry(bus=bus, enabled=True)
        reg.register_baseline("test.slot", "Hello {name}.")
        reg.add_variant("test.slot", "Hi {name}.")

        calls = bus.publish_sync.call_args_list
        created_calls = [
            c for c in calls if c.args[0] == "prompt.variant_created"
        ]
        assert created_calls[0].args[1]["strategy"] == "manual"

    def test_promote_variant_works(self):
        reg = PromptRegistry(enabled=True)
        reg.register_baseline("test.slot", "Hello {name}.")
        variant = reg.add_variant("test.slot", "Hi {name}.", rollout_pct=10.0)

        result = reg.promote_variant(variant.variant_id, 50.0)
        assert result is True
        assert variant.rollout_pct == 50.0

    def test_promote_variant_not_found(self):
        reg = PromptRegistry(enabled=True)
        result = reg.promote_variant("nonexistent_id", 50.0)
        assert result is False

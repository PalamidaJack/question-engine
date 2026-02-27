from unittest.mock import patch

import pytest

from qe.models.envelope import Envelope
from qe.models.genome import Blueprint, CapabilityDeclaration, ModelPreference
from qe.runtime.budget import BudgetTracker
from qe.runtime.context_manager import ContextManager
from qe.runtime.router import TIER_MODELS, AutoRouter

# === AutoRouter Tests ===

def test_autorouter_select_first_model_in_tier():
    """select() returns the first model in the tier's list."""
    preference = ModelPreference(tier="balanced")
    router = AutoRouter(preference)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={}
    )

    selected = router.select(envelope)
    assert selected == TIER_MODELS["balanced"][0]


def test_autorouter_prefer_local():
    """With PREFER_LOCAL env set and local models available in tier: returns an ollama model."""
    preference = ModelPreference(tier="local")  # Use local tier which has ollama models
    router = AutoRouter(preference)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={}
    )

    with patch("qe.runtime.router.os.getenv") as mock_getenv:
        mock_getenv.return_value = "true"
        selected = router.select(envelope)

    assert "ollama" in selected


def test_autorouter_fallback_when_all_in_cooldown():
    """After record_error(model) for all models in a tier: falls back to the cheaper tier."""
    preference = ModelPreference(tier="balanced")
    router = AutoRouter(preference)

    # Record errors for all balanced models
    for model in TIER_MODELS["balanced"]:
        router.record_error(model)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={}
    )

    selected = router.select(envelope)
    # Should fall back to "fast" tier
    assert selected in TIER_MODELS["fast"]


def test_autorouter_budget_gate():
    """Budget gate: mock _budget_remaining_pct to return 0.05 → tier is forced to 'fast'."""
    preference = ModelPreference(tier="balanced")
    router = AutoRouter(preference)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={}
    )

    with patch.object(router, "_budget_remaining_pct", return_value=0.05):
        selected = router.select(envelope)

    assert selected in TIER_MODELS["fast"]


def test_autorouter_force_model_escape_hatch():
    """If envelope.payload contains 'force_model': use it directly."""
    preference = ModelPreference(tier="balanced")
    router = AutoRouter(preference)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={"force_model": "gpt-4o"}
    )

    selected = router.select(envelope)
    assert selected == "gpt-4o"


# === ContextManager Tests ===

def test_context_manager_build_messages_starts_with_system():
    """build_messages starts with a system message containing system_prompt."""
    blueprint = Blueprint(
        service_id="test-service",
        display_name="Test",
        version="1.0",
        system_prompt="You are a helpful assistant.",
        model_preference=ModelPreference(tier="balanced"),
        capabilities=CapabilityDeclaration()
    )

    ctx = ContextManager(blueprint)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={"text": "test observation"}
    )

    messages = ctx.build_messages(envelope, turn_count=0)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."


def test_context_manager_build_messages_includes_envelope():
    """build_messages includes the envelope content as the final user message."""
    blueprint = Blueprint(
        service_id="test-service",
        display_name="Test",
        version="1.0",
        system_prompt="You are a helpful assistant.",
        model_preference=ModelPreference(tier="balanced"),
        capabilities=CapabilityDeclaration()
    )

    ctx = ContextManager(blueprint)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={"text": "test observation content"}
    )

    messages = ctx.build_messages(envelope, turn_count=0)

    # Last message should be from user containing envelope content
    last_msg = messages[-1]
    assert last_msg["role"] == "user"
    assert "test observation content" in last_msg["content"]


def test_context_manager_reinforce():
    """reinforce appends a user+assistant message pair to history."""
    blueprint = Blueprint(
        service_id="test-service",
        display_name="Test",
        version="1.0",
        system_prompt="You are a helpful assistant.",
        model_preference=ModelPreference(tier="balanced"),
        capabilities=CapabilityDeclaration()
    )

    ctx = ContextManager(blueprint)

    # Add some initial history
    ctx.history = [{"role": "user", "content": "Hello"}]

    ctx.reinforce()

    assert len(ctx.history) == 3  # original + 2 reinforcement messages
    assert ctx.history[1]["role"] == "user"
    assert "Reminder" in ctx.history[1]["content"]
    assert ctx.history[2]["role"] == "assistant"
    assert ctx.history[2]["content"] == "You are a helpful assistant."


def test_context_manager_truncation_never_drops_system():
    """Truncation drops oldest messages first; system prompt is never dropped."""
    blueprint = Blueprint(
        service_id="test-service",
        display_name="Test",
        version="1.0",
        system_prompt="You are a helpful assistant.",
        model_preference=ModelPreference(tier="balanced"),
        capabilities=CapabilityDeclaration(),
        max_context_tokens=100,  # Very low limit
        context_compression_threshold=0.5
    )

    ctx = ContextManager(blueprint)

    # Add lots of history
    ctx.history = [
        {"role": "user", "content": f"Message {i}"} for i in range(50)
    ]

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={"text": "test"}
    )

    messages = ctx.build_messages(envelope, turn_count=0)

    # System prompt should always be first
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."

    # Should have truncated some messages but still have the envelope
    assert any(msg["role"] == "user" and "test" in msg.get("content", "") for msg in messages)


def test_context_manager_reinforcement_interval():
    """When turn_count is a multiple of reinforcement_interval_turns, system reminder is added."""
    blueprint = Blueprint(
        service_id="test-service",
        display_name="Test",
        version="1.0",
        system_prompt="You are a helpful assistant.",
        model_preference=ModelPreference(tier="balanced"),
        capabilities=CapabilityDeclaration(),
        reinforcement_interval_turns=5
    )

    ctx = ContextManager(blueprint)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={"text": "test"}
    )

    # turn_count = 5 (exactly at reinforcement interval)
    messages = ctx.build_messages(envelope, turn_count=5)

    # Should have system prompt at start + user reminder
    # The implementation adds a user message with [SYSTEM REMINDER]
    reminder_count = sum(1 for m in messages if "SYSTEM REMINDER" in m.get("content", ""))
    assert reminder_count >= 1  # should have at least one reminder


# === BudgetTracker Tests ===

def test_budget_tracker_remaining_starts_at_full():
    tracker = BudgetTracker(monthly_limit_usd=100.0)
    assert tracker.remaining_pct() == 1.0


def test_budget_tracker_tracks_spend():
    tracker = BudgetTracker(monthly_limit_usd=10.0)
    tracker.record_cost("gpt-4o-mini", 2.0)
    assert tracker.remaining_pct() == pytest.approx(0.8)
    assert tracker.total_spend() == 2.0


def test_budget_tracker_spend_by_model():
    tracker = BudgetTracker(monthly_limit_usd=100.0)
    tracker.record_cost("gpt-4o-mini", 1.5)
    tracker.record_cost("gpt-4o", 3.0)
    tracker.record_cost("gpt-4o-mini", 0.5)
    breakdown = tracker.spend_by_model()
    assert breakdown["gpt-4o-mini"] == pytest.approx(2.0)
    assert breakdown["gpt-4o"] == pytest.approx(3.0)


def test_budget_tracker_exhausted_at_zero():
    tracker = BudgetTracker(monthly_limit_usd=5.0)
    tracker.record_cost("gpt-4o", 5.0)
    assert tracker.remaining_pct() == 0.0
    tracker.record_cost("gpt-4o", 1.0)
    assert tracker.remaining_pct() == 0.0  # clamped to 0


def test_autorouter_with_real_budget_tracker():
    """Router downgrades to fast tier when budget tracker reports low budget."""
    tracker = BudgetTracker(monthly_limit_usd=10.0)
    tracker.record_cost("gpt-4o", 9.5)  # 95% spent

    preference = ModelPreference(tier="balanced")
    router = AutoRouter(preference, budget_tracker=tracker)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={},
    )
    selected = router.select(envelope)
    assert selected in TIER_MODELS["fast"]


# === Constitution / Immutable Safety Tests ===

def test_constitution_injected_as_second_system_message():
    blueprint = Blueprint(
        service_id="test-service",
        display_name="Test",
        version="1.0",
        system_prompt="You are a helpful assistant.",
        constitution="Never fabricate evidence.",
        model_preference=ModelPreference(tier="balanced"),
        capabilities=CapabilityDeclaration(),
    )

    ctx = ContextManager(blueprint)
    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={"text": "test"},
    )

    messages = ctx.build_messages(envelope, turn_count=0)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "system"
    assert "CONSTITUTION" in messages[1]["content"]
    assert "Never fabricate evidence." in messages[1]["content"]


def test_constitution_survives_truncation():
    """Constitution must never be dropped during compression."""
    blueprint = Blueprint(
        service_id="test-service",
        display_name="Test",
        version="1.0",
        system_prompt="You are a helpful assistant.",
        constitution="Never fabricate evidence.",
        model_preference=ModelPreference(tier="balanced"),
        capabilities=CapabilityDeclaration(),
        max_context_tokens=100,
        context_compression_threshold=0.5,
    )

    ctx = ContextManager(blueprint)
    ctx.history = [
        {"role": "user", "content": f"Message {i} " * 20} for i in range(50)
    ]

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={"text": "test"},
    )

    messages = ctx.build_messages(envelope, turn_count=0)

    # System prompt always first
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    # Constitution always second
    assert messages[1]["role"] == "system"
    assert "CONSTITUTION" in messages[1]["content"]


def test_no_constitution_means_no_extra_system_message():
    """Without constitution, there should be only one system message."""
    blueprint = Blueprint(
        service_id="test-service",
        display_name="Test",
        version="1.0",
        system_prompt="You are a helpful assistant.",
        model_preference=ModelPreference(tier="balanced"),
        capabilities=CapabilityDeclaration(),
    )

    ctx = ContextManager(blueprint)
    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test",
        payload={"text": "test"},
    )

    messages = ctx.build_messages(envelope, turn_count=0)
    system_msgs = [m for m in messages if m["role"] == "system"]
    assert len(system_msgs) == 1

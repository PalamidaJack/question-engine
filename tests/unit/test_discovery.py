"""Tests for the model discovery system."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.envelope import Envelope
from qe.models.genome import ModelPreference
from qe.runtime.discovery.providers import (
    _infer_capabilities,
    _infer_quality_tier,
)
from qe.runtime.discovery.schemas import (
    DiscoveredModel,
    ModelHealthMetrics,
    TierAssignment,
)
from qe.runtime.discovery.service import ModelDiscoveryService

# ── Schema tests ──────────────────────────────────────────────────────────


def test_discovered_model_defaults():
    m = DiscoveredModel(
        model_id="groq/llama-3.3-70b",
        provider="groq",
        base_model_name="llama-3.3-70b",
    )
    assert m.is_free is True
    assert m.status == "active"
    assert m.context_length == 4096
    assert m.cost_per_m_input == 0.0


def test_model_health_defaults():
    h = ModelHealthMetrics(model_id="test/model")
    assert h.total_calls == 0
    assert h.avg_latency_ms == 0.0


def test_tier_assignment():
    t = TierAssignment(
        tier="fast",
        primary="groq/llama-3.3-70b",
        fallbacks=["cerebras/llama-3.3-70b"],
        reason="auto",
    )
    assert t.auto_assigned is True
    assert len(t.fallbacks) == 1


# ── Tier inference tests ──────────────────────────────────────────────────


def test_infer_tier_powerful_70b():
    assert _infer_quality_tier("llama-3.3-70b-versatile") == "powerful"


def test_infer_tier_powerful_opus():
    assert _infer_quality_tier("claude-opus-4") == "powerful"


def test_infer_tier_powerful_pro():
    assert _infer_quality_tier("gemini-2.5-pro-preview") == "powerful"


def test_infer_tier_fast_8b():
    assert _infer_quality_tier("llama-3.1-8b-instant") == "fast"


def test_infer_tier_fast_mini():
    assert _infer_quality_tier("gpt-4o-mini") == "fast"


def test_infer_tier_fast_flash():
    assert _infer_quality_tier("gemini-2.0-flash") == "fast"


def test_infer_tier_balanced_default():
    assert _infer_quality_tier("some-unknown-model") == "balanced"


def test_infer_tier_context_bump():
    # 8b is normally fast, but > 100k context bumps it up
    assert _infer_quality_tier("llama-3.1-8b", context_length=128_000) == "balanced"


def test_infer_tier_balanced_bumped_to_powerful():
    assert _infer_quality_tier("some-model", context_length=200_000) == "powerful"


# ── Capability inference tests ────────────────────────────────────────────


def test_infer_capabilities_gpt4():
    caps = _infer_capabilities({"id": "gpt-4o-mini"})
    assert caps["supports_tool_calling"] is True
    assert caps["supports_json_mode"] is True


def test_infer_capabilities_claude():
    caps = _infer_capabilities({"id": "claude-3-5-sonnet"})
    assert caps["supports_tool_calling"] is True


def test_infer_capabilities_unknown():
    caps = _infer_capabilities({"id": "my-custom-gguf"})
    assert caps["supports_tool_calling"] is False


# ── Service tests ─────────────────────────────────────────────────────────


def _make_bus() -> MemoryBus:
    return MemoryBus()


def _make_models(n: int = 3, tier: str = "fast", provider: str = "groq") -> list[DiscoveredModel]:
    return [
        DiscoveredModel(
            model_id=f"{provider}/model-{i}",
            provider=provider,
            base_model_name=f"model-{i}",
            quality_tier=tier,
            context_length=8192,
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_service_discover_all_no_providers():
    """With no env vars set, discover_all returns empty and doesn't crash."""
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)
    with patch.dict("os.environ", {}, clear=True):
        result = await svc.discover_all()
    assert result == {}


@pytest.mark.asyncio
async def test_service_get_available_models():
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    # Manually inject models
    models = _make_models(3, tier="fast")
    for m in models:
        svc._models[m.model_id] = m

    available = svc.get_available_models(tier="fast")
    assert len(available) == 3


@pytest.mark.asyncio
async def test_service_get_available_filters_gone():
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    models = _make_models(3, tier="fast")
    models[1].status = "gone"
    for m in models:
        svc._models[m.model_id] = m

    available = svc.get_available_models(tier="fast")
    assert len(available) == 2


@pytest.mark.asyncio
async def test_service_get_model():
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    m = _make_models(1)[0]
    svc._models[m.model_id] = m

    assert svc.get_model(m.model_id) is not None
    assert svc.get_model("nonexistent") is None


@pytest.mark.asyncio
async def test_service_get_model_gone():
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    m = _make_models(1)[0]
    m.status = "gone"
    svc._models[m.model_id] = m

    assert svc.get_model(m.model_id) is None


@pytest.mark.asyncio
async def test_reassign_tiers_basic():
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    # Add fast-tier models
    for m in _make_models(3, tier="fast"):
        svc._models[m.model_id] = m
    # Add balanced-tier models
    for m in _make_models(2, tier="balanced", provider="cerebras"):
        svc._models[m.model_id] = m

    await svc.reassign_tiers()

    fast_assign = svc.get_tier_assignment("fast")
    assert fast_assign is not None
    assert fast_assign.primary.startswith("groq/")
    assert fast_assign.auto_assigned is True

    balanced_assign = svc.get_tier_assignment("balanced")
    assert balanced_assign is not None
    assert balanced_assign.primary.startswith("cerebras/")


@pytest.mark.asyncio
async def test_reassign_tiers_user_override():
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    for m in _make_models(3, tier="fast"):
        svc._models[m.model_id] = m

    svc.set_user_override("fast", "my/custom-model")
    await svc.reassign_tiers()

    assign = svc.get_tier_assignment("fast")
    assert assign is not None
    assert assign.primary == "my/custom-model"
    assert assign.auto_assigned is False


@pytest.mark.asyncio
async def test_reassign_tiers_clear_override():
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    for m in _make_models(3, tier="fast"):
        svc._models[m.model_id] = m

    svc.set_user_override("fast", "pinned/model")
    svc.set_user_override("fast", None)  # clear
    await svc.reassign_tiers()

    assign = svc.get_tier_assignment("fast")
    assert assign is not None
    assert assign.primary != "pinned/model"


@pytest.mark.asyncio
async def test_record_call_success():
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    m = _make_models(1)[0]
    svc._models[m.model_id] = m

    svc.record_call(m.model_id, latency_ms=100.0, success=True)

    h = svc._health[m.model_id]
    assert h.total_calls == 1
    assert h.success_count == 1
    assert h.avg_latency_ms == 100.0


@pytest.mark.asyncio
async def test_record_call_error():
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    m = _make_models(1)[0]
    svc._models[m.model_id] = m

    svc.record_call(m.model_id, latency_ms=50.0, success=False, error="timeout")

    h = svc._health[m.model_id]
    assert h.error_count == 1
    assert h.last_error == "timeout"


@pytest.mark.asyncio
async def test_model_degraded_on_high_error_rate():
    """Model status → degraded when error rate exceeds threshold."""
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    m = _make_models(1)[0]
    svc._models[m.model_id] = m

    # Record 10 calls with > 30% error rate
    for i in range(10):
        svc.record_call(m.model_id, latency_ms=100.0, success=(i < 5), error="" if i < 5 else "err")

    # Give the background reassign task time to run
    await asyncio.sleep(0.05)

    assert svc._models[m.model_id].status == "degraded"


@pytest.mark.asyncio
async def test_reassign_prefers_healthy_models():
    """Healthier models are preferred as primary in tier assignment."""
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    models = _make_models(2, tier="fast")
    for m in models:
        svc._models[m.model_id] = m

    # Make model-0 have worse health
    h0 = ModelHealthMetrics(model_id=models[0].model_id)
    h0.total_calls = 10
    h0.error_count = 4
    h0.success_count = 6
    svc._health[models[0].model_id] = h0

    # model-1 is healthy
    h1 = ModelHealthMetrics(model_id=models[1].model_id)
    h1.total_calls = 10
    h1.error_count = 0
    h1.success_count = 10
    svc._health[models[1].model_id] = h1

    await svc.reassign_tiers()

    assign = svc.get_tier_assignment("fast")
    assert assign is not None
    assert assign.primary == models[1].model_id  # healthier one is primary


@pytest.mark.asyncio
async def test_bus_event_on_tier_update():
    """Verify bus event is published when tiers change."""
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    received = []

    async def handler(envelope: Envelope) -> None:
        received.append(envelope)

    bus.subscribe("models.tiers_updated", handler)

    for m in _make_models(2, tier="fast"):
        svc._models[m.model_id] = m

    await svc.reassign_tiers()
    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert "fast" in received[0].payload


@pytest.mark.asyncio
async def test_start_and_stop():
    """Service starts polling and stops cleanly."""
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus, poll_interval=3600)

    with patch.dict("os.environ", {}, clear=True):
        await svc.start()
        assert svc._poll_task is not None
        await svc.stop()
        assert svc._poll_task is None


@pytest.mark.asyncio
async def test_discover_provider_unknown():
    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)
    result = await svc.discover_provider("nonexistent_provider")
    assert result == []


# ── Router integration tests ─────────────────────────────────────────────


def test_router_uses_discovery_tier():
    """AutoRouter prefers discovery tier assignment over hardcoded."""
    from qe.runtime.router import AutoRouter

    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    # Set up tier assignment
    svc._tier_assignments["fast"] = TierAssignment(
        tier="fast",
        primary="groq/llama-3.3-70b",
        fallbacks=["cerebras/llama-3.3-70b"],
        reason="test",
    )

    pref = ModelPreference(tier="fast")
    router = AutoRouter(preference=pref, discovery=svc)

    env = Envelope(topic="test", source_service_id="test", payload={})
    selected = router.select(env)
    assert selected == "groq/llama-3.3-70b"


def test_router_falls_back_to_hardcoded():
    """When discovery has no assignment, fall back to TIER_MODELS."""
    from qe.runtime.router import AutoRouter

    bus = _make_bus()
    svc = ModelDiscoveryService(bus=bus)

    pref = ModelPreference(tier="fast")
    router = AutoRouter(preference=pref, discovery=svc)

    env = Envelope(topic="test", source_service_id="test", payload={})
    selected = router.select(env)
    assert selected == "gpt-4o-mini"  # first in hardcoded fast tier


def test_router_no_discovery():
    """Without discovery, original behavior is preserved."""
    from qe.runtime.router import AutoRouter

    pref = ModelPreference(tier="balanced")
    router = AutoRouter(preference=pref)

    env = Envelope(topic="test", source_service_id="test", payload={})
    selected = router.select(env)
    assert selected == "gpt-4o"


def test_router_record_success_with_discovery():
    """record_success calls discovery.record_call."""
    from qe.runtime.router import AutoRouter

    mock_discovery = MagicMock()
    pref = ModelPreference(tier="fast")
    router = AutoRouter(preference=pref, discovery=mock_discovery)

    # Trigger a select to set _select_start
    mock_discovery.get_tier_assignment.return_value = None
    env = Envelope(topic="test", source_service_id="test", payload={})
    router.select(env)

    router.record_success("gpt-4o-mini")
    mock_discovery.record_call.assert_called_once()
    call_args = mock_discovery.record_call.call_args
    assert call_args[0][0] == "gpt-4o-mini"
    assert call_args[1]["success"] is True


def test_router_record_error_with_discovery():
    """record_error calls discovery.record_call with success=False."""
    from qe.runtime.router import AutoRouter

    mock_discovery = MagicMock()
    pref = ModelPreference(tier="fast")
    router = AutoRouter(preference=pref, discovery=mock_discovery)

    mock_discovery.get_tier_assignment.return_value = None
    env = Envelope(topic="test", source_service_id="test", payload={})
    router.select(env)

    router.record_error("gpt-4o-mini")
    mock_discovery.record_call.assert_called_once()
    call_args = mock_discovery.record_call.call_args
    assert call_args[1]["success"] is False


# ── Cost governor integration tests ──────────────────────────────────────


def test_cost_governor_uses_discovery():
    """CostGovernor returns 0 for free discovered models."""
    from qe.runtime.cost_governor import CostGovernor

    mock_discovery = MagicMock()
    mock_model = MagicMock()
    mock_model.cost_per_m_input = 0.0
    mock_model.cost_per_m_output = 0.0
    mock_discovery.get_model.return_value = mock_model

    gov = CostGovernor(discovery=mock_discovery)
    est = gov.estimate_cost("groq/llama-3.3-70b", input_tokens=1000, output_tokens=500)
    assert est.estimated_cost_usd == 0.0


def test_cost_governor_fallback_without_discovery():
    """Without discovery, CostGovernor uses hardcoded rates."""
    from qe.runtime.cost_governor import CostGovernor

    gov = CostGovernor()
    est = gov.estimate_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
    assert est.estimated_cost_usd == pytest.approx(0.15, abs=0.01)


# ── Model capabilities integration tests ─────────────────────────────────


def test_capabilities_from_discovery():
    """ModelCapabilities uses discovery for unknown models."""
    from qe.runtime.model_capabilities import ModelCapabilities

    mock_discovery = MagicMock()
    mock_model = MagicMock()
    mock_model.supports_json_mode = True
    mock_model.supports_tool_calling = True
    mock_model.supports_system_messages = True
    mock_model.context_length = 131072
    mock_model.quality_tier = "powerful"
    mock_discovery.get_model.return_value = mock_model

    caps = ModelCapabilities(discovery=mock_discovery)
    profile = caps.get_profile("groq/llama-3.3-70b")
    assert profile.supports_tool_calling is True
    assert profile.max_context_tokens == 131072
    assert profile.estimated_quality_tier == "powerful"


def test_capabilities_fallback_known():
    """Without discovery, known profiles still work."""
    from qe.runtime.model_capabilities import ModelCapabilities

    caps = ModelCapabilities()
    profile = caps.get_profile("gpt-4o")
    assert profile.supports_tool_calling is True
    assert profile.estimated_quality_tier == "powerful"


# ── Rate limiter integration tests ────────────────────────────────────────


def test_rate_limiter_discovery_rpm():
    """RateLimiter uses discovery RPM when available."""
    from qe.runtime.rate_limiter import RateLimiter

    mock_discovery = MagicMock()
    mock_model = MagicMock()
    mock_model.rate_limit_rpm = 15
    mock_discovery.get_model.return_value = mock_model

    limiter = RateLimiter(discovery=mock_discovery)
    rpm = limiter._get_rpm(model="gemini/gemini-2.0-flash", provider="gemini/")
    assert rpm == 15


def test_rate_limiter_fallback_rpm():
    """Without discovery, RateLimiter uses hardcoded RPMs."""
    from qe.runtime.rate_limiter import RateLimiter

    limiter = RateLimiter()
    rpm = limiter._get_rpm(model="groq/something", provider="groq/")
    assert rpm == 30

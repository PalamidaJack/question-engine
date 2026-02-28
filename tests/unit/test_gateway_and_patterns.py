"""Tests for gateway middleware, retry hardening, burst, and circuit breaker."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from qe.bus.memory_bus import (
    _DEFAULT_RETRY_BASE_DELAY,
    MemoryBus,
    is_retryable,
)
from qe.kernel.supervisor import (
    _CIRCUIT_COOLDOWN_SECONDS,
    CircuitState,
    Supervisor,
)
from qe.models.envelope import Envelope
from qe.runtime.rate_limiter import RateLimiter, TokenBucket


def _make_envelope(
    topic: str = "system.heartbeat", **kwargs
) -> Envelope:
    return Envelope(
        topic=topic,
        source_service_id="test",
        payload={"msg": "test"},
        **kwargs,
    )

# ── Retry Hardening Tests ────────────────────────────────────────────────


class TestRetryJitter:
    """Verify that retry delays include jitter (not pure exponential)."""

    @pytest.mark.asyncio
    async def test_retry_delay_includes_jitter(self):
        """Delay should be > pure exponential (jitter adds positive offset)."""
        bus = MemoryBus(max_retries=1)
        delays: list[float] = []

        async def capture_delay(d: float) -> None:
            delays.append(d)

        async def failing_handler(_env: Envelope) -> None:
            raise RuntimeError("transient")

        bus.subscribe("system.heartbeat", failing_handler)

        with patch("qe.bus.memory_bus.asyncio.sleep", side_effect=capture_delay):
            bus.publish(_make_envelope())
            await asyncio.sleep(0.1)

        assert len(delays) == 1
        pure_exp = _DEFAULT_RETRY_BASE_DELAY * (2**0)
        assert delays[0] >= pure_exp

    @pytest.mark.asyncio
    async def test_jitter_bounded(self):
        """Delay should be < exponential + base_delay."""
        bus = MemoryBus(max_retries=1)
        delays: list[float] = []

        async def capture_delay(d: float) -> None:
            delays.append(d)

        async def failing_handler(_env: Envelope) -> None:
            raise RuntimeError("transient")

        bus.subscribe("system.heartbeat", failing_handler)

        with patch("qe.bus.memory_bus.asyncio.sleep", side_effect=capture_delay):
            bus.publish(_make_envelope())
            await asyncio.sleep(0.1)

        assert len(delays) == 1
        upper = _DEFAULT_RETRY_BASE_DELAY * (2**0) + _DEFAULT_RETRY_BASE_DELAY
        assert delays[0] < upper


class TestNonRetryableErrors:
    """Verify that non-retryable errors skip retries and go to DLQ."""

    @pytest.mark.asyncio
    async def test_value_error_skips_retries(self):
        """ValueError is non-retryable — DLQ immediately, call_count=1."""
        bus = MemoryBus(max_retries=2)
        call_count = 0

        async def failing_handler(_env: Envelope) -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        bus.subscribe("system.heartbeat", failing_handler)
        bus.publish(_make_envelope())
        await asyncio.sleep(0.2)

        assert call_count == 1, "Non-retryable error should not retry"
        assert bus.dlq_size() == 1

    @pytest.mark.asyncio
    async def test_runtime_error_still_retried(self):
        """RuntimeError is retryable — should exhaust all attempts."""
        bus = MemoryBus(max_retries=2)
        call_count = 0

        async def failing_handler(_env: Envelope) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("transient")

        bus.subscribe("system.heartbeat", failing_handler)
        bus.publish(_make_envelope())
        # Allow retries with backoff (total ~0.5s with jitter)
        await asyncio.sleep(1.0)

        assert call_count == 3

    def test_is_retryable_classification(self):
        """Unit test for is_retryable function."""
        assert not is_retryable(ValueError("x"))
        assert not is_retryable(TypeError("x"))
        assert not is_retryable(KeyError("x"))
        assert not is_retryable(AttributeError("x"))
        assert not is_retryable(NotImplementedError("x"))

        assert is_retryable(RuntimeError("x"))
        assert is_retryable(OSError("x"))
        assert is_retryable(Exception("x"))
        assert is_retryable(ConnectionError("x"))


# ── Burst Allowance Tests ────────────────────────────────────────────────


class TestBurstAllowance:
    """Verify burst capacity in TokenBucket."""

    def test_burst_capacity_exceeds_rpm(self):
        """capacity = rpm + burst."""
        bucket = TokenBucket(provider="test", rpm=100, burst=20)
        assert bucket.capacity == 120

    def test_default_burst_zero(self):
        """Default burst is 0 for backward compat."""
        bucket = TokenBucket(provider="test", rpm=100)
        assert bucket.burst == 0
        assert bucket.capacity == 100

    def test_burst_refill_caps_at_capacity(self):
        """Tokens should not exceed rpm + burst after refill."""
        bucket = TokenBucket(provider="test", rpm=60, burst=10)
        # Capacity should be 70
        assert bucket.tokens == 70.0

        # Drain some tokens
        for _ in range(5):
            bucket.try_acquire()

        assert bucket.tokens == pytest.approx(65.0, abs=0.01)

        # Simulate time passing (enough to fully refill)
        bucket.last_refill = time.monotonic() - 120  # 2 minutes ago
        bucket._refill()

        # Should cap at capacity, not exceed it
        assert bucket.tokens == pytest.approx(70.0, abs=0.01)

    def test_rate_limiter_burst_propagated(self):
        """RateLimiter passes burst_allowance to created buckets."""
        rl = RateLimiter(burst_allowance=15)
        # Trigger bucket creation via try_acquire
        rl.try_acquire("gpt-4o")

        # Find the bucket
        bucket = list(rl._buckets.values())[0]
        assert bucket.burst == 15


# ── Half-Open Circuit Breaker Tests ──────────────────────────────────────


class TestCircuitState:
    """Verify CircuitState probing logic."""

    def test_circuit_state_should_probe_after_cooldown(self):
        """should_probe returns True after cooldown period elapses."""
        opened = datetime.now(UTC) - timedelta(seconds=_CIRCUIT_COOLDOWN_SECONDS + 1)
        state = CircuitState(status="open", opened_at=opened)
        assert state.should_probe(datetime.now(UTC)) is True

    def test_circuit_state_no_probe_before_cooldown(self):
        """should_probe returns False before cooldown period elapses."""
        opened = datetime.now(UTC) - timedelta(seconds=_CIRCUIT_COOLDOWN_SECONDS - 10)
        state = CircuitState(status="open", opened_at=opened)
        assert state.should_probe(datetime.now(UTC)) is False

    def test_half_open_no_probe_when_already_half_open(self):
        """Half-open state should not trigger should_probe."""
        opened = datetime.now(UTC) - timedelta(seconds=_CIRCUIT_COOLDOWN_SECONDS + 1)
        state = CircuitState(status="half_open", opened_at=opened)
        # should_probe only returns True when status == "open"
        assert state.should_probe(datetime.now(UTC)) is False


class TestHalfOpenCircuitBreaker:
    """Integration tests for half-open circuit breaker in Supervisor."""

    def _make_supervisor(self):
        bus = MemoryBus()
        sup = Supervisor.__new__(Supervisor)
        sup.bus = bus
        sup.registry = MagicMock()
        sup._circuits = {}
        sup._pub_history = {}
        return sup

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self):
        """A successful probe in half-open state should close the circuit."""
        sup = self._make_supervisor()
        sid = "test-service"

        # Set up an open circuit that is past cooldown
        opened = datetime.now(UTC) - timedelta(seconds=_CIRCUIT_COOLDOWN_SECONDS + 1)
        sup._circuits[sid] = CircuitState(status="open", opened_at=opened)

        handler = AsyncMock()
        service = MagicMock()
        service.blueprint.service_id = sid

        wrapped = sup._wrap_service_handler(handler, service)

        env = Envelope(topic="system.heartbeat", source_service_id="test", payload={})
        await wrapped(env)

        # Handler should have been called (probe)
        handler.assert_awaited_once()
        # Circuit should be closed (removed)
        assert sid not in sup._circuits

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        """A failed probe in half-open state should reopen the circuit."""
        sup = self._make_supervisor()
        sid = "test-service"

        opened = datetime.now(UTC) - timedelta(seconds=_CIRCUIT_COOLDOWN_SECONDS + 1)
        sup._circuits[sid] = CircuitState(status="open", opened_at=opened)

        handler = AsyncMock(side_effect=RuntimeError("still broken"))
        service = MagicMock()
        service.blueprint.service_id = sid

        wrapped = sup._wrap_service_handler(handler, service)

        env = Envelope(topic="system.heartbeat", source_service_id="test", payload={})
        await wrapped(env)

        # Circuit should still exist and be "open" again
        assert sid in sup._circuits
        assert sup._circuits[sid].status == "open"
        # opened_at should be reset (newer than the original)
        assert sup._circuits[sid].opened_at > opened

    def test_loop_detection_creates_circuit_state(self):
        """_check_loop should create a CircuitState with status='open'."""
        sup = self._make_supervisor()

        from qe.models.envelope import Envelope

        # Trigger loop detection by publishing identical envelopes
        for _ in range(6):
            env = Envelope(
                topic="system.heartbeat",
                source_service_id="looper",
                payload={"x": 1},
            )
            sup._check_loop("looper", env)

        assert "looper" in sup._circuits
        assert sup._circuits["looper"].status == "open"
        assert sup._circuits["looper"].opened_at is not None

    def test_backward_compat_property(self):
        """_circuit_broken property should return set of circuit keys."""
        sup = self._make_supervisor()
        sup._circuits["svc-a"] = CircuitState(status="open", opened_at=datetime.now(UTC))
        sup._circuits["svc-b"] = CircuitState(status="half_open", opened_at=datetime.now(UTC))

        broken = sup._circuit_broken
        assert isinstance(broken, set)
        assert broken == {"svc-a", "svc-b"}


# ── Gateway Middleware Tests ─────────────────────────────────────────────


def _make_test_app():
    """Create a minimal FastAPI app with our middleware for testing."""
    from fastapi import FastAPI

    from qe.api.middleware import AuthMiddleware, RateLimitMiddleware

    app = FastAPI()

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/data")
    async def get_data():
        return {"data": "secret"}

    @app.post("/api/data")
    async def post_data():
        return {"created": True}

    @app.get("/api/audit")
    async def audit():
        return {"entries": []}

    # Add middleware (last-added = outermost)
    app.add_middleware(RateLimitMiddleware, rpm=5, burst=2)
    app.add_middleware(AuthMiddleware)

    return app


class TestAuthMiddleware:
    """Test authentication middleware."""

    def test_auth_middleware_passthrough_when_disabled(self):
        """When no keys are configured, all requests pass through."""
        app = _make_test_app()
        client = TestClient(app)

        with patch("qe.api.middleware.get_auth_provider") as mock_provider:
            provider = MagicMock()
            provider.enabled = False
            mock_provider.return_value = provider

            resp = client.get("/api/data")
            assert resp.status_code == 200

    def test_auth_middleware_rejects_without_key(self):
        """401 when auth is enabled and no X-API-Key header."""
        app = _make_test_app()
        client = TestClient(app)

        with patch("qe.api.middleware.get_auth_provider") as mock_provider:
            provider = MagicMock()
            provider.enabled = True
            mock_provider.return_value = provider

            resp = client.get("/api/data")
            assert resp.status_code == 401
            assert "X-API-Key" in resp.json()["error"]

    def test_auth_middleware_accepts_valid_key(self):
        """200 with correct key."""
        from qe.api.auth import AuthContext, Scope

        app = _make_test_app()
        client = TestClient(app)

        with patch("qe.api.middleware.get_auth_provider") as mock_provider:
            provider = MagicMock()
            provider.enabled = True
            provider.validate_key.return_value = AuthContext(
                scope=Scope.ADMIN, key_id="test"
            )
            mock_provider.return_value = provider

            resp = client.get("/api/data", headers={"X-API-Key": "valid-key"})
            assert resp.status_code == 200

    def test_auth_middleware_scope_enforcement(self):
        """POST with READ-only key should return 403."""
        from qe.api.auth import AuthContext, Scope

        app = _make_test_app()
        client = TestClient(app)

        with patch("qe.api.middleware.get_auth_provider") as mock_provider:
            provider = MagicMock()
            provider.enabled = True
            provider.validate_key.return_value = AuthContext(
                scope=Scope.READ, key_id="readonly"
            )
            mock_provider.return_value = provider

            resp = client.post("/api/data", headers={"X-API-Key": "read-key"})
            assert resp.status_code == 403
            assert "scope" in resp.json()["error"].lower()

    def test_auth_middleware_public_paths_exempt(self):
        """Health endpoints always pass regardless of auth config."""
        app = _make_test_app()
        client = TestClient(app)

        with patch("qe.api.middleware.get_auth_provider") as mock_provider:
            provider = MagicMock()
            provider.enabled = True
            mock_provider.return_value = provider

            resp = client.get("/api/health")
            assert resp.status_code == 200


class TestRateLimitMiddleware:
    """Test HTTP rate limiting middleware."""

    def test_rate_limit_returns_429(self):
        """Exhausted bucket should return 429."""
        app = _make_test_app()
        client = TestClient(app)

        with patch("qe.api.middleware.get_auth_provider") as mock_provider:
            provider = MagicMock()
            provider.enabled = False
            mock_provider.return_value = provider

            # rpm=5, burst=2 → capacity=7, so 8th request should 429
            for i in range(7):
                resp = client.get("/api/data")
                assert resp.status_code == 200, f"Request {i+1} should pass"

            resp = client.get("/api/data")
            assert resp.status_code == 429
            body = resp.json()
            assert "retry_after_seconds" in body

    def test_rate_limit_allows_within_budget(self):
        """Requests within budget should pass through."""
        app = _make_test_app()
        client = TestClient(app)

        with patch("qe.api.middleware.get_auth_provider") as mock_provider:
            provider = MagicMock()
            provider.enabled = False
            mock_provider.return_value = provider

            resp = client.get("/api/data")
            assert resp.status_code == 200

    def test_rate_limit_health_exempt(self):
        """Health endpoints should not be rate limited."""
        app = _make_test_app()
        client = TestClient(app)

        with patch("qe.api.middleware.get_auth_provider") as mock_provider:
            provider = MagicMock()
            provider.enabled = False
            mock_provider.return_value = provider

            # Exhaust the regular budget
            for _ in range(10):
                client.get("/api/data")

            # Health should still work
            resp = client.get("/api/health")
            assert resp.status_code == 200

"""API request middleware: timing, auth, rate limiting, structured logging, error wrapping.

Records request latency in MetricsCollector and emits structured
log entries for every HTTP request.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from qe.api.auth import Scope, get_auth_provider
from qe.runtime.metrics import get_metrics

log = logging.getLogger(__name__)

# ── Paths exempt from auth and rate limiting ──────────────────────────────

_PUBLIC_PATHS: set[str] = {
    "/",
    "/api/health",
    "/api/health/ready",
    "/api/health/live",
    "/api/setup/status",
    "/api/setup/providers",
    "/api/setup/save",
    "/docs",
    "/openapi.json",
    "/redoc",
}

_PUBLIC_PREFIXES: tuple[str, ...] = ("/static/", "/ws")

_ADMIN_PREFIXES: tuple[str, ...] = ("/api/audit", "/api/services/")

_METHOD_SCOPE: dict[str, Scope] = {
    "GET": Scope.READ,
    "HEAD": Scope.READ,
    "OPTIONS": Scope.READ,
    "POST": Scope.WRITE,
    "PUT": Scope.WRITE,
    "DELETE": Scope.WRITE,
    "PATCH": Scope.WRITE,
}


def _is_public(path: str) -> bool:
    return path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES)


# ── Auth Middleware ────────────────────────────────────────────────────────


class AuthMiddleware(BaseHTTPMiddleware):
    """Enforce API key authentication when the AuthProvider has keys configured.

    Public/health/setup paths are always exempt. When auth is disabled
    (no keys configured), all requests pass through at zero cost.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        path = request.url.path

        # Public paths always pass
        if _is_public(path):
            return await call_next(request)

        provider = get_auth_provider()
        if not provider.enabled:
            return await call_next(request)

        # Extract and validate API key
        api_key = request.headers.get("x-api-key")
        if not api_key:
            return JSONResponse(
                {"error": "X-API-Key header required"},
                status_code=401,
            )

        ctx = provider.validate_key(api_key)
        if ctx is None:
            return JSONResponse(
                {"error": "Invalid API key"},
                status_code=401,
            )

        # Scope check
        if path.startswith(_ADMIN_PREFIXES):
            required = Scope.ADMIN
        else:
            required = _METHOD_SCOPE.get(request.method, Scope.READ)

        if not ctx.has_scope(required):
            return JSONResponse(
                {"error": f"Insufficient scope: {required} required"},
                status_code=403,
            )

        # Attach auth context for downstream handlers
        request.state.auth = ctx
        return await call_next(request)


# ── HTTP Rate Limiting Middleware ─────────────────────────────────────────


class _HTTPTokenBucket:
    """Simple per-client token bucket for HTTP rate limiting."""

    __slots__ = ("rpm", "burst", "_tokens", "_last_refill")

    def __init__(self, rpm: int, burst: int) -> None:
        self.rpm = rpm
        self.burst = burst
        self._tokens: float = float(rpm + burst)
        self._last_refill: float = time.monotonic()

    @property
    def _capacity(self) -> int:
        return self.rpm + self.burst

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * (self.rpm / 60.0))
        self._last_refill = now

    def try_acquire(self) -> bool:
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    @property
    def retry_after_seconds(self) -> float:
        """Seconds until the next token is available."""
        if self._tokens >= 1.0:
            return 0.0
        return (1.0 - self._tokens) / (self.rpm / 60.0)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client-IP HTTP rate limiter using token buckets.

    Public/health paths are exempt from rate limiting.
    Returns 429 with retry_after_seconds when a client exceeds their budget.
    """

    def __init__(self, app: Any, rpm: int = 120, burst: int = 20) -> None:
        super().__init__(app)
        self._rpm = rpm
        self._burst = burst
        self._buckets: dict[str, _HTTPTokenBucket] = {}

    def _get_bucket(self, client_ip: str) -> _HTTPTokenBucket:
        if client_ip not in self._buckets:
            self._buckets[client_ip] = _HTTPTokenBucket(self._rpm, self._burst)
        return self._buckets[client_ip]

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        path = request.url.path

        if _is_public(path):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        bucket = self._get_bucket(client_ip)

        if not bucket.try_acquire():
            retry_after = bucket.retry_after_seconds
            return JSONResponse(
                {
                    "error": "Rate limit exceeded",
                    "retry_after_seconds": round(retry_after, 1),
                },
                status_code=429,
            )

        return await call_next(request)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Middleware that times requests, logs them, and wraps errors.

    For every request:
    1. Records start time
    2. Calls the route handler
    3. Logs method, path, status, and duration
    4. Records latency in MetricsCollector histogram
    5. Catches unhandled exceptions and returns consistent JSON errors
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        start = time.monotonic()
        method = request.method
        path = request.url.path

        metrics = get_metrics()
        metrics.counter("api_requests_total").inc()

        try:
            response = await call_next(request)
            elapsed_ms = (time.monotonic() - start) * 1000
            status = response.status_code

            metrics.histogram("api_latency_ms").observe(elapsed_ms)

            if status >= 400:
                metrics.counter("api_errors_total").inc()

            log.info(
                "api.request method=%s path=%s status=%d duration_ms=%.1f",
                method,
                path,
                status,
                elapsed_ms,
            )

            return response

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            metrics.counter("api_errors_total").inc()
            metrics.histogram("api_latency_ms").observe(elapsed_ms)

            log.error(
                "api.unhandled_error method=%s path=%s "
                "error=%s duration_ms=%.1f",
                method,
                path,
                str(exc),
                elapsed_ms,
            )

            return JSONResponse(
                {"error": "Internal server error", "detail": str(exc)},
                status_code=500,
            )

"""Provider-level rate limiting with token bucket algorithm.

Prevents 429 errors by proactively throttling requests per provider.
Supports configurable RPM (requests per minute) limits and async waiting.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# Default RPM limits per provider prefix
_DEFAULT_RPM: dict[str, int] = {
    "gpt-": 500,
    "o1": 200,
    "claude-": 400,
    "gemini/": 300,
    "groq/": 30,
    "ollama/": 10000,  # local, effectively unlimited
    "together_ai/": 100,
}


@dataclass
class TokenBucket:
    """Token bucket rate limiter for a single provider.

    Tokens refill at a constant rate. Each request consumes one token.
    When empty, callers wait until a token becomes available.
    """

    provider: str
    rpm: int  # requests per minute
    burst: int = 0
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self._lock = asyncio.Lock()

    @property
    def capacity(self) -> int:
        """Maximum tokens: rpm + burst allowance."""
        return self.rpm + self.burst

    @property
    def refill_rate(self) -> float:
        """Tokens per second."""
        return self.rpm / 60.0

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    async def acquire(self, max_wait: float = 30.0) -> bool:
        """Acquire a token, waiting if necessary.

        Returns True if acquired within max_wait, False if timed out.
        """
        deadline = time.monotonic() + max_wait

        async with self._lock:
            self._refill()

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True

            # Calculate wait time for next token
            wait_time = (1.0 - self.tokens) / self.refill_rate
            if time.monotonic() + wait_time > deadline:
                log.warning(
                    "rate_limiter.timeout provider=%s rpm=%d wait=%.2fs",
                    self.provider,
                    self.rpm,
                    wait_time,
                )
                return False

            log.debug(
                "rate_limiter.waiting provider=%s wait=%.2fs tokens=%.1f",
                self.provider,
                wait_time,
                self.tokens,
            )
            await asyncio.sleep(wait_time)

            self._refill()
            self.tokens -= 1.0
            return True

    def try_acquire(self) -> bool:
        """Non-blocking acquire. Returns True if token available."""
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def available_tokens(self) -> float:
        """Current available tokens (without consuming any)."""
        self._refill()
        return self.tokens


class RateLimiter:
    """Per-provider rate limiter for LLM API calls.

    Maintains a token bucket per provider, identified by model prefix.
    Auto-creates buckets on first use with default or custom RPM limits.
    """

    def __init__(
        self,
        custom_limits: dict[str, int] | None = None,
        enabled: bool = True,
        burst_allowance: int = 0,
    ) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._custom_limits = custom_limits or {}
        self._enabled = enabled
        self._burst_allowance = burst_allowance
        self._total_waits = 0
        self._total_requests = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def _get_provider(self, model: str) -> str:
        """Extract provider prefix from model name."""
        for prefix in _DEFAULT_RPM:
            if model.startswith(prefix):
                return prefix
        return model.split("/")[0] if "/" in model else model

    def _get_rpm(self, provider: str) -> int:
        """Get RPM limit for a provider."""
        if provider in self._custom_limits:
            return self._custom_limits[provider]
        return _DEFAULT_RPM.get(provider, 200)

    def _get_bucket(self, model: str) -> TokenBucket:
        """Get or create a token bucket for a model's provider."""
        provider = self._get_provider(model)
        if provider not in self._buckets:
            rpm = self._get_rpm(provider)
            self._buckets[provider] = TokenBucket(
                provider=provider, rpm=rpm, burst=self._burst_allowance
            )
            log.debug(
                "rate_limiter.new_bucket provider=%s rpm=%d",
                provider,
                rpm,
            )
        return self._buckets[provider]

    async def acquire(self, model: str, max_wait: float = 30.0) -> bool:
        """Acquire a rate limit token for the given model.

        Blocks until a token is available or max_wait is reached.
        Returns True if acquired, False if timed out.
        """
        if not self._enabled:
            return True

        self._total_requests += 1
        bucket = self._get_bucket(model)
        old_tokens = bucket.available_tokens

        acquired = await bucket.acquire(max_wait=max_wait)

        if old_tokens < 1.0:
            self._total_waits += 1

        return acquired

    def try_acquire(self, model: str) -> bool:
        """Non-blocking acquire for the given model."""
        if not self._enabled:
            return True
        bucket = self._get_bucket(model)
        return bucket.try_acquire()

    def set_rpm(self, provider: str, rpm: int) -> None:
        """Update RPM limit for a provider. Takes effect on next bucket creation."""
        self._custom_limits[provider] = rpm
        # Update existing bucket if present
        if provider in self._buckets:
            self._buckets[provider].rpm = rpm

    def stats(self) -> dict[str, Any]:
        """Return rate limiter statistics."""
        return {
            "enabled": self._enabled,
            "total_requests": self._total_requests,
            "total_waits": self._total_waits,
            "buckets": {
                provider: {
                    "rpm": bucket.rpm,
                    "available_tokens": round(bucket.available_tokens, 1),
                }
                for provider, bucket in self._buckets.items()
            },
        }


# ── Singleton ──────────────────────────────────────────────────────────────

_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter

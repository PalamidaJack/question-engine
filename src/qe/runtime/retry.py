"""Retry policy and simple circuit breaker utilities.

Small, well-tested primitives for configurable retry/backoff and a
lightweight circuit breaker to prevent thundering failures.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock


class CBState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 0.1  # seconds
    max_delay: float = 10.0  # seconds
    multiplier: float = 2.0
    jitter: bool = True

    def get_delay(self, attempt: int) -> float:
        """Return delay in seconds for the given 0-based attempt index."""
        if attempt <= 0:
            return 0.0
        delay = self.base_delay * (self.multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay


class CircuitBreaker:
    """Simple non-distributed circuit breaker.

    Usage:
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout_s=30)
        if not cb.allow_request():
            raise RuntimeError("service unavailable")

        try:
            do_work()
            cb.record_success()
        except Exception:
            cb.record_failure()
            raise
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout_s: int = 30):
        self._failure_threshold = max(1, int(failure_threshold))
        self._recovery_timeout_s = max(1, int(recovery_timeout_s))
        self._failures = 0
        self._state: CBState = CBState.CLOSED
        self._opened_at: float | None = None
        self._lock = Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self._state == CBState.OPEN:
                assert self._opened_at is not None
                if time.time() - self._opened_at >= self._recovery_timeout_s:
                    # Move to half-open to try a probe
                    self._state = CBState.HALF_OPEN
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = CBState.CLOSED
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = CBState.OPEN
        self._opened_at = time.time()

    def state(self) -> CBState:
        with self._lock:
            return self._state

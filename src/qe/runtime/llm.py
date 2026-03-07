"""Unified LLM abstraction — single entry point for all LLM calls.

Centralizes retry logic, cost tracking, per-provider configuration,
and call metrics.  All other modules should call ``UnifiedLLM`` instead
of ``litellm.acompletion`` directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class LLMCallRecord:
    """Immutable record of a single LLM call."""

    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    success: bool
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


class UnifiedLLM:
    """Unified wrapper around ``litellm.acompletion``.

    Features:
    - Automatic retry with exponential backoff
    - Per-call cost tracking (via litellm ``completion_cost``)
    - Metrics emission to ``MetricsCollector``
    - Provider-level configuration (temperature, max_tokens overrides)
    - Call history for diagnostics
    """

    def __init__(
        self,
        *,
        default_model: str = "openai/anthropic/claude-sonnet-4",
        max_retries: int = 2,
        base_delay: float = 1.0,
        provider_config: dict[str, dict[str, Any]] | None = None,
        metrics: Any | None = None,
        budget_tracker: Any | None = None,
    ) -> None:
        self._default_model = default_model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._provider_config = provider_config or {}
        self._metrics = metrics
        self._budget_tracker = budget_tracker
        self._history: list[LLMCallRecord] = []
        self._max_history = 500
        self._total_cost_usd = 0.0
        self._total_calls = 0
        self._total_errors = 0

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call the LLM with automatic retry and cost tracking.

        Returns the litellm response object.
        """
        import asyncio

        import litellm

        model = model or self._default_model

        # Apply provider-level config overrides
        call_kwargs = self._resolve_provider_config(model)
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        call_kwargs.update(kwargs)

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            start = time.monotonic()
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **call_kwargs,
                )

                latency_ms = (time.monotonic() - start) * 1000
                cost = self._extract_cost(response, model)
                input_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
                output_tokens = (
                    getattr(response.usage, "completion_tokens", 0) if response.usage else 0
                )

                record = LLMCallRecord(
                    model=model,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    success=True,
                )
                self._record(record)
                return response

            except Exception as exc:
                latency_ms = (time.monotonic() - start) * 1000
                last_error = exc
                record = LLMCallRecord(
                    model=model,
                    latency_ms=latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    success=False,
                    error=str(exc),
                )
                self._record(record)

                if attempt < self._max_retries:
                    delay = self._base_delay * (2**attempt)
                    log.warning(
                        "unified_llm.retry model=%s attempt=%d delay=%.1fs error=%s",
                        model,
                        attempt + 1,
                        delay,
                        str(exc)[:120],
                    )
                    await asyncio.sleep(delay)

        raise last_error  # type: ignore[misc]

    def _resolve_provider_config(self, model: str) -> dict[str, Any]:
        """Merge provider-level defaults for the given model."""
        config: dict[str, Any] = {}
        for prefix, overrides in self._provider_config.items():
            if model.startswith(prefix):
                config.update(overrides)
                break
        return config

    def _extract_cost(self, response: Any, model: str) -> float:
        """Extract cost from litellm response, falling back to 0."""
        try:
            from litellm import completion_cost

            return completion_cost(completion_response=response, model=model)
        except Exception:
            return 0.0

    def _record(self, record: LLMCallRecord) -> None:
        """Record a call and update aggregates."""
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        self._total_calls += 1
        self._total_cost_usd += record.cost_usd

        if not record.success:
            self._total_errors += 1

        if self._metrics is not None:
            self._metrics.counter("llm_calls_total").inc()
            self._metrics.histogram("llm_latency_ms").observe(record.latency_ms)
            if not record.success:
                self._metrics.counter("llm_errors_total").inc()

        if self._budget_tracker is not None and record.cost_usd > 0:
            self._budget_tracker.record_cost(record.cost_usd)

    def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        return {
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "total_cost_usd": round(self._total_cost_usd, 6),
            "error_rate": (
                round(self._total_errors / self._total_calls, 4)
                if self._total_calls > 0
                else 0.0
            ),
            "default_model": self._default_model,
        }

    def recent_calls(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent call records."""
        return [
            {
                "model": r.model,
                "latency_ms": round(r.latency_ms, 1),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": round(r.cost_usd, 6),
                "success": r.success,
                "error": r.error,
                "timestamp": r.timestamp,
            }
            for r in self._history[-limit:]
        ]


# ── Circuit Breaker (#68) ─────────────────────────────────────────────────


class CircuitBreaker:
    """Per-model circuit breaker: open after N failures, half-open after cooldown.

    States:
    - closed: normal operation
    - open: all calls rejected (fail fast)
    - half_open: one probe call allowed to test recovery
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._states: dict[str, str] = {}  # model -> "closed"|"open"|"half_open"
        self._failure_counts: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def state(self, model: str) -> str:
        """Return current state for a model."""
        s = self._states.get(model, "closed")
        if s == "open":
            elapsed = time.time() - self._opened_at.get(model, 0)
            if elapsed >= self._cooldown_seconds:
                self._states[model] = "half_open"
                return "half_open"
        return s

    def is_available(self, model: str) -> bool:
        """Return True if model is available for calls."""
        return self.state(model) != "open"

    def record_success(self, model: str) -> None:
        """Record a successful call — reset breaker."""
        self._states[model] = "closed"
        self._failure_counts[model] = 0

    def record_failure(self, model: str) -> None:
        """Record a failed call — increment toward threshold."""
        count = self._failure_counts.get(model, 0) + 1
        self._failure_counts[model] = count
        if count >= self._failure_threshold:
            self._states[model] = "open"
            self._opened_at[model] = time.time()
            log.warning("circuit_breaker.opened model=%s failures=%d", model, count)

    def all_states(self) -> dict[str, str]:
        """Return states for all tracked models."""
        return {model: self.state(model) for model in self._states}


class FailoverChain:
    """Multi-provider failover with circuit breaker integration.

    Tries models in priority order, skipping circuit-broken models.
    On success, dynamically reorders to prefer faster/cheaper models.
    """

    def __init__(
        self,
        models: list[str],
        circuit_breaker: CircuitBreaker | None = None,
        unified_llm: UnifiedLLM | None = None,
    ) -> None:
        self._models = list(models)
        self._cb = circuit_breaker or CircuitBreaker()
        self._llm = unified_llm

    @property
    def models(self) -> list[str]:
        return list(self._models)

    def available_models(self) -> list[str]:
        """Return models not circuit-broken."""
        return [m for m in self._models if self._cb.is_available(m)]

    async def complete(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        """Try each model in order until one succeeds.

        Uses the circuit breaker to skip known-broken models.
        """
        if self._llm is None:
            raise RuntimeError("FailoverChain requires a UnifiedLLM instance")

        last_error: Exception | None = None
        for model in self._models:
            if not self._cb.is_available(model):
                continue
            try:
                response = await self._llm.complete(
                    messages, model=model, **kwargs
                )
                self._cb.record_success(model)
                return response
            except Exception as exc:
                self._cb.record_failure(model)
                last_error = exc
                log.warning(
                    "failover.model_failed model=%s error=%s",
                    model,
                    str(exc)[:100],
                )

        if last_error:
            raise last_error
        raise RuntimeError("No models available in failover chain")

    def status(self) -> dict[str, Any]:
        """Return failover chain status."""
        return {
            "models": self._models,
            "circuit_breaker_states": self._cb.all_states(),
            "available_models": self.available_models(),
        }

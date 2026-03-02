"""Model discovery service — polls providers, profiles models, assigns tiers."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from qe.models.envelope import Envelope
from qe.runtime.discovery.providers import PROVIDER_FETCHERS
from qe.runtime.discovery.schemas import DiscoveredModel, ModelHealthMetrics, TierAssignment

if TYPE_CHECKING:
    from qe.bus.memory_bus import MemoryBus

log = logging.getLogger(__name__)

# Error rate threshold (fraction) over the recent window to trigger deprioritization
_ERROR_RATE_THRESHOLD = 0.30
_HEALTH_WINDOW_CALLS = 10


class ModelDiscoveryService:
    """Discovers free models from provider APIs and manages tier assignments.

    Graceful degradation: all consumers check ``if discovery is not None``
    before calling methods.  If no providers are configured, the service
    is effectively a no-op and existing hardcoded behaviour is preserved.
    """

    def __init__(
        self,
        bus: MemoryBus,
        poll_interval: int = 3600,
    ) -> None:
        self._bus = bus
        self._models: dict[str, DiscoveredModel] = {}
        self._health: dict[str, ModelHealthMetrics] = {}
        self._tier_assignments: dict[str, TierAssignment] = {}
        self._poll_interval = poll_interval
        self._poll_task: asyncio.Task[None] | None = None
        self._user_overrides: dict[str, str] = {}  # tier → model_id (manual pins)

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Run initial discovery then start background polling."""
        await self.discover_all()
        await self.reassign_tiers()
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info(
            "discovery.started models=%d providers=%d poll=%ds",
            len(self._models),
            len(self._configured_providers()),
            self._poll_interval,
        )

    async def stop(self) -> None:
        """Cancel background polling."""
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        log.info("discovery.stopped")

    # ── Discovery ─────────────────────────────────────────────────────

    async def discover_all(self) -> dict[str, list[DiscoveredModel]]:
        """Poll all configured providers in parallel.

        Returns mapping of provider → discovered models.
        """
        providers = self._configured_providers()
        if not providers:
            log.debug("discovery.no_providers_configured")
            return {}

        tasks: dict[str, asyncio.Task[list[DiscoveredModel]]] = {}
        for name, (env_vars, fetcher) in providers.items():
            keys = [os.environ[v] for v in env_vars]
            tasks[name] = asyncio.create_task(self._safe_fetch(name, fetcher, keys))

        results: dict[str, list[DiscoveredModel]] = {}
        for name, task in tasks.items():
            models = await task
            results[name] = models
            for m in models:
                existing = self._models.get(m.model_id)
                if existing:
                    m.discovered_at = existing.discovered_at
                self._models[m.model_id] = m

        # Mark models not returned by their provider as gone
        seen_ids = {m.model_id for ms in results.values() for m in ms}
        for model_id, model in self._models.items():
            if model.provider in results and model_id not in seen_ids:
                model.status = "gone"

        total = sum(len(ms) for ms in results.values())
        log.info(
            "Discovered %d free models from %d providers",
            total,
            len(results),
        )

        # Publish bus event
        self._bus.publish(Envelope(
            topic="models.discovered",
            source_service_id="model_discovery",
            payload={
                "total_models": total,
                "providers": {k: len(v) for k, v in results.items()},
            },
        ))

        return results

    async def discover_provider(self, provider: str) -> list[DiscoveredModel]:
        """Poll a single provider and update internal state."""
        entry = PROVIDER_FETCHERS.get(provider)
        if not entry:
            log.warning("discovery.unknown_provider name=%s", provider)
            return []

        env_vars, fetcher = entry
        keys = []
        for v in env_vars:
            val = os.environ.get(v)
            if not val:
                log.debug("discovery.missing_key provider=%s var=%s", provider, v)
                return []
            keys.append(val)

        models = await self._safe_fetch(provider, fetcher, keys)

        seen_ids = {m.model_id for m in models}
        for m in models:
            existing = self._models.get(m.model_id)
            if existing:
                m.discovered_at = existing.discovered_at
            self._models[m.model_id] = m

        # Mark gone
        for model_id, model in self._models.items():
            if model.provider == provider and model_id not in seen_ids:
                model.status = "gone"

        return models

    # ── Queries ────────────────────────────────────────────────────────

    def get_available_models(
        self,
        tier: str | None = None,
        free_only: bool = True,
    ) -> list[DiscoveredModel]:
        """Return active models, optionally filtered by tier."""
        result = []
        for m in self._models.values():
            if m.status == "gone":
                continue
            if free_only and not m.is_free:
                continue
            if tier and m.quality_tier != tier:
                continue
            result.append(m)
        return result

    def get_model(self, model_id: str) -> DiscoveredModel | None:
        """Lookup a single discovered model by ID."""
        m = self._models.get(model_id)
        if m and m.status != "gone":
            return m
        return None

    def get_tier_assignment(self, tier: str) -> TierAssignment | None:
        """Get current tier → model mapping."""
        return self._tier_assignments.get(tier)

    # ── Tier assignment ───────────────────────────────────────────────

    async def reassign_tiers(self) -> None:
        """Re-evaluate tier assignments based on health, quality, and overrides."""
        old_assignments = dict(self._tier_assignments)

        for tier in ("fast", "balanced", "powerful"):
            # 1. User overrides always win
            if tier in self._user_overrides:
                override_id = self._user_overrides[tier]
                self._tier_assignments[tier] = TierAssignment(
                    tier=tier,
                    primary=override_id,
                    fallbacks=[],
                    reason=f"user override → {override_id}",
                    auto_assigned=False,
                )
                continue

            # 2. Get candidates matching this tier
            candidates = [
                m for m in self._models.values()
                if m.status != "gone" and m.quality_tier == tier and m.is_free
            ]

            if not candidates:
                continue

            # 3. Sort by health (fewer errors first), then context length
            def _score(m: DiscoveredModel) -> tuple[float, int]:
                h = self._health.get(m.model_id)
                error_rate = 0.0
                if h and h.total_calls >= 3:
                    error_rate = h.error_count / h.total_calls
                return (error_rate, -m.context_length)

            candidates.sort(key=_score)

            primary = candidates[0]
            fallbacks = [c.model_id for c in candidates[1:5]]

            self._tier_assignments[tier] = TierAssignment(
                tier=tier,
                primary=primary.model_id,
                fallbacks=fallbacks,
                reason=f"auto: best of {len(candidates)} candidates",
                auto_assigned=True,
            )

        # Publish event if assignments changed
        if self._tier_assignments != old_assignments:
            self._bus.publish(Envelope(
                topic="models.tiers_updated",
                source_service_id="model_discovery",
                payload={
                    tier: {
                        "primary": a.primary,
                        "fallbacks": a.fallbacks,
                        "reason": a.reason,
                    }
                    for tier, a in self._tier_assignments.items()
                },
            ))
            log.info(
                "discovery.tiers_updated %s",
                {t: a.primary for t, a in self._tier_assignments.items()},
            )

    # ── Health tracking ───────────────────────────────────────────────

    def record_call(
        self,
        model_id: str,
        latency_ms: float,
        success: bool,
        error: str = "",
    ) -> None:
        """Update health metrics for a model after an LLM call."""
        h = self._health.get(model_id)
        if not h:
            h = ModelHealthMetrics(model_id=model_id)
            self._health[model_id] = h

        h.total_calls += 1
        if success:
            h.success_count += 1
        else:
            h.error_count += 1
            h.last_error = error
            h.last_error_at = datetime.now()

        # Update latency stats
        latencies = h._latencies
        latencies.append(latency_ms)
        # Keep only last 100 measurements
        if len(latencies) > 100:
            h._latencies = latencies[-100:]
            latencies = h._latencies

        h.avg_latency_ms = sum(latencies) / len(latencies)
        if len(latencies) >= 5:
            sorted_lats = sorted(latencies)
            idx = int(len(sorted_lats) * 0.95)
            h.p95_latency_ms = sorted_lats[min(idx, len(sorted_lats) - 1)]

        # Mark degraded if error rate exceeds threshold in recent calls
        if h.total_calls >= _HEALTH_WINDOW_CALLS:
            recent_errors = h.error_count
            recent_total = h.total_calls
            if recent_total > 0 and recent_errors / recent_total > _ERROR_RATE_THRESHOLD:
                model = self._models.get(model_id)
                if model and model.status == "active":
                    model.status = "degraded"
                    log.warning(
                        "discovery.model_degraded model=%s error_rate=%.1f%%",
                        model_id,
                        100 * recent_errors / recent_total,
                    )
                    # Trigger async tier reassignment
                    asyncio.create_task(self.reassign_tiers())

    # ── User overrides ────────────────────────────────────────────────

    def set_user_override(self, tier: str, model_id: str | None) -> None:
        """Pin a tier to a specific model, or clear the pin."""
        if model_id is None:
            self._user_overrides.pop(tier, None)
        else:
            self._user_overrides[tier] = model_id

    # ── Internal helpers ──────────────────────────────────────────────

    def _configured_providers(self) -> dict[str, tuple[list[str], Any]]:
        """Return provider fetchers where all required env vars are set."""
        result: dict[str, tuple[list[str], Any]] = {}
        for name, (env_vars, fetcher) in PROVIDER_FETCHERS.items():
            if all(os.environ.get(v) for v in env_vars):
                result[name] = (env_vars, fetcher)
        return result

    async def _safe_fetch(
        self,
        provider: str,
        fetcher: Any,
        keys: list[str],
    ) -> list[DiscoveredModel]:
        """Call a provider fetcher, catching and logging errors."""
        try:
            return await fetcher(*keys)
        except Exception:
            log.exception("discovery.fetch_failed provider=%s", provider)
            return []

    async def _poll_loop(self) -> None:
        """Background loop: discover_all → reassign_tiers every interval."""
        while True:
            await asyncio.sleep(self._poll_interval)
            try:
                await self.discover_all()
                await self.reassign_tiers()
            except Exception:
                log.exception("discovery.poll_loop_error")

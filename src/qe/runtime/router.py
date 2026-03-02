from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from qe.models.envelope import Envelope
from qe.models.genome import ModelPreference

if TYPE_CHECKING:
    from qe.runtime.budget import BudgetTracker
    from qe.runtime.discovery.service import ModelDiscoveryService

TIER_MODELS = {
    "fast": ["gpt-4o-mini", "claude-haiku-3-5"],
    "balanced": ["gpt-4o", "claude-sonnet-3-5"],
    "powerful": ["o1-preview", "claude-opus-3"],
    "local": ["ollama/qwen3:8b", "ollama/llama3.1:8b"],
}


class AutoRouter:
    def __init__(
        self,
        preference: ModelPreference,
        budget_tracker: BudgetTracker | None = None,
        discovery: ModelDiscoveryService | None = None,
    ) -> None:
        self.preference = preference
        self.budget_tracker = budget_tracker
        self.discovery = discovery
        self._error_timestamps: dict[str, datetime] = {}

    def select(self, envelope: Envelope) -> str:
        """
        Selection algorithm (in priority order):
        1. If envelope.payload contains "force_model": use it directly
        2. If budget remaining < 10%: downgrade to "fast"
        3. If discovery available, use TierAssignment (primary + fallbacks)
        4. Otherwise fall back to hardcoded TIER_MODELS
        5. If PREFER_LOCAL env set, prefer local models
        Return: the litellm model string (e.g. "gpt-4o-mini")
        """
        self._select_start = time.monotonic()

        # 1. Escape hatch for testing
        if "force_model" in envelope.payload:
            return envelope.payload["force_model"]

        tier = self.preference.tier

        # 2. Budget gate
        if self._budget_remaining_pct() < 0.10:
            tier = "fast"

        # 3. Prefer local if requested (before discovery)
        if os.getenv("PREFER_LOCAL"):
            local = [m for m in TIER_MODELS.get("local", []) if not self._is_cooling_down(m)]
            if local:
                return local[0]

        # 4. Try discovery-based selection
        if self.discovery is not None:
            assignment = self.discovery.get_tier_assignment(tier)
            if assignment is not None:
                # Try primary, then fallbacks
                candidates = [assignment.primary, *assignment.fallbacks]
                for model_id in candidates:
                    if not self._is_cooling_down(model_id):
                        return model_id

        # 5. Fallback to hardcoded TIER_MODELS
        available = [m for m in TIER_MODELS.get(tier, []) if not self._is_cooling_down(m)]
        if not available:
            # Escalate to cheaper tier
            tiers = list(TIER_MODELS.keys())
            if tier in tiers:
                idx = tiers.index(tier)
                fallback_tier = tiers[max(0, idx - 1)]
                available = TIER_MODELS.get(fallback_tier, [])

        if not available:
            raise RuntimeError("No models available — all in cooldown and fallback empty")

        return available[0]

    def _is_cooling_down(self, model: str) -> bool:
        """
        Returns True if this model had a non-rate-limit error in the last 5 minutes.
        """
        if model not in self._error_timestamps:
            return False

        cooldown_window = datetime.now(UTC) - timedelta(minutes=5)
        return self._error_timestamps[model] > cooldown_window

    def _budget_remaining_pct(self) -> float:
        """Return budget remaining percentage from the budget tracker."""
        if self.budget_tracker is None:
            return 1.0
        return self.budget_tracker.remaining_pct()

    def record_error(self, model: str) -> None:
        """Record the current timestamp for a model error."""
        self._error_timestamps[model] = datetime.now(UTC)
        if self.discovery is not None:
            elapsed = self._elapsed_ms()
            self.discovery.record_call(model, elapsed, success=False, error="model_error")

    def record_success(self, model: str) -> None:
        """Record a successful call with latency for discovery health tracking."""
        if self.discovery is not None:
            elapsed = self._elapsed_ms()
            self.discovery.record_call(model, elapsed, success=True)

    def _elapsed_ms(self) -> float:
        """Milliseconds since last select() call."""
        start = getattr(self, "_select_start", None)
        if start is None:
            return 0.0
        return (time.monotonic() - start) * 1000

import os
from datetime import datetime, timedelta

from qe.models.genome import ModelPreference
from qe.models.envelope import Envelope


TIER_MODELS = {
    "fast": ["gpt-4o-mini", "claude-haiku-3-5"],
    "balanced": ["gpt-4o", "claude-sonnet-3-5"],
    "powerful": ["o1-preview", "claude-opus-3"],
    "local": ["ollama/llama3.2", "ollama/mistral"],
}


class AutoRouter:
    def __init__(self, preference: ModelPreference) -> None:
        self.preference = preference
        self._error_timestamps: dict[str, datetime] = {}

    def select(self, envelope: Envelope) -> str:
        """
        Selection algorithm (in priority order):
        1. If envelope.payload contains "force_model": use it directly
        2. If budget remaining < 10%: downgrade to "fast"
        3. If preferred tier has no available models: try next tier up (cheaper)
        4. From available models, select lowest cost
        5. If PREFER_LOCAL env set, prefer local models
        Return: the litellm model string (e.g. "gpt-4o-mini")
        """
        # 1. Escape hatch for testing
        if "force_model" in envelope.payload:
            return envelope.payload["force_model"]

        tier = self.preference.tier

        # 2. Budget gate
        if self._budget_remaining_pct() < 0.10:
            tier = "fast"

        # 3. Availability check
        available = [m for m in TIER_MODELS[tier] if not self._is_cooling_down(m)]
        if not available:
            # Escalate to cheaper tier
            tiers = list(TIER_MODELS.keys())
            idx = tiers.index(tier)
            fallback_tier = tiers[max(0, idx - 1)]  # go cheaper, not more expensive
            available = TIER_MODELS[fallback_tier]

        if not available:
            raise RuntimeError("No models available — all in cooldown and fallback empty")

        # 4. Prefer local if requested
        if os.getenv("PREFER_LOCAL"):
            local = [m for m in available if m.startswith("ollama/")]
            if local:
                return local[0]

        # 5. Default: first in list (cheapest by convention)
        return available[0]

    def _is_cooling_down(self, model: str) -> bool:
        """
        Returns True if this model had a non-rate-limit error in the last 5 minutes.
        """
        if model not in self._error_timestamps:
            return False

        cooldown_window = datetime.utcnow() - timedelta(minutes=5)
        return self._error_timestamps[model] > cooldown_window

    def _budget_remaining_pct(self) -> float:
        """
        Return budget remaining percentage.
        TODO Phase 1: implement real budget tracking via litellm.
        """
        # TODO Phase 1: implement real budget tracking via litellm
        return 1.0

    def record_error(self, model: str) -> None:
        """Record the current timestamp for a model error."""
        self._error_timestamps[model] = datetime.utcnow()

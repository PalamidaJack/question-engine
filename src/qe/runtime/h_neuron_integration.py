"""Integration of H-Neuron signals with verification and calibration."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qe.runtime.h_neurons import _DEFAULT_PROFILE_DIR, HNeuronProfiler

log = logging.getLogger(__name__)

# Risk thresholds for mapping continuous risk to discrete verification profiles
_HIGH_RISK_THRESHOLD = 0.7
_MODERATE_RISK_THRESHOLD = 0.4


class HNeuronIntegration:
    """Integrates H-Neuron hallucination risk signals with the verification
    and calibration subsystems.

    This acts as the bridge between low-level H-Neuron monitoring (which requires
    torch and a local model) and the rest of the Question Engine pipeline (which
    operates on dictionaries and risk scores).
    """

    def __init__(self, profile_dir: str = _DEFAULT_PROFILE_DIR) -> None:
        self._profile_dir = Path(profile_dir)
        self._profiler = HNeuronProfiler(profile_dir)
        self._torch_available: bool | None = None

    def is_available(self) -> bool:
        """Check whether H-Neuron integration is usable.

        Returns True only if:
        1. torch is importable, AND
        2. at least one cached H-Neuron profile exists.
        """
        if self._torch_available is None:
            try:
                import torch  # type: ignore[import-untyped]  # noqa: F401

                self._torch_available = True
            except ImportError:
                self._torch_available = False

        if not self._torch_available:
            return False

        profiles = self._profiler.list_profiles()
        return len(profiles) > 0

    def get_risk_adjusted_profile(self, risk_score: float) -> str:
        """Map a continuous hallucination risk score to a verification profile name.

        Args:
            risk_score: Float between 0.0 and 1.0 from ``HNeuronMonitor.get_hallucination_risk()``.

        Returns:
            One of ``"high_risk"``, ``"moderate_risk"``, or ``"standard"``.
        """
        risk_score = max(0.0, min(1.0, risk_score))

        if risk_score >= _HIGH_RISK_THRESHOLD:
            return "high_risk"
        elif risk_score >= _MODERATE_RISK_THRESHOLD:
            return "moderate_risk"
        else:
            return "standard"

    async def enhance_verification(
        self,
        subtask_result: dict[str, Any],
        risk_score: float,
    ) -> dict[str, Any]:
        """Enhance a subtask result dict with H-Neuron risk metadata.

        This is called after a local model produces a subtask result but before
        verification is run.  It adds:
        - ``h_neuron_risk``: the raw risk score (0.0 - 1.0)
        - ``verification_profile``: the risk-adjusted verification profile name
        - ``h_neuron_timestamp``: ISO timestamp of when the risk was assessed
        - Optionally adjusts ``confidence`` downward if risk is high

        Args:
            subtask_result: The subtask result dictionary to enhance.
            risk_score: Hallucination risk from the monitor.

        Returns:
            The enhanced subtask result dictionary (mutated in place and returned).
        """
        risk_score = max(0.0, min(1.0, risk_score))
        profile_name = self.get_risk_adjusted_profile(risk_score)

        subtask_result["h_neuron_risk"] = risk_score
        subtask_result["verification_profile"] = profile_name
        subtask_result["h_neuron_timestamp"] = datetime.now(UTC).isoformat()

        # If the subtask has a confidence value, adjust it downward proportionally to risk
        existing_confidence = subtask_result.get("confidence")
        if existing_confidence is not None:
            try:
                conf = float(existing_confidence)
                # Reduce confidence based on risk: high risk -> larger reduction
                # Formula: adjusted = original * (1 - risk * penalty_weight)
                # penalty_weight = 0.5 means at risk=1.0, confidence halved
                penalty_weight = 0.5
                adjusted = conf * (1.0 - risk_score * penalty_weight)
                subtask_result["confidence"] = round(max(0.0, adjusted), 4)
                subtask_result["confidence_original"] = conf
                log.debug(
                    "h_neuron_integration.confidence_adjusted "
                    "original=%.4f adjusted=%.4f risk=%.4f profile=%s",
                    conf,
                    subtask_result["confidence"],
                    risk_score,
                    profile_name,
                )
            except (TypeError, ValueError):
                log.warning(
                    "h_neuron_integration.confidence_not_numeric value=%r",
                    existing_confidence,
                )

        # Flag for additional verification if risk is elevated
        if profile_name == "high_risk":
            subtask_result.setdefault("flags", [])
            if isinstance(subtask_result["flags"], list):
                subtask_result["flags"].append("h_neuron_high_risk")
            subtask_result["requires_additional_verification"] = True
            log.info(
                "h_neuron_integration.high_risk_flagged risk=%.4f",
                risk_score,
            )
        elif profile_name == "moderate_risk":
            subtask_result.setdefault("flags", [])
            if isinstance(subtask_result["flags"], list):
                subtask_result["flags"].append("h_neuron_moderate_risk")

        return subtask_result

    def get_calibration_features(self, risk_score: float) -> dict[str, Any]:
        """Return a feature dictionary suitable for the calibration tracker.

        These features can be incorporated into calibration models to improve
        confidence estimates when H-Neuron data is available.

        Args:
            risk_score: Hallucination risk score from the monitor.

        Returns:
            Dictionary of calibration features.
        """
        risk_score = max(0.0, min(1.0, risk_score))
        profile_name = self.get_risk_adjusted_profile(risk_score)

        return {
            "h_neuron_risk": risk_score,
            "h_neuron_profile": profile_name,
            "h_neuron_risk_bucket": _risk_bucket(risk_score),
            "h_neuron_is_high_risk": risk_score >= _HIGH_RISK_THRESHOLD,
            "h_neuron_is_moderate_risk": (
                _MODERATE_RISK_THRESHOLD <= risk_score < _HIGH_RISK_THRESHOLD
            ),
            "h_neuron_available": True,
        }


def _risk_bucket(risk: float) -> int:
    """Map a risk score to one of 10 buckets (0-9)."""
    return min(int(risk * 10), 9)

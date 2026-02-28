"""Conformal prediction: distribution-free uncertainty quantification."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class PredictionSet:
    """A prediction set with coverage guarantees."""

    items: list[Any] = field(default_factory=list)
    coverage_target: float = 0.90
    effective_coverage: float = 0.0
    threshold: float = 0.0


class ConformalPredictor:
    """Split-conformal predictor with finite-sample coverage guarantees.

    Uses nonconformity scores derived from a calibration set to construct
    prediction sets that contain the true answer with probability >= coverage.
    """

    def __init__(self, coverage: float = 0.90) -> None:
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"Coverage must be in (0, 1), got {coverage}")
        self._coverage = coverage
        self._scores: list[float] = []
        self._threshold: float | None = None

    def calibrate(self, calibration_set: list[tuple[float, bool]]) -> None:
        """Compute nonconformity scores and quantile threshold.

        Args:
            calibration_set: list of (confidence_score, was_correct) pairs.
                confidence_score is the model's reported confidence in [0, 1].
                was_correct indicates whether the prediction was actually correct.

        The nonconformity score measures how "non-conforming" a prediction is:
        - For correct predictions: 1 - confidence  (low confidence = high nonconformity)
        - For incorrect predictions: confidence     (high confidence = high nonconformity)
        """
        if not calibration_set:
            raise ValueError("Calibration set must not be empty")

        self._scores = []
        for confidence, was_correct in calibration_set:
            if was_correct:
                self._scores.append(1.0 - confidence)
            else:
                self._scores.append(confidence)

        self._scores.sort()

        n = len(self._scores)
        # Quantile level: ceil((n+1) * coverage) / n
        quantile_index = math.ceil((n + 1) * self._coverage) / n
        # Clamp to [0, 1] for index computation
        quantile_index = min(quantile_index, 1.0)
        idx = min(int(math.ceil(quantile_index * n)) - 1, n - 1)
        idx = max(idx, 0)

        self._threshold = self._scores[idx]
        log.info(
            "Conformal calibration complete: n=%d, threshold=%.4f, coverage=%.2f",
            n,
            self._threshold,
            self._coverage,
        )

    def prediction_set(
        self,
        outputs: list[Any],
        confidence_scores: list[float],
    ) -> PredictionSet:
        """Build a prediction set from candidate outputs and their confidence scores.

        Includes all outputs whose nonconformity score (1 - confidence) falls
        below the calibrated threshold.

        Args:
            outputs: candidate predictions/answers.
            confidence_scores: model confidence for each output, in [0, 1].

        Returns:
            PredictionSet containing the included items and coverage metadata.
        """
        if self._threshold is None:
            raise RuntimeError("Must call calibrate() before prediction_set()")

        if len(outputs) != len(confidence_scores):
            raise ValueError(
                f"outputs ({len(outputs)}) and confidence_scores "
                f"({len(confidence_scores)}) must have the same length"
            )

        included: list[Any] = []
        for output, score in zip(outputs, confidence_scores, strict=True):
            nonconformity = 1.0 - score
            if nonconformity <= self._threshold:
                included.append(output)

        # Effective coverage: fraction of calibration scores at or below threshold
        n_below = sum(1 for s in self._scores if s <= self._threshold)
        effective = n_below / len(self._scores) if self._scores else 0.0

        return PredictionSet(
            items=included,
            coverage_target=self._coverage,
            effective_coverage=effective,
            threshold=self._threshold,
        )

    def is_calibrated(self) -> bool:
        """Return True if the predictor has been calibrated."""
        return self._threshold is not None

    @staticmethod
    def required_calibration_size(
        coverage: float = 0.90,
        error_tolerance: float = 0.05,
    ) -> int:
        """Estimate the minimum calibration set size for desired coverage and tolerance.

        Uses the relation n >= ceil(1 / (2 * error_tolerance^2) * ln(2 / (1 - coverage)))
        derived from Hoeffding's inequality for finite-sample coverage guarantees.

        Args:
            coverage: target coverage probability (e.g. 0.90).
            error_tolerance: acceptable deviation from target coverage.

        Returns:
            Minimum number of calibration examples needed.
        """
        if not 0.0 < coverage < 1.0:
            raise ValueError(f"Coverage must be in (0, 1), got {coverage}")
        if not 0.0 < error_tolerance < 1.0:
            raise ValueError(f"Error tolerance must be in (0, 1), got {error_tolerance}")

        # Hoeffding-based bound
        n = math.ceil(math.log(2.0 / (1.0 - coverage)) / (2.0 * error_tolerance**2))
        return max(n, 1)

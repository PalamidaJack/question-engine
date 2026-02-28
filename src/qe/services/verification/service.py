"""Verification service: validates subtask outputs before acceptance."""

from __future__ import annotations

import logging
import statistics
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class CheckResult(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


class VerificationReport(BaseModel):
    """Detailed verification report for a subtask output."""

    subtask_id: str
    goal_id: str
    overall: CheckResult = CheckResult.PASS
    structural: CheckResult = CheckResult.SKIP
    contract: CheckResult = CheckResult.SKIP
    anomaly: CheckResult = CheckResult.SKIP
    belief: CheckResult = CheckResult.SKIP
    adversarial: CheckResult = CheckResult.SKIP
    details: list[str] = Field(default_factory=list)
    confidence_adjustment: float = 0.0


# Verification strictness by model tier
VERIFICATION_PROFILES: dict[str, dict[str, bool]] = {
    "powerful": {
        "structural_checks": True,
        "contract_checks": True,
        "anomaly_checks": True,
        "belief_checks": True,
        "adversarial_verification": False,
    },
    "balanced": {
        "structural_checks": True,
        "contract_checks": True,
        "anomaly_checks": True,
        "belief_checks": True,
        "adversarial_verification": False,
    },
    "fast": {
        "structural_checks": True,
        "contract_checks": True,
        "anomaly_checks": True,
        "belief_checks": True,
        "adversarial_verification": True,
    },
    "local": {
        "structural_checks": True,
        "contract_checks": True,
        "anomaly_checks": True,
        "belief_checks": True,
        "adversarial_verification": True,
    },
}


class VerificationService:
    """Validates subtask outputs through a multi-layer checking pipeline."""

    def __init__(self, substrate: Any = None) -> None:
        self.substrate = substrate
        # Historical distributions for anomaly detection
        self._token_history: dict[str, list[int]] = {}
        self._latency_history: dict[str, list[int]] = {}
        self._confidence_history: dict[str, list[float]] = {}

    async def verify(
        self,
        subtask_id: str,
        goal_id: str,
        output: dict[str, Any],
        contract: dict[str, Any] | None = None,
        model_tier: str = "balanced",
    ) -> VerificationReport:
        """Run the full verification pipeline on a subtask output."""
        profile = VERIFICATION_PROFILES.get(model_tier, VERIFICATION_PROFILES["balanced"])
        report = VerificationReport(subtask_id=subtask_id, goal_id=goal_id)

        # Layer 1: Structural checks
        if profile["structural_checks"]:
            report.structural = self._check_structural(output, report)

        # Layer 2: Contract checks
        if profile["contract_checks"] and contract:
            report.contract = self._check_contract(output, contract, report)

        # Layer 3: Anomaly checks
        if profile["anomaly_checks"]:
            report.anomaly = self._check_anomaly(
                output, model_tier, report
            )

        # Determine overall result
        checks = [report.structural, report.contract, report.anomaly]
        if CheckResult.FAIL in checks:
            report.overall = CheckResult.FAIL
        elif CheckResult.WARN in checks:
            report.overall = CheckResult.WARN

        log.info(
            "verification.done subtask=%s goal=%s result=%s checks=[s=%s c=%s a=%s]",
            subtask_id,
            goal_id,
            report.overall,
            report.structural,
            report.contract,
            report.anomaly,
        )

        return report

    def _check_structural(
        self, output: dict[str, Any], report: VerificationReport
    ) -> CheckResult:
        """Structural checks: output is well-formed."""
        if not output:
            report.details.append("structural: output is empty")
            return CheckResult.FAIL

        # Check for required fields
        if "result" in output and output["result"] is None:
            report.details.append("structural: result field is null")
            return CheckResult.FAIL

        # Check for non-empty text content
        text_fields = [v for v in output.values() if isinstance(v, str)]
        if text_fields and all(not t.strip() for t in text_fields):
            report.details.append("structural: all text fields are empty")
            return CheckResult.FAIL

        return CheckResult.PASS

    def _check_contract(
        self,
        output: dict[str, Any],
        contract: dict[str, Any],
        report: VerificationReport,
    ) -> CheckResult:
        """Contract checks: postconditions are met."""
        postconditions = contract.get("postconditions", [])
        if not postconditions:
            return CheckResult.SKIP

        result = CheckResult.PASS
        for condition in postconditions:
            if not self._evaluate_postcondition(condition, output):
                report.details.append(f"contract: failed postcondition: {condition}")
                result = CheckResult.FAIL

        return result

    def _evaluate_postcondition(
        self, condition: str, output: dict[str, Any]
    ) -> bool:
        """Safely evaluate a postcondition against the output."""
        try:
            # Simple condition patterns
            if ">=" in condition:
                field, threshold = condition.split(">=")
                field = field.strip()
                threshold = float(threshold.strip())
                value = self._resolve_field(field, output)
                if value is None:
                    return False
                return float(value) >= threshold

            if "==" in condition:
                field, expected = condition.split("==")
                field = field.strip()
                expected = expected.strip()
                value = self._resolve_field(field, output)
                return str(value) == expected

            if "<=" in condition:
                field, threshold = condition.split("<=")
                field = field.strip()
                threshold = float(threshold.strip())
                value = self._resolve_field(field, output)
                if value is None:
                    return False
                return float(value) <= threshold

        except (ValueError, TypeError, AttributeError):
            return False

        # Unknown condition format — pass by default
        return True

    def _resolve_field(self, field_path: str, data: dict) -> Any:
        """Resolve a dotted field path against a dict."""
        parts = field_path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    if part == "length":
                        return len(current)
                    return None
            else:
                return None
        return current

    def _check_anomaly(
        self,
        output: dict[str, Any],
        model_tier: str,
        report: VerificationReport,
    ) -> CheckResult:
        """Statistical anomaly detection based on historical distributions."""
        key = model_tier

        # Track confidence if present
        confidence = output.get("confidence")
        if confidence is not None:
            self._confidence_history.setdefault(key, []).append(float(confidence))
            history = self._confidence_history[key]
            if len(history) >= 10:
                mean = statistics.mean(history)
                stdev = statistics.stdev(history)
                if stdev > 0 and abs(float(confidence) - mean) > 2 * stdev:
                    report.details.append(
                        f"anomaly: confidence {confidence:.2f} is >2σ from mean "
                        f"{mean:.2f} (σ={stdev:.2f})"
                    )
                    return CheckResult.WARN

        # Track output size
        output_text = str(output)
        self._token_history.setdefault(key, []).append(len(output_text))
        history = self._token_history[key]
        if len(history) >= 10:
            mean = statistics.mean(history)
            stdev = statistics.stdev(history)
            if stdev > 0 and abs(len(output_text) - mean) > 2 * stdev:
                report.details.append(
                    f"anomaly: output size {len(output_text)} is >2σ from mean "
                    f"{mean:.0f}"
                )
                return CheckResult.WARN

        return CheckResult.PASS

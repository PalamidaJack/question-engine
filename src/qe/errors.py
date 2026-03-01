"""Structured error taxonomy for Question Engine.

Every error in QE carries a machine-readable code, severity, and
retryability flag so that recovery logic can make automated decisions.

Error code format: QE_<DOMAIN>_<ISSUE>
Domains: LLM, BUS, CONFIG, SECURITY, SUBSTRATE, GOAL, API
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    CRITICAL = "critical"  # system cannot continue
    ERROR = "error"  # operation failed
    WARN = "warn"  # degraded but operational


class ErrorDomain(StrEnum):
    LLM = "LLM"
    BUS = "BUS"
    CONFIG = "CONFIG"
    SECURITY = "SECURITY"
    SUBSTRATE = "SUBSTRATE"
    GOAL = "GOAL"
    API = "API"
    INQUIRY = "INQUIRY"


# ── Base exception ─────────────────────────────────────────────────────────


class QEError(Exception):
    """Base exception for all Question Engine errors.

    Carries structured metadata for automated recovery decisions.
    """

    code: str = "QE_UNKNOWN"
    domain: ErrorDomain = ErrorDomain.API
    severity: Severity = Severity.ERROR
    is_retryable: bool = False
    retry_delay_ms: int = 0

    def __init__(
        self,
        message: str = "",
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.code
        self.context: dict[str, Any] = context or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "domain": self.domain.value,
            "severity": self.severity.value,
            "message": self.message,
            "is_retryable": self.is_retryable,
            "retry_delay_ms": self.retry_delay_ms,
            "context": self.context,
        }


# ── LLM errors ─────────────────────────────────────────────────────────────


class LLMRateLimitError(QEError):
    code = "QE_LLM_RATE_LIMIT"
    domain = ErrorDomain.LLM
    severity = Severity.WARN
    is_retryable = True
    retry_delay_ms = 5000


class LLMTimeoutError(QEError):
    code = "QE_LLM_TIMEOUT"
    domain = ErrorDomain.LLM
    severity = Severity.WARN
    is_retryable = True
    retry_delay_ms = 2000


class LLMTokenLimitError(QEError):
    code = "QE_LLM_TOKEN_LIMIT"
    domain = ErrorDomain.LLM
    severity = Severity.ERROR
    is_retryable = False


class LLMProviderError(QEError):
    code = "QE_LLM_PROVIDER_DOWN"
    domain = ErrorDomain.LLM
    severity = Severity.ERROR
    is_retryable = True
    retry_delay_ms = 10000


class LLMParseError(QEError):
    code = "QE_LLM_PARSE_FAILED"
    domain = ErrorDomain.LLM
    severity = Severity.ERROR
    is_retryable = True
    retry_delay_ms = 1000


# ── Bus errors ─────────────────────────────────────────────────────────────


class BusHandlerError(QEError):
    code = "QE_BUS_HANDLER_FAILED"
    domain = ErrorDomain.BUS
    severity = Severity.ERROR
    is_retryable = True
    retry_delay_ms = 100


class BusDeliveryError(QEError):
    code = "QE_BUS_DELIVERY_FAILED"
    domain = ErrorDomain.BUS
    severity = Severity.ERROR
    is_retryable = False


class BusTopicError(QEError):
    code = "QE_BUS_INVALID_TOPIC"
    domain = ErrorDomain.BUS
    severity = Severity.ERROR
    is_retryable = False


# ── Config errors ──────────────────────────────────────────────────────────


class ConfigValidationError(QEError):
    code = "QE_CONFIG_INVALID"
    domain = ErrorDomain.CONFIG
    severity = Severity.CRITICAL
    is_retryable = False


class ConfigMissingError(QEError):
    code = "QE_CONFIG_MISSING"
    domain = ErrorDomain.CONFIG
    severity = Severity.CRITICAL
    is_retryable = False


# ── Security errors ────────────────────────────────────────────────────────


class AuthenticationError(QEError):
    code = "QE_SECURITY_AUTH_FAILED"
    domain = ErrorDomain.SECURITY
    severity = Severity.WARN
    is_retryable = False


class AuthorizationError(QEError):
    code = "QE_SECURITY_FORBIDDEN"
    domain = ErrorDomain.SECURITY
    severity = Severity.WARN
    is_retryable = False


class GuardrailTripError(QEError):
    code = "QE_SECURITY_GUARDRAIL_TRIPPED"
    domain = ErrorDomain.SECURITY
    severity = Severity.WARN
    is_retryable = False


# ── Substrate errors ───────────────────────────────────────────────────────


class SubstrateConnectionError(QEError):
    code = "QE_SUBSTRATE_CONNECTION"
    domain = ErrorDomain.SUBSTRATE
    severity = Severity.CRITICAL
    is_retryable = True
    retry_delay_ms = 5000


class SubstrateQueryError(QEError):
    code = "QE_SUBSTRATE_QUERY"
    domain = ErrorDomain.SUBSTRATE
    severity = Severity.ERROR
    is_retryable = True
    retry_delay_ms = 1000


# ── Goal errors ────────────────────────────────────────────────────────────


class GoalBudgetExceededError(QEError):
    code = "QE_GOAL_BUDGET_EXCEEDED"
    domain = ErrorDomain.GOAL
    severity = Severity.WARN
    is_retryable = False


class GoalInvalidTransitionError(QEError):
    code = "QE_GOAL_INVALID_TRANSITION"
    domain = ErrorDomain.GOAL
    severity = Severity.ERROR
    is_retryable = False


class GoalTimeoutError(QEError):
    code = "QE_GOAL_TIMEOUT"
    domain = ErrorDomain.GOAL
    severity = Severity.WARN
    is_retryable = False


# ── API errors ─────────────────────────────────────────────────────────────


class APIValidationError(QEError):
    code = "QE_API_VALIDATION"
    domain = ErrorDomain.API
    severity = Severity.WARN
    is_retryable = False


class APINotReadyError(QEError):
    code = "QE_API_NOT_READY"
    domain = ErrorDomain.API
    severity = Severity.ERROR
    is_retryable = True
    retry_delay_ms = 5000


# ── Inquiry errors ────────────────────────────────────────────────────────


class InquiryPhaseError(QEError):
    code = "QE_INQUIRY_PHASE_FAILED"
    domain = ErrorDomain.INQUIRY
    severity = Severity.ERROR
    is_retryable = True
    retry_delay_ms = 1000


class InquiryConfigError(QEError):
    code = "QE_INQUIRY_CONFIG_INVALID"
    domain = ErrorDomain.INQUIRY
    severity = Severity.ERROR
    is_retryable = False


class InquiryTimeoutError(QEError):
    code = "QE_INQUIRY_TIMEOUT"
    domain = ErrorDomain.INQUIRY
    severity = Severity.WARN
    is_retryable = False


# ── Error classification helper ────────────────────────────────────────────

_RETRYABLE_KEYWORDS = {
    "timeout": LLMTimeoutError,
    "rate_limit": LLMRateLimitError,
    "rate limit": LLMRateLimitError,
    "429": LLMRateLimitError,
    "503": LLMProviderError,
    "504": LLMProviderError,
    "connection": SubstrateConnectionError,
    "token": LLMTokenLimitError,
    "context length": LLMTokenLimitError,
    "parse": LLMParseError,
    "json": LLMParseError,
    "validation": LLMParseError,
}


def classify_error(exc: Exception) -> QEError:
    """Classify a raw exception into a structured QEError.

    Uses keyword matching on the error message to determine the most
    appropriate error type. Falls back to QEError if no match.
    """
    if isinstance(exc, QEError):
        return exc

    msg = str(exc).lower()
    for keyword, error_cls in _RETRYABLE_KEYWORDS.items():
        if keyword in msg:
            return error_cls(str(exc))

    return QEError(str(exc))

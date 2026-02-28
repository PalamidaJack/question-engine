"""Input sanitization for prompt injection detection."""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

log = logging.getLogger(__name__)


class SanitizeResult(NamedTuple):
    """Result of input sanitization."""

    text: str
    risk_score: float  # 0.0 = clean, 1.0 = high risk
    matches: list[str]  # matched pattern names


# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS: list[tuple[str, str, float]] = [
    # (name, regex_pattern, risk_weight)
    ("ignore_instructions", r"ignore\s+(all\s+)?previous\s+instructions", 0.9),
    ("role_switch", r"you\s+are\s+now\s+", 0.8),
    ("system_tag", r"<\|.*?\|>", 0.7),
    ("system_block", r"```system", 0.7),
    ("system_colon", r"^system:\s*", 0.6),
    ("new_instructions", r"new\s+instructions?:\s*", 0.8),
    ("forget_everything", r"forget\s+(everything|all)", 0.9),
    ("pretend_to_be", r"pretend\s+(to\s+be|you\s+are)", 0.7),
    ("jailbreak", r"(jailbreak|DAN|do\s+anything\s+now)", 0.9),
    ("override", r"override\s+(your|system|safety)", 0.8),
    ("act_as", r"act\s+as\s+(if|a|an)\s+", 0.5),
    ("developer_mode", r"developer\s+mode", 0.7),
    ("hidden_prompt", r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", 0.8),
]

_COMPILED_PATTERNS = [
    (name, re.compile(pattern, re.IGNORECASE | re.MULTILINE), weight)
    for name, pattern, weight in _INJECTION_PATTERNS
]


class InputSanitizer:
    """Detects and neutralizes prompt injection in input payloads.

    Applied at system boundaries (API endpoints, CLI commands)
    before any text reaches an LLM.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def sanitize(self, text: str) -> SanitizeResult:
        """Scan text for injection patterns.

        Returns the original text with a risk score and matched patterns.
        Text is NOT modified — callers decide how to handle high risk.
        """
        matches: list[str] = []
        max_risk = 0.0

        for name, pattern, weight in _COMPILED_PATTERNS:
            if pattern.search(text):
                matches.append(name)
                max_risk = max(max_risk, weight)

        if matches:
            log.warning(
                "sanitizer.detected patterns=%s risk=%.2f text_preview=%s",
                matches,
                max_risk,
                text[:100],
            )

        return SanitizeResult(text=text, risk_score=max_risk, matches=matches)

    def wrap_untrusted(self, text: str) -> str:
        """Wrap untrusted content in delimiters.

        The system prompt instructs the LLM to treat content between
        these delimiters as data, not instructions.
        """
        return (
            "[UNTRUSTED_CONTENT_START]\n"
            f"{text}\n"
            "[UNTRUSTED_CONTENT_END]"
        )

    def is_safe(self, text: str) -> bool:
        """Quick check if text is below the risk threshold."""
        result = self.sanitize(text)
        return result.risk_score < self.threshold

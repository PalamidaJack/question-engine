"""Guardrails pipeline: input/output checks and built-in rules.

Provides a simple, extensible pipeline of GuardrailRule objects. The
implementation here is intentionally conservative and test-friendly, using
synchronous checks and small async wrappers where needed.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel

log = logging.getLogger(__name__)


class GuardrailResult(BaseModel):
    passed: bool
    rule_name: str
    message: str = ""
    severity: str = "info"  # info | warning | block


class GuardrailRule:
    name: str = "base"
    enabled: bool = True

    async def check(  # type: ignore[override]
        self, content: str, context: dict[str, Any],
    ) -> GuardrailResult:
        return GuardrailResult(passed=True, rule_name=self.name)


# Built-in rules
class ContentFilterRule(GuardrailRule):
    name = "ContentFilter"

    def __init__(self, patterns: list[str] | None = None):
        self._patterns = patterns or [r"drop table", r"\bexec\b", r"<script>"]

    async def check(self, content: str, context: dict[str, Any]) -> GuardrailResult:
        for p in self._patterns:
            if re.search(p, content, re.IGNORECASE):
                return GuardrailResult(
                    passed=False, rule_name=self.name,
                    message=f"pattern matched: {p}",
                    severity="block",
                )
        return GuardrailResult(passed=True, rule_name=self.name)


class PiiDetectorRule(GuardrailRule):
    name = "PiiDetector"

    def __init__(self):
        # simple regexes: email, phone, ssn-like
        self._email = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}")
        self._phone = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
        self._ssn = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    async def check(self, content: str, context: dict[str, Any]) -> GuardrailResult:
        if self._email.search(content) or self._phone.search(content) or self._ssn.search(content):
            return GuardrailResult(
                passed=False, rule_name=self.name,
                message="PII detected", severity="warning",
            )
        return GuardrailResult(passed=True, rule_name=self.name)


class CostGuardRule(GuardrailRule):
    name = "CostGuard"

    def __init__(self, threshold_usd: float = 5.0):
        self._threshold = threshold_usd

    async def check(self, content: str, context: dict[str, Any]) -> GuardrailResult:
        # budget tracker lives in context optionally;
        # we simulate by checking context.get('estimated_cost')
        est = float(context.get("estimated_cost_usd", 0.0))
        if est > self._threshold:
            return GuardrailResult(
                passed=False, rule_name=self.name,
                message=f"estimated_cost_usd={est} > threshold",
                severity="block",
            )
        return GuardrailResult(passed=True, rule_name=self.name)


class OutputSchemaValidatorRule(GuardrailRule):
    name = "OutputSchemaValidator"

    async def check(self, content: str, context: dict[str, Any]) -> GuardrailResult:
        # If a response schema is present in context, perform a
        # lightweight check (presence of key names)
        schema_keys = context.get("expected_schema_keys") or []
        missing = [k for k in schema_keys if f'"{k}"' not in content and f"'{k}'" not in content]
        if missing:
            return GuardrailResult(
                passed=False, rule_name=self.name,
                message=f"missing keys: {missing}",
                severity="warning",
            )
        return GuardrailResult(passed=True, rule_name=self.name)


class HallucinationGuardRule(GuardrailRule):
    name = "HallucinationGuard"

    async def check(self, content: str, context: dict[str, Any]) -> GuardrailResult:
        # Basic heuristic: if content contains phrases like
        # "I think" or "might be" with facts, mark warning
        if re.search(r"\b(I think|might be|possibly|could be)\b", content, re.IGNORECASE):
            return GuardrailResult(
                passed=True, rule_name=self.name,
                message="hedging language detected",
                severity="warning",
            )
        return GuardrailResult(passed=True, rule_name=self.name)


class ConstraintGuardrails:
    """Constraint-based guardrails: max_tool_calls, max_cost, max_tokens, domain restrictions.

    Tracks per-session usage counters and enforces limits.
    Gated behind ``constraint_guardrails`` feature flag.
    """

    def __init__(
        self,
        max_tool_calls: int = 50,
        max_cost_usd: float = 5.0,
        max_tokens: int = 100_000,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> None:
        self.max_tool_calls = max_tool_calls
        self.max_cost_usd = max_cost_usd
        self.max_tokens = max_tokens
        self.allowed_domains = allowed_domains or []
        self.blocked_domains = blocked_domains or []
        # Per-session counters: session_id -> counts
        self._sessions: dict[str, dict[str, Any]] = {}

    def _get_session(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "tool_calls": 0,
                "cost_usd": 0.0,
                "tokens_used": 0,
            }
        return self._sessions[session_id]

    def record_tool_call(self, session_id: str) -> None:
        self._get_session(session_id)["tool_calls"] += 1

    def record_cost(self, session_id: str, cost_usd: float) -> None:
        self._get_session(session_id)["cost_usd"] += cost_usd

    def record_tokens(self, session_id: str, tokens: int) -> None:
        self._get_session(session_id)["tokens_used"] += tokens

    def check_tool_calls(self, session_id: str) -> GuardrailResult:
        s = self._get_session(session_id)
        if s["tool_calls"] >= self.max_tool_calls:
            return GuardrailResult(
                passed=False, rule_name="ConstraintGuardrails",
                message=f"Tool call limit reached: {s['tool_calls']}/{self.max_tool_calls}",
                severity="block",
            )
        return GuardrailResult(passed=True, rule_name="ConstraintGuardrails")

    def check_cost(self, session_id: str) -> GuardrailResult:
        s = self._get_session(session_id)
        if s["cost_usd"] >= self.max_cost_usd:
            return GuardrailResult(
                passed=False, rule_name="ConstraintGuardrails",
                message=f"Cost limit reached: ${s['cost_usd']:.4f}/${self.max_cost_usd}",
                severity="block",
            )
        return GuardrailResult(passed=True, rule_name="ConstraintGuardrails")

    def check_tokens(self, session_id: str) -> GuardrailResult:
        s = self._get_session(session_id)
        if s["tokens_used"] >= self.max_tokens:
            return GuardrailResult(
                passed=False, rule_name="ConstraintGuardrails",
                message=f"Token limit reached: {s['tokens_used']}/{self.max_tokens}",
                severity="block",
            )
        return GuardrailResult(passed=True, rule_name="ConstraintGuardrails")

    def check_domain(self, url: str) -> GuardrailResult:
        """Check if a URL domain is allowed/blocked."""
        domain = _extract_domain(url)
        if self.blocked_domains and domain in self.blocked_domains:
            return GuardrailResult(
                passed=False, rule_name="ConstraintGuardrails",
                message=f"Domain blocked: {domain}",
                severity="block",
            )
        if self.allowed_domains and domain not in self.allowed_domains:
            return GuardrailResult(
                passed=False, rule_name="ConstraintGuardrails",
                message=f"Domain not in allowlist: {domain}",
                severity="block",
            )
        return GuardrailResult(passed=True, rule_name="ConstraintGuardrails")

    def check_all(self, session_id: str) -> list[GuardrailResult]:
        """Run all constraint checks. Returns list of results."""
        return [
            self.check_tool_calls(session_id),
            self.check_cost(session_id),
            self.check_tokens(session_id),
        ]

    def session_status(self, session_id: str) -> dict[str, Any]:
        s = self._get_session(session_id)
        return {
            "tool_calls": s["tool_calls"],
            "max_tool_calls": self.max_tool_calls,
            "cost_usd": round(s["cost_usd"], 4),
            "max_cost_usd": self.max_cost_usd,
            "tokens_used": s["tokens_used"],
            "max_tokens": self.max_tokens,
        }

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def _extract_domain(url: str) -> str:
    """Extract domain from a URL string."""
    # Simple extraction without urllib for speed
    url = url.lower().strip()
    if "://" in url:
        url = url.split("://", 1)[1]
    return url.split("/", 1)[0].split(":", 1)[0]


class GuardrailsPipeline:
    def __init__(self, rules: list[GuardrailRule] | None = None, bus: Any | None = None):
        self._bus = bus
        self._rules: list[GuardrailRule] = rules or []

    @classmethod
    def default_pipeline(
        cls, config: Any | None = None, bus: Any | None = None,
    ) -> GuardrailsPipeline:
        cfg = config or {}
        rules: list[GuardrailRule] = []
        rules.append(ContentFilterRule())
        if getattr(cfg, "pii_detection_enabled", False):
            rules.append(PiiDetectorRule())
        if getattr(cfg, "cost_guard_enabled", True):
            rules.append(CostGuardRule(threshold_usd=getattr(cfg, "cost_guard_threshold_usd", 5.0)))
        rules.append(OutputSchemaValidatorRule())
        if getattr(cfg, "hallucination_guard_enabled", False):
            rules.append(HallucinationGuardRule())
        return cls(rules=rules, bus=bus)

    async def run_input(self, text: str, context: dict[str, Any]) -> list[GuardrailResult]:
        results: list[GuardrailResult] = []
        for r in self._rules:
            if not r.enabled:
                continue
            try:
                res = await r.check(text, context)
                results.append(res)
                # publish if blocked
                if not res.passed and res.severity == "block":
                    self._publish_trigger(text, context, results)
                    self._publish_block(text, context, res)
                    return results
            except Exception:
                log.exception("guardrail.rule_failed name=%s", r.name)
        # publish normal triggered
        self._publish_trigger(text, context, results)
        return results

    async def run_output(self, text: str, context: dict[str, Any]) -> list[GuardrailResult]:
        # for now, same as run_input
        return await self.run_input(text, context)

    def _publish_trigger(
        self, text: str, context: dict[str, Any],
        results: list[GuardrailResult],
    ) -> None:
        if self._bus is None:
            return
        try:
            payload = {
                "request_id": context.get("request_id", ""),
                "origin": context.get("origin", ""),
                "results": [r.model_dump() for r in results],
            }
            self._bus.publish({"topic": "guardrails.triggered", "payload": payload})
        except Exception:
            log.debug("guardrails.publish_trigger_failed")

    def _publish_block(self, text: str, context: dict[str, Any], blocking: GuardrailResult) -> None:
        if self._bus is None:
            return
        try:
            payload = {
                "request_id": context.get("request_id", ""),
                "origin": context.get("origin", ""),
                "blocking_rule": blocking.rule_name,
                "reason": blocking.message,
            }
            self._bus.publish({"topic": "guardrails.blocked", "payload": payload})
        except Exception:
            log.debug("guardrails.publish_block_failed")

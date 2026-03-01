"""Comprehensive tests for all P1 features.

Covers: error taxonomy, config validation, cost governance,
auth middleware, event versioning, graceful degradation,
contract response models, admin audit trail, observability metrics/SLOs.
"""

from __future__ import annotations

import hashlib
import time
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from qe.api.auth import AuthContext, AuthProvider, Scope
from qe.api.responses import (
    BudgetInfo,
    ClaimResponse,
    ClaimsListResponse,
    DLQEntry,
    DLQListResponse,
    ErrorResponse,
    EventsListResponse,
    GoalSubmitResponse,
    GoalSummary,
    HealthResponse,
    ProjectSummary,
    ServiceInfo,
    StatusResponse,
    SubmitResponse,
    SuccessResponse,
)
from qe.audit import AuditEntry, AuditLog
from qe.bus.schemas import (
    TOPIC_SCHEMAS,
    ClaimCommittedPayload,
    ClaimProposedPayload,
    DLQPayload,
    GateDeniedPayload,
    GoalCompletedPayload,
    GoalFailedPayload,
    GoalSubmittedPayload,
    HeartbeatPayload,
    SystemCircuitBreakPayload,
    SystemErrorPayload,
    TaskDispatchedPayload,
    get_schema,
    validate_payload,
)
from qe.config import (
    BudgetConfig,
    BusConfig,
    ModelsConfig,
    QEConfig,
    RuntimeConfig,
    SecurityConfig,
    SubstrateConfig,
    load_config,
)
from qe.errors import (
    APINotReadyError,
    APIValidationError,
    AuthenticationError,
    AuthorizationError,
    BusDeliveryError,
    BusHandlerError,
    BusTopicError,
    ConfigMissingError,
    ConfigValidationError,
    ErrorDomain,
    GoalBudgetExceededError,
    GoalInvalidTransitionError,
    GoalTimeoutError,
    GuardrailTripError,
    LLMParseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTokenLimitError,
    QEError,
    Severity,
    SubstrateConnectionError,
    classify_error,
)
from qe.runtime.cost_governor import CostGovernor, GoalBudget
from qe.runtime.degradation import (
    DegradationLevel,
    DegradationPolicy,
    FallbackChain,
)
from qe.runtime.metrics import (
    SLO,
    Counter,
    Gauge,
    Histogram,
    MetricsCollector,
    get_metrics,
)

# ── P1 #8: Error taxonomy ─────────────────────────────────────────────────


class TestErrorTaxonomy:
    def test_base_error_defaults(self):
        e = QEError("something broke")
        assert e.code == "QE_UNKNOWN"
        assert e.domain == ErrorDomain.API
        assert e.severity == Severity.ERROR
        assert e.is_retryable is False
        assert e.message == "something broke"
        assert str(e) == "something broke"

    def test_base_error_default_message(self):
        e = QEError()
        assert e.message == "QE_UNKNOWN"

    def test_to_dict(self):
        e = LLMRateLimitError("rate limited", context={"model": "gpt-4o"})
        d = e.to_dict()
        assert d["code"] == "QE_LLM_RATE_LIMIT"
        assert d["domain"] == "LLM"
        assert d["severity"] == "warn"
        assert d["is_retryable"] is True
        assert d["retry_delay_ms"] == 5000
        assert d["context"] == {"model": "gpt-4o"}

    def test_llm_errors(self):
        assert LLMRateLimitError.is_retryable is True
        assert LLMTimeoutError.is_retryable is True
        assert LLMTokenLimitError.is_retryable is False
        assert LLMProviderError.retry_delay_ms == 10000
        assert LLMParseError.is_retryable is True

    def test_bus_errors(self):
        assert BusHandlerError.domain == ErrorDomain.BUS
        assert BusHandlerError.is_retryable is True
        assert BusDeliveryError.is_retryable is False
        assert BusTopicError.is_retryable is False

    def test_config_errors_critical(self):
        assert ConfigValidationError.severity == Severity.CRITICAL
        assert ConfigMissingError.severity == Severity.CRITICAL

    def test_security_errors_not_retryable(self):
        assert AuthenticationError.is_retryable is False
        assert AuthorizationError.is_retryable is False
        assert GuardrailTripError.is_retryable is False

    def test_goal_errors(self):
        assert GoalBudgetExceededError.code == "QE_GOAL_BUDGET_EXCEEDED"
        assert GoalInvalidTransitionError.code == "QE_GOAL_INVALID_TRANSITION"
        assert GoalTimeoutError.code == "QE_GOAL_TIMEOUT"

    def test_api_errors(self):
        assert APIValidationError.is_retryable is False
        assert APINotReadyError.is_retryable is True
        assert APINotReadyError.retry_delay_ms == 5000

    def test_classify_rate_limit(self):
        e = classify_error(ValueError("rate limit exceeded"))
        assert isinstance(e, LLMRateLimitError)

    def test_classify_timeout(self):
        e = classify_error(TimeoutError("request timeout"))
        assert isinstance(e, LLMTimeoutError)

    def test_classify_429(self):
        e = classify_error(RuntimeError("HTTP 429 too many requests"))
        assert isinstance(e, LLMRateLimitError)

    def test_classify_503(self):
        e = classify_error(RuntimeError("HTTP 503 service unavailable"))
        assert isinstance(e, LLMProviderError)

    def test_classify_connection(self):
        e = classify_error(ConnectionError("connection refused"))
        assert isinstance(e, SubstrateConnectionError)

    def test_classify_token_limit(self):
        e = classify_error(ValueError("token limit exceeded"))
        assert isinstance(e, LLMTokenLimitError)

    def test_classify_json_parse(self):
        e = classify_error(ValueError("invalid json response"))
        assert isinstance(e, LLMParseError)

    def test_classify_unknown_falls_back(self):
        e = classify_error(RuntimeError("something weird"))
        assert type(e) is QEError

    def test_classify_qe_error_passthrough(self):
        original = BusTopicError("bad topic")
        assert classify_error(original) is original

    def test_error_context(self):
        e = QEError("fail", context={"key": "value"})
        assert e.context == {"key": "value"}

    def test_error_is_exception(self):
        with pytest.raises(QEError):
            raise LLMRateLimitError("test")


# ── P1 #9: Config validation ─────────────────────────────────────────────


class TestConfigValidation:
    def test_default_config(self):
        cfg = QEConfig()
        assert cfg.budget.monthly_limit_usd == 50.0
        assert cfg.budget.alert_at_pct == 0.80
        assert cfg.runtime.log_level == "INFO"
        assert cfg.runtime.hil_timeout_seconds == 3600
        assert cfg.models.fast == "gpt-4o-mini"

    def test_budget_limit_must_be_positive(self):
        with pytest.raises(ValidationError):
            BudgetConfig(monthly_limit_usd=-10.0)

    def test_budget_limit_zero_invalid(self):
        with pytest.raises(ValidationError):
            BudgetConfig(monthly_limit_usd=0.0)

    def test_alert_pct_range(self):
        BudgetConfig(alert_at_pct=0.0)  # OK
        BudgetConfig(alert_at_pct=1.0)  # OK
        with pytest.raises(ValidationError):
            BudgetConfig(alert_at_pct=1.5)
        with pytest.raises(ValidationError):
            BudgetConfig(alert_at_pct=-0.1)

    def test_log_level_literal(self):
        RuntimeConfig(log_level="DEBUG")
        RuntimeConfig(log_level="WARNING")
        with pytest.raises(ValidationError):
            RuntimeConfig(log_level="TRACE")

    def test_hil_timeout_positive(self):
        with pytest.raises(ValidationError):
            RuntimeConfig(hil_timeout_seconds=0)

    def test_model_name_stripped(self):
        m = ModelsConfig(fast="  gpt-4o-mini  ", balanced="gpt-4o")
        assert m.fast == "gpt-4o-mini"

    def test_bus_type_literal(self):
        BusConfig(type="memory")
        BusConfig(type="redis")
        with pytest.raises(ValidationError):
            BusConfig(type="kafka")

    def test_extra_fields_allowed(self):
        cfg = QEConfig.model_validate({"unknown_section": {"key": "val"}})
        assert cfg.budget.monthly_limit_usd == 50.0

    def test_load_config_missing_file(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert cfg.budget.monthly_limit_usd == 50.0
        assert cfg.runtime.log_level == "INFO"

    def test_load_config_from_file(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[budget]\nmonthly_limit_usd = 100.0\nalert_at_pct = 0.90\n'
            '[runtime]\nlog_level = "DEBUG"\n',
            encoding="utf-8",
        )
        cfg = load_config(config_file)
        assert cfg.budget.monthly_limit_usd == 100.0
        assert cfg.budget.alert_at_pct == 0.90
        assert cfg.runtime.log_level == "DEBUG"

    def test_load_config_invalid_raises(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[budget]\nmonthly_limit_usd = -5.0\n',
            encoding="utf-8",
        )
        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_security_config_defaults(self):
        s = SecurityConfig()
        assert s.require_auth is False
        assert s.api_key is None

    def test_substrate_config_defaults(self):
        s = SubstrateConfig()
        assert s.db_path == "data/qe.db"


# ── P1 #10: Cost governance ──────────────────────────────────────────────


class TestCostGovernor:
    def test_register_goal_default_cap(self):
        gov = CostGovernor(default_goal_cap_usd=5.0)
        budget = gov.register_goal("g1")
        assert budget.cap_usd == 5.0
        assert budget.spent_usd == 0.0
        assert budget.remaining_usd == 5.0
        assert budget.exhausted is False

    def test_register_goal_custom_cap(self):
        gov = CostGovernor()
        budget = gov.register_goal("g1", cap_usd=10.0)
        assert budget.cap_usd == 10.0

    def test_record_call_within_budget(self):
        gov = CostGovernor()
        gov.register_goal("g1", cap_usd=1.0)
        ok = gov.record_call("g1", 0.50)
        assert ok is True
        budget = gov.get_goal_budget("g1")
        assert budget.spent_usd == 0.50
        assert budget.call_count == 1

    def test_record_call_exceeds_budget(self):
        gov = CostGovernor()
        gov.register_goal("g1", cap_usd=0.10)
        gov.record_call("g1", 0.05)
        ok = gov.record_call("g1", 0.06)
        assert ok is False
        budget = gov.get_goal_budget("g1")
        assert budget.exhausted is True

    def test_record_call_unregistered_goal(self):
        gov = CostGovernor()
        ok = gov.record_call("unknown", 1.0)
        assert ok is True  # unregistered goals are uncapped

    def test_estimate_cost_known_model(self):
        gov = CostGovernor()
        gov.register_goal("g1", cap_usd=1.0)
        est = gov.estimate_cost("gpt-4o-mini", 1000, 500, goal_id="g1")
        assert est.model == "gpt-4o-mini"
        assert est.within_budget is True
        assert est.estimated_cost_usd > 0
        assert est.remaining_goal_budget_usd == 1.0

    def test_estimate_cost_exceeds_goal_budget(self):
        gov = CostGovernor()
        gov.register_goal("g1", cap_usd=0.001)
        est = gov.estimate_cost("gpt-4o", 100_000, 50_000, goal_id="g1")
        assert est.within_budget is False

    def test_estimate_cost_no_goal(self):
        gov = CostGovernor()
        est = gov.estimate_cost("gpt-4o-mini", 1000, 500)
        assert est.within_budget is True
        assert est.remaining_goal_budget_usd is None

    def test_estimate_cost_global_budget_exhausted(self):
        tracker = MagicMock()
        tracker.remaining_pct.return_value = 0.0
        gov = CostGovernor(budget_tracker=tracker)
        est = gov.estimate_cost("gpt-4o-mini", 1000, 500)
        assert est.within_budget is False

    def test_suggest_downgrade(self):
        gov = CostGovernor()
        assert gov.suggest_downgrade("gpt-4o") == "gpt-4o-mini"
        assert gov.suggest_downgrade("o1-preview") == "gpt-4o"
        assert gov.suggest_downgrade("gpt-4o-mini") is None

    def test_goal_summary(self):
        gov = CostGovernor()
        gov.register_goal("g1", cap_usd=2.0)
        gov.record_call("g1", 0.50)
        summary = gov.goal_summary("g1")
        assert summary["goal_id"] == "g1"
        assert summary["cap_usd"] == 2.0
        assert summary["spent_usd"] == 0.50
        assert summary["remaining_usd"] == 1.50
        assert summary["call_count"] == 1
        assert summary["exhausted"] is False

    def test_goal_summary_unknown(self):
        gov = CostGovernor()
        assert gov.goal_summary("nope") is None

    def test_goal_budget_remaining_floor(self):
        gb = GoalBudget(goal_id="g1", cap_usd=1.0, spent_usd=5.0)
        assert gb.remaining_usd == 0.0
        assert gb.exhausted is True

    def test_estimate_unknown_model_fallback_rate(self):
        gov = CostGovernor()
        est = gov.estimate_cost("unknown-model-xyz", 1000, 500)
        # Fallback rate is $1/M tokens
        expected = (1000 * 1.0 + 500 * 1.0) / 1_000_000
        assert est.estimated_cost_usd == pytest.approx(expected, abs=0.0001)


# ── P1 #11: Auth middleware ──────────────────────────────────────────────


class TestAuthMiddleware:
    def test_scope_hierarchy_admin_has_all(self):
        ctx = AuthContext(scope=Scope.ADMIN, key_id="admin")
        assert ctx.has_scope(Scope.READ) is True
        assert ctx.has_scope(Scope.WRITE) is True
        assert ctx.has_scope(Scope.ADMIN) is True

    def test_scope_hierarchy_write_has_read(self):
        ctx = AuthContext(scope=Scope.WRITE, key_id="writer")
        assert ctx.has_scope(Scope.READ) is True
        assert ctx.has_scope(Scope.WRITE) is True
        assert ctx.has_scope(Scope.ADMIN) is False

    def test_scope_hierarchy_read_only(self):
        ctx = AuthContext(scope=Scope.READ, key_id="reader")
        assert ctx.has_scope(Scope.READ) is True
        assert ctx.has_scope(Scope.WRITE) is False
        assert ctx.has_scope(Scope.ADMIN) is False

    def test_configure_api_key(self):
        provider = AuthProvider()
        provider.configure(api_key="test-key-123")
        assert provider.enabled is True

        ctx = provider.validate_key("test-key-123")
        assert ctx is not None
        assert ctx.scope == Scope.WRITE
        assert ctx.key_id == "default"

    def test_configure_admin_key(self):
        provider = AuthProvider()
        provider.configure(admin_api_key="admin-key-456")
        assert provider.enabled is True

        ctx = provider.validate_key("admin-key-456")
        assert ctx is not None
        assert ctx.scope == Scope.ADMIN
        assert ctx.key_id == "admin"

    def test_validate_invalid_key(self):
        provider = AuthProvider()
        provider.configure(api_key="real-key")
        assert provider.validate_key("wrong-key") is None

    def test_no_auth_disabled(self):
        provider = AuthProvider()
        assert provider.enabled is False

    def test_configure_clears_previous(self):
        provider = AuthProvider()
        provider.configure(api_key="key1")
        assert provider.validate_key("key1") is not None
        provider.configure(api_key="key2")
        assert provider.validate_key("key1") is None
        assert provider.validate_key("key2") is not None

    def test_generate_key(self):
        key = AuthProvider.generate_key()
        assert key.startswith("qe_")
        assert len(key) > 10
        # Keys should be unique
        assert AuthProvider.generate_key() != AuthProvider.generate_key()

    def test_keys_stored_as_hash(self):
        provider = AuthProvider()
        provider.configure(api_key="secret")
        # The internal dict should use hashed keys
        for stored_hash in provider._keys:
            assert stored_hash == hashlib.sha256(b"secret").hexdigest()

    def test_both_keys_configured(self):
        provider = AuthProvider()
        provider.configure(api_key="write-key", admin_api_key="admin-key")
        w = provider.validate_key("write-key")
        a = provider.validate_key("admin-key")
        assert w.scope == Scope.WRITE
        assert a.scope == Scope.ADMIN


# ── P1 #12: Event versioning ────────────────────────────────────────────


class TestEventVersioning:
    def test_claim_proposed_valid(self):
        payload = validate_payload("claims.proposed", {
            "subject_entity_id": "Apple",
            "predicate": "has_revenue",
            "object_value": "$100B",
            "confidence": 0.9,
        })
        assert isinstance(payload, ClaimProposedPayload)
        assert payload.schema_version == "1.0"

    def test_claim_proposed_invalid_confidence(self):
        with pytest.raises(ValidationError):
            validate_payload("claims.proposed", {
                "subject_entity_id": "X",
                "predicate": "p",
                "object_value": "v",
                "confidence": 1.5,
            })

    def test_claim_committed_inherits_proposed(self):
        payload = validate_payload("claims.committed", {
            "subject_entity_id": "Y",
            "predicate": "p",
            "object_value": "v",
            "confidence": 0.8,
            "created_at": "2025-01-01",
        })
        assert isinstance(payload, ClaimCommittedPayload)
        assert payload.created_at == "2025-01-01"

    def test_goal_submitted(self):
        payload = validate_payload("goals.submitted", {
            "goal_id": "g1",
            "description": "Research AI",
        })
        assert isinstance(payload, GoalSubmittedPayload)

    def test_goal_completed(self):
        payload = validate_payload("goals.completed", {
            "goal_id": "g1",
            "subtask_count": 3,
        })
        assert isinstance(payload, GoalCompletedPayload)

    def test_goal_failed(self):
        payload = validate_payload("goals.failed", {
            "goal_id": "g1",
            "reason": "budget exceeded",
        })
        assert isinstance(payload, GoalFailedPayload)

    def test_task_dispatched(self):
        payload = validate_payload("tasks.dispatched", {
            "goal_id": "g1",
            "subtask_id": "st1",
            "description": "search the web",
            "task_type": "web_search",
        })
        assert isinstance(payload, TaskDispatchedPayload)
        assert payload.model_tier == "balanced"

    def test_system_error(self):
        payload = validate_payload("system.error", {
            "envelope_id": "e1",
            "topic": "claims.proposed",
            "error": "handler failed",
        })
        assert isinstance(payload, SystemErrorPayload)

    def test_system_circuit_break(self):
        payload = validate_payload("system.circuit_break", {
            "service_id": "svc1",
            "reason": "too many errors",
        })
        assert isinstance(payload, SystemCircuitBreakPayload)

    def test_heartbeat(self):
        payload = validate_payload("system.heartbeat", {
            "turn_count": 42,
            "status": "alive",
        })
        assert isinstance(payload, HeartbeatPayload)

    def test_dlq_payload(self):
        payload = validate_payload("system.dlq", {
            "envelope_id": "e1",
            "topic": "test",
            "error": "boom",
            "attempts": 3,
        })
        assert isinstance(payload, DLQPayload)

    def test_gate_denied(self):
        payload = validate_payload("system.gate_denied", {
            "reason": "unsafe content",
            "risk_score": 0.95,
        })
        assert isinstance(payload, GateDeniedPayload)

    def test_unknown_topic_returns_none(self):
        assert validate_payload("unknown.topic", {"foo": "bar"}) is None

    def test_get_schema_known(self):
        assert get_schema("claims.proposed") is ClaimProposedPayload

    def test_get_schema_unknown(self):
        assert get_schema("nonexistent") is None

    def test_all_registered_topics(self):
        expected = {
            "claims.proposed", "claims.committed",
            "goals.submitted", "goals.completed", "goals.failed",
            "tasks.dispatched",
            "system.error", "system.circuit_break", "system.heartbeat",
            "system.dlq", "system.gate_denied",
            "agents.registered",
            "coordination.vote_request", "coordination.vote_response",
            "tasks.delegated",
            "inquiry.started", "inquiry.phase_completed",
            "inquiry.question_generated", "inquiry.investigation_completed",
            "inquiry.hypothesis_generated", "inquiry.hypothesis_updated",
            "inquiry.insight_generated", "inquiry.completed",
            "inquiry.failed", "inquiry.budget_warning",
            "strategy.selected", "strategy.switch_requested",
            "strategy.evaluated",
            "pool.scale_recommended", "pool.scale_executed",
            "pool.health_check",
            "prompt.variant_selected", "prompt.outcome_recorded",
            "prompt.variant_created", "prompt.variant_deactivated",
            "prompt.mutation_cycle_completed", "prompt.variant_promoted",
            "knowledge.consolidation_completed", "knowledge.belief_promoted",
            "knowledge.hypothesis_updated",
            "bridge.strategy_outcome_recorded",
        }
        assert set(TOPIC_SCHEMAS.keys()) == expected


# ── P1 #13: Graceful degradation ────────────────────────────────────────


class TestGracefulDegradation:
    def test_default_level_healthy(self):
        policy = DegradationPolicy()
        assert policy.level == DegradationLevel.HEALTHY

    def test_set_level(self):
        policy = DegradationPolicy()
        policy.set_level(DegradationLevel.DEGRADED)
        assert policy.level == DegradationLevel.DEGRADED

    def test_feature_available_at_healthy(self):
        policy = DegradationPolicy()
        assert policy.is_feature_enabled("web_search") is True
        assert policy.is_feature_enabled("llm_calls") is True

    def test_feature_disabled_at_degraded(self):
        policy = DegradationPolicy()
        policy.set_level(DegradationLevel.DEGRADED)
        # web_search has min_level=HEALTHY, so disabled at DEGRADED
        assert policy.is_feature_enabled("web_search") is False
        # llm_calls has min_level=DEGRADED, so still enabled
        assert policy.is_feature_enabled("llm_calls") is True

    def test_feature_disabled_at_fallback(self):
        policy = DegradationPolicy()
        policy.set_level(DegradationLevel.FALLBACK)
        assert policy.is_feature_enabled("web_search") is False
        assert policy.is_feature_enabled("llm_calls") is False
        assert policy.is_feature_enabled("local_models") is True

    def test_feature_disabled_at_minimal(self):
        policy = DegradationPolicy()
        policy.set_level(DegradationLevel.MINIMAL)
        assert policy.is_feature_enabled("llm_calls") is False
        assert policy.is_feature_enabled("goal_execution") is False
        assert policy.is_feature_enabled("read_api") is True
        assert policy.is_feature_enabled("hil_approvals") is True

    def test_manual_disable_overrides(self):
        policy = DegradationPolicy()
        policy.disable_feature("llm_calls")
        assert policy.is_feature_enabled("llm_calls") is False

    def test_enable_removes_override(self):
        policy = DegradationPolicy()
        policy.disable_feature("llm_calls")
        assert policy.is_feature_enabled("llm_calls") is False
        policy.enable_feature("llm_calls")
        assert policy.is_feature_enabled("llm_calls") is True

    def test_unknown_feature_defaults_enabled(self):
        policy = DegradationPolicy()
        assert policy.is_feature_enabled("nonexistent_feature") is True

    def test_fallback_chain_advance(self):
        chain = FallbackChain(capability="test", chain=["a", "b", "c"])
        assert chain.current == "a"
        assert chain.advance() == "b"
        assert chain.advance() == "c"
        assert chain.advance() is None  # exhausted

    def test_fallback_chain_reset(self):
        chain = FallbackChain(capability="test", chain=["a", "b"])
        chain.advance()
        chain.reset()
        assert chain.current == "a"

    def test_policy_get_fallback(self):
        policy = DegradationPolicy()
        model = policy.get_fallback("powerful")
        assert model is not None
        assert "claude" in model or "gpt" in model

    def test_policy_advance_fallback(self):
        policy = DegradationPolicy()
        first = policy.get_fallback("fast")
        second = policy.advance_fallback("fast")
        assert first != second

    def test_policy_reset_fallbacks(self):
        policy = DegradationPolicy()
        policy.advance_fallback("fast")
        policy.reset_fallbacks()
        # Should be back to first
        assert policy.get_fallback("fast") is not None

    def test_get_fallback_unknown_tier(self):
        policy = DegradationPolicy()
        assert policy.get_fallback("nonexistent") is None
        assert policy.advance_fallback("nonexistent") is None

    def test_status_snapshot(self):
        policy = DegradationPolicy()
        s = policy.status()
        assert s["level"] == "HEALTHY"
        assert "features" in s
        assert "fallback_chains" in s
        assert "llm_calls" in s["features"]

    def test_assess_from_health_all_pass(self):
        policy = DegradationPolicy()
        report = {"checks": [
            {"status": "pass"}, {"status": "pass"}, {"status": "pass"},
        ]}
        assert policy.assess_from_health(report) == DegradationLevel.HEALTHY

    def test_assess_from_health_warnings(self):
        policy = DegradationPolicy()
        report = {"checks": [
            {"status": "warn"}, {"status": "warn"}, {"status": "pass"},
        ]}
        assert policy.assess_from_health(report) == DegradationLevel.DEGRADED

    def test_assess_from_health_one_fail(self):
        policy = DegradationPolicy()
        report = {"checks": [
            {"status": "fail"}, {"status": "pass"},
        ]}
        assert policy.assess_from_health(report) == DegradationLevel.FALLBACK

    def test_assess_from_health_many_fails(self):
        policy = DegradationPolicy()
        report = {"checks": [
            {"status": "fail"}, {"status": "fail"}, {"status": "fail"},
        ]}
        assert policy.assess_from_health(report) == DegradationLevel.MINIMAL

    def test_degradation_level_ordering(self):
        assert DegradationLevel.HEALTHY < DegradationLevel.DEGRADED
        assert DegradationLevel.DEGRADED < DegradationLevel.FALLBACK
        assert DegradationLevel.FALLBACK < DegradationLevel.MINIMAL
        assert DegradationLevel.MINIMAL < DegradationLevel.OFFLINE


# ── P1 #14: Contract response models ────────────────────────────────────


class TestContractModels:
    def test_health_response(self):
        r = HealthResponse(status="ok", timestamp="2025-01-01T00:00:00Z")
        assert r.status == "ok"

    def test_service_info(self):
        s = ServiceInfo(
            service_id="s1",
            display_name="Test",
            status="alive",
            turn_count=5,
            circuit_broken=False,
        )
        assert s.service_id == "s1"

    def test_budget_info(self):
        b = BudgetInfo(
            total_spend=10.0,
            remaining_pct=0.80,
            limit_usd=50.0,
            by_model={"gpt-4o": 8.0, "gpt-4o-mini": 2.0},
        )
        assert b.remaining_pct == 0.80

    def test_status_response(self):
        r = StatusResponse(
            services=[ServiceInfo(
                service_id="s1", display_name="S",
                status="alive", turn_count=0, circuit_broken=False,
            )],
            budget=BudgetInfo(
                total_spend=0, remaining_pct=1.0,
                limit_usd=50, by_model={},
            ),
        )
        assert len(r.services) == 1

    def test_claim_response(self):
        c = ClaimResponse(
            claim_id="c1",
            subject_entity_id="Apple",
            predicate="has_ceo",
            object_value="Tim Cook",
            confidence=0.95,
            created_at="2025-01-01",
        )
        assert c.superseded_by is None
        assert c.tags == []

    def test_claims_list(self):
        r = ClaimsListResponse(
            claims=[ClaimResponse(
                claim_id="c1", subject_entity_id="X",
                predicate="p", object_value="v",
                confidence=0.9, created_at="now",
            )],
            count=1,
        )
        assert r.count == 1

    def test_goal_summary(self):
        g = GoalSummary(
            goal_id="g1",
            description="Test goal",
            status="executing",
            subtask_count=3,
            created_at="2025-01-01",
        )
        assert g.completed_at is None

    def test_goal_submit_response(self):
        r = GoalSubmitResponse(
            goal_id="g1",
            status="planning",
            subtask_count=2,
            strategy="parallel",
        )
        assert r.strategy == "parallel"

    def test_project_summary(self):
        p = ProjectSummary(
            project_id="p1",
            name="My Project",
            status="active",
        )
        assert p.goal_count == 0
        assert p.completed_goals == 0

    def test_dlq_entry(self):
        d = DLQEntry(
            envelope_id="e1",
            topic="test.topic",
            error="handler crash",
            attempts=3,
        )
        assert d.source_service_id == ""

    def test_error_response(self):
        e = ErrorResponse(error="not found", code="QE_API_NOT_FOUND")
        assert e.error == "not found"

    def test_success_response(self):
        s = SuccessResponse(status="ok")
        assert s.status == "ok"

    def test_submit_response(self):
        r = SubmitResponse(envelope_id="e1", status="submitted")
        assert r.envelope_id == "e1"

    def test_events_list_response(self):
        r = EventsListResponse(events=[{"topic": "test"}], count=1)
        assert r.count == 1

    def test_dlq_list_response(self):
        r = DLQListResponse(entries=[], count=0)
        assert r.count == 0


# ── P1 #15: Admin audit trail ────────────────────────────────────────────


class TestAuditTrail:
    def test_record_action(self):
        audit = AuditLog()
        entry = audit.record("settings.update", resource="budget")
        assert entry.action == "settings.update"
        assert entry.resource == "budget"
        assert entry.result == "success"
        assert audit.count() == 1

    def test_record_with_actor_and_detail(self):
        audit = AuditLog()
        entry = audit.record(
            "circuit.reset",
            resource="service/researcher",
            actor="admin_key",
            detail={"reason": "manual reset"},
        )
        assert entry.actor == "admin_key"
        assert entry.detail == {"reason": "manual reset"}

    def test_record_with_error_result(self):
        audit = AuditLog()
        entry = audit.record(
            "settings.update",
            result="error",
            detail={"error": "validation failed"},
        )
        assert entry.result == "error"

    def test_query_all(self):
        audit = AuditLog()
        audit.record("a.one")
        audit.record("a.two")
        audit.record("a.three")
        results = audit.query()
        assert len(results) == 3
        # Most recent first
        assert results[0]["action"] == "a.three"

    def test_query_filter_action(self):
        audit = AuditLog()
        audit.record("settings.update")
        audit.record("circuit.reset")
        audit.record("settings.update")
        results = audit.query(action="settings.update")
        assert len(results) == 2

    def test_query_filter_actor(self):
        audit = AuditLog()
        audit.record("a.one", actor="alice")
        audit.record("a.two", actor="bob")
        audit.record("a.three", actor="alice")
        results = audit.query(actor="alice")
        assert len(results) == 2

    def test_query_limit(self):
        audit = AuditLog()
        for i in range(10):
            audit.record(f"action.{i}")
        results = audit.query(limit=3)
        assert len(results) == 3

    def test_max_entries_eviction(self):
        audit = AuditLog(max_entries=5)
        for i in range(10):
            audit.record(f"action.{i}")
        assert audit.count() == 5
        results = audit.query()
        # Oldest entries should be evicted
        assert results[-1]["action"] == "action.5"

    def test_entry_has_timestamp(self):
        audit = AuditLog()
        before = time.time()
        entry = audit.record("test")
        after = time.time()
        assert before <= entry.timestamp <= after

    def test_audit_entry_model(self):
        entry = AuditEntry(
            action="test",
            resource="r",
            actor="a",
            detail={"k": "v"},
            result="success",
        )
        d = entry.model_dump()
        assert d["action"] == "test"
        assert "timestamp" in d


# ── P1 #7: Observability metrics & SLOs ──────────────────────────────────


class TestMetricsPrimitives:
    def test_counter_inc(self):
        c = Counter(name="test")
        assert c.value == 0
        c.inc()
        assert c.value == 1
        c.inc(5)
        assert c.value == 6

    def test_gauge_set_inc_dec(self):
        g = Gauge(name="test")
        g.set(10.0)
        assert g.value == 10.0
        g.inc(3.0)
        assert g.value == 13.0
        g.dec(5.0)
        assert g.value == 8.0

    def test_histogram_observe(self):
        h = Histogram(name="test", buckets=[10, 50, 100])
        h.observe(5)
        h.observe(30)
        h.observe(90)
        h.observe(200)
        assert h._count == 4
        assert h._sum == 325.0

    def test_histogram_percentiles(self):
        h = Histogram(name="test", buckets=[10, 50, 100])
        for _ in range(50):
            h.observe(5)
        for _ in range(45):
            h.observe(30)
        for _ in range(5):
            h.observe(90)
        # 100 samples: 50 in <=10, 45 in <=50 (cum 95), 5 in <=100 (cum 100)
        assert h.p50 == 10   # target=50 → cum 50 at bucket 10
        assert h.p95 == 50   # target=95 → cum 95 at bucket 50
        assert h.p99 == 100  # target=99 → cum 100 at bucket 100

    def test_histogram_avg(self):
        h = Histogram(name="test", buckets=[100])
        h.observe(10)
        h.observe(20)
        h.observe(30)
        assert h.avg == 20.0

    def test_histogram_empty_percentile(self):
        h = Histogram(name="test")
        assert h.p50 == 0.0
        assert h.avg == 0.0

    def test_histogram_snapshot(self):
        h = Histogram(name="test", buckets=[10, 50])
        h.observe(5)
        h.observe(30)
        snap = h.snapshot()
        assert snap["count"] == 2
        assert snap["sum"] == 35.0
        assert "avg" in snap
        assert "p50" in snap
        assert "p95" in snap
        assert "p99" in snap

    def test_histogram_inf_bucket(self):
        h = Histogram(name="test", buckets=[10])
        h.observe(100)  # beyond all buckets
        assert h._counts[-1] == 1


class TestSLO:
    def test_slo_lte_pass(self):
        slo = SLO(name="test", metric="m", target=100.0, comparator="lte")
        assert slo.evaluate(50.0) is True
        assert slo.evaluate(100.0) is True
        assert slo.evaluate(101.0) is False

    def test_slo_gte_pass(self):
        slo = SLO(name="test", metric="m", target=0.10, comparator="gte")
        assert slo.evaluate(0.50) is True
        assert slo.evaluate(0.10) is True
        assert slo.evaluate(0.05) is False

    def test_slo_lt(self):
        slo = SLO(name="test", metric="m", target=100.0, comparator="lt")
        assert slo.evaluate(99.0) is True
        assert slo.evaluate(100.0) is False

    def test_slo_gt(self):
        slo = SLO(name="test", metric="m", target=0.0, comparator="gt")
        assert slo.evaluate(0.1) is True
        assert slo.evaluate(0.0) is False

    def test_slo_unknown_comparator(self):
        slo = SLO(name="test", metric="m", target=100.0, comparator="eq")
        assert slo.evaluate(100.0) is False


class TestMetricsCollector:
    def test_counter_registration(self):
        mc = MetricsCollector()
        c = mc.counter("test_counter")
        assert c.value == 0
        c.inc()
        assert mc.counter("test_counter").value == 1

    def test_histogram_registration(self):
        mc = MetricsCollector()
        h = mc.histogram("test_hist")
        h.observe(42.0)
        assert mc.histogram("test_hist")._count == 1

    def test_gauge_registration(self):
        mc = MetricsCollector()
        g = mc.gauge("test_gauge")
        g.set(7.5)
        assert mc.gauge("test_gauge").value == 7.5

    def test_default_counters_registered(self):
        mc = MetricsCollector()
        assert mc.counter("llm_calls_total").value == 0
        assert mc.counter("bus_published_total").value == 0
        assert mc.counter("api_requests_total").value == 0

    def test_default_histograms_registered(self):
        mc = MetricsCollector()
        assert mc.histogram("llm_latency_ms")._count == 0

    def test_default_gauges_registered(self):
        mc = MetricsCollector()
        assert mc.gauge("active_services").value == 0.0

    def test_snapshot(self):
        mc = MetricsCollector()
        mc.counter("llm_calls_total").inc(10)
        mc.gauge("active_services").set(3)

        snap = mc.snapshot()
        assert "uptime_seconds" in snap
        assert snap["counters"]["llm_calls_total"] == 10
        assert snap["gauges"]["active_services"] == 3.0
        assert "histograms" in snap
        assert "slos" in snap

    def test_evaluate_slos(self):
        mc = MetricsCollector()
        results = mc.evaluate_slos()
        assert len(results) >= 5
        for r in results:
            assert "name" in r
            assert "target" in r
            assert "actual" in r
            assert "passing" in r

    def test_slo_all_passing_at_zero(self):
        mc = MetricsCollector()
        results = mc.evaluate_slos()
        for r in results:
            if r["comparator"] == "lte":
                assert r["passing"] is True  # 0.0 <= target
            elif r["comparator"] == "gte" and r["name"] == "budget_remaining":
                assert r["passing"] is False  # 0.0 < 0.10 target

    def test_singleton_get_metrics(self):
        m = get_metrics()
        assert isinstance(m, MetricsCollector)

    def test_snapshot_uptime(self):
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert snap["uptime_seconds"] >= 0.0

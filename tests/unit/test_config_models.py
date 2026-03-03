"""Tests for QE typed configuration models (src/qe/config.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qe.config import (
    A2AConfig,
    BudgetConfig,
    BusConfig,
    GuardrailsConfig,
    HarvestConfig,
    ModelsConfig,
    OpenTelemetryConfig,
    QEConfig,
    RuntimeConfig,
    ScoutConfig,
    SecurityConfig,
    SubstrateConfig,
    load_config,
)

# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------


class TestBudgetConfig:
    def test_defaults(self):
        cfg = BudgetConfig()
        assert cfg.monthly_limit_usd == 50.0
        assert cfg.alert_at_pct == 0.80

    def test_custom_values(self):
        cfg = BudgetConfig(monthly_limit_usd=100.0, alert_at_pct=0.50)
        assert cfg.monthly_limit_usd == 100.0
        assert cfg.alert_at_pct == 0.50

    def test_limit_must_be_positive(self):
        with pytest.raises(ValidationError):
            BudgetConfig(monthly_limit_usd=0)
        with pytest.raises(ValidationError):
            BudgetConfig(monthly_limit_usd=-1)

    def test_alert_pct_bounds(self):
        # lower bound
        cfg = BudgetConfig(alert_at_pct=0.0)
        assert cfg.alert_at_pct == 0.0
        # upper bound
        cfg = BudgetConfig(alert_at_pct=1.0)
        assert cfg.alert_at_pct == 1.0
        # below lower bound
        with pytest.raises(ValidationError):
            BudgetConfig(alert_at_pct=-0.01)
        # above upper bound
        with pytest.raises(ValidationError):
            BudgetConfig(alert_at_pct=1.01)


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------


class TestRuntimeConfig:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.log_level == "INFO"
        assert cfg.log_json is False
        assert cfg.log_dir is None
        assert cfg.hil_timeout_seconds == 3600
        assert cfg.prefer_local_models is False
        assert cfg.module_levels is None

    def test_valid_log_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            cfg = RuntimeConfig(log_level=level)
            assert cfg.log_level == level

    def test_invalid_log_level(self):
        with pytest.raises(ValidationError):
            RuntimeConfig(log_level="TRACE")

    def test_optional_fields(self):
        cfg = RuntimeConfig(log_dir="/tmp/logs", module_levels={"qe.bus": "DEBUG"})
        assert cfg.log_dir == "/tmp/logs"
        assert cfg.module_levels == {"qe.bus": "DEBUG"}

    def test_hil_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            RuntimeConfig(hil_timeout_seconds=0)


# ---------------------------------------------------------------------------
# ModelsConfig
# ---------------------------------------------------------------------------


class TestModelsConfig:
    def test_defaults(self):
        cfg = ModelsConfig()
        assert cfg.fast == "gpt-4o-mini"
        assert cfg.balanced == "gpt-4o"
        assert cfg.powerful == "o1-preview"
        assert cfg.local is None

    def test_strip_model_name_strips_whitespace(self):
        cfg = ModelsConfig(fast="  gpt-4o-mini  ", balanced="\tgpt-4o\n")
        assert cfg.fast == "gpt-4o-mini"
        assert cfg.balanced == "gpt-4o"

    def test_strip_validator_passes_through_none(self):
        cfg = ModelsConfig(local=None)
        assert cfg.local is None

    def test_custom_models(self):
        cfg = ModelsConfig(
            fast="claude-haiku", balanced="claude-sonnet",
            powerful="claude-opus", local="llama3",
        )
        assert cfg.fast == "claude-haiku"
        assert cfg.local == "llama3"


# ---------------------------------------------------------------------------
# SubstrateConfig
# ---------------------------------------------------------------------------


class TestSubstrateConfig:
    def test_defaults(self):
        cfg = SubstrateConfig()
        assert cfg.db_path == "data/qe.db"
        assert cfg.cold_storage_path is None

    def test_custom_paths(self):
        cfg = SubstrateConfig(db_path="/tmp/test.db", cold_storage_path="/tmp/cold")
        assert cfg.db_path == "/tmp/test.db"
        assert cfg.cold_storage_path == "/tmp/cold"


# ---------------------------------------------------------------------------
# BusConfig
# ---------------------------------------------------------------------------


class TestBusConfig:
    def test_defaults(self):
        cfg = BusConfig()
        assert cfg.type == "memory"
        assert cfg.redis_url is None

    def test_redis_type(self):
        cfg = BusConfig(type="redis", redis_url="redis://localhost:6379")
        assert cfg.type == "redis"
        assert cfg.redis_url == "redis://localhost:6379"

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            BusConfig(type="kafka")


# ---------------------------------------------------------------------------
# SecurityConfig
# ---------------------------------------------------------------------------


class TestSecurityConfig:
    def test_defaults_no_auth(self):
        cfg = SecurityConfig()
        assert cfg.api_key is None
        assert cfg.admin_api_key is None
        assert cfg.require_auth is False

    def test_auth_enabled(self):
        cfg = SecurityConfig(api_key="secret", admin_api_key="admin-secret", require_auth=True)
        assert cfg.require_auth is True
        assert cfg.api_key == "secret"
        assert cfg.admin_api_key == "admin-secret"


# ---------------------------------------------------------------------------
# ScoutConfig
# ---------------------------------------------------------------------------


class TestScoutConfig:
    def test_defaults(self):
        cfg = ScoutConfig()
        assert cfg.enabled is False
        assert cfg.poll_interval_seconds == 3600
        assert cfg.max_findings_per_cycle == 20
        assert cfg.max_proposals_per_cycle == 3
        assert cfg.min_composite_score == 0.5
        assert cfg.budget_limit_per_cycle_usd == 1.0
        assert cfg.hil_timeout_seconds == 86400
        assert cfg.max_pending_proposals == 10
        assert len(cfg.search_topics) == 5

    def test_poll_interval_must_be_positive(self):
        with pytest.raises(ValidationError):
            ScoutConfig(poll_interval_seconds=0)

    def test_min_composite_score_bounds(self):
        ScoutConfig(min_composite_score=0.0)
        ScoutConfig(min_composite_score=1.0)
        with pytest.raises(ValidationError):
            ScoutConfig(min_composite_score=-0.1)
        with pytest.raises(ValidationError):
            ScoutConfig(min_composite_score=1.1)

    def test_budget_limit_must_be_positive(self):
        with pytest.raises(ValidationError):
            ScoutConfig(budget_limit_per_cycle_usd=0)

    def test_max_fields_must_be_positive(self):
        with pytest.raises(ValidationError):
            ScoutConfig(max_findings_per_cycle=0)
        with pytest.raises(ValidationError):
            ScoutConfig(max_proposals_per_cycle=0)
        with pytest.raises(ValidationError):
            ScoutConfig(max_pending_proposals=0)


# ---------------------------------------------------------------------------
# HarvestConfig
# ---------------------------------------------------------------------------


class TestHarvestConfig:
    def test_defaults(self):
        cfg = HarvestConfig()
        assert cfg.enabled is False
        assert cfg.poll_interval_seconds == 1800
        assert cfg.max_claims_per_cycle == 10
        assert cfg.consensus_model_count == 5
        assert cfg.adversarial_confidence_threshold == 0.85
        assert cfg.low_confidence_threshold == 0.5
        assert cfg.premium_sprint_enabled is True
        assert cfg.model_profile_enabled is True
        assert cfg.budget_limit_per_cycle_usd == 0.0
        assert cfg.max_concurrent_harvest == 5
        assert cfg.cycle_timeout_seconds == 300

    def test_consensus_model_count_bounds(self):
        HarvestConfig(consensus_model_count=2)
        HarvestConfig(consensus_model_count=20)
        with pytest.raises(ValidationError):
            HarvestConfig(consensus_model_count=1)
        with pytest.raises(ValidationError):
            HarvestConfig(consensus_model_count=21)

    def test_adversarial_threshold_bounds(self):
        with pytest.raises(ValidationError):
            HarvestConfig(adversarial_confidence_threshold=-0.01)
        with pytest.raises(ValidationError):
            HarvestConfig(adversarial_confidence_threshold=1.01)

    def test_poll_interval_must_be_positive(self):
        with pytest.raises(ValidationError):
            HarvestConfig(poll_interval_seconds=0)

    def test_budget_limit_allows_zero(self):
        cfg = HarvestConfig(budget_limit_per_cycle_usd=0.0)
        assert cfg.budget_limit_per_cycle_usd == 0.0

    def test_budget_limit_rejects_negative(self):
        with pytest.raises(ValidationError):
            HarvestConfig(budget_limit_per_cycle_usd=-1.0)

    def test_max_concurrent_harvest_must_be_positive(self):
        with pytest.raises(ValidationError):
            HarvestConfig(max_concurrent_harvest=0)

    def test_cycle_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            HarvestConfig(cycle_timeout_seconds=0)


# ---------------------------------------------------------------------------
# GuardrailsConfig
# ---------------------------------------------------------------------------


class TestGuardrailsConfig:
    def test_defaults(self):
        cfg = GuardrailsConfig()
        assert cfg.enabled is True
        assert cfg.content_filter_enabled is True
        assert cfg.pii_detection_enabled is False
        assert cfg.cost_guard_enabled is True
        assert cfg.cost_guard_threshold_usd == 5.0
        assert cfg.hallucination_guard_enabled is False

    def test_cost_guard_threshold_must_be_positive(self):
        with pytest.raises(ValidationError):
            GuardrailsConfig(cost_guard_threshold_usd=0)
        with pytest.raises(ValidationError):
            GuardrailsConfig(cost_guard_threshold_usd=-1)

    def test_all_guards_disabled(self):
        cfg = GuardrailsConfig(
            enabled=False,
            content_filter_enabled=False,
            cost_guard_enabled=False,
        )
        assert cfg.enabled is False
        assert cfg.content_filter_enabled is False


# ---------------------------------------------------------------------------
# OpenTelemetryConfig
# ---------------------------------------------------------------------------


class TestOpenTelemetryConfig:
    def test_defaults(self):
        cfg = OpenTelemetryConfig()
        assert cfg.enabled is False
        assert cfg.exporter == "console"
        assert cfg.otlp_endpoint is None
        assert cfg.service_name == "question-engine"

    def test_otlp_exporter(self):
        cfg = OpenTelemetryConfig(enabled=True, exporter="otlp", otlp_endpoint="http://localhost:4317")
        assert cfg.exporter == "otlp"
        assert cfg.otlp_endpoint == "http://localhost:4317"

    def test_invalid_exporter(self):
        with pytest.raises(ValidationError):
            OpenTelemetryConfig(exporter="jaeger")


# ---------------------------------------------------------------------------
# A2AConfig
# ---------------------------------------------------------------------------


class TestA2AConfig:
    def test_defaults(self):
        cfg = A2AConfig()
        assert cfg.enabled is False
        assert cfg.agent_name == "Question Engine"
        assert cfg.agent_description == "Cognitive architecture for autonomous knowledge discovery"
        assert cfg.require_auth is True

    def test_extra_fields_allowed(self):
        cfg = A2AConfig(enabled=True, custom_field="hello", another=42)
        assert cfg.enabled is True
        assert cfg.custom_field == "hello"  # type: ignore[attr-defined]
        assert cfg.another == 42  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# QEConfig (root)
# ---------------------------------------------------------------------------


class TestQEConfig:
    def test_all_sub_configs_present_with_defaults(self):
        cfg = QEConfig()
        assert isinstance(cfg.budget, BudgetConfig)
        assert isinstance(cfg.runtime, RuntimeConfig)
        assert isinstance(cfg.models, ModelsConfig)
        assert isinstance(cfg.substrate, SubstrateConfig)
        assert isinstance(cfg.bus, BusConfig)
        assert isinstance(cfg.security, SecurityConfig)
        assert isinstance(cfg.scout, ScoutConfig)
        assert isinstance(cfg.harvest, HarvestConfig)
        assert isinstance(cfg.guardrails, GuardrailsConfig)
        assert isinstance(cfg.otel, OpenTelemetryConfig)
        assert isinstance(cfg.a2a, A2AConfig)

    def test_construct_from_empty_dict(self):
        cfg = QEConfig.model_validate({})
        assert cfg.budget.monthly_limit_usd == 50.0
        assert cfg.runtime.log_level == "INFO"
        assert cfg.models.fast == "gpt-4o-mini"

    def test_partial_override(self):
        cfg = QEConfig.model_validate({"budget": {"monthly_limit_usd": 200.0}})
        assert cfg.budget.monthly_limit_usd == 200.0
        # other sub-configs still defaults
        assert cfg.runtime.log_level == "INFO"
        assert cfg.models.fast == "gpt-4o-mini"

    def test_invalid_nested_field(self):
        with pytest.raises(ValidationError):
            QEConfig.model_validate({"budget": {"monthly_limit_usd": -5}})


# ---------------------------------------------------------------------------
# load_config()
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_returns_defaults_for_missing_file(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.toml")
        assert cfg.budget.monthly_limit_usd == 50.0
        assert cfg.runtime.log_level == "INFO"

    def test_loads_from_file(self, tmp_path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            '[budget]\nmonthly_limit_usd = 999.0\n\n'
            '[runtime]\nlog_level = "DEBUG"\n'
        )
        cfg = load_config(toml_file)
        assert cfg.budget.monthly_limit_usd == 999.0
        assert cfg.runtime.log_level == "DEBUG"
        # unspecified sections keep defaults
        assert cfg.models.fast == "gpt-4o-mini"

    def test_validates_invalid_toml_values(self, tmp_path):
        toml_file = tmp_path / "bad.toml"
        toml_file.write_text('[budget]\nmonthly_limit_usd = -10.0\n')
        with pytest.raises(ValidationError):
            load_config(toml_file)

    def test_loads_all_sections(self, tmp_path):
        toml_file = tmp_path / "full.toml"
        toml_file.write_text(
            '[budget]\nmonthly_limit_usd = 100.0\nalert_at_pct = 0.90\n\n'
            '[runtime]\nlog_level = "WARNING"\n\n'
            '[models]\nfast = "custom-fast"\nbalanced = "custom-balanced"\n'
            'powerful = "custom-powerful"\n\n'
            '[substrate]\ndb_path = "/tmp/test.db"\n\n'
            '[bus]\ntype = "redis"\nredis_url = "redis://localhost"\n\n'
            '[security]\nrequire_auth = true\napi_key = "key123"\n\n'
            '[scout]\nenabled = true\npoll_interval_seconds = 7200\n\n'
            '[guardrails]\npii_detection_enabled = true\n\n'
            '[otel]\nenabled = true\nexporter = "otlp"\n\n'
            '[a2a]\nenabled = true\n'
        )
        cfg = load_config(toml_file)
        assert cfg.budget.monthly_limit_usd == 100.0
        assert cfg.budget.alert_at_pct == 0.90
        assert cfg.runtime.log_level == "WARNING"
        assert cfg.models.fast == "custom-fast"
        assert cfg.substrate.db_path == "/tmp/test.db"
        assert cfg.bus.type == "redis"
        assert cfg.security.require_auth is True
        assert cfg.scout.enabled is True
        assert cfg.scout.poll_interval_seconds == 7200
        assert cfg.guardrails.pii_detection_enabled is True
        assert cfg.otel.enabled is True
        assert cfg.otel.exporter == "otlp"
        assert cfg.a2a.enabled is True

    def test_defaults_for_none_path(self, tmp_path, monkeypatch):
        # When path=None and no config.toml in cwd, should return defaults
        monkeypatch.chdir(tmp_path)
        cfg = load_config(None)
        assert cfg.budget.monthly_limit_usd == 50.0

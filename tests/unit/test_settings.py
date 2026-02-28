"""Tests for the Settings panel backend: setup helpers, budget limits, API endpoints."""

from __future__ import annotations

import textwrap
from unittest.mock import patch

import pytest

from qe.api.setup import get_settings, save_settings
from qe.runtime.budget import BudgetTracker

# ── get_settings / save_settings ─────────────────────────────────────────


class TestGetSettings:
    def test_defaults_when_config_missing(self, tmp_path):
        """Returns defaults when config.toml doesn't exist."""
        with patch("qe.api.setup.CONFIG_PATH", tmp_path / "missing.toml"):
            result = get_settings()
        assert result["budget"]["monthly_limit_usd"] == 50.0
        assert result["budget"]["alert_at_pct"] == 0.80
        assert result["runtime"]["log_level"] == "INFO"
        assert result["runtime"]["hil_timeout_seconds"] == 3600

    def test_reads_existing_values(self, tmp_path):
        """Reads values from an existing config.toml."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [budget]
            monthly_limit_usd = 100.0
            alert_at_pct = 0.90

            [runtime]
            log_level = "DEBUG"
            hil_timeout_seconds = 7200
        """))
        with patch("qe.api.setup.CONFIG_PATH", config_file):
            result = get_settings()
        assert result["budget"]["monthly_limit_usd"] == 100.0
        assert result["budget"]["alert_at_pct"] == 0.90
        assert result["runtime"]["log_level"] == "DEBUG"
        assert result["runtime"]["hil_timeout_seconds"] == 7200

    def test_fills_defaults_for_missing_keys(self, tmp_path):
        """When config has partial sections, missing keys get defaults."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[budget]\nmonthly_limit_usd = 25.0\n")
        with patch("qe.api.setup.CONFIG_PATH", config_file):
            result = get_settings()
        assert result["budget"]["monthly_limit_usd"] == 25.0
        assert result["budget"]["alert_at_pct"] == 0.80  # default
        assert result["runtime"]["log_level"] == "INFO"  # default


class TestSaveSettings:
    def test_persists_budget_and_runtime(self, tmp_path):
        """Saves budget and runtime sections to config.toml."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        with patch("qe.api.setup.CONFIG_PATH", config_file):
            save_settings({
                "budget": {"monthly_limit_usd": 75.0, "alert_at_pct": 0.85},
                "runtime": {"log_level": "WARNING"},
            })
            result = get_settings()
        assert result["budget"]["monthly_limit_usd"] == 75.0
        assert result["budget"]["alert_at_pct"] == 0.85
        assert result["runtime"]["log_level"] == "WARNING"

    def test_preserves_existing_sections(self, tmp_path):
        """Existing sections like [models] survive a save_settings call."""
        config_file = tmp_path / "config.toml"
        config_file.write_text(textwrap.dedent("""\
            [models]
            fast = "gpt-4o-mini"
            balanced = "gpt-4o"

            [budget]
            monthly_limit_usd = 50.0
        """))
        with patch("qe.api.setup.CONFIG_PATH", config_file):
            save_settings({"budget": {"monthly_limit_usd": 100.0}})
            # Re-read raw to check models section survived
            import tomllib
            with config_file.open("rb") as f:
                raw = tomllib.load(f)
        assert raw["models"]["fast"] == "gpt-4o-mini"
        assert raw["models"]["balanced"] == "gpt-4o"
        assert raw["budget"]["monthly_limit_usd"] == 100.0

    def test_ignores_unknown_sections(self, tmp_path):
        """save_settings only writes budget and runtime, ignores anything else."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        with patch("qe.api.setup.CONFIG_PATH", config_file):
            save_settings({
                "budget": {"monthly_limit_usd": 10.0},
                "models": {"fast": "evil-model"},  # should be ignored
            })
            import tomllib
            with config_file.open("rb") as f:
                raw = tomllib.load(f)
        assert "models" not in raw
        assert raw["budget"]["monthly_limit_usd"] == 10.0


# ── BudgetTracker.update_limits ──────────────────────────────────────────


class TestBudgetTrackerUpdateLimits:
    def test_updates_monthly_limit(self):
        bt = BudgetTracker(monthly_limit_usd=50.0)
        bt.update_limits(monthly_limit_usd=100.0)
        assert bt.monthly_limit_usd == 100.0

    def test_updates_alert_threshold(self):
        bt = BudgetTracker(alert_at_pct=0.80)
        bt.update_limits(alert_at_pct=0.90)
        assert bt.alert_at_pct == 0.90

    def test_resets_alert_flag_on_threshold_change(self):
        bt = BudgetTracker(monthly_limit_usd=10.0, alert_at_pct=0.80)
        # Record enough to trigger alert
        bt.record_cost("test-model", 9.0)
        assert bt._alerted is True
        # Change threshold — should re-arm
        bt.update_limits(alert_at_pct=0.95)
        assert bt._alerted is False

    def test_none_values_leave_current(self):
        bt = BudgetTracker(monthly_limit_usd=50.0, alert_at_pct=0.80)
        bt.update_limits(monthly_limit_usd=None, alert_at_pct=None)
        assert bt.monthly_limit_usd == 50.0
        assert bt.alert_at_pct == 0.80

    def test_updates_both_at_once(self):
        bt = BudgetTracker(monthly_limit_usd=50.0, alert_at_pct=0.80)
        bt.update_limits(monthly_limit_usd=200.0, alert_at_pct=0.95)
        assert bt.monthly_limit_usd == 200.0
        assert bt.alert_at_pct == 0.95

    def test_remaining_pct_reflects_new_limit(self):
        bt = BudgetTracker(monthly_limit_usd=100.0)
        bt.record_cost("m", 50.0)
        assert bt.remaining_pct() == pytest.approx(0.5)
        bt.update_limits(monthly_limit_usd=200.0)
        assert bt.remaining_pct() == pytest.approx(0.75)


# ── API Endpoints ────────────────────────────────────────────────────────


class TestSettingsEndpoints:
    @pytest.fixture
    def client(self):
        from starlette.testclient import TestClient

        from qe.api.app import app
        return TestClient(app, raise_server_exceptions=False)

    def test_get_settings_returns_config(self, client, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("[budget]\nmonthly_limit_usd = 42.0\n")
        with patch("qe.api.setup.CONFIG_PATH", config_file):
            res = client.get("/api/settings")
        assert res.status_code == 200
        assert res.json()["budget"]["monthly_limit_usd"] == 42.0

    def test_post_settings_returns_503_without_engine(self, client, tmp_path):
        """POST /api/settings saves to config even without engine."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        with patch("qe.api.setup.CONFIG_PATH", config_file):
            res = client.post("/api/settings", json={
                "budget": {"monthly_limit_usd": 99.0},
            })
        # Should succeed (config is written regardless of engine state)
        assert res.status_code == 200
        assert res.json()["status"] == "saved"

    def test_reset_circuit_returns_503_without_engine(self, client):
        res = client.post("/api/services/researcher/reset-circuit")
        assert res.status_code == 503

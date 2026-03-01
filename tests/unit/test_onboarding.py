"""Tests for onboarding flow: health/ready phases, CHANNELS constant validation."""

import pytest
from fastapi.testclient import TestClient

from qe.api.setup import CHANNELS


@pytest.fixture
def client():
    from qe.api.app import app

    return TestClient(app, raise_server_exceptions=False)


class TestHealthReady:
    def test_health_ready_returns_phases(self, client):
        """GET /api/health/ready returns phases dict for hatching screen polling."""
        resp = client.get("/api/health/ready")
        data = resp.json()
        assert "phases" in data
        phases = data["phases"]
        assert "substrate_ready" in phases
        assert "event_log_ready" in phases
        assert "services_subscribed" in phases
        assert "supervisor_ready" in phases
        assert isinstance(phases["substrate_ready"], bool)

    def test_health_ready_returns_ready_flag(self, client):
        """GET /api/health/ready includes top-level 'ready' boolean."""
        resp = client.get("/api/health/ready")
        data = resp.json()
        assert "ready" in data
        assert isinstance(data["ready"], bool)


class TestChannelsConstant:
    def test_expected_channel_ids(self):
        """CHANNELS constant has entries for web, telegram, slack, email."""
        ids = [ch["id"] for ch in CHANNELS]
        assert "web" in ids
        assert "telegram" in ids
        assert "slack" in ids
        assert "email" in ids

    def test_web_channel_always_on(self):
        """Web channel has always_on=True."""
        web = [ch for ch in CHANNELS if ch["id"] == "web"][0]
        assert web.get("always_on") is True
        assert web["env_vars"] == []

    def test_channel_env_var_keys_are_strings(self):
        """All channel env_var keys are non-empty strings."""
        for ch in CHANNELS:
            for ev in ch.get("env_vars", []):
                assert isinstance(ev["key"], str)
                assert len(ev["key"]) > 0
                assert isinstance(ev["label"], str)
                assert ev["type"] in ("text", "password")

    def test_channel_env_var_keys_match_expected(self):
        """Channel env_var keys match what adapters expect from os.environ."""
        # Telegram
        tg = [ch for ch in CHANNELS if ch["id"] == "telegram"][0]
        tg_keys = [ev["key"] for ev in tg["env_vars"]]
        assert "TELEGRAM_BOT_TOKEN" in tg_keys

        # Slack
        slack = [ch for ch in CHANNELS if ch["id"] == "slack"][0]
        slack_keys = [ev["key"] for ev in slack["env_vars"]]
        assert "SLACK_BOT_TOKEN" in slack_keys
        assert "SLACK_APP_TOKEN" in slack_keys

        # Email
        email = [ch for ch in CHANNELS if ch["id"] == "email"][0]
        email_keys = [ev["key"] for ev in email["env_vars"]]
        assert "EMAIL_IMAP_HOST" in email_keys
        assert "EMAIL_SMTP_HOST" in email_keys
        assert "EMAIL_USERNAME" in email_keys
        assert "EMAIL_PASSWORD" in email_keys

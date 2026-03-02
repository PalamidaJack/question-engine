"""Tests for the first-run setup module."""

import pytest
from fastapi.testclient import TestClient

from qe.api.setup import (
    get_configured_channels,
    get_configured_providers,
    is_setup_complete,
    mask_key,
    save_setup,
)

# ── Unit tests for setup helpers ────────────────────────────────────────────


class TestMaskKey:
    def test_long_key(self):
        assert mask_key("sk-abcdefghijklmnop") == "sk-...mnop"

    def test_short_key(self):
        assert mask_key("short") == "****"

    def test_exactly_eight(self):
        assert mask_key("12345678") == "****"

    def test_nine_chars(self):
        assert mask_key("123456789") == "123...6789"


class TestIsSetupComplete:
    def test_no_env_file(self, tmp_path):
        assert is_setup_complete(env_path=tmp_path / ".env") is False

    def test_empty_env_file(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("")
        assert is_setup_complete(env_path=env) is False

    def test_placeholder_key(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=sk-your-key-here\n")
        assert is_setup_complete(env_path=env) is False

    def test_real_key(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=sk-realkey1234567890\n")
        assert is_setup_complete(env_path=env) is True

    def test_comment_only(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("# OPENAI_API_KEY=sk-realkey\n")
        assert is_setup_complete(env_path=env) is False

    def test_multiple_keys(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=sk-abc123456789\nANTHROPIC_API_KEY=sk-ant-xyz\n")
        assert is_setup_complete(env_path=env) is True


class TestGetConfiguredProviders:
    def test_no_env(self, tmp_path):
        result = get_configured_providers(env_path=tmp_path / ".env")
        # Ollama should still be configured (no key needed)
        ollama = [p for p in result if p["name"] == "Ollama (local)"]
        assert len(ollama) == 1
        assert ollama[0]["configured"] is True
        assert ollama[0]["masked_key"] is None

    def test_with_key(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=sk-abcdefghijklmnop\n")
        result = get_configured_providers(env_path=env)
        openai = [p for p in result if p["name"] == "OpenAI"][0]
        assert openai["configured"] is True
        assert openai["masked_key"] == "sk-...mnop"
        # Full key must never appear
        for p in result:
            if p["masked_key"]:
                assert "abcdefghijklmnop" not in p["masked_key"]

    def test_masks_all_keys(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=sk-abcdefghijklmnop\nANTHROPIC_API_KEY=sk-ant-very-long-key-value\n")
        result = get_configured_providers(env_path=env)
        for p in result:
            if p["masked_key"]:
                assert len(p["masked_key"]) < 15  # masked is short


class TestSaveSetup:
    def test_writes_env(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        config = tmp_path / "config.toml"
        config.write_text(
            '[models]\nfast = "gpt-4o-mini"\nbalanced = "gpt-4o"\npowerful = "o1-preview"\n'
        )
        monkeypatch.setattr("qe.api.setup.CONFIG_PATH", config)

        save_setup(
            providers={"OPENAI_API_KEY": "sk-testkey123456789"},
            tier_config={"fast": {"provider": "OpenAI", "model": "gpt-4o-mini"}},
            env_path=env,
        )

        content = env.read_text()
        assert "OPENAI_API_KEY=sk-testkey123456789" in content

    def test_writes_config_toml(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        config = tmp_path / "config.toml"
        config.write_text(
            '[models]\nfast = "gpt-4o-mini"\nbalanced = "gpt-4o"\npowerful = "o1-preview"\n'
        )
        monkeypatch.setattr("qe.api.setup.CONFIG_PATH", config)

        save_setup(
            providers={"ANTHROPIC_API_KEY": "sk-ant-key"},
            tier_config={
                "fast": {"provider": "Anthropic", "model": "claude-haiku-4-5-20251001"},
                "balanced": {"provider": "Anthropic", "model": "claude-sonnet-4-20250514"},
            },
            env_path=env,
        )

        content = config.read_text()
        assert "claude-haiku" in content
        assert "claude-sonnet" in content

    def test_preserves_existing_env(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("EXISTING_VAR=keep-me\n")
        config = tmp_path / "config.toml"
        config.write_text("[models]\n")
        monkeypatch.setattr("qe.api.setup.CONFIG_PATH", config)

        save_setup(
            providers={"OPENAI_API_KEY": "sk-new"},
            tier_config={},
            env_path=env,
        )

        content = env.read_text()
        assert "EXISTING_VAR=keep-me" in content
        assert "OPENAI_API_KEY=sk-new" in content

    def test_ignores_invalid_tiers(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        config = tmp_path / "config.toml"
        config.write_text('[models]\nfast = "gpt-4o-mini"\n')
        monkeypatch.setattr("qe.api.setup.CONFIG_PATH", config)

        save_setup(
            providers={},
            tier_config={"invalid_tier": {"provider": "x", "model": "y"}},
            env_path=env,
        )

        content = config.read_text()
        assert "invalid_tier" not in content


class TestSaveSetupWithChannels:
    def test_writes_channel_env_vars(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        config = tmp_path / "config.toml"
        config.write_text("[models]\n")
        monkeypatch.setattr("qe.api.setup.CONFIG_PATH", config)

        save_setup(
            providers={},
            tier_config={},
            env_path=env,
            channels={"TELEGRAM_BOT_TOKEN": "123:ABC-xyz"},
        )

        content = env.read_text()
        assert "TELEGRAM_BOT_TOKEN=123:ABC-xyz" in content

    def test_writes_multiple_channel_vars(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        config = tmp_path / "config.toml"
        config.write_text("[models]\n")
        monkeypatch.setattr("qe.api.setup.CONFIG_PATH", config)

        save_setup(
            providers={"OPENAI_API_KEY": "sk-testkey123456789"},
            tier_config={},
            env_path=env,
            channels={
                "SLACK_BOT_TOKEN": "xoxb-bot-token-value",
                "SLACK_APP_TOKEN": "xapp-app-token-value",
            },
        )

        content = env.read_text()
        assert "SLACK_BOT_TOKEN=xoxb-bot-token-value" in content
        assert "SLACK_APP_TOKEN=xapp-app-token-value" in content
        assert "OPENAI_API_KEY=sk-testkey123456789" in content

    def test_empty_channels_ignored(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        config = tmp_path / "config.toml"
        config.write_text("[models]\n")
        monkeypatch.setattr("qe.api.setup.CONFIG_PATH", config)

        save_setup(
            providers={"OPENAI_API_KEY": "sk-testkey123456789"},
            tier_config={},
            env_path=env,
            channels={},
        )

        content = env.read_text()
        assert "OPENAI_API_KEY=sk-testkey123456789" in content

    def test_none_channels_handled(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        config = tmp_path / "config.toml"
        config.write_text("[models]\n")
        monkeypatch.setattr("qe.api.setup.CONFIG_PATH", config)

        save_setup(
            providers={"OPENAI_API_KEY": "sk-testkey123456789"},
            tier_config={},
            env_path=env,
            channels=None,
        )

        content = env.read_text()
        assert "OPENAI_API_KEY=sk-testkey123456789" in content


class TestGetConfiguredChannels:
    def test_no_env_web_only(self, tmp_path):
        result = get_configured_channels(env_path=tmp_path / ".env")
        web = [ch for ch in result if ch["id"] == "web"]
        assert len(web) == 1
        assert web[0]["configured"] is True
        assert web[0]["always_on"] is True
        # Non-web channels should not be configured
        for ch in result:
            if ch["id"] != "web":
                assert ch["configured"] is False

    def test_telegram_configured(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11\n")
        result = get_configured_channels(env_path=env)
        tg = [ch for ch in result if ch["id"] == "telegram"][0]
        assert tg["configured"] is True
        assert tg["env_vars"][0]["has_value"] is True
        assert tg["env_vars"][0]["masked_value"] is not None
        # Masked — should not contain full token
        assert "ABC-DEF" not in tg["env_vars"][0]["masked_value"]

    def test_slack_needs_both_tokens(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("SLACK_BOT_TOKEN=xoxb-bot-token-value\n")
        result = get_configured_channels(env_path=env)
        slack = [ch for ch in result if ch["id"] == "slack"][0]
        assert slack["configured"] is False  # missing SLACK_APP_TOKEN

    def test_slack_fully_configured(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("SLACK_BOT_TOKEN=xoxb-bot-token-value\nSLACK_APP_TOKEN=xapp-app-token-value\n")
        result = get_configured_channels(env_path=env)
        slack = [ch for ch in result if ch["id"] == "slack"][0]
        assert slack["configured"] is True

    def test_email_fully_configured(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "EMAIL_IMAP_HOST=imap.example.com\n"
            "EMAIL_SMTP_HOST=smtp.example.com\n"
            "EMAIL_USERNAME=user@example.com\n"
            "EMAIL_PASSWORD=supersecretpassword12345\n"
        )
        result = get_configured_channels(env_path=env)
        email = [ch for ch in result if ch["id"] == "email"][0]
        assert email["configured"] is True
        # Check text fields show full value, password fields are masked
        for ev in email["env_vars"]:
            assert ev["has_value"] is True
            if ev["type"] == "text":
                raw = env.read_text().split(f"{ev['key']}=")[1].split("\n")[0]
                assert ev["masked_value"] == raw
            elif ev["type"] == "password":
                assert len(ev["masked_value"]) < len("supersecretpassword12345")

    def test_masked_values_for_passwords(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("TELEGRAM_BOT_TOKEN=123456789:very-long-bot-token-string\n")
        result = get_configured_channels(env_path=env)
        tg = [ch for ch in result if ch["id"] == "telegram"][0]
        # Password type should be masked
        assert tg["env_vars"][0]["type"] == "password"
        assert tg["env_vars"][0]["masked_value"] is not None
        assert len(tg["env_vars"][0]["masked_value"]) <= 15


# ── API endpoint tests ──────────────────────────────────────────────────────


@pytest.fixture
def client():
    from qe.api.app import app

    return TestClient(app, raise_server_exceptions=False)


def test_setup_status_endpoint(client):
    resp = client.get("/api/setup/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "complete" in data
    assert "providers" in data
    assert "tiers" in data
    assert isinstance(data["providers"], list)
    assert isinstance(data["tiers"], dict)


def test_setup_providers_endpoint(client):
    resp = client.get("/api/setup/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert "providers" in data
    providers = data["providers"]
    assert len(providers) == 15
    names = [p["name"] for p in providers]
    assert "OpenAI" in names
    assert "Anthropic" in names
    assert "Kilo Code" in names
    assert "Ollama (local)" in names
    # Each provider has tier_defaults
    for p in providers:
        assert "tier_defaults" in p
        assert "fast" in p["tier_defaults"]


def test_setup_save_requires_body(client):
    from unittest.mock import patch

    with patch("qe.api.app.is_setup_complete", return_value=False):
        resp = client.post("/api/setup/save", json={})
    assert resp.status_code == 400


def test_setup_save_blocked_after_complete(client):
    from unittest.mock import patch

    with patch("qe.api.app.is_setup_complete", return_value=True):
        resp = client.post("/api/setup/save", json={"providers": {"k": "v"}})
    assert resp.status_code == 403
    assert "already complete" in resp.json()["error"].lower()


def test_setup_status_never_returns_full_keys(client):
    """Ensure masked keys never contain a full key value."""
    resp = client.get("/api/setup/status")
    data = resp.json()
    for p in data["providers"]:
        if p["masked_key"]:
            # Masked keys should be short
            assert len(p["masked_key"]) <= 15


def test_setup_status_includes_channels(client):
    """GET /api/setup/status includes channels in response."""
    resp = client.get("/api/setup/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "channels" in data
    channels = data["channels"]
    assert isinstance(channels, list)
    assert len(channels) >= 4
    ids = [ch["id"] for ch in channels]
    assert "web" in ids
    assert "telegram" in ids


def test_setup_channels_endpoint(client):
    """GET /api/setup/channels returns static channel list."""
    resp = client.get("/api/setup/channels")
    assert resp.status_code == 200
    data = resp.json()
    assert "channels" in data
    channels = data["channels"]
    assert len(channels) == 4
    web = [ch for ch in channels if ch["id"] == "web"][0]
    assert web["always_on"] is True
    assert web["env_vars"] == []
    # Telegram has one env_var
    tg = [ch for ch in channels if ch["id"] == "telegram"][0]
    assert len(tg["env_vars"]) == 1
    assert tg["env_vars"][0]["key"] == "TELEGRAM_BOT_TOKEN"
    # Slack has two env_vars
    slack = [ch for ch in channels if ch["id"] == "slack"][0]
    assert len(slack["env_vars"]) == 2


def test_setup_reconfigure_works_after_complete(client):
    """POST /api/setup/reconfigure works when setup is already complete."""
    from unittest.mock import patch

    with patch("qe.api.app.is_setup_complete", return_value=True), \
         patch("qe.api.app.save_setup") as mock_save:
        resp = client.post("/api/setup/reconfigure", json={
            "providers": {"OPENAI_API_KEY": "sk-new-key-value-123456"},
            "tiers": {"fast": {"model": "gpt-4o-mini"}},
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "saved"
    assert "note" in data
    mock_save.assert_called_once()


def test_setup_reconfigure_rejects_empty_body(client):
    """POST /api/setup/reconfigure rejects empty body."""
    resp = client.post("/api/setup/reconfigure", json={})
    assert resp.status_code == 400


def test_setup_save_403_message_points_to_reconfigure(client):
    """POST /api/setup/save 403 message mentions reconfigure endpoint."""
    from unittest.mock import patch

    with patch("qe.api.app.is_setup_complete", return_value=True):
        resp = client.post("/api/setup/save", json={"providers": {"k": "v"}})
    assert resp.status_code == 403
    assert "reconfigure" in resp.json()["error"].lower()

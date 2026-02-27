"""Tests for the first-run setup module."""

import pytest
from fastapi.testclient import TestClient

from qe.api.setup import (
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
    assert len(providers) == 8
    names = [p["name"] for p in providers]
    assert "OpenAI" in names
    assert "Anthropic" in names
    assert "Ollama (local)" in names
    # Each provider has tier_defaults
    for p in providers:
        assert "tier_defaults" in p
        assert "fast" in p["tier_defaults"]


def test_setup_save_requires_body(client):
    resp = client.post("/api/setup/save", json={})
    assert resp.status_code == 400


def test_setup_status_never_returns_full_keys(client):
    """Ensure masked keys never contain a full key value."""
    resp = client.get("/api/setup/status")
    data = resp.json()
    for p in data["providers"]:
        if p["masked_key"]:
            # Masked keys should be short
            assert len(p["masked_key"]) <= 15

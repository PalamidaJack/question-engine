"""Tests for new CLI command groups: a2a, memory, arena, models, doctor, init.

Uses typer.testing.CliRunner with mocked httpx calls and config loading
to verify CLI commands produce expected output without hitting real services.

Since CLI functions use lazy imports (``import httpx`` inside function bodies),
we patch ``httpx.get`` / ``httpx.post`` / ``httpx.delete`` directly on the
httpx module.  Similarly, ``load_config`` is patched at ``qe.config.load_config``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from qe.cli.main import app

runner = CliRunner()


# ── Helpers ────────────────────────────────────────────────────────────────


def _mock_response(json_data: dict | list, status_code: int = 200) -> MagicMock:
    """Build a mock httpx.Response with .json() and .status_code."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = json.dumps(json_data)
    resp.raise_for_status = MagicMock()
    return resp


def _mock_config() -> MagicMock:
    """Build a mock QEConfig with all sections the CLI reads."""
    cfg = MagicMock()
    cfg.models.fast = "gpt-4o-mini"
    cfg.models.balanced = "gpt-4o"
    cfg.models.powerful = "o1-preview"
    cfg.models.local = None
    cfg.runtime.log_level = "INFO"
    cfg.budget.monthly_limit_usd = 50.0
    cfg.guardrails.enabled = True
    cfg.otel.enabled = False
    cfg.a2a.enabled = True
    cfg.scout.enabled = False
    cfg.guardrails.content_filter_enabled = True
    cfg.guardrails.pii_detection_enabled = False
    cfg.guardrails.cost_guard_enabled = True
    cfg.guardrails.cost_guard_threshold_usd = 5.0
    cfg.guardrails.model_dump.return_value = {
        "enabled": True,
        "content_filter_enabled": True,
        "pii_detection_enabled": False,
        "cost_guard_enabled": True,
        "cost_guard_threshold_usd": 5.0,
        "hallucination_guard_enabled": False,
    }
    cfg.otel.exporter = "console"
    cfg.otel.otlp_endpoint = None
    cfg.otel.service_name = "question-engine"
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# A2A Peer Commands
# ═══════════════════════════════════════════════════════════════════════════


class TestA2AListPeers:
    """Tests for ``a2a list-peers``."""

    @patch("httpx.get")
    def test_list_peers_with_results(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({
            "total_peers": 2,
            "peers": [
                {
                    "peer_id": "peer_001",
                    "name": "Alpha Agent",
                    "url": "http://alpha:8000",
                    "healthy": True,
                    "capabilities": ["research", "analysis"],
                },
                {
                    "peer_id": "peer_002",
                    "name": "Beta Agent",
                    "url": "http://beta:8000",
                    "healthy": False,
                    "capabilities": ["search"],
                },
            ],
        })
        result = runner.invoke(app, ["a2a", "list-peers"])
        assert result.exit_code == 0
        assert "peer_001" in result.output
        assert "Alpha Agent" in result.output
        assert "peer_002" in result.output
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert "/api/a2a/peers" in call_url

    @patch("httpx.get")
    def test_list_peers_empty(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({"total_peers": 0, "peers": []})
        result = runner.invoke(app, ["a2a", "list-peers"])
        assert result.exit_code == 0
        assert "No peers registered" in result.output

    @patch("httpx.get")
    def test_list_peers_custom_url(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({"total_peers": 0, "peers": []})
        result = runner.invoke(app, ["a2a", "list-peers", "--url", "http://myhost:9000"])
        assert result.exit_code == 0
        call_url = mock_get.call_args[0][0]
        assert call_url.startswith("http://myhost:9000")

    @patch("httpx.get")
    def test_list_peers_connect_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        result = runner.invoke(app, ["a2a", "list-peers"])
        assert result.exit_code == 1
        assert "Cannot connect" in result.output


class TestA2ARegisterPeer:
    """Tests for ``a2a register-peer``."""

    @patch("httpx.post")
    def test_register_success(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _mock_response(
            {"peer_id": "peer_new", "name": "New Agent"}, status_code=200,
        )
        result = runner.invoke(app, ["a2a", "register-peer", "http://new-agent:8000"])
        assert result.exit_code == 0
        assert "Registered peer" in result.output
        assert "peer_new" in result.output

    @patch("httpx.post")
    def test_register_error_response(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _mock_response(
            {"detail": "Discovery failed"}, status_code=400,
        )
        result = runner.invoke(app, ["a2a", "register-peer", "http://bad-agent:8000"])
        assert result.exit_code == 0
        assert "Error" in result.output

    @patch("httpx.post")
    def test_register_connect_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(app, ["a2a", "register-peer", "http://x:8000"])
        assert result.exit_code == 1
        assert "Cannot connect" in result.output


class TestA2ARemovePeer:
    """Tests for ``a2a remove-peer``."""

    @patch("httpx.delete")
    def test_remove_success(self, mock_delete: MagicMock) -> None:
        mock_delete.return_value = _mock_response({}, status_code=200)
        result = runner.invoke(app, ["a2a", "remove-peer", "peer_001"])
        assert result.exit_code == 0
        assert "Removed peer" in result.output
        assert "peer_001" in result.output

    @patch("httpx.delete")
    def test_remove_not_found(self, mock_delete: MagicMock) -> None:
        mock_delete.return_value = _mock_response(
            {"detail": "not found"}, status_code=404,
        )
        result = runner.invoke(app, ["a2a", "remove-peer", "peer_999"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower()

    @patch("httpx.delete")
    def test_remove_connect_error(self, mock_delete: MagicMock) -> None:
        mock_delete.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(app, ["a2a", "remove-peer", "peer_001"])
        assert result.exit_code == 1


class TestA2ACheckPeer:
    """Tests for ``a2a check-peer``."""

    @patch("httpx.get")
    def test_check_healthy(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({"healthy": True})
        result = runner.invoke(app, ["a2a", "check-peer", "peer_001"])
        assert result.exit_code == 0
        assert "healthy" in result.output.lower()

    @patch("httpx.get")
    def test_check_unhealthy(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({"healthy": False})
        result = runner.invoke(app, ["a2a", "check-peer", "peer_001"])
        assert result.exit_code == 0
        assert "unreachable" in result.output.lower()

    @patch("httpx.get")
    def test_check_connect_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(app, ["a2a", "check-peer", "peer_001"])
        assert result.exit_code == 1


# ═══════════════════════════════════════════════════════════════════════════
# Memory Commands
# ═══════════════════════════════════════════════════════════════════════════


class TestMemorySearch:
    """Tests for ``memory search``."""

    @patch("httpx.post")
    def test_search_default_params(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _mock_response({"results": [{"text": "found"}]})
        result = runner.invoke(app, ["memory", "search", "lithium battery"])
        assert result.exit_code == 0
        body = mock_post.call_args[1]["json"]
        assert body["query"] == "lithium battery"
        assert body["tier"] == "all"
        assert body["top_k"] == 10

    @patch("httpx.post")
    def test_search_custom_tier_and_topk(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _mock_response({"results": []})
        result = runner.invoke(
            app,
            ["memory", "search", "query text", "--tier", "episodic", "--top-k", "5"],
        )
        assert result.exit_code == 0
        body = mock_post.call_args[1]["json"]
        assert body["tier"] == "episodic"
        assert body["top_k"] == 5

    @patch("httpx.post")
    def test_search_connect_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(app, ["memory", "search", "test"])
        assert result.exit_code == 1
        assert "Cannot connect" in result.output


class TestMemoryTiers:
    """Tests for ``memory tiers``."""

    @patch("httpx.get")
    def test_tiers_display(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({
            "episodic": {"count": 42, "hot_count": 10},
            "belief": {"count": 100, "hypotheses": 5},
            "procedural": {"templates": 20, "sequences": 8},
        })
        result = runner.invoke(app, ["memory", "tiers"])
        assert result.exit_code == 0
        assert "episodic" in result.output
        assert "belief" in result.output
        assert "procedural" in result.output

    @patch("httpx.get")
    def test_tiers_connect_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(app, ["memory", "tiers"])
        assert result.exit_code == 1


class TestMemoryExport:
    """Tests for ``memory export``."""

    @patch("httpx.get")
    def test_export_to_stdout(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({"episodic": [], "beliefs": []})
        result = runner.invoke(app, ["memory", "export"])
        assert result.exit_code == 0

    @patch("httpx.get")
    def test_export_to_file(self, mock_get: MagicMock, tmp_path: Path) -> None:
        mock_get.return_value = _mock_response(
            {"episodic": [{"id": 1}], "beliefs": [{"id": 2}]},
        )
        out_file = tmp_path / "mem_export.json"
        result = runner.invoke(app, ["memory", "export", "--output", str(out_file)])
        assert result.exit_code == 0
        assert "exported" in result.output.lower()
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "episodic" in data

    @patch("httpx.get")
    def test_export_connect_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(app, ["memory", "export"])
        assert result.exit_code == 1


# ═══════════════════════════════════════════════════════════════════════════
# Arena Commands
# ═══════════════════════════════════════════════════════════════════════════


class TestArenaStatus:
    """Tests for ``arena status``."""

    @patch("httpx.get")
    def test_arena_status_with_rankings(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({
            "rankings": [
                {"agent_id": "agent_A", "elo": 1250, "wins": 5, "losses": 2, "draws": 1},
                {"agent_id": "agent_B", "elo": 1180, "wins": 3, "losses": 4, "draws": 1},
            ],
        })
        result = runner.invoke(app, ["arena", "status"])
        assert result.exit_code == 0
        assert "agent_A" in result.output
        assert "1250" in result.output
        assert "agent_B" in result.output

    @patch("httpx.get")
    def test_arena_status_no_data(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({"rankings": []})
        result = runner.invoke(app, ["arena", "status"])
        assert result.exit_code == 0
        assert "No arena data" in result.output

    @patch("httpx.get")
    def test_arena_status_connect_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(app, ["arena", "status"])
        assert result.exit_code == 1


# ═══════════════════════════════════════════════════════════════════════════
# Models Commands
# ═══════════════════════════════════════════════════════════════════════════


class TestModelsList:
    """Tests for ``models list``."""

    @patch("qe.config.load_config")
    def test_models_list_all_tiers(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_config()
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert "gpt-4o-mini" in result.output
        assert "gpt-4o" in result.output
        assert "o1-preview" in result.output

    @patch("qe.config.load_config")
    def test_models_list_local_not_set(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_config()
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert "(not set)" in result.output

    @patch("qe.config.load_config")
    def test_models_list_local_set(self, mock_load: MagicMock) -> None:
        cfg = _mock_config()
        cfg.models.local = "ollama/llama3"
        mock_load.return_value = cfg
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert "ollama/llama3" in result.output


class TestModelsCheck:
    """Tests for ``models check``."""

    @patch("httpx.get")
    def test_check_shows_budget_and_breakers(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({
            "budget": {"used_usd": 12.50, "limit_usd": 50.0},
            "circuit_breakers": {"llm_primary": "closed", "llm_fallback": "open"},
        })
        result = runner.invoke(app, ["models", "check"])
        assert result.exit_code == 0
        assert "12.50" in result.output
        assert "50.00" in result.output
        assert "llm_primary" in result.output
        assert "closed" in result.output
        assert "llm_fallback" in result.output
        assert "open" in result.output

    @patch("httpx.get")
    def test_check_no_circuit_breakers(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({
            "budget": {"used_usd": 0.0, "limit_usd": 50.0},
            "circuit_breakers": {},
        })
        result = runner.invoke(app, ["models", "check"])
        assert result.exit_code == 0
        assert "0.00" in result.output

    @patch("httpx.get")
    def test_check_connect_error(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(app, ["models", "check"])
        assert result.exit_code == 1


# ═══════════════════════════════════════════════════════════════════════════
# Doctor Commands
# ═══════════════════════════════════════════════════════════════════════════


class TestDoctorCheck:
    """Tests for ``doctor check``."""

    @patch("qe.config.load_config")
    def test_doctor_check_runs(self, mock_load: MagicMock) -> None:
        """doctor check should not crash even with default mocks."""
        mock_load.return_value = _mock_config()
        result = runner.invoke(app, ["doctor", "check"])
        assert result.exit_code == 0
        assert "QE Doctor" in result.output
        assert "Python" in result.output

    @patch("qe.config.load_config")
    def test_doctor_check_reports_valid_config(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_config()
        result = runner.invoke(app, ["doctor", "check"])
        assert result.exit_code == 0
        assert "config.toml is valid" in result.output

    @patch("qe.config.load_config")
    def test_doctor_check_reports_invalid_config(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = ValueError("bad config")
        result = runner.invoke(app, ["doctor", "check"])
        assert result.exit_code == 0  # doctor doesn't exit on bad config
        assert "invalid" in result.output.lower()

    @patch("qe.config.load_config")
    def test_doctor_check_shows_config_summary(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_config()
        result = runner.invoke(app, ["doctor", "check"])
        assert "Config Summary" in result.output
        assert "INFO" in result.output
        assert "50.00" in result.output

    @patch("qe.config.load_config")
    def test_doctor_check_reports_dependencies(self, mock_load: MagicMock) -> None:
        mock_load.return_value = _mock_config()
        result = runner.invoke(app, ["doctor", "check"])
        for pkg in ["fastapi", "pydantic", "litellm", "instructor"]:
            assert pkg in result.output


class TestDoctorConnectivity:
    """Tests for ``doctor connectivity``."""

    @patch("httpx.get")
    def test_connectivity_success(self, mock_get: MagicMock) -> None:
        health_resp = _mock_response({"status": "ok"})
        ready_resp = _mock_response({
            "ready": True,
            "phases": {"bus": True, "substrate": True, "services": True},
        })
        mock_get.side_effect = [health_resp, ready_resp]
        result = runner.invoke(app, ["doctor", "connectivity"])
        assert result.exit_code == 0
        assert "Connectivity Check" in result.output
        assert "ok" in result.output

    @patch("httpx.get")
    def test_connectivity_server_unreachable(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(app, ["doctor", "connectivity"])
        assert result.exit_code == 1
        assert "unreachable" in result.output.lower()

    @patch("httpx.get")
    def test_connectivity_shows_readiness_phases(self, mock_get: MagicMock) -> None:
        health_resp = _mock_response({"status": "ok"})
        ready_resp = _mock_response({
            "ready": False,
            "phases": {"bus": True, "substrate": True, "services": False},
        })
        mock_get.side_effect = [health_resp, ready_resp]
        result = runner.invoke(app, ["doctor", "connectivity"])
        assert result.exit_code == 0
        assert "not ready" in result.output

    @patch("httpx.get")
    def test_connectivity_readiness_check_failure_non_fatal(
        self, mock_get: MagicMock,
    ) -> None:
        """If the readiness endpoint fails, connectivity still reports the health check."""
        health_resp = _mock_response({"status": "ok"})
        mock_get.side_effect = [health_resp, Exception("readiness failed")]
        result = runner.invoke(app, ["doctor", "connectivity"])
        assert result.exit_code == 0
        assert "ok" in result.output
        assert "check failed" in result.output.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Init Command
# ═══════════════════════════════════════════════════════════════════════════


class TestInitWizard:
    """Tests for the ``init`` command.

    Uses ``monkeypatch.chdir(tmp_path)`` so that the relative paths
    ``Path("config.toml")``, ``Path(".env")``, ``Path("data")`` used
    inside ``init_wizard`` resolve under a temp directory.
    """

    def test_init_creates_config_when_none_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("typer.prompt") as mock_prompt, patch("typer.confirm"):
            mock_prompt.side_effect = [
                "gpt-4o-mini", "gpt-4o", "o1-preview",
                50.0, "INFO", "sk-test-key",
            ]
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        config_path = tmp_path / "config.toml"
        assert config_path.exists()
        content = config_path.read_text()
        assert "gpt-4o-mini" in content
        assert "gpt-4o" in content
        assert "o1-preview" in content
        assert "50.0" in content

    def test_init_writes_env_file_with_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("typer.prompt") as mock_prompt, patch("typer.confirm"):
            mock_prompt.side_effect = [
                "gpt-4o-mini", "gpt-4o", "o1-preview",
                50.0, "INFO", "sk-my-api-key",
            ]
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        env_path = tmp_path / ".env"
        assert env_path.exists()
        assert "sk-my-api-key" in env_path.read_text()

    def test_init_skips_env_when_key_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("typer.prompt") as mock_prompt, patch("typer.confirm"):
            mock_prompt.side_effect = [
                "gpt-4o-mini", "gpt-4o", "o1-preview",
                50.0, "INFO", "",
            ]
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert not (tmp_path / ".env").exists()

    def test_init_aborts_on_existing_config_no_overwrite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "config.toml"
        config_path.write_text("[budget]\nmonthly_limit_usd = 10.0\n")

        with patch("typer.confirm", return_value=False):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "Keeping existing config" in result.output
        assert "10.0" in config_path.read_text()

    def test_init_creates_data_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("typer.prompt") as mock_prompt, patch("typer.confirm"):
            mock_prompt.side_effect = [
                "gpt-4o-mini", "gpt-4o", "o1-preview",
                50.0, "INFO", "",
            ]
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert (tmp_path / "data").exists()
        assert "Setup complete" in result.output

    def test_init_output_mentions_qe_serve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with patch("typer.prompt") as mock_prompt, patch("typer.confirm"):
            mock_prompt.side_effect = [
                "gpt-4o-mini", "gpt-4o", "o1-preview",
                50.0, "INFO", "",
            ]
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "qe serve" in result.output


# ═══════════════════════════════════════════════════════════════════════════
# A2A Discover and Send (existing commands - supplementary coverage)
# ═══════════════════════════════════════════════════════════════════════════


class TestA2ADiscover:
    """Tests for ``a2a discover``."""

    @patch("httpx.get")
    def test_discover_success(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({
            "name": "Remote Agent",
            "capabilities": ["research"],
        })
        result = runner.invoke(app, ["a2a", "discover", "http://remote:8000"])
        assert result.exit_code == 0
        call_url = mock_get.call_args[0][0]
        assert "/.well-known/agent.json" in call_url

    @patch("httpx.get")
    def test_discover_failure(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = Exception("timeout")
        result = runner.invoke(app, ["a2a", "discover", "http://bad:8000"])
        assert result.exit_code == 0
        assert "Failed" in result.output

    @patch("httpx.get")
    def test_discover_strips_trailing_slash(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _mock_response({"name": "Agent"})
        runner.invoke(app, ["a2a", "discover", "http://remote:8000/"])
        call_url = mock_get.call_args[0][0]
        assert "http://remote:8000/.well-known/agent.json" == call_url


class TestA2ASend:
    """Tests for ``a2a send``."""

    @patch("httpx.post")
    def test_send_success(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _mock_response({"task_id": "t_123"})
        result = runner.invoke(
            app, ["a2a", "send", "http://remote:8000", "Research lithium"],
        )
        assert result.exit_code == 0
        call_args = mock_post.call_args
        assert "/api/a2a/tasks" in call_args[0][0]
        assert call_args[1]["json"]["description"] == "Research lithium"

    @patch("httpx.post")
    def test_send_failure(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = Exception("network error")
        result = runner.invoke(
            app, ["a2a", "send", "http://remote:8000", "Do something"],
        )
        assert result.exit_code == 0
        assert "Failed" in result.output

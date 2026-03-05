"""Tests for the /playground endpoint, SSE typed event models, and A2A peer API endpoints."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest  # noqa: F401
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _patch_peer_registry(value):
    """Patch app.state.peer_registry (the attribute endpoints actually read)."""
    from qe.api.app import app

    old = getattr(app.state, "peer_registry", None)
    app.state.peer_registry = value
    try:
        yield
    finally:
        app.state.peer_registry = old


@pytest.fixture()
def client():
    from qe.api.app import app

    return TestClient(app, raise_server_exceptions=False)


# ===================================================================
# 1. Playground endpoint tests
# ===================================================================


class TestPlaygroundEndpoint:
    """Tests for GET /playground."""

    def test_playground_returns_200(self, client):
        resp = client.get("/playground")
        assert resp.status_code == 200

    def test_playground_returns_html_content_type(self, client):
        resp = client.get("/playground")
        assert "text/html" in resp.headers["content-type"]

    def test_playground_contains_title(self, client):
        resp = client.get("/playground")
        assert "QE API Playground" in resp.text

    def test_playground_contains_endpoints_js_array(self, client):
        """The playground HTML embeds a JS ENDPOINTS array for sidebar rendering."""
        resp = client.get("/playground")
        assert "const ENDPOINTS" in resp.text

    def test_playground_contains_endpoint_groups(self, client):
        """Key endpoint groups should be present in the JS listing."""
        resp = client.get("/playground")
        html = resp.text
        for group in ("Health", "Goals", "Memory", "Chat", "A2A", "Guardrails"):
            assert f'group:"{group}"' in html

    def test_playground_contains_send_request_function(self, client):
        """The playground should have the sendRequest() JS function."""
        resp = client.get("/playground")
        assert "async function sendRequest()" in resp.text

    def test_playground_contains_sidebar_element(self, client):
        resp = client.get("/playground")
        assert 'id="sidebar"' in resp.text

    def test_playground_is_excluded_from_openapi_schema(self, client):
        """include_in_schema=False means /playground is absent from OpenAPI."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        paths = schema.get("paths", {})
        assert "/playground" not in paths


# ===================================================================
# 2. SSE typed event model tests
# ===================================================================


class TestChatProgressEvent:
    """Tests for ChatProgressEvent base model."""

    def test_construction_with_defaults(self):
        from qe.services.chat.events import ChatProgressEvent

        evt = ChatProgressEvent(type="llm_start")
        assert evt.type == "llm_start"
        assert evt.iteration == 0
        assert evt.timestamp.endswith("Z")

    def test_type_literal_enforcement(self):
        """Only the six defined literal types are accepted."""
        from qe.services.chat.events import ChatProgressEvent

        for t in ("llm_start", "llm_complete", "tool_start", "tool_complete", "error", "complete"):
            evt = ChatProgressEvent(type=t)
            assert evt.type == t

    def test_custom_iteration(self):
        from qe.services.chat.events import ChatProgressEvent

        evt = ChatProgressEvent(type="llm_start", iteration=5)
        assert evt.iteration == 5

    def test_serialization_round_trip(self):
        from qe.services.chat.events import ChatProgressEvent

        evt = ChatProgressEvent(type="tool_start", iteration=3)
        data = evt.model_dump()
        assert data["type"] == "tool_start"
        assert data["iteration"] == 3
        assert "timestamp" in data


class TestLLMCompleteEvent:
    """Tests for LLMCompleteEvent model."""

    def test_default_type_is_llm_complete(self):
        from qe.services.chat.events import LLMCompleteEvent

        evt = LLMCompleteEvent()
        assert evt.type == "llm_complete"

    def test_fields_with_values(self):
        from qe.services.chat.events import LLMCompleteEvent

        evt = LLMCompleteEvent(
            model="claude-3.5-haiku",
            call_tokens={"input": 100, "output": 50},
            call_cost_usd=0.002,
            has_tool_calls=True,
            iteration=2,
        )
        assert evt.model == "claude-3.5-haiku"
        assert evt.call_tokens == {"input": 100, "output": 50}
        assert evt.call_cost_usd == pytest.approx(0.002)
        assert evt.has_tool_calls is True
        assert evt.iteration == 2

    def test_inherits_timestamp(self):
        from qe.services.chat.events import LLMCompleteEvent

        evt = LLMCompleteEvent()
        assert evt.timestamp.endswith("Z")

    def test_defaults(self):
        from qe.services.chat.events import LLMCompleteEvent

        evt = LLMCompleteEvent()
        assert evt.model == ""
        assert evt.call_tokens == {}
        assert evt.call_cost_usd == 0.0
        assert evt.has_tool_calls is False


class TestToolCompleteEvent:
    """Tests for ToolCompleteEvent model."""

    def test_default_type(self):
        from qe.services.chat.events import ToolCompleteEvent

        evt = ToolCompleteEvent()
        assert evt.type == "tool_complete"

    def test_fields_with_values(self):
        from qe.services.chat.events import ToolCompleteEvent

        evt = ToolCompleteEvent(
            tool_name="web_search",
            result_preview="Found 5 results for lithium",
            duration_ms=342.5,
        )
        assert evt.tool_name == "web_search"
        assert evt.result_preview == "Found 5 results for lithium"
        assert evt.duration_ms == pytest.approx(342.5)

    def test_defaults(self):
        from qe.services.chat.events import ToolCompleteEvent

        evt = ToolCompleteEvent()
        assert evt.tool_name == ""
        assert evt.result_preview == ""
        assert evt.duration_ms == 0.0


class TestErrorEvent:
    """Tests for ErrorEvent model."""

    def test_default_type(self):
        from qe.services.chat.events import ErrorEvent

        evt = ErrorEvent()
        assert evt.type == "error"

    def test_message_field(self):
        from qe.services.chat.events import ErrorEvent

        evt = ErrorEvent(message="LLM timeout after 30s")
        assert evt.message == "LLM timeout after 30s"

    def test_defaults(self):
        from qe.services.chat.events import ErrorEvent

        evt = ErrorEvent()
        assert evt.message == ""


class TestCompleteEvent:
    """Tests for CompleteEvent model."""

    def test_default_type(self):
        from qe.services.chat.events import CompleteEvent

        evt = CompleteEvent()
        assert evt.type == "complete"

    def test_summary_field(self):
        from qe.services.chat.events import CompleteEvent

        evt = CompleteEvent(summary="Analysis completed in 3 iterations")
        assert evt.summary == "Analysis completed in 3 iterations"

    def test_defaults(self):
        from qe.services.chat.events import CompleteEvent

        evt = CompleteEvent()
        assert evt.summary == ""

    def test_serialization(self):
        from qe.services.chat.events import CompleteEvent

        evt = CompleteEvent(summary="done", iteration=7)
        data = evt.model_dump()
        assert data["type"] == "complete"
        assert data["summary"] == "done"
        assert data["iteration"] == 7


# ===================================================================
# 3. A2A Peer API endpoint tests
# ===================================================================


class TestListPeers:
    """Tests for GET /api/a2a/peers."""

    def test_list_peers_when_registry_is_none(self, client):
        """When _peer_registry is None, returns empty list with zero counts."""
        with _patch_peer_registry(None):
            resp = client.get("/api/a2a/peers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_peers"] == 0
        assert data["healthy_peers"] == 0
        assert data["peers"] == []

    def test_list_peers_delegates_to_status(self, client):
        """When registry is present, delegates to registry.status()."""
        mock_registry = MagicMock()
        mock_registry.status.return_value = {
            "total_peers": 2,
            "healthy_peers": 1,
            "unhealthy_peers": 1,
            "peers": [
                {"peer_id": "abc", "url": "http://a.example.com", "healthy": True},
                {"peer_id": "def", "url": "http://b.example.com", "healthy": False},
            ],
        }
        with _patch_peer_registry(mock_registry):
            resp = client.get("/api/a2a/peers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_peers"] == 2
        assert data["healthy_peers"] == 1
        assert len(data["peers"]) == 2
        mock_registry.status.assert_called_once()


class TestRegisterPeer:
    """Tests for POST /api/a2a/peers."""

    def test_register_peer_when_registry_is_none(self, client):
        """Returns 503 when _peer_registry is None."""
        with _patch_peer_registry(None):
            resp = client.post("/api/a2a/peers", json={"url": "http://x.example.com"})
        assert resp.status_code == 503
        assert "not initialized" in resp.json()["error"]

    def test_register_peer_missing_url(self, client):
        """Returns 400 when url is not provided."""
        mock_registry = MagicMock()
        with _patch_peer_registry(mock_registry):
            resp = client.post("/api/a2a/peers", json={})
        assert resp.status_code == 400
        assert "url is required" in resp.json()["error"]

    def test_register_peer_empty_url(self, client):
        """Returns 400 when url is empty string."""
        mock_registry = MagicMock()
        with _patch_peer_registry(mock_registry):
            resp = client.post("/api/a2a/peers", json={"url": ""})
        assert resp.status_code == 400

    def test_register_peer_discovery_failure(self, client):
        """Returns 502 when discover_and_register returns None."""
        mock_registry = MagicMock()
        mock_registry.discover_and_register = AsyncMock(return_value=None)
        with _patch_peer_registry(mock_registry):
            resp = client.post("/api/a2a/peers", json={"url": "http://bad.example.com"})
        assert resp.status_code == 502
        assert "Failed to discover" in resp.json()["error"]

    def test_register_peer_success(self, client):
        """Returns peer data on successful registration."""
        from qe.runtime.peer_registry import PeerAgent

        peer = PeerAgent(
            peer_id="p123",
            url="http://good.example.com",
            name="TestAgent",
            capabilities=["research"],
        )
        mock_registry = MagicMock()
        mock_registry.discover_and_register = AsyncMock(return_value=peer)
        with _patch_peer_registry(mock_registry):
            resp = client.post("/api/a2a/peers", json={"url": "http://good.example.com"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["peer_id"] == "p123"
        assert data["url"] == "http://good.example.com"
        assert data["name"] == "TestAgent"
        assert "research" in data["capabilities"]


class TestRemovePeer:
    """Tests for DELETE /api/a2a/peers/{peer_id}."""

    def test_remove_peer_when_registry_is_none(self, client):
        """Returns 503 when _peer_registry is None."""
        with _patch_peer_registry(None):
            resp = client.delete("/api/a2a/peers/abc123")
        assert resp.status_code == 503

    def test_remove_peer_not_found(self, client):
        """Returns 404 when peer_id does not exist."""
        mock_registry = MagicMock()
        mock_registry.unregister.return_value = False
        with _patch_peer_registry(mock_registry):
            resp = client.delete("/api/a2a/peers/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"].lower()

    def test_remove_peer_success(self, client):
        """Returns removed peer_id on success."""
        mock_registry = MagicMock()
        mock_registry.unregister.return_value = True
        with _patch_peer_registry(mock_registry):
            resp = client.delete("/api/a2a/peers/abc123")
        assert resp.status_code == 200
        assert resp.json()["removed"] == "abc123"
        mock_registry.unregister.assert_called_once_with("abc123")


class TestCheckPeerHealth:
    """Tests for GET /api/a2a/peers/{peer_id}/health."""

    def test_health_when_registry_is_none(self, client):
        """Returns 503 when _peer_registry is None."""
        with _patch_peer_registry(None):
            resp = client.get("/api/a2a/peers/abc123/health")
        assert resp.status_code == 503

    def test_health_peer_not_found(self, client):
        """Returns 404 when peer is not in the registry."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        with _patch_peer_registry(mock_registry):
            resp = client.get("/api/a2a/peers/nonexistent/health")
        assert resp.status_code == 404

    def test_health_check_healthy(self, client):
        """Returns healthy=True when check_health succeeds."""
        from qe.runtime.peer_registry import PeerAgent

        peer = PeerAgent(peer_id="p1", url="http://healthy.example.com")
        mock_registry = MagicMock()
        mock_registry.get.return_value = peer
        mock_registry.check_health = AsyncMock(return_value=True)
        with _patch_peer_registry(mock_registry):
            resp = client.get("/api/a2a/peers/p1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["peer_id"] == "p1"
        assert data["healthy"] is True
        assert data["url"] == "http://healthy.example.com"

    def test_health_check_unhealthy(self, client):
        """Returns healthy=False when check_health fails."""
        from qe.runtime.peer_registry import PeerAgent

        peer = PeerAgent(peer_id="p2", url="http://down.example.com")
        mock_registry = MagicMock()
        mock_registry.get.return_value = peer
        mock_registry.check_health = AsyncMock(return_value=False)
        with _patch_peer_registry(mock_registry):
            resp = client.get("/api/a2a/peers/p2/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["peer_id"] == "p2"
        assert data["healthy"] is False

    def test_health_check_calls_registry(self, client):
        """Verifies registry.check_health is called with the correct peer_id."""
        from qe.runtime.peer_registry import PeerAgent

        peer = PeerAgent(peer_id="p3", url="http://check.example.com")
        mock_registry = MagicMock()
        mock_registry.get.return_value = peer
        mock_registry.check_health = AsyncMock(return_value=True)
        with _patch_peer_registry(mock_registry):
            client.get("/api/a2a/peers/p3/health")
        mock_registry.get.assert_called_once_with("p3")
        mock_registry.check_health.assert_called_once_with("p3")

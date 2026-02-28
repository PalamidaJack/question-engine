"""Tests for the chat interface."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from qe.services.chat.schemas import ChatIntent, CommandAction
from qe.services.chat.service import ChatService, ChatSession

# ── ChatSession tests ───────────────────────────────────────────────────────


class TestChatSession:
    def test_creation(self):
        session = ChatSession("test-1")
        assert session.session_id == "test-1"
        assert session.history == []

    def test_add_messages(self):
        session = ChatSession("test-1")
        session.add_user_message("hello")
        session.add_assistant_message("hi there")
        assert len(session.history) == 2
        assert session.history[0]["role"] == "user"
        assert session.history[1]["role"] == "assistant"

    def test_trim_at_max(self):
        session = ChatSession("test-1")
        for i in range(60):
            session.add_user_message(f"message {i}")
        assert len(session.history) <= 50

    def test_last_active_updates(self):
        session = ChatSession("test-1")
        before = session.last_active
        session.add_user_message("hello")
        assert session.last_active >= before


# ── Command parsing tests ───────────────────────────────────────────────────


class TestParseCommand:
    @pytest.fixture
    def svc(self):
        """Create a ChatService without full init for command parsing tests."""
        obj = ChatService.__new__(ChatService)
        return obj

    def test_help(self, svc):
        cmd = svc._parse_command("help")
        assert cmd.action == CommandAction.HELP

    def test_list_claims(self, svc):
        cmd = svc._parse_command("list claims about SpaceX")
        assert cmd.action == CommandAction.LIST_CLAIMS
        assert cmd.target == "spacex"

    def test_show_claims(self, svc):
        cmd = svc._parse_command("show all claims")
        assert cmd.action == CommandAction.LIST_CLAIMS

    def test_retract(self, svc):
        cmd = svc._parse_command("retract claim clm_abc123")
        assert cmd.action == CommandAction.RETRACT_CLAIM
        assert "clm_abc123" in cmd.target

    def test_show_budget(self, svc):
        cmd = svc._parse_command("show budget")
        assert cmd.action == CommandAction.SHOW_BUDGET

    def test_cost_keyword(self, svc):
        cmd = svc._parse_command("how much did it cost")
        assert cmd.action == CommandAction.SHOW_BUDGET

    def test_list_entities(self, svc):
        cmd = svc._parse_command("show entities")
        assert cmd.action == CommandAction.LIST_ENTITIES

    def test_show_entity(self, svc):
        cmd = svc._parse_command("show entity SpaceX")
        assert cmd.action == CommandAction.SHOW_ENTITY
        assert "spacex" in cmd.target

    def test_unknown(self, svc):
        cmd = svc._parse_command("do something weird")
        assert cmd.action == CommandAction.UNKNOWN


# ── Budget exhaustion ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_budget_exhausted():
    mock_budget = MagicMock()
    mock_budget.remaining_pct.return_value = 0.0

    svc = ChatService(
        substrate=MagicMock(),
        bus=MagicMock(),
        budget_tracker=mock_budget,
    )

    response = await svc.handle_message("session-1", "hello")
    assert response.error == "budget_exhausted"
    assert response.intent == ChatIntent.CONVERSATION


# ── Session management ──────────────────────────────────────────────────────


def test_get_or_create_session():
    svc = ChatService(substrate=MagicMock(), bus=MagicMock())
    s1 = svc.get_or_create_session("s1")
    s2 = svc.get_or_create_session("s1")
    assert s1 is s2

    s3 = svc.get_or_create_session("s3")
    assert s3 is not s1


def test_cleanup_stale_sessions():
    svc = ChatService(substrate=MagicMock(), bus=MagicMock())
    s1 = svc.get_or_create_session("s1")
    s1.last_active = datetime.now(UTC) - timedelta(hours=48)

    s2 = svc.get_or_create_session("s2")
    s2.last_active = datetime.now(UTC)

    removed = svc.cleanup_stale_sessions(max_age_hours=24)
    assert removed == 1
    assert "s1" not in svc._sessions
    assert "s2" in svc._sessions


# ── API endpoint tests ──────────────────────────────────────────────────────


@pytest.fixture
def client():
    from qe.api.app import app

    return TestClient(app, raise_server_exceptions=False)


def test_chat_requires_message(client):
    resp = client.post("/api/chat", json={})
    assert resp.status_code in (400, 503)


def test_chat_returns_503_when_not_started(client):
    resp = client.post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 503

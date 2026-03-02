"""Tests for the chat interface."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from qe.services.chat.schemas import ChatIntent, CommandAction
from qe.services.chat.service import CHAT_INQUIRY_CONFIG, ChatService, ChatSession
from qe.services.inquiry.schemas import InquiryResult, Question

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


# ── Inquiry routing tests ──────────────────────────────────────────────────


def _make_inquiry_result(**overrides) -> InquiryResult:
    defaults = dict(
        inquiry_id="inq_test",
        goal_id="chat_msg1",
        status="completed",
        termination_reason="confidence_met",
        iterations_completed=1,
        total_questions_generated=4,
        total_questions_answered=3,
        findings_summary="Dark matter makes up ~27% of the universe.",
        insights=[
            {"insight_id": "ins_1", "headline": "Dark matter is invisible"},
            {"insight_id": "ins_2", "headline": "It interacts via gravity"},
        ],
        question_tree=[
            Question(text="What is dark energy?", status="pending"),
            Question(text="How is it detected?", status="answered"),
            Question(text="What are WIMPs?", status="pending"),
        ],
    )
    defaults.update(overrides)
    return InquiryResult(**defaults)


@pytest.mark.asyncio
async def test_handle_question_uses_inquiry_when_available():
    mock_engine = MagicMock()
    mock_engine.run_inquiry = AsyncMock(return_value=_make_inquiry_result())

    svc = ChatService(
        substrate=MagicMock(), bus=MagicMock(), inquiry_engine=mock_engine,
    )

    response = await svc._handle_question("What is dark matter?", "msg1")

    mock_engine.run_inquiry.assert_awaited_once_with(
        goal_id="chat_msg1",
        goal_description="What is dark matter?",
        config=CHAT_INQUIRY_CONFIG,
    )
    assert response.intent == ChatIntent.QUESTION
    assert "27%" in response.reply_text
    assert response.confidence == 3 / 4
    assert len(response.claims) == 2
    assert "confidence_met" in response.reasoning


@pytest.mark.asyncio
async def test_handle_question_fallback_when_no_inquiry_engine():
    svc = ChatService(substrate=MagicMock(), bus=MagicMock())

    with patch("qe.services.chat.service.answer_question", new_callable=AsyncMock) as mock_aq:
        mock_aq.return_value = {
            "answer": "Not enough info",
            "reasoning": "Empty ledger",
            "confidence": 0.1,
            "supporting_claims": [],
        }
        response = await svc._handle_question("What is dark matter?", "msg1")

    mock_aq.assert_awaited_once()
    assert response.reply_text.startswith("Not enough info")
    assert response.intent == ChatIntent.QUESTION


@pytest.mark.asyncio
async def test_handle_question_degrades_on_inquiry_exception():
    mock_engine = MagicMock()
    mock_engine.run_inquiry = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    svc = ChatService(
        substrate=MagicMock(), bus=MagicMock(), inquiry_engine=mock_engine,
    )

    with patch("qe.services.chat.service.answer_question", new_callable=AsyncMock) as mock_aq:
        mock_aq.return_value = {
            "answer": "Fallback answer",
            "reasoning": None,
            "confidence": 0.2,
            "supporting_claims": [],
        }
        response = await svc._handle_question("What is dark matter?", "msg1")

    mock_engine.run_inquiry.assert_awaited_once()
    mock_aq.assert_awaited_once()
    assert response.reply_text == "Fallback answer"


@pytest.mark.asyncio
async def test_handle_question_inquiry_failed_status():
    result = _make_inquiry_result(
        status="failed",
        findings_summary="",
        total_questions_generated=0,
        total_questions_answered=0,
        insights=[],
        question_tree=[],
    )
    mock_engine = MagicMock()
    mock_engine.run_inquiry = AsyncMock(return_value=result)

    svc = ChatService(
        substrate=MagicMock(), bus=MagicMock(), inquiry_engine=mock_engine,
    )

    response = await svc._handle_question("What is dark matter?", "msg1")

    assert "could not find a definitive answer" in response.reply_text
    assert response.confidence == 0.0
    assert "failed" in response.reasoning


@pytest.mark.asyncio
async def test_map_inquiry_result_suggestions_from_tree():
    result = _make_inquiry_result(
        question_tree=[
            Question(text="Q1 pending", status="pending"),
            Question(text="Q2 answered", status="answered"),
            Question(text="Q3 pending", status="pending"),
            Question(text="Q4 pending", status="pending"),
        ],
    )

    svc = ChatService(substrate=MagicMock(), bus=MagicMock())
    payload = svc._map_inquiry_result(result, "msg1")

    assert len(payload.suggestions) == 3
    assert payload.suggestions[0] == "Q1 pending"
    assert payload.suggestions[1] == "Q3 pending"
    assert payload.suggestions[2] == "Submit a new observation"


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

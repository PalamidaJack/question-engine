"""Tests for chat follow-up suggestion generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.services.chat.schemas import (
    ChatIntent,
    ChatResponsePayload,
    ConversationalResponse,
)
from qe.services.chat.service import ChatService

# ── Schema tests ─────────────────────────────────────────────────────────────


class TestSuggestionSchemas:
    def test_chat_response_payload_default_suggestions(self):
        payload = ChatResponsePayload(
            message_id="m1",
            reply_text="hello",
            intent=ChatIntent.CONVERSATION,
        )
        assert payload.suggestions == []

    def test_chat_response_payload_with_suggestions(self):
        payload = ChatResponsePayload(
            message_id="m1",
            reply_text="hello",
            intent=ChatIntent.CONVERSATION,
            suggestions=["Ask me anything", "Show claims"],
        )
        assert payload.suggestions == ["Ask me anything", "Show claims"]

    def test_conversational_response_includes_suggestions(self):
        resp = ConversationalResponse(
            reply="Hi there!",
            suggestions=["What can you do?", "Show entities"],
        )
        assert resp.suggestions == ["What can you do?", "Show entities"]

    def test_conversational_response_default_suggestions(self):
        resp = ConversationalResponse(reply="Hi there!")
        assert resp.suggestions == []


# ── Handler tests ────────────────────────────────────────────────────────────


def _make_service(**kwargs) -> ChatService:
    """Create a ChatService with mocked dependencies."""
    return ChatService(
        substrate=kwargs.get("substrate", MagicMock()),
        bus=kwargs.get("bus", MagicMock()),
        budget_tracker=kwargs.get("budget_tracker", None),
        model="gpt-4o-mini",
    )


class TestObservationSuggestions:
    @pytest.mark.asyncio
    async def test_handle_observation_returns_suggestions(self):
        svc = _make_service()
        response = await svc._handle_observation(
            "Mars has water", "I heard Mars has water", "msg-1"
        )
        assert len(response.suggestions) == 3
        assert "What do we know so far?" in response.suggestions
        assert "Submit another observation" in response.suggestions
        assert "Show all claims" in response.suggestions


class TestCommandSuggestions:
    @pytest.mark.asyncio
    async def test_help_returns_suggestions(self):
        svc = _make_service()
        response = await svc._handle_command("help", "msg-1")
        assert len(response.suggestions) == 3
        assert "What do we know about exoplanets?" in response.suggestions
        assert "Show all claims" in response.suggestions
        assert "Show budget" in response.suggestions

    @pytest.mark.asyncio
    async def test_list_claims_returns_suggestions(self):
        mock_substrate = MagicMock()
        mock_substrate.get_claims = AsyncMock(return_value=[])
        svc = _make_service(substrate=mock_substrate)
        response = await svc._handle_command("list claims", "msg-1")
        assert "Show entities" in response.suggestions

    @pytest.mark.asyncio
    async def test_list_entities_returns_suggestions(self):
        mock_substrate = MagicMock()
        mock_substrate.entity_resolver.list_entities = AsyncMock(
            return_value=[{"canonical_name": "SpaceX", "claim_count": 5}]
        )
        svc = _make_service(substrate=mock_substrate)
        response = await svc._handle_command("show entities", "msg-1")
        assert "Show claims" in response.suggestions
        assert "Tell me about SpaceX" in response.suggestions

    @pytest.mark.asyncio
    async def test_show_entity_returns_suggestions(self):
        mock_substrate = MagicMock()
        mock_substrate.entity_resolver.resolve = AsyncMock(return_value="SpaceX")
        mock_claim = MagicMock()
        mock_claim.model_dump.return_value = {"claim_id": "c1"}
        mock_substrate.get_claims = AsyncMock(return_value=[mock_claim])
        svc = _make_service(substrate=mock_substrate)
        response = await svc._handle_command("show entity SpaceX", "msg-1")
        assert "Show all entities" in response.suggestions
        assert "What else do we know about SpaceX?" in response.suggestions

    @pytest.mark.asyncio
    async def test_show_budget_returns_suggestions(self):
        mock_budget = MagicMock()
        mock_budget.total_spend.return_value = 0.5
        mock_budget.monthly_limit_usd = 5.0
        mock_budget.remaining_pct.return_value = 0.9
        svc = _make_service(budget_tracker=mock_budget)
        response = await svc._handle_command("show budget", "msg-1")
        assert "Show all claims" in response.suggestions
        assert "List entities" in response.suggestions

    @pytest.mark.asyncio
    async def test_retract_claim_returns_suggestions(self):
        mock_substrate = MagicMock()
        mock_substrate.retract_claim = AsyncMock(return_value=True)
        svc = _make_service(substrate=mock_substrate)
        response = await svc._handle_command("retract claim clm_abc", "msg-1")
        assert "Show all claims" in response.suggestions
        assert "List entities" in response.suggestions


class TestQuestionSuggestions:
    @pytest.mark.asyncio
    async def test_suggestions_from_supporting_claims(self):
        svc = _make_service()
        mock_result = {
            "answer": "SpaceX launched Starship.",
            "reasoning": "Based on claims.",
            "confidence": 0.9,
            "supporting_claims": [
                {"subject_entity_id": "SpaceX", "claim_id": "c1"},
                {"subject_entity_id": "Starship", "claim_id": "c2"},
            ],
        }
        with patch(
            "qe.services.chat.service.answer_question",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = await svc._handle_question("What about SpaceX?", "msg-1")
        assert len(response.suggestions) <= 3
        assert "Tell me more about SpaceX" in response.suggestions
        assert "Tell me more about Starship" in response.suggestions
        assert "Submit a new observation" in response.suggestions

    @pytest.mark.asyncio
    async def test_fallback_suggestion_when_no_entities(self):
        svc = _make_service()
        mock_result = {
            "answer": "I don't have information about that.",
            "reasoning": None,
            "confidence": 0.1,
            "supporting_claims": [],
        }
        with patch(
            "qe.services.chat.service.answer_question",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            response = await svc._handle_question("What about nothing?", "msg-1")
        assert "What else do we know?" in response.suggestions
        assert "Submit a new observation" in response.suggestions


class TestConversationSuggestions:
    @pytest.mark.asyncio
    async def test_passes_llm_suggestions_to_payload(self):
        mock_substrate = MagicMock()
        mock_substrate.get_claims = AsyncMock(return_value=[])
        mock_substrate.entity_resolver.list_entities = AsyncMock(return_value=[])
        svc = _make_service(substrate=mock_substrate)
        mock_conv = ConversationalResponse(
            reply="Hello! How can I help?",
            suggestions=["Show claims", "Ask a question", "Submit an observation"],
        )
        with patch(
            "qe.services.chat.service.instructor"
        ) as mock_instructor:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_conv)
            mock_instructor.from_litellm.return_value = mock_client

            session = svc.get_or_create_session("s1")
            response = await svc._handle_conversation("hello", "msg-1", session)

        assert len(response.suggestions) <= 3
        assert response.suggestions == [
            "Show claims",
            "Ask a question",
            "Submit an observation",
        ]

"""Tests for chat suggestion schemas and agent response suggestions."""

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

    def test_new_fields_default_values(self):
        payload = ChatResponsePayload(
            message_id="m1",
            reply_text="hello",
            intent=ChatIntent.CONVERSATION,
        )
        assert payload.tool_calls_made == []
        assert payload.cognitive_process_used is False

    def test_new_fields_populated(self):
        payload = ChatResponsePayload(
            message_id="m1",
            reply_text="hello",
            intent=ChatIntent.CONVERSATION,
            tool_calls_made=[{"tool": "query_beliefs", "blocked": False}],
            cognitive_process_used=True,
        )
        assert len(payload.tool_calls_made) == 1
        assert payload.cognitive_process_used is True


# ── Agent response suggestion tests ─────────────────────────────────────────


def _mock_llm_text_response(text: str):
    """Create a mock litellm response with plain text."""
    message = MagicMock()
    message.content = text
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_service(**kwargs) -> ChatService:
    substrate = kwargs.get("substrate", MagicMock())
    if not isinstance(getattr(substrate, "get_claims", None), AsyncMock):
        substrate.get_claims = AsyncMock(return_value=[])
    if not isinstance(
        getattr(substrate.entity_resolver, "list_entities", None), AsyncMock
    ):
        substrate.entity_resolver.list_entities = AsyncMock(return_value=[])

    return ChatService(
        substrate=substrate,
        bus=kwargs.get("bus", MagicMock()),
        budget_tracker=kwargs.get("budget_tracker", None),
        model="gpt-4o-mini",
        inquiry_engine=kwargs.get("inquiry_engine", None),
        tool_registry=kwargs.get("tool_registry", None),
        tool_gate=kwargs.get("tool_gate", None),
        episodic_memory=kwargs.get("episodic_memory", None),
    )


class TestAgentSuggestions:
    @pytest.mark.asyncio
    async def test_agent_response_includes_suggestions_in_text(self):
        """Agent includes suggestions in its reply text (formatted as > lines)."""
        svc = _make_service()
        reply = (
            "Hello! I'm the Question Engine assistant.\n\n"
            "> What do we know?\n"
            "> Show entities\n"
            "> Submit an observation"
        )
        mock_resp = _mock_llm_text_response(reply)

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            response = await svc.handle_message("s1", "hello")

        assert "Question Engine" in response.reply_text
        assert response.intent == ChatIntent.CONVERSATION

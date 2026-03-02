"""Tests for context compression with LLM summarization."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.runtime.context_manager import ContextManager, ConversationSummary


def _make_blueprint(
    system_prompt: str = "You are a test agent.",
    constitution: str = "Safety first.",
    max_tokens: int = 8000,
) -> MagicMock:
    bp = MagicMock()
    bp.system_prompt = system_prompt
    bp.constitution = constitution
    bp.max_context_tokens = max_tokens
    bp.context_compression_threshold = 0.75
    bp.reinforcement_interval_turns = 10
    return bp


class TestConversationSummaryModel:
    def test_defaults(self):
        s = ConversationSummary()
        assert s.summary == ""
        assert s.key_facts == []
        assert s.open_questions == []

    def test_with_data(self):
        s = ConversationSummary(
            summary="The conversation discussed AI.",
            key_facts=["AI is growing", "LLMs are useful"],
            open_questions=["What about AGI?"],
        )
        assert len(s.key_facts) == 2
        assert len(s.open_questions) == 1


class TestCompressNoOp:
    @pytest.mark.asyncio
    async def test_noop_when_history_short(self):
        cm = ContextManager(_make_blueprint())
        cm.history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        await cm.compress(keep_recent=3)
        assert len(cm.history) == 2  # unchanged

    @pytest.mark.asyncio
    async def test_noop_when_exactly_at_limit(self):
        cm = ContextManager(_make_blueprint())
        cm.history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        await cm.compress(keep_recent=3)
        assert len(cm.history) == 3  # unchanged


class TestCompressWithLLM:
    @pytest.mark.asyncio
    async def test_compresses_older_messages(self):
        cm = ContextManager(_make_blueprint())
        cm.history = [
            {"role": "user", "content": "message 1"},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "message 2"},
            {"role": "assistant", "content": "response 2"},
            {"role": "user", "content": "message 3"},
            {"role": "assistant", "content": "response 3"},
        ]

        mock_summary = ConversationSummary(
            summary="Discussed research topics.",
            key_facts=["Fact A", "Fact B"],
            open_questions=["What next?"],
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_summary)

        with patch("qe.runtime.context_manager.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            await cm.compress(keep_recent=3)

        # Should be: 1 summary + 3 recent = 4 messages
        assert len(cm.history) == 4
        assert cm.history[0]["role"] == "system"
        assert "[CONVERSATION SUMMARY]" in cm.history[0]["content"]
        assert "Discussed research topics." in cm.history[0]["content"]
        assert "Fact A" in cm.history[0]["content"]
        assert "What next?" in cm.history[0]["content"]

        # Recent messages preserved
        assert cm.history[1]["content"] == "response 2"
        assert cm.history[2]["content"] == "message 3"
        assert cm.history[3]["content"] == "response 3"

    @pytest.mark.asyncio
    async def test_summary_includes_key_facts(self):
        cm = ContextManager(_make_blueprint())
        cm.history = [
            {"role": "user", "content": "old 1"},
            {"role": "assistant", "content": "old 2"},
            {"role": "user", "content": "old 3"},
            {"role": "assistant", "content": "old 4"},
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
        ]

        mock_summary = ConversationSummary(
            summary="Key discussion.",
            key_facts=["Important fact"],
            open_questions=[],
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_summary)

        with patch("qe.runtime.context_manager.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            await cm.compress(keep_recent=3)

        assert "Key facts: Important fact" in cm.history[0]["content"]


class TestCompressFallback:
    @pytest.mark.asyncio
    async def test_fallback_to_truncation_on_llm_failure(self):
        cm = ContextManager(_make_blueprint())
        cm.history = [
            {"role": "user", "content": "old 1"},
            {"role": "assistant", "content": "old 2"},
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("LLM unavailable")
        )

        with patch("qe.runtime.context_manager.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            await cm.compress(keep_recent=3)

        # Fallback: just recent messages
        assert len(cm.history) == 3
        assert cm.history[0]["content"] == "recent 1"
        assert cm.history[1]["content"] == "recent 2"
        assert cm.history[2]["content"] == "recent 3"


class TestCompressCustomKeepRecent:
    @pytest.mark.asyncio
    async def test_keep_recent_2(self):
        cm = ContextManager(_make_blueprint())
        cm.history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
        ]

        mock_summary = ConversationSummary(summary="Summary.")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_summary)

        with patch("qe.runtime.context_manager.instructor") as mock_instructor:
            mock_instructor.from_litellm.return_value = mock_client
            await cm.compress(keep_recent=2)

        # 1 summary + 2 recent
        assert len(cm.history) == 3
        assert cm.history[1]["content"] == "recent 1"
        assert cm.history[2]["content"] == "recent 2"

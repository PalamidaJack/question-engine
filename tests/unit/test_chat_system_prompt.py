"""Snapshot tests for the chat system prompt."""

from unittest.mock import MagicMock

from qe.services.chat.service import ChatService


def _make_service(access_mode: str = "balanced", model: str = "gpt-4o-mini") -> ChatService:
    substrate = MagicMock()
    return ChatService(
        substrate=substrate,
        bus=MagicMock(),
        model=model,
        access_mode=access_mode,
    )


class TestSystemPromptContent:
    def test_contains_identity(self):
        svc = _make_service()
        prompt = svc._build_system_prompt()
        assert "Question Engine assistant" in prompt

    def test_contains_model_name(self):
        svc = _make_service(model="gpt-4o")
        prompt = svc._build_system_prompt()
        assert "gpt-4o" in prompt

    def test_strict_mode_disables_filesystem(self):
        svc = _make_service(access_mode="strict")
        prompt = svc._build_system_prompt()
        assert "disabled" in prompt.lower()

    def test_balanced_mode_mentions_sandbox(self):
        svc = _make_service(access_mode="balanced")
        prompt = svc._build_system_prompt()
        assert "sandbox" in prompt.lower()

    def test_full_mode_enables_code_execution(self):
        svc = _make_service(access_mode="full")
        prompt = svc._build_system_prompt()
        assert "execute code" in prompt.lower()

    def test_tool_descriptions_present(self):
        svc = _make_service()
        prompt = svc._build_system_prompt()
        assert "query_beliefs" in prompt
        assert "deep_research" in prompt
        assert "swarm_research" in prompt

    def test_length_sanity(self):
        """System prompt should be between 500 and 5000 chars."""
        svc = _make_service()
        prompt = svc._build_system_prompt()
        assert 500 <= len(prompt) <= 5000

    def test_modes_produce_different_prompts(self):
        strict = _make_service(access_mode="strict")._build_system_prompt()
        balanced = _make_service(access_mode="balanced")._build_system_prompt()
        full = _make_service(access_mode="full")._build_system_prompt()
        assert strict != balanced
        assert balanced != full

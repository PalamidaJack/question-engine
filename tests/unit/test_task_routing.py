"""Tests for task-aware routing in ChatService."""

from unittest.mock import AsyncMock, MagicMock

from qe.services.chat.service import ChatService


def _make_service(**kwargs) -> ChatService:
    substrate = kwargs.get("substrate", MagicMock())
    if not isinstance(getattr(substrate, "get_claims", None), AsyncMock):
        substrate.get_claims = AsyncMock(return_value=[])
    if not isinstance(getattr(substrate.entity_resolver, "list_entities", None), AsyncMock):
        substrate.entity_resolver.list_entities = AsyncMock(return_value=[])
    return ChatService(
        substrate=substrate,
        bus=kwargs.get("bus", MagicMock()),
        model="gpt-4o",
        fast_model=kwargs.get("fast_model", "gpt-4o-mini"),
        router=kwargs.get("router", None),
    )


class TestTaskClassification:
    def test_extraction_keywords(self):
        svc = _make_service()
        messages = [{"role": "user", "content": "extract all entities from this text"}]
        assert svc._classify_task(messages) == "extraction"

    def test_summarization_keywords(self):
        svc = _make_service()
        messages = [{"role": "user", "content": "summarize this article for me"}]
        assert svc._classify_task(messages) == "summarization"

    def test_analysis_keywords(self):
        svc = _make_service()
        messages = [{"role": "user", "content": "analyze the trends and compare results"}]
        assert svc._classify_task(messages) == "analysis"

    def test_reasoning_keywords(self):
        svc = _make_service()
        messages = [{"role": "user", "content": "explain why this happened and how"}]
        assert svc._classify_task(messages) == "reasoning"

    def test_no_match_returns_none(self):
        svc = _make_service()
        messages = [{"role": "user", "content": "hello"}]
        assert svc._classify_task(messages) is None

    def test_empty_messages_returns_none(self):
        svc = _make_service()
        assert svc._classify_task([]) is None


class TestTaskAwareRouting:
    def test_task_hint_passed_to_select(self):
        svc = _make_service()
        # Without router or flag, should fall back to default logic
        model = svc._select_model_for_iteration(0, False, 8, task_hint="extraction")
        assert model == "gpt-4o"  # first iteration → balanced

    def test_fast_model_mid_chain(self):
        svc = _make_service(fast_model="gpt-4o-mini")
        model = svc._select_model_for_iteration(2, True, 8)
        assert model == "gpt-4o-mini"

    def test_balanced_model_first_iteration(self):
        svc = _make_service(fast_model="gpt-4o-mini")
        model = svc._select_model_for_iteration(0, False, 8)
        assert model == "gpt-4o"

    def test_balanced_model_last_iteration(self):
        svc = _make_service(fast_model="gpt-4o-mini")
        model = svc._select_model_for_iteration(7, True, 8)
        assert model == "gpt-4o"

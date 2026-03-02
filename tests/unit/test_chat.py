"""Tests for the chat interface — agent loop architecture."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from qe.services.chat.schemas import ChatIntent
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


# ── Helper: mock LLM response ──────────────────────────────────────────────


def _mock_llm_text_response(text: str):
    """Create a mock litellm response with plain text (no tool calls)."""
    message = MagicMock()
    message.content = text
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_llm_tool_response(tool_calls_data: list[dict]):
    """Create a mock litellm response with tool calls.

    tool_calls_data: list of {"name": ..., "arguments": ..., "id": ...}
    """
    tool_calls = []
    for tc in tool_calls_data:
        fn = MagicMock()
        fn.name = tc["name"]
        fn.arguments = tc.get("arguments", "{}")
        tool_call = MagicMock()
        tool_call.function = fn
        tool_call.id = tc.get("id", f"call_{tc['name']}")
        tool_calls.append(tool_call)

    message = MagicMock()
    message.content = None
    message.tool_calls = tool_calls
    message.model_dump.return_value = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ],
    }
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_service(**kwargs) -> ChatService:
    """Create a ChatService with mocked dependencies."""
    substrate = kwargs.get("substrate", MagicMock())
    # Default: get_claims returns [], list_entities returns []
    if not hasattr(substrate.get_claims, "return_value") or not isinstance(
        substrate.get_claims, AsyncMock
    ):
        substrate.get_claims = AsyncMock(return_value=[])
    if not hasattr(
        substrate.entity_resolver.list_entities, "return_value"
    ) or not isinstance(substrate.entity_resolver.list_entities, AsyncMock):
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


# ── Agent loop tests ────────────────────────────────────────────────────────


class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_agent_responds_directly_for_chat(self):
        """LLM returns plain text → reply_text populated, no tool_calls."""
        svc = _make_service()
        mock_resp = _mock_llm_text_response("Hello! How can I help you today?")

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            response = await svc.handle_message("s1", "hello")

        assert response.reply_text == "Hello! How can I help you today?"
        assert response.tool_calls_made == []
        assert response.cognitive_process_used is False

    @pytest.mark.asyncio
    async def test_agent_loop_calls_tool_and_loops(self):
        """LLM returns tool_call, then text → tool executed, result fed back."""
        svc = _make_service()
        svc.substrate.hybrid_search = AsyncMock(return_value=[])

        tool_resp = _mock_llm_tool_response([
            {"name": "query_beliefs", "arguments": '{"query": "dark matter"}'},
        ])
        text_resp = _mock_llm_text_response("I found no claims about dark matter.")

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_resp
            return text_resp

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            response = await svc.handle_message("s1", "What is dark matter?")

        assert "no claims" in response.reply_text.lower()
        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0]["tool"] == "query_beliefs"

    @pytest.mark.asyncio
    async def test_agent_loop_respects_max_iterations(self):
        """After max iterations, loop stops and returns last text."""
        svc = _make_service()
        svc.substrate.hybrid_search = AsyncMock(return_value=[])

        # Always return tool calls — should stop at max_iterations
        tool_resp = _mock_llm_tool_response([
            {"name": "query_beliefs", "arguments": '{"query": "test"}'},
        ])

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=tool_resp)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)

            messages = await svc._build_messages(svc.get_or_create_session("s1"))
            tool_schemas = svc._get_chat_tools()
            reply, audit = await svc._chat_tool_loop(
                messages, tool_schemas, max_iterations=3
            )

        assert len(audit) == 3  # 3 iterations, 1 tool call each
        assert reply  # Should have some fallback text

    @pytest.mark.asyncio
    async def test_context_includes_belief_state(self):
        """_build_messages() includes claim count and top claims in system context."""
        mock_claim = MagicMock()
        mock_claim.confidence = 0.9
        mock_claim.subject_entity_id = "Mars"
        mock_claim.predicate = "has"
        mock_claim.object_value = "water"

        substrate = MagicMock()
        substrate.get_claims = AsyncMock(return_value=[mock_claim])
        substrate.entity_resolver.list_entities = AsyncMock(
            return_value=[{"canonical_name": "Mars"}]
        )

        svc = _make_service(substrate=substrate)
        session = svc.get_or_create_session("s1")
        session.add_user_message("test")

        messages = await svc._build_messages(session)

        system_msg = messages[0]["content"]
        assert "1 claims" in system_msg
        assert "1 entities" in system_msg
        assert "Mars" in system_msg

    @pytest.mark.asyncio
    async def test_context_includes_memory(self):
        """_build_messages() includes episodic memory entries."""
        mock_episode = MagicMock()
        mock_episode.episode_type = "observation"
        mock_episode.summary = "User discussed Mars exploration"

        mock_memory = MagicMock()
        mock_memory.recall = AsyncMock(return_value=[mock_episode])

        svc = _make_service(episodic_memory=mock_memory)
        session = svc.get_or_create_session("s1")
        session.add_user_message("tell me about Mars")

        messages = await svc._build_messages(session)

        system_msg = messages[0]["content"]
        assert "Mars exploration" in system_msg
        assert "observation" in system_msg


# ── Tool handler tests ──────────────────────────────────────────────────────


class TestToolHandlers:
    @pytest.mark.asyncio
    async def test_tool_query_beliefs(self):
        """_tool_query_beliefs calls substrate.hybrid_search, returns formatted claims."""
        mock_claim = MagicMock()
        mock_claim.claim_id = "clm_001"
        mock_claim.subject_entity_id = "Mars"
        mock_claim.predicate = "has"
        mock_claim.object_value = "water"
        mock_claim.confidence = 0.85

        substrate = MagicMock()
        substrate.hybrid_search = AsyncMock(return_value=[mock_claim])

        svc = _make_service(substrate=substrate)
        result = await svc._tool_query_beliefs("Mars water")

        substrate.hybrid_search.assert_awaited_once_with("Mars water")
        assert "clm_001" in result
        assert "Mars" in result
        assert "85%" in result

    @pytest.mark.asyncio
    async def test_tool_submit_observation(self):
        """_tool_submit_observation publishes to bus, returns envelope_id."""
        mock_bus = MagicMock()
        svc = _make_service(bus=mock_bus)

        result = await svc._tool_submit_observation("NASA found water on Mars")

        mock_bus.publish.assert_called_once()
        envelope = mock_bus.publish.call_args[0][0]
        assert envelope.topic == "observations.structured"
        assert envelope.payload["text"] == "NASA found water on Mars"
        assert "Tracking ID:" in result
        assert envelope.envelope_id in result

    @pytest.mark.asyncio
    async def test_tool_deep_research(self):
        """_tool_deep_research calls inquiry_engine.run_inquiry with correct config."""
        mock_result = MagicMock()
        mock_result.findings_summary = "Mars has subsurface water."
        mock_result.insights = [{"headline": "Water found"}]
        mock_result.iterations_completed = 2
        mock_result.total_questions_generated = 4
        mock_result.status = "completed"

        mock_engine = MagicMock()
        mock_engine.run_inquiry = AsyncMock(return_value=mock_result)

        svc = _make_service(inquiry_engine=mock_engine)
        result = await svc._tool_deep_research("Is there water on Mars?")

        mock_engine.run_inquiry.assert_awaited_once()
        call_kwargs = mock_engine.run_inquiry.call_args[1]
        assert call_kwargs["goal_description"] == "Is there water on Mars?"
        assert call_kwargs["config"].max_iterations == 2
        assert call_kwargs["config"].inquiry_timeout_seconds == 30.0
        assert "Mars has subsurface water" in result
        assert "Water found" in result

    @pytest.mark.asyncio
    async def test_tool_list_entities(self):
        """_tool_list_entities calls entity_resolver.list_entities."""
        substrate = MagicMock()
        substrate.entity_resolver.list_entities = AsyncMock(
            return_value=[{"canonical_name": "Mars"}, {"canonical_name": "SpaceX"}]
        )

        svc = _make_service(substrate=substrate)
        result = await svc._tool_list_entities()

        substrate.entity_resolver.list_entities.assert_awaited_once()
        assert "2 entities" in result
        assert "Mars" in result
        assert "SpaceX" in result

    @pytest.mark.asyncio
    async def test_tool_retract_claim(self):
        """_tool_retract_claim calls substrate.retract_claim."""
        substrate = MagicMock()
        substrate.retract_claim = AsyncMock(return_value=True)

        svc = _make_service(substrate=substrate)
        result = await svc._tool_retract_claim("clm_abc123")

        substrate.retract_claim.assert_awaited_once_with("clm_abc123")
        assert "retracted" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_retract_claim_not_found(self):
        substrate = MagicMock()
        substrate.retract_claim = AsyncMock(return_value=False)

        svc = _make_service(substrate=substrate)
        result = await svc._tool_retract_claim("clm_nope")

        assert "not found" in result.lower()


# ── Tool gate tests ─────────────────────────────────────────────────────────


class TestToolGate:
    @pytest.mark.asyncio
    async def test_tool_gate_blocks_denied_tool(self):
        """Tool gate returns DENY → tool not executed, error in audit."""
        from qe.runtime.tool_gate import GateDecision, GateResult

        mock_gate = MagicMock()
        mock_gate.validate.return_value = GateResult(
            decision=GateDecision.DENY,
            reason="Not allowed",
            policy_name="test",
        )

        svc = _make_service(tool_gate=mock_gate)
        svc.substrate.hybrid_search = AsyncMock(return_value=[])

        tool_resp = _mock_llm_tool_response([
            {"name": "query_beliefs", "arguments": '{"query": "test"}'},
        ])
        text_resp = _mock_llm_text_response("Tool was blocked.")

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_resp
            return text_resp

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            response = await svc.handle_message("s1", "test")

        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0]["blocked"] is True
        # hybrid_search should NOT have been called
        svc.substrate.hybrid_search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_gate_allows_tool(self):
        """Tool gate returns ALLOW → tool executed normally."""
        from qe.runtime.tool_gate import GateDecision, GateResult

        mock_gate = MagicMock()
        mock_gate.validate.return_value = GateResult(
            decision=GateDecision.ALLOW,
            reason="",
            policy_name="test",
        )

        svc = _make_service(tool_gate=mock_gate)
        svc.substrate.hybrid_search = AsyncMock(return_value=[])

        tool_resp = _mock_llm_tool_response([
            {"name": "query_beliefs", "arguments": '{"query": "test"}'},
        ])
        text_resp = _mock_llm_text_response("No results.")

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_resp
            return text_resp

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            response = await svc.handle_message("s1", "test")

        assert len(response.tool_calls_made) == 1
        assert response.tool_calls_made[0]["blocked"] is False
        svc.substrate.hybrid_search.assert_awaited_once()


# ── Response builder tests ──────────────────────────────────────────────────


class TestResponseBuilder:
    def test_response_includes_tracking_id(self):
        """When submit_observation used, tracking_envelope_id populated."""
        svc = _make_service()
        tool_audit = [
            {
                "tool": "submit_observation",
                "params": {"text": "test"},
                "result": "Observation submitted. Tracking ID: abc-123\nPipeline: ...",
                "blocked": False,
            },
        ]
        response = svc._build_response("msg-1", "Done!", tool_audit)
        assert response.tracking_envelope_id == "abc-123"

    def test_response_marks_cognitive_process(self):
        """When deep_research used, cognitive_process_used is True."""
        svc = _make_service()
        tool_audit = [
            {
                "tool": "deep_research",
                "params": {"question": "test"},
                "result": "Findings...",
                "blocked": False,
            },
        ]
        response = svc._build_response("msg-1", "Here are my findings.", tool_audit)
        assert response.cognitive_process_used is True

    def test_response_no_tools(self):
        """No tools used → clean response."""
        svc = _make_service()
        response = svc._build_response("msg-1", "Hello!", [])
        assert response.tool_calls_made == []
        assert response.cognitive_process_used is False
        assert response.tracking_envelope_id is None


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

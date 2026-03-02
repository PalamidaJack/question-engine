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
    resp.usage = None
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
    resp.usage = None
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
        fast_model=kwargs.get("fast_model", None),
        inquiry_engine=kwargs.get("inquiry_engine", None),
        tool_registry=kwargs.get("tool_registry", None),
        tool_gate=kwargs.get("tool_gate", None),
        episodic_memory=kwargs.get("episodic_memory", None),
        cognitive_pool=kwargs.get("cognitive_pool", None),
        competitive_arena=kwargs.get("competitive_arena", None),
        planner=kwargs.get("planner", None),
        dispatcher=kwargs.get("dispatcher", None),
        goal_store=kwargs.get("goal_store", None),
        epistemic_reasoner=kwargs.get("epistemic_reasoner", None),
        dialectic_engine=kwargs.get("dialectic_engine", None),
        insight_crystallizer=kwargs.get("insight_crystallizer", None),
        knowledge_loop=kwargs.get("knowledge_loop", None),
        procedural_memory=kwargs.get("procedural_memory", None),
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
            reply, audit, _cost = await svc._chat_tool_loop(
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
        assert call_kwargs["config"].max_iterations == 5
        assert call_kwargs["config"].inquiry_timeout_seconds == 120.0
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


# ── Cognitive infrastructure tool tests ─────────────────────────────────────


class TestCognitiveTools:
    @pytest.mark.asyncio
    async def test_tool_swarm_research(self):
        """swarm_research spawns agents, runs parallel inquiry, merges results."""
        mock_agent = MagicMock()
        mock_agent.agent_id = "agent_1"

        mock_merged = MagicMock()
        mock_merged.findings_summary = "Merged findings from swarm."
        mock_merged.insights = [{"headline": "Insight A"}]

        mock_pool = MagicMock()
        mock_pool.spawn_agent = AsyncMock(return_value=mock_agent)
        mock_pool.run_parallel_inquiry = AsyncMock(return_value=[MagicMock(), MagicMock()])
        mock_pool.merge_results = AsyncMock(return_value=mock_merged)
        mock_pool.retire_agent = AsyncMock()

        svc = _make_service(cognitive_pool=mock_pool)
        result = await svc._tool_swarm_research("What is dark matter?", num_agents=3)

        assert mock_pool.spawn_agent.await_count == 3
        mock_pool.run_parallel_inquiry.assert_awaited_once()
        mock_pool.merge_results.assert_awaited_once()
        assert "Merged findings from swarm" in result
        assert "Insight A" in result
        assert mock_pool.retire_agent.await_count == 3

    @pytest.mark.asyncio
    async def test_tool_swarm_research_not_available(self):
        """Returns not-available message when cognitive_pool is None."""
        svc = _make_service(cognitive_pool=None)
        result = await svc._tool_swarm_research("test")
        assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_swarm_research_with_arena(self):
        """swarm_research uses competitive arena when available."""
        mock_agent = MagicMock()
        mock_agent.agent_id = "agent_1"

        mock_pool = MagicMock()
        mock_pool.spawn_agent = AsyncMock(return_value=mock_agent)
        mock_pool.run_parallel_inquiry = AsyncMock(
            return_value=[MagicMock(), MagicMock()]
        )
        mock_pool.retire_agent = AsyncMock()

        mock_arena_result = MagicMock()
        mock_arena_result.winner_agent_id = "agent_1"
        mock_arena_result.summary = "Agent 1 wins."
        mock_arena = MagicMock()
        mock_arena.run_tournament = AsyncMock(return_value=mock_arena_result)

        svc = _make_service(cognitive_pool=mock_pool, competitive_arena=mock_arena)
        result = await svc._tool_swarm_research("test", num_agents=2)

        mock_arena.run_tournament.assert_awaited_once()
        assert "tournament" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_plan_and_execute(self):
        """plan_and_execute calls planner.decompose + dispatcher.submit_goal."""
        mock_decomp = MagicMock()
        mock_decomp.subtasks = [MagicMock(), MagicMock()]

        mock_state = MagicMock()
        mock_state.goal_id = "goal_abc"
        mock_state.decomposition = mock_decomp

        mock_planner = MagicMock()
        mock_planner.decompose = AsyncMock(return_value=mock_state)

        mock_bus = MagicMock()
        # Simulate bus.subscribe capturing the callback
        subscriber_holder = {}

        def fake_subscribe(topic, cb):
            subscriber_holder[topic] = cb

        mock_bus.subscribe = fake_subscribe
        mock_bus.unsubscribe = MagicMock()

        mock_dispatcher = MagicMock()

        async def fake_submit(state):
            # Simulate the synthesizer publishing the result
            envelope = MagicMock()
            envelope.payload = {
                "goal_id": "goal_abc",
                "synthesis": "All subtasks completed successfully.",
            }
            cb = subscriber_holder.get("goals.synthesized")
            if cb:
                await cb(envelope)

        mock_dispatcher.submit_goal = AsyncMock(side_effect=fake_submit)

        svc = _make_service(
            bus=mock_bus,
            planner=mock_planner,
            dispatcher=mock_dispatcher,
        )
        result = await svc._tool_plan_and_execute("Analyze climate data")

        mock_planner.decompose.assert_awaited_once_with("Analyze climate data")
        mock_dispatcher.submit_goal.assert_awaited_once_with(mock_state)
        assert "completed" in result.lower()
        assert "All subtasks completed" in result

    @pytest.mark.asyncio
    async def test_tool_plan_and_execute_timeout(self):
        """Returns timeout message on TimeoutError."""
        mock_state = MagicMock()
        mock_state.goal_id = "goal_timeout"
        mock_state.decomposition = MagicMock()

        mock_planner = MagicMock()
        mock_planner.decompose = AsyncMock(return_value=mock_state)

        mock_dispatcher = MagicMock()
        mock_dispatcher.submit_goal = AsyncMock()  # Never triggers done event

        mock_bus = MagicMock()
        mock_bus.subscribe = MagicMock()
        mock_bus.unsubscribe = MagicMock()

        svc = _make_service(
            bus=mock_bus,
            planner=mock_planner,
            dispatcher=mock_dispatcher,
        )

        # Patch wait_for to immediately raise TimeoutError
        with patch("qe.services.chat.service.asyncio.wait_for", side_effect=TimeoutError):
            result = await svc._tool_plan_and_execute("Slow goal")

        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_plan_and_execute_not_available(self):
        """Returns not-available when planner is None."""
        svc = _make_service(planner=None, dispatcher=None)
        result = await svc._tool_plan_and_execute("test")
        assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_reason_about(self):
        """reason_about calls epistemic + dialectic methods."""
        mock_uncertainty = MagicMock()
        mock_uncertainty.confidence_level = "low"
        mock_uncertainty.evidence_quality = "secondary"
        mock_uncertainty.potential_biases = ["confirmation bias"]
        mock_uncertainty.information_gaps = ["missing data"]
        mock_uncertainty.could_be_wrong_because = ["small sample"]

        mock_surprise = MagicMock()
        mock_surprise.surprise_magnitude = 0.8
        mock_surprise.finding = "Unexpected result"
        mock_surprise.implications = ["Rethink assumptions"]

        mock_dialectic = MagicMock()
        mock_dialectic.revised_confidence = 0.4
        mock_dialectic.counterarguments = []
        mock_dialectic.assumptions_challenged = []
        mock_dialectic.synthesis = "The claim needs more evidence."

        mock_epistemic = MagicMock()
        mock_epistemic.assess_uncertainty = AsyncMock(return_value=mock_uncertainty)
        mock_epistemic.detect_surprise = AsyncMock(return_value=mock_surprise)

        mock_dialectic_engine = MagicMock()
        mock_dialectic_engine.full_dialectic = AsyncMock(return_value=mock_dialectic)

        svc = _make_service(
            epistemic_reasoner=mock_epistemic,
            dialectic_engine=mock_dialectic_engine,
        )
        result = await svc._tool_reason_about("AI will replace all jobs")

        mock_epistemic.assess_uncertainty.assert_awaited_once()
        mock_epistemic.detect_surprise.assert_awaited_once()
        mock_dialectic_engine.full_dialectic.assert_awaited_once()
        assert "low" in result
        assert "confirmation bias" in result
        assert "Surprise detected" in result
        assert "claim needs more evidence" in result

    @pytest.mark.asyncio
    async def test_tool_reason_about_not_available(self):
        """Returns not-available when epistemic_reasoner is None."""
        svc = _make_service(epistemic_reasoner=None, dialectic_engine=None)
        result = await svc._tool_reason_about("test claim")
        assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_crystallize_insights(self):
        """crystallize_insights calls crystallizer, formats CrystallizedInsight."""
        mock_mechanism = MagicMock()
        mock_mechanism.explanation = "Mechanism X causes Y"

        mock_novelty = MagicMock()
        mock_novelty.novelty_score = 0.85

        mock_insight = MagicMock()
        mock_insight.headline = "Novel finding about X"
        mock_insight.confidence = 0.9
        mock_insight.mechanism = mock_mechanism
        mock_insight.novelty = mock_novelty
        mock_insight.actionability_score = 0.7
        mock_insight.actionability_description = "Can be applied to Y"
        mock_insight.cross_domain_connections = ["biology", "economics"]

        mock_crystallizer = MagicMock()
        mock_crystallizer.crystallize = AsyncMock(return_value=mock_insight)

        svc = _make_service(insight_crystallizer=mock_crystallizer)
        result = await svc._tool_crystallize_insights("Finding about X", domain="science")

        mock_crystallizer.crystallize.assert_awaited_once()
        call_kwargs = mock_crystallizer.crystallize.call_args[1]
        assert call_kwargs["finding"] == "Finding about X"
        assert call_kwargs["domain"] == "science"
        assert "Novel finding about X" in result
        assert "Mechanism X causes Y" in result
        assert "biology" in result
        assert "economics" in result

    @pytest.mark.asyncio
    async def test_tool_crystallize_insights_not_novel(self):
        """Returns 'not novel' when crystallizer returns None."""
        mock_crystallizer = MagicMock()
        mock_crystallizer.crystallize = AsyncMock(return_value=None)

        svc = _make_service(insight_crystallizer=mock_crystallizer)
        result = await svc._tool_crystallize_insights("Obvious fact")

        assert "not novel" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_consolidate_knowledge(self):
        """consolidate_knowledge triggers consolidation + returns status."""
        mock_loop = MagicMock()
        mock_loop.trigger_consolidation = AsyncMock()
        mock_loop.status.return_value = {
            "running": True,
            "last_cycle_result": {
                "episodes_scanned": 42,
                "patterns_detected": 3,
                "beliefs_promoted": 1,
                "hypotheses_reviewed": 5,
                "contradictions_found": 0,
            },
        }

        svc = _make_service(knowledge_loop=mock_loop)
        result = await svc._tool_consolidate_knowledge()

        mock_loop.trigger_consolidation.assert_awaited_once()
        assert "consolidation completed" in result.lower()
        assert "42" in result
        assert "3" in result

    @pytest.mark.asyncio
    async def test_tool_consolidate_knowledge_not_available(self):
        """Returns not-available when knowledge_loop is None."""
        svc = _make_service(knowledge_loop=None)
        result = await svc._tool_consolidate_knowledge()
        assert "not available" in result.lower()

    def test_deep_research_config_uncrippled(self):
        """Verify config has max_iterations=5 and timeout=120s."""
        from qe.services.chat.service import _CHAT_INQUIRY_CONFIG

        assert _CHAT_INQUIRY_CONFIG.max_iterations == 5
        assert _CHAT_INQUIRY_CONFIG.inquiry_timeout_seconds == 120.0

    @pytest.mark.asyncio
    async def test_system_prompt_full_capabilities(self):
        """System prompt mentions swarm, plan, epistemic, dialectic, crystallize."""
        svc = _make_service()
        prompt = svc._build_system_prompt()
        assert "swarm" in prompt.lower()
        assert "plan_and_execute" in prompt
        assert "epistemic" in prompt.lower()
        assert "dialectic" in prompt.lower()
        assert "crystallize_insights" in prompt
        assert "consolidate_knowledge" in prompt

    def test_cognitive_tools_in_build_response(self):
        """New cognitive tools marked as cognitive_process_used."""
        svc = _make_service()
        for tool_name in [
            "swarm_research", "plan_and_execute", "reason_about",
            "crystallize_insights", "consolidate_knowledge",
        ]:
            audit = [{"tool": tool_name, "params": {}, "result": "ok", "blocked": False}]
            response = svc._build_response("msg-1", "Done.", audit)
            assert response.cognitive_process_used is True, f"{tool_name} not marked cognitive"


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


# ── Progress events tests ──────────────────────────────────────────────────


class TestProgressEvents:
    @pytest.mark.asyncio
    async def test_progress_events_emitted_for_simple_response(self):
        """No tool calls → [llm_start, llm_complete, complete]."""
        import asyncio

        svc = _make_service()
        mock_resp = _mock_llm_text_response("Hello!")

        queue: asyncio.Queue[dict] = asyncio.Queue()

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            messages = await svc._build_messages(svc.get_or_create_session("s1"))
            tool_schemas = svc._get_chat_tools()
            reply, audit, _cost = await svc._chat_tool_loop(
                messages, tool_schemas, max_iterations=5,
                progress_queue=queue,
            )

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        phases = [e["phase"] for e in events]
        assert phases == ["llm_start", "llm_complete", "complete"]
        assert events[-1]["total_tool_calls"] == 0
        assert "elapsed_ms" in events[0]
        assert "timestamp" in events[0]

    @pytest.mark.asyncio
    async def test_progress_events_emitted_for_tool_loop(self):
        """With tool calls → includes tool_start, tool_complete."""
        import asyncio

        svc = _make_service()
        svc.substrate.hybrid_search = AsyncMock(return_value=[])

        tool_resp = _mock_llm_tool_response([
            {"name": "query_beliefs", "arguments": '{"query": "test"}'},
        ])
        text_resp = _mock_llm_text_response("No results.")

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            return tool_resp if call_count == 1 else text_resp

        queue: asyncio.Queue[dict] = asyncio.Queue()

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            messages = await svc._build_messages(svc.get_or_create_session("s1"))
            tool_schemas = svc._get_chat_tools()
            reply, audit, _cost = await svc._chat_tool_loop(
                messages, tool_schemas, max_iterations=5,
                progress_queue=queue,
            )

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())

        phases = [e["phase"] for e in events]
        assert "tool_start" in phases
        assert "tool_complete" in phases
        assert phases[0] == "llm_start"
        assert phases[-1] == "complete"

        tool_start_evt = next(e for e in events if e["phase"] == "tool_start")
        assert tool_start_evt["tool_name"] == "query_beliefs"
        assert tool_start_evt["tool_index"] == 0

        tool_complete_evt = next(e for e in events if e["phase"] == "tool_complete")
        assert "duration_ms" in tool_complete_evt
        assert tool_complete_evt["tool_name"] == "query_beliefs"

    @pytest.mark.asyncio
    async def test_no_progress_when_queue_is_none(self):
        """Backward compat: no errors when progress_queue is None."""
        svc = _make_service()
        mock_resp = _mock_llm_text_response("Hello!")

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            messages = await svc._build_messages(svc.get_or_create_session("s1"))
            tool_schemas = svc._get_chat_tools()
            reply, audit, _cost = await svc._chat_tool_loop(
                messages, tool_schemas, max_iterations=5,
                progress_queue=None,
            )

        assert reply == "Hello!"

    @pytest.mark.asyncio
    async def test_handle_message_passes_queue(self):
        """Queue gets events when passed to handle_message."""
        import asyncio

        svc = _make_service()
        mock_resp = _mock_llm_text_response("Hi there!")

        queue: asyncio.Queue[dict] = asyncio.Queue()

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            response = await svc.handle_message(
                "s1", "hello", progress_queue=queue,
            )

        assert response.reply_text == "Hi there!"
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        assert len(events) >= 3
        assert events[-1]["phase"] == "complete"


class TestRecordCost:
    def test_record_cost_uses_actual_response(self):
        """Verify _record_cost calls completion_cost(completion_response=response)."""
        mock_budget = MagicMock()
        svc = _make_service(budget_tracker=mock_budget)

        mock_response = MagicMock()

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.completion_cost = MagicMock(return_value=0.005)
            svc._record_cost(mock_response)

        mock_litellm.completion_cost.assert_called_once_with(
            completion_response=mock_response,
        )
        mock_budget.record_cost.assert_called_once_with(
            svc.model, 0.005, service_id="chat",
        )


# ── Improvement 1: Plan State Injection ──────────────────────────────────


class TestPlanStateInjection:
    @pytest.mark.asyncio
    async def test_state_message_injected_on_second_call(self):
        """After first tool round, a system message with agent state is injected."""
        svc = _make_service()
        svc.substrate.hybrid_search = AsyncMock(return_value=[])

        tool_resp = _mock_llm_tool_response([
            {"name": "query_beliefs", "arguments": '{"query": "test"}'},
        ])
        text_resp = _mock_llm_text_response("Done.")

        call_count = 0
        captured_messages = []

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            captured_messages.append(list(kwargs["messages"]))
            return tool_resp if call_count == 1 else text_resp

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            messages = await svc._build_messages(svc.get_or_create_session("s1"))
            tool_schemas = svc._get_chat_tools()
            await svc._chat_tool_loop(messages, tool_schemas, max_iterations=5)

        # Second LLM call should have the agent state system message
        second_call_msgs = captured_messages[1]
        state_msgs = [
            m for m in second_call_msgs
            if isinstance(m, dict) and m.get("role") == "system"
            and "AGENT STATE" in m.get("content", "")
        ]
        assert len(state_msgs) == 1
        assert "Iteration 1/" in state_msgs[0]["content"]
        assert "query_beliefs" in state_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_state_message_correct_iteration(self):
        """Agent state shows correct iteration numbers across multiple rounds."""
        svc = _make_service()
        svc.substrate.hybrid_search = AsyncMock(return_value=[])

        tool_resp = _mock_llm_tool_response([
            {"name": "query_beliefs", "arguments": '{"query": "test"}'},
        ])
        text_resp = _mock_llm_text_response("Done.")

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            return tool_resp if call_count <= 2 else text_resp

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            messages = await svc._build_messages(svc.get_or_create_session("s1"))
            tool_schemas = svc._get_chat_tools()
            reply, audit, _cost = await svc._chat_tool_loop(
                messages, tool_schemas, max_iterations=5,
            )

        assert len(audit) == 2  # Two tool call rounds


# ── Improvement 2: Smart Context Compaction ──────────────────────────────


class TestContextCompaction:
    @pytest.mark.asyncio
    async def test_triggers_at_threshold(self):
        """Compaction triggers when history exceeds threshold."""
        svc = _make_service()
        session = svc.get_or_create_session("s1")

        # Add enough messages to exceed _COMPACTION_THRESHOLD (24)
        for i in range(30):
            session.history.append({"role": "user", "content": f"msg {i}"})

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Summary of conversation."

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            await svc._compact_history(session)

        assert session.context_summary == "Summary of conversation."
        assert len(session.history) == 16  # _MAX_CONTEXT_MESSAGES

    @pytest.mark.asyncio
    async def test_skips_below_threshold(self):
        """Compaction does not trigger below threshold."""
        svc = _make_service()
        session = svc.get_or_create_session("s1")

        for i in range(10):
            session.history.append({"role": "user", "content": f"msg {i}"})

        await svc._compact_history(session)

        assert session.context_summary is None
        assert len(session.history) == 10

    @pytest.mark.asyncio
    async def test_preserves_recent_messages(self):
        """After compaction, the most recent 16 messages are preserved."""
        svc = _make_service()
        session = svc.get_or_create_session("s1")

        for i in range(30):
            session.history.append({"role": "user", "content": f"msg {i}"})

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Summary"

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            await svc._compact_history(session)

        # Most recent message should be "msg 29"
        assert session.history[-1]["content"] == "msg 29"

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        """On LLM failure, falls back to FIFO trim without summary."""
        svc = _make_service()
        session = svc.get_or_create_session("s1")

        for i in range(30):
            session.history.append({"role": "user", "content": f"msg {i}"})

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=Exception("LLM error"))
            await svc._compact_history(session)

        assert session.context_summary is None
        assert len(session.history) == 16

    @pytest.mark.asyncio
    async def test_summary_in_built_messages(self):
        """Context summary appears in messages built for LLM."""
        svc = _make_service()
        session = svc.get_or_create_session("s1")
        session.context_summary = "User asked about Mars exploration."
        session.add_user_message("tell me more")

        messages = await svc._build_messages(session)

        system_msg = messages[0]["content"]
        assert "Mars exploration" in system_msg
        assert "Conversation context (summarized)" in system_msg

    def test_default_context_summary_is_none(self):
        """New sessions have context_summary=None."""
        session = ChatSession("test-1")
        assert session.context_summary is None


# ── Improvement 3: Dual-Model Routing ────────────────────────────────────


class TestDualModelRouting:
    def test_first_iteration_uses_balanced(self):
        """Iteration 0 always uses the balanced model."""
        svc = _make_service(fast_model="gpt-4o-mini")
        svc.model = "gpt-4o"
        model = svc._select_model_for_iteration(0, False, 10)
        assert model == "gpt-4o"

    def test_mid_chain_uses_fast(self):
        """After tool results, mid-chain iterations use fast model."""
        svc = _make_service(fast_model="gpt-4o-mini")
        svc.model = "gpt-4o"
        model = svc._select_model_for_iteration(2, True, 10)
        assert model == "gpt-4o-mini"

    def test_fallback_when_no_fast_model(self):
        """Without fast_model, always returns balanced."""
        svc = _make_service(fast_model=None)
        svc.model = "gpt-4o"
        model = svc._select_model_for_iteration(2, True, 10)
        assert model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_tool_loop_uses_fast_after_tools(self):
        """After first tool call, LLM is called with fast model."""
        svc = _make_service(fast_model="gpt-4o-mini")
        svc.model = "gpt-4o"
        svc.substrate.hybrid_search = AsyncMock(return_value=[])

        tool_resp = _mock_llm_tool_response([
            {"name": "query_beliefs", "arguments": '{"query": "test"}'},
        ])
        text_resp = _mock_llm_text_response("Done.")

        call_count = 0
        used_models = []

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            used_models.append(kwargs["model"])
            return tool_resp if call_count == 1 else text_resp

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            messages = await svc._build_messages(svc.get_or_create_session("s1"))
            tool_schemas = svc._get_chat_tools()
            await svc._chat_tool_loop(messages, tool_schemas, max_iterations=5)

        assert used_models[0] == "gpt-4o"  # First call: balanced
        assert used_models[1] == "gpt-4o-mini"  # After tools: fast


# ── Improvement 4: Mid-Execution Interjection ────────────────────────────


class TestInterjection:
    @pytest.mark.asyncio
    async def test_interjection_injected_into_messages(self):
        """User interjection is injected as system message."""
        import asyncio

        svc = _make_service()
        svc.substrate.hybrid_search = AsyncMock(return_value=[])

        tool_resp = _mock_llm_tool_response([
            {"name": "query_beliefs", "arguments": '{"query": "test"}'},
        ])
        text_resp = _mock_llm_text_response("Adjusted.")

        call_count = 0
        captured_messages = []

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            captured_messages.append(list(kwargs["messages"]))
            return tool_resp if call_count == 1 else text_resp

        interjection_queue: asyncio.Queue[str] = asyncio.Queue()
        await interjection_queue.put("Actually, search for Mars instead")

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            messages = await svc._build_messages(svc.get_or_create_session("s1"))
            tool_schemas = svc._get_chat_tools()
            await svc._chat_tool_loop(
                messages, tool_schemas, max_iterations=5,
                interjection_queue=interjection_queue,
            )

        # Second call should contain the interjection
        second_call_msgs = captured_messages[1]
        interjection_msgs = [
            m for m in second_call_msgs
            if isinstance(m, dict) and "USER INTERJECTION" in (m.get("content") or "")
        ]
        assert len(interjection_msgs) == 1
        assert "Mars" in interjection_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_empty_queue_is_noop(self):
        """Empty interjection queue doesn't affect the loop."""
        import asyncio

        svc = _make_service()
        mock_resp = _mock_llm_text_response("Hello!")

        interjection_queue: asyncio.Queue[str] = asyncio.Queue()

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            messages = await svc._build_messages(svc.get_or_create_session("s1"))
            tool_schemas = svc._get_chat_tools()
            reply, audit, _cost = await svc._chat_tool_loop(
                messages, tool_schemas, max_iterations=5,
                interjection_queue=interjection_queue,
            )

        assert reply == "Hello!"

    @pytest.mark.asyncio
    async def test_interjection_emits_progress_event(self):
        """Interjection triggers a progress event."""
        import asyncio

        svc = _make_service()
        svc.substrate.hybrid_search = AsyncMock(return_value=[])

        tool_resp = _mock_llm_tool_response([
            {"name": "query_beliefs", "arguments": '{"query": "test"}'},
        ])
        text_resp = _mock_llm_text_response("Adjusted.")

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            return tool_resp if call_count == 1 else text_resp

        interjection_queue: asyncio.Queue[str] = asyncio.Queue()
        await interjection_queue.put("Change direction")

        progress_queue: asyncio.Queue[dict] = asyncio.Queue()

        with patch("qe.services.chat.service.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=side_effect)
            mock_litellm.completion_cost = MagicMock(return_value=0.001)
            messages = await svc._build_messages(svc.get_or_create_session("s1"))
            tool_schemas = svc._get_chat_tools()
            await svc._chat_tool_loop(
                messages, tool_schemas, max_iterations=5,
                progress_queue=progress_queue,
                interjection_queue=interjection_queue,
            )

        events = []
        while not progress_queue.empty():
            events.append(progress_queue.get_nowait())

        phases = [e["phase"] for e in events]
        assert "interjection_received" in phases
        inj_event = next(e for e in events if e["phase"] == "interjection_received")
        assert "Change direction" in inj_event["interjection"]


# ── Improvement 5: Session Pattern Extraction ────────────────────────────


class TestSessionPatterns:
    @pytest.mark.asyncio
    async def test_records_sequence(self):
        """Successful tool use records pattern in procedural memory."""
        mock_pm = MagicMock()
        mock_pm.record_sequence_outcome = AsyncMock()

        svc = _make_service(procedural_memory=mock_pm)
        audit = [
            {"tool": "query_beliefs", "params": {}, "result": "Found 3 claims.", "blocked": False},
            {"tool": "deep_research", "params": {}, "result": "Research done.", "blocked": False},
        ]
        await svc._extract_session_patterns(audit, 0.005)

        mock_pm.record_sequence_outcome.assert_awaited_once()
        call_kwargs = mock_pm.record_sequence_outcome.call_args[1]
        assert call_kwargs["tool_names"] == ["query_beliefs", "deep_research"]
        assert call_kwargs["success"] is True
        assert call_kwargs["cost_usd"] == 0.005
        assert call_kwargs["domain"] == "chat"

    @pytest.mark.asyncio
    async def test_skips_blocked_tools(self):
        """Blocked tools are excluded from the recorded sequence."""
        mock_pm = MagicMock()
        mock_pm.record_sequence_outcome = AsyncMock()

        svc = _make_service(procedural_memory=mock_pm)
        audit = [
            {"tool": "query_beliefs", "params": {}, "result": "OK", "blocked": False},
            {"tool": "submit_observation", "params": {}, "result": "blocked", "blocked": True},
        ]
        await svc._extract_session_patterns(audit, 0.001)

        call_kwargs = mock_pm.record_sequence_outcome.call_args[1]
        assert call_kwargs["tool_names"] == ["query_beliefs"]

    @pytest.mark.asyncio
    async def test_noop_without_procedural_memory(self):
        """No error when procedural_memory is None."""
        svc = _make_service(procedural_memory=None)
        audit = [{"tool": "query_beliefs", "params": {}, "result": "OK", "blocked": False}]
        # Should not raise
        await svc._extract_session_patterns(audit, 0.001)

    @pytest.mark.asyncio
    async def test_detects_errors_as_failure(self):
        """Tool results containing 'error' mark the sequence as failed."""
        mock_pm = MagicMock()
        mock_pm.record_sequence_outcome = AsyncMock()

        svc = _make_service(procedural_memory=mock_pm)
        audit = [
            {"tool": "deep_research", "params": {}, "result": "Tool error: timeout", "blocked": False},
        ]
        await svc._extract_session_patterns(audit, 0.002)

        call_kwargs = mock_pm.record_sequence_outcome.call_args[1]
        assert call_kwargs["success"] is False

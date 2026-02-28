"""Tests for multi-agent orchestration components."""

from __future__ import annotations

import time

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.envelope import Envelope
from qe.models.goal import Subtask
from qe.runtime.agent_pool import AgentPool, AgentRecord
from qe.runtime.coordination import CoordinationProtocol
from qe.runtime.working_memory import WorkingMemory

# ── AgentRecord tests ──────────────────────────────────────────────────────


class TestAgentRecord:
    def test_success_rate_default(self):
        """New agent has optimistic 1.0 success rate."""
        record = AgentRecord(agent_id="a1")
        assert record.success_rate == 1.0

    def test_success_rate_after_tasks(self):
        """8 completed + 2 failed = 0.8 success rate."""
        record = AgentRecord(agent_id="a1", tasks_completed=8, tasks_failed=2)
        assert record.success_rate == pytest.approx(0.8)

    def test_available_slots(self):
        """Available slots = max_concurrency - current_load."""
        record = AgentRecord(agent_id="a1", max_concurrency=5, current_load=3)
        assert record.available_slots == 2

    def test_load_pct(self):
        """Load percentage = current_load / max_concurrency."""
        record = AgentRecord(agent_id="a1", max_concurrency=4, current_load=2)
        assert record.load_pct == pytest.approx(0.5)


# ── AgentPool tests ────────────────────────────────────────────────────────


class TestAgentPool:
    def test_register_and_get(self):
        pool = AgentPool()
        record = AgentRecord(agent_id="a1", service_id="svc1")
        pool.register(record)
        assert pool.get("a1") is record
        assert len(pool.all_agents()) == 1

    def test_deregister(self):
        pool = AgentPool()
        pool.register(AgentRecord(agent_id="a1"))
        pool.deregister("a1")
        assert pool.get("a1") is None

    def test_select_agent_by_task_type(self):
        pool = AgentPool()
        pool.register(
            AgentRecord(agent_id="a1", task_types={"research"}, max_concurrency=5)
        )
        pool.register(
            AgentRecord(agent_id="a2", task_types={"analysis"}, max_concurrency=5)
        )
        subtask = Subtask(description="test", task_type="research")
        selected = pool.select_agent(subtask)
        assert selected is not None
        assert selected.agent_id == "a1"

    def test_select_agent_filters_by_capabilities(self):
        pool = AgentPool()
        pool.register(
            AgentRecord(
                agent_id="a1",
                task_types={"research"},
                capabilities={"web_search"},
                max_concurrency=5,
            )
        )
        pool.register(
            AgentRecord(
                agent_id="a2",
                task_types={"research"},
                capabilities={"web_search", "code_exec"},
                max_concurrency=5,
            )
        )
        subtask = Subtask(
            description="test",
            task_type="research",
            tools_required=["web_search", "code_exec"],
        )
        selected = pool.select_agent(subtask)
        assert selected is not None
        assert selected.agent_id == "a2"

    def test_select_agent_prefers_less_loaded(self):
        pool = AgentPool()
        pool.register(
            AgentRecord(
                agent_id="a1",
                task_types={"research"},
                max_concurrency=5,
                current_load=4,
            )
        )
        pool.register(
            AgentRecord(
                agent_id="a2",
                task_types={"research"},
                max_concurrency=5,
                current_load=0,
            )
        )
        subtask = Subtask(description="test", task_type="research")
        selected = pool.select_agent(subtask)
        assert selected is not None
        assert selected.agent_id == "a2"

    def test_select_agent_returns_none_when_no_match(self):
        pool = AgentPool()
        pool.register(
            AgentRecord(agent_id="a1", task_types={"analysis"}, max_concurrency=5)
        )
        subtask = Subtask(description="test", task_type="research")
        assert pool.select_agent(subtask) is None

    def test_select_agent_respects_capacity(self):
        pool = AgentPool()
        pool.register(
            AgentRecord(
                agent_id="a1",
                task_types={"research"},
                max_concurrency=2,
                current_load=2,
            )
        )
        subtask = Subtask(description="test", task_type="research")
        assert pool.select_agent(subtask) is None

    def test_acquire_and_release(self):
        pool = AgentPool()
        pool.register(AgentRecord(agent_id="a1", max_concurrency=2))
        assert pool.acquire("a1") is True
        assert pool.get("a1").current_load == 1
        assert pool.acquire("a1") is True
        assert pool.get("a1").current_load == 2
        # At capacity
        assert pool.acquire("a1") is False
        pool.release("a1")
        assert pool.get("a1").current_load == 1

    def test_record_completion(self):
        pool = AgentPool()
        pool.register(AgentRecord(agent_id="a1"))
        pool.record_completion("a1", latency_ms=100.0, cost_usd=0.01, success=True)
        pool.record_completion("a1", latency_ms=200.0, cost_usd=0.02, success=False)
        agent = pool.get("a1")
        assert agent.tasks_completed == 1
        assert agent.tasks_failed == 1
        assert agent.total_latency_ms == pytest.approx(300.0)
        assert agent.total_cost_usd == pytest.approx(0.03)

    def test_status(self):
        pool = AgentPool()
        pool.register(
            AgentRecord(agent_id="a1", service_id="svc1", task_types={"research"})
        )
        status = pool.status()
        assert status["total_agents"] == 1
        assert len(status["agents"]) == 1
        assert status["agents"][0]["agent_id"] == "a1"


# ── WorkingMemory tests ───────────────────────────────────────────────────


class TestWorkingMemory:
    def test_put_and_get(self):
        mem = WorkingMemory()
        mem.put("g1", "key1", "value1")
        assert mem.get("g1", "key1") == "value1"

    def test_get_missing_returns_default(self):
        mem = WorkingMemory()
        assert mem.get("g1", "missing") is None
        assert mem.get("g1", "missing", "fallback") == "fallback"

    def test_get_all(self):
        mem = WorkingMemory()
        mem.put("g1", "a", 1)
        mem.put("g1", "b", 2)
        result = mem.get_all("g1")
        assert result == {"a": 1, "b": 2}

    def test_ttl_expiry(self):
        mem = WorkingMemory()
        mem.put("g1", "key1", "value1", ttl_seconds=0.01)
        time.sleep(0.02)
        assert mem.get("g1", "key1") is None

    def test_store_subtask_result(self):
        mem = WorkingMemory()
        mem.store_subtask_result("g1", "sub1", {"content": "hello", "score": 0.9})
        # Stored under subtask: prefix
        assert mem.get("g1", "subtask:sub1") == {"content": "hello", "score": 0.9}
        # Individual keys
        assert mem.get("g1", "sub1.content") == "hello"
        assert mem.get("g1", "sub1.score") == 0.9

    def test_build_context_for_subtask(self):
        mem = WorkingMemory()
        mem.store_subtask_result("g1", "dep1", {"content": "result1"})
        mem.store_subtask_result("g1", "dep2", {"content": "result2"})
        context = mem.build_context_for_subtask("g1", "sub3", ["dep1", "dep2"])
        assert "dep1" in context
        assert "dep2" in context
        assert context["dep1"]["content"] == "result1"

    def test_clear_goal(self):
        mem = WorkingMemory()
        mem.put("g1", "key1", "value1")
        mem.clear_goal("g1")
        assert mem.get("g1", "key1") is None

    def test_max_entries_eviction(self):
        mem = WorkingMemory(max_entries_per_goal=3)
        mem.put("g1", "a", 1)
        time.sleep(0.001)
        mem.put("g1", "b", 2)
        time.sleep(0.001)
        mem.put("g1", "c", 3)
        time.sleep(0.001)
        # This should evict "a" (LRU)
        mem.put("g1", "d", 4)
        assert mem.get("g1", "a") is None
        assert mem.get("g1", "d") == 4

    def test_isolation_between_goals(self):
        mem = WorkingMemory()
        mem.put("g1", "key", "val1")
        mem.put("g2", "key", "val2")
        assert mem.get("g1", "key") == "val1"
        assert mem.get("g2", "key") == "val2"


# ── CoordinationProtocol tests ────────────────────────────────────────────


class TestCoordinationProtocol:
    @pytest.mark.asyncio
    async def test_vote_basic_consensus(self):
        """Two agents vote the same → unanimous."""
        bus = MemoryBus()
        coord = CoordinationProtocol(bus)
        await coord.start()

        # Simulate two agents that vote "optionA"
        async def agent_voter(envelope: Envelope) -> None:
            payload = envelope.payload
            bus.publish(
                Envelope(
                    topic="coordination.vote_response",
                    source_service_id="agent",
                    correlation_id=envelope.envelope_id,
                    payload={
                        "vote_id": payload["vote_id"],
                        "agent_id": "agent1",
                        "choice": "optionA",
                        "confidence": 0.9,
                    },
                )
            )
            bus.publish(
                Envelope(
                    topic="coordination.vote_response",
                    source_service_id="agent",
                    correlation_id=envelope.envelope_id,
                    payload={
                        "vote_id": payload["vote_id"],
                        "agent_id": "agent2",
                        "choice": "optionA",
                        "confidence": 0.8,
                    },
                )
            )

        bus.subscribe("coordination.vote_request", agent_voter)

        result = await coord.request_vote(
            question="Which option?",
            options=["optionA", "optionB"],
            min_voters=2,
            timeout_seconds=5.0,
        )

        assert result.winning_choice == "optionA"
        assert result.unanimous is True
        assert result.total_votes == 2

        await coord.stop()

    @pytest.mark.asyncio
    async def test_vote_timeout_returns_partial(self):
        """Only 1 of 2 min_voters responds → partial result after timeout."""
        bus = MemoryBus()
        coord = CoordinationProtocol(bus)
        await coord.start()

        async def single_voter(envelope: Envelope) -> None:
            payload = envelope.payload
            bus.publish(
                Envelope(
                    topic="coordination.vote_response",
                    source_service_id="agent",
                    payload={
                        "vote_id": payload["vote_id"],
                        "agent_id": "agent1",
                        "choice": "optionA",
                        "confidence": 0.9,
                    },
                )
            )

        bus.subscribe("coordination.vote_request", single_voter)

        result = await coord.request_vote(
            question="Which option?",
            options=["optionA", "optionB"],
            min_voters=2,
            timeout_seconds=0.2,
        )

        assert result.total_votes == 1
        assert result.winning_choice == "optionA"

        await coord.stop()

    @pytest.mark.asyncio
    async def test_vote_split_uses_confidence_weighting(self):
        """High-confidence vote wins in a split."""
        bus = MemoryBus()
        coord = CoordinationProtocol(bus)
        await coord.start()

        async def split_voter(envelope: Envelope) -> None:
            payload = envelope.payload
            # Agent 1 votes optionA with low confidence
            bus.publish(
                Envelope(
                    topic="coordination.vote_response",
                    source_service_id="agent",
                    payload={
                        "vote_id": payload["vote_id"],
                        "agent_id": "agent1",
                        "choice": "optionA",
                        "confidence": 0.3,
                    },
                )
            )
            # Agent 2 votes optionB with high confidence
            bus.publish(
                Envelope(
                    topic="coordination.vote_response",
                    source_service_id="agent",
                    payload={
                        "vote_id": payload["vote_id"],
                        "agent_id": "agent2",
                        "choice": "optionB",
                        "confidence": 0.9,
                    },
                )
            )

        bus.subscribe("coordination.vote_request", split_voter)

        result = await coord.request_vote(
            question="Which option?",
            options=["optionA", "optionB"],
            min_voters=2,
            timeout_seconds=5.0,
        )

        assert result.winning_choice == "optionB"
        assert result.unanimous is False

        await coord.stop()


# ── Bus Request/Response tests ─────────────────────────────────────────────


class TestBusRequestResponse:
    @pytest.mark.asyncio
    async def test_request_receives_correlated_reply(self):
        bus = MemoryBus()

        # Use tasks.delegated as request topic and tasks.delegation_result as reply
        async def responder(envelope: Envelope) -> None:
            bus.publish(
                Envelope(
                    topic="tasks.delegation_result",
                    source_service_id="responder",
                    correlation_id=envelope.envelope_id,
                    payload={"answer": 42},
                )
            )

        bus.subscribe("tasks.delegated", responder)

        envelope = Envelope(
            topic="tasks.delegated",
            source_service_id="requester",
            payload={"goal_id": "g1", "subtask_id": "s1"},
        )

        reply = await bus.request(
            envelope,
            reply_topic="tasks.delegation_result",
            timeout_seconds=5.0,
        )
        assert reply is not None
        assert reply.payload["answer"] == 42
        assert reply.correlation_id == envelope.envelope_id

    @pytest.mark.asyncio
    async def test_request_timeout_returns_none(self):
        bus = MemoryBus()
        # No responder registered

        envelope = Envelope(
            topic="tasks.delegated",
            source_service_id="requester",
            payload={"goal_id": "g1", "subtask_id": "s1"},
        )

        reply = await bus.request(
            envelope,
            reply_topic="tasks.delegation_result",
            timeout_seconds=0.1,
        )
        assert reply is None

    @pytest.mark.asyncio
    async def test_request_ignores_uncorrelated_replies(self):
        bus = MemoryBus()

        # Responder sends reply with wrong correlation_id
        async def bad_responder(envelope: Envelope) -> None:
            bus.publish(
                Envelope(
                    topic="tasks.delegation_result",
                    source_service_id="responder",
                    correlation_id="wrong_id",
                    payload={"answer": "wrong"},
                )
            )

        bus.subscribe("tasks.delegated", bad_responder)

        envelope = Envelope(
            topic="tasks.delegated",
            source_service_id="requester",
            payload={"goal_id": "g1", "subtask_id": "s1"},
        )

        reply = await bus.request(
            envelope,
            reply_topic="tasks.delegation_result",
            timeout_seconds=0.2,
        )
        assert reply is None

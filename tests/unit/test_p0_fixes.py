"""Tests for P0 bug fixes: loop detector, planner genome guard, executor pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.envelope import Envelope
from qe.models.goal import SubtaskResult

# ── P0-1: Loop Detector Tests ─────────────────────────────────────────────


class TestPublishListeners:
    """MemoryBus.add_publish_listener and listener invocation."""

    def test_listener_called_on_publish(self):
        bus = MemoryBus()
        observed: list[Envelope] = []
        bus.add_publish_listener(lambda env: observed.append(env))

        env = Envelope(
            topic="observations.structured",
            source_service_id="svc-a",
            payload={"data": 1},
        )
        bus.publish(env)

        assert len(observed) == 1
        assert observed[0].envelope_id == env.envelope_id

    def test_listener_not_called_for_dedup(self):
        bus = MemoryBus()
        observed: list[Envelope] = []
        bus.add_publish_listener(lambda env: observed.append(env))

        env = Envelope(
            topic="observations.structured",
            source_service_id="svc-a",
            payload={"data": 1},
        )
        bus.publish(env)
        bus.publish(env)  # duplicate — should be deduped

        assert len(observed) == 1

    def test_multiple_listeners(self):
        bus = MemoryBus()
        calls_a: list[str] = []
        calls_b: list[str] = []
        bus.add_publish_listener(lambda env: calls_a.append(env.topic))
        bus.add_publish_listener(lambda env: calls_b.append(env.topic))

        env = Envelope(
            topic="claims.proposed",
            source_service_id="svc",
            payload={},
        )
        bus.publish(env)

        assert calls_a == ["claims.proposed"]
        assert calls_b == ["claims.proposed"]


class TestLoopDetectorObservesOutbound:
    """Supervisor loop detection observes outbound publishes, not inbound."""

    @pytest.mark.asyncio
    async def test_wrap_handler_does_not_call_check_loop(self):
        """_wrap_service_handler should NOT call _check_loop."""
        from qe.kernel.supervisor import Supervisor

        bus = MemoryBus()
        sup = Supervisor(bus=bus, substrate=None)

        service = MagicMock()
        service.blueprint.service_id = "test-svc"
        handler = AsyncMock()
        service._handle_envelope = handler

        wrapped = sup._wrap_service_handler(handler, service)

        env = Envelope(
            topic="observations.structured",
            source_service_id="other",
            payload={"x": 1},
        )

        await wrapped(env)

        # _check_loop should NOT have been called by the wrapper
        assert "test-svc" not in sup._pub_history

    def test_on_publish_calls_check_loop(self):
        """_on_publish calls _check_loop for non-supervisor sources."""
        from qe.kernel.supervisor import Supervisor

        bus = MemoryBus()
        sup = Supervisor(bus=bus, substrate=None)

        env = Envelope(
            topic="claims.proposed",
            source_service_id="researcher",
            payload={"claim": "test"},
        )

        sup._on_publish(env)
        assert "researcher" in sup._pub_history

    def test_on_publish_ignores_supervisor(self):
        """_on_publish skips envelopes from the supervisor itself."""
        from qe.kernel.supervisor import Supervisor

        bus = MemoryBus()
        sup = Supervisor(bus=bus, substrate=None)

        env = Envelope(
            topic="system.circuit_break",
            source_service_id="supervisor",
            payload={},
        )

        sup._on_publish(env)
        assert "supervisor" not in sup._pub_history


# ── P0-2: Planner Genome Bootstrap Tests ──────────────────────────────────


class TestPlannerGenomeGuard:
    """Supervisor skips genomes that are not BaseService subclasses."""

    @pytest.mark.asyncio
    async def test_non_baseservice_genome_skipped(self, tmp_path: Path):
        """Genome pointing to non-BaseService class is skipped."""
        from qe.kernel.supervisor import Supervisor

        bus = MemoryBus()
        sup = Supervisor(bus=bus, substrate=None)

        genome = tmp_path / "planner.toml"
        genome.write_text(
            'service_id = "planner"\n'
            'display_name = "Planner"\n'
            'version = "1.0"\n'
            'system_prompt = "Plan goals"\n'
            'service_class = "qe.services.planner:PlannerService"\n'
            "\n"
            "[model_preference]\n"
            'tier = "balanced"\n'
            "\n"
            "[capabilities]\n"
            'bus_topics_subscribe = ["goals.submitted"]\n'
            'bus_topics_publish = ["goals.planned"]\n'
        )

        task = asyncio.create_task(sup.start([genome]))
        await asyncio.sleep(0.1)
        await sup.stop()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert sup.registry.get("planner") is None

    @pytest.mark.asyncio
    async def test_instantiation_exception_skipped(self, tmp_path: Path):
        """Genome whose service_class cannot be imported is skipped."""
        from qe.kernel.supervisor import Supervisor

        bus = MemoryBus()
        sup = Supervisor(bus=bus, substrate=None)

        genome = tmp_path / "broken.toml"
        genome.write_text(
            'service_id = "broken"\n'
            'display_name = "Broken"\n'
            'version = "1.0"\n'
            'system_prompt = "x"\n'
            'service_class = "qe.nonexistent.module:FakeClass"\n'
            "\n"
            "[model_preference]\n"
            'tier = "balanced"\n'
            "\n"
            "[capabilities]\n"
            'bus_topics_subscribe = ["observations.structured"]\n'
            "bus_topics_publish = []\n"
        )

        task = asyncio.create_task(sup.start([genome]))
        await asyncio.sleep(0.1)
        await sup.stop()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert sup.registry.get("broken") is None


# ── P0-3: Executor Pipeline Tests ────────────────────────────────────────


class TestExecutorService:
    """ExecutorService subscribes to tasks.dispatched and publishes results."""

    @pytest.mark.asyncio
    async def test_executor_subscribes_on_start(self):
        from qe.services.executor import ExecutorService

        bus = MemoryBus()
        executor = ExecutorService(
            bus=bus, substrate=None, model="gpt-4o-mini"
        )
        await executor.start()

        assert len(bus._subscribers.get("tasks.dispatched", [])) == 1

    @pytest.mark.asyncio
    async def test_executor_publishes_completed_on_success(self):
        from qe.services.executor import ExecutorService

        bus = MemoryBus()
        executor = ExecutorService(
            bus=bus, substrate=None, model="gpt-4o-mini"
        )

        published: list[Envelope] = []
        bus.add_publish_listener(lambda env: published.append(env))

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Task result content"

        acompletion_path = (
            "qe.services.executor.service.litellm.acompletion"
        )
        limiter_path = (
            "qe.services.executor.service.get_rate_limiter"
        )
        with (
            patch(acompletion_path, new_callable=AsyncMock,
                  return_value=mock_response),
            patch(limiter_path) as mock_rl,
        ):
            mock_rl.return_value.acquire = AsyncMock()
            await executor.start()

            dispatched_env = Envelope(
                topic="tasks.dispatched",
                source_service_id="dispatcher",
                correlation_id="goal_123",
                payload={
                    "goal_id": "goal_123",
                    "subtask_id": "sub_abc",
                    "task_type": "research",
                    "description": "Research quantum computing",
                    "model_tier": "balanced",
                    "depends_on": [],
                    "contract": {},
                },
            )

            await executor._handle_dispatched(dispatched_env)

        completed = [
            e for e in published if e.topic == "tasks.completed"
        ]
        assert len(completed) == 1
        result = SubtaskResult.model_validate(completed[0].payload)
        assert result.subtask_id == "sub_abc"
        assert result.goal_id == "goal_123"
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_executor_publishes_failed_on_error(self):
        from qe.services.executor import ExecutorService

        bus = MemoryBus()
        executor = ExecutorService(
            bus=bus, substrate=None, model="gpt-4o-mini"
        )

        published: list[Envelope] = []
        bus.add_publish_listener(lambda env: published.append(env))

        acompletion_path = (
            "qe.services.executor.service.litellm.acompletion"
        )
        limiter_path = (
            "qe.services.executor.service.get_rate_limiter"
        )
        with (
            patch(acompletion_path, new_callable=AsyncMock,
                  side_effect=RuntimeError("LLM unavailable")),
            patch(limiter_path) as mock_rl,
        ):
            mock_rl.return_value.acquire = AsyncMock()
            await executor.start()

            dispatched_env = Envelope(
                topic="tasks.dispatched",
                source_service_id="dispatcher",
                correlation_id="goal_456",
                payload={
                    "goal_id": "goal_456",
                    "subtask_id": "sub_def",
                    "task_type": "analysis",
                    "description": "Analyse data",
                    "model_tier": "balanced",
                    "depends_on": [],
                    "contract": {},
                },
            )

            await executor._handle_dispatched(dispatched_env)

        failed = [e for e in published if e.topic == "tasks.failed"]
        assert len(failed) == 1
        result = SubtaskResult.model_validate(failed[0].payload)
        assert result.subtask_id == "sub_def"
        assert result.status == "failed"
        assert "LLM unavailable" in result.output["error"]


class TestDispatcherWiring:
    """Dispatcher receives tasks.completed/failed via bus subscription."""

    @pytest.mark.asyncio
    async def test_task_result_reaches_dispatcher(self):
        """tasks.completed -> dispatcher.handle_subtask_completed."""
        from qe.models.goal import (
            GoalDecomposition,
            GoalState,
            Subtask,
        )
        from qe.services.dispatcher.service import Dispatcher
        from qe.substrate.goal_store import GoalStore

        bus = MemoryBus()

        mock_store = MagicMock(spec=GoalStore)
        mock_store.save_goal = AsyncMock()
        mock_store.save_checkpoint = AsyncMock(return_value="ckpt_1")

        dispatcher = Dispatcher(bus=bus, goal_store=mock_store)

        async def _on_task_result(envelope: Envelope) -> None:
            r = SubtaskResult.model_validate(envelope.payload)
            await dispatcher.handle_subtask_completed(r.goal_id, r)

        bus.subscribe("tasks.completed", _on_task_result)
        bus.subscribe("tasks.failed", _on_task_result)

        state = GoalState(
            goal_id="goal_test",
            description="test goal",
            status="executing",
            decomposition=GoalDecomposition(
                goal_id="goal_test",
                original_description="test goal",
                strategy="single subtask",
                subtasks=[
                    Subtask(
                        subtask_id="sub_1",
                        description="do thing",
                        task_type="research",
                    )
                ],
            ),
            subtask_states={"sub_1": "dispatched"},
        )
        dispatcher._active_goals["goal_test"] = state

        result = SubtaskResult(
            subtask_id="sub_1",
            goal_id="goal_test",
            status="completed",
            output={"content": "done"},
        )
        tasks = bus.publish(
            Envelope(
                topic="tasks.completed",
                source_service_id="executor",
                correlation_id="goal_test",
                payload=result.model_dump(mode="json"),
            )
        )
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        assert state.status == "completed"
        assert state.subtask_states["sub_1"] == "completed"

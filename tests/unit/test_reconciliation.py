"""Tests for startup reconciliation of in-flight goals."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.models.goal import GoalState
from qe.services.dispatcher.service import Dispatcher
from qe.substrate.goal_store import GoalStore


class TestReconciliation:
    @pytest.fixture
    def mock_goal_store(self):
        store = AsyncMock(spec=GoalStore)
        store.list_active_goals = AsyncMock(return_value=[])
        store.save_goal = AsyncMock()
        return store

    @pytest.fixture
    def dispatcher(self, mock_goal_store):
        bus = MagicMock()
        bus.publish = MagicMock()
        return Dispatcher(bus=bus, goal_store=mock_goal_store)

    @pytest.mark.asyncio
    async def test_no_active_goals(self, dispatcher):
        result = await dispatcher.reconcile()
        assert result == {"resumed": 0, "failed_stale": 0}

    @pytest.mark.asyncio
    async def test_resumes_executing_goals(self, dispatcher, mock_goal_store):
        state = GoalState(
            description="test goal",
            status="executing",
            subtask_states={"sub_1": "pending"},
        )
        mock_goal_store.list_active_goals.return_value = [state]

        result = await dispatcher.reconcile()
        assert result["resumed"] == 1
        assert state.goal_id in dispatcher._active_goals

    @pytest.mark.asyncio
    async def test_fails_stale_planning_goals(self, dispatcher, mock_goal_store):
        state = GoalState(description="interrupted", status="planning")
        mock_goal_store.list_active_goals.return_value = [state]

        result = await dispatcher.reconcile()
        assert result["failed_stale"] == 1
        assert state.status == "failed"
        assert state.completed_at is not None
        mock_goal_store.save_goal.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_skips_already_tracked_goals(self, dispatcher, mock_goal_store):
        state = GoalState(description="already tracked", status="executing")
        dispatcher._active_goals[state.goal_id] = state
        mock_goal_store.list_active_goals.return_value = [state]

        result = await dispatcher.reconcile()
        # Should not count as resumed (already tracked)
        assert result == {"resumed": 0, "failed_stale": 0}

    @pytest.mark.asyncio
    async def test_mixed_goals(self, dispatcher, mock_goal_store):
        executing = GoalState(description="exec", status="executing")
        planning = GoalState(description="plan", status="planning")
        mock_goal_store.list_active_goals.return_value = [executing, planning]

        result = await dispatcher.reconcile()
        assert result == {"resumed": 1, "failed_stale": 1}
        assert executing.goal_id in dispatcher._active_goals
        assert planning.status == "failed"

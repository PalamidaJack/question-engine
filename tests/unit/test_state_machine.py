"""Tests for GoalState state machine enforcement."""

from __future__ import annotations

import pytest

from qe.models.goal import GoalState, InvalidTransition


class TestGoalTransitions:
    def test_planning_to_executing(self):
        state = GoalState(description="test")
        assert state.status == "planning"
        state.transition_to("executing")
        assert state.status == "executing"
        assert state.started_at is not None

    def test_planning_to_failed(self):
        state = GoalState(description="test")
        state.transition_to("failed")
        assert state.status == "failed"
        assert state.completed_at is not None

    def test_planning_to_paused(self):
        state = GoalState(description="test")
        state.transition_to("paused")
        assert state.status == "paused"

    def test_executing_to_completed(self):
        state = GoalState(description="test", status="executing")
        state.transition_to("completed")
        assert state.status == "completed"
        assert state.completed_at is not None

    def test_executing_to_failed(self):
        state = GoalState(description="test", status="executing")
        state.transition_to("failed")
        assert state.status == "failed"

    def test_executing_to_paused(self):
        state = GoalState(description="test", status="executing")
        state.transition_to("paused")
        assert state.status == "paused"

    def test_paused_to_executing(self):
        state = GoalState(description="test", status="paused")
        state.transition_to("executing")
        assert state.status == "executing"

    def test_paused_to_failed(self):
        state = GoalState(description="test", status="paused")
        state.transition_to("failed")
        assert state.status == "failed"


class TestInvalidTransitions:
    def test_completed_is_terminal(self):
        state = GoalState(description="test", status="completed")
        with pytest.raises(InvalidTransition, match="terminal state"):
            state.transition_to("executing")

    def test_failed_is_terminal(self):
        state = GoalState(description="test", status="failed")
        with pytest.raises(InvalidTransition, match="terminal state"):
            state.transition_to("planning")

    def test_planning_to_completed_invalid(self):
        state = GoalState(description="test")
        with pytest.raises(InvalidTransition):
            state.transition_to("completed")

    def test_paused_to_completed_invalid(self):
        state = GoalState(description="test", status="paused")
        with pytest.raises(InvalidTransition):
            state.transition_to("completed")

    def test_paused_to_planning_invalid(self):
        state = GoalState(description="test", status="paused")
        with pytest.raises(InvalidTransition):
            state.transition_to("planning")

    def test_executing_to_planning_invalid(self):
        state = GoalState(description="test", status="executing")
        with pytest.raises(InvalidTransition):
            state.transition_to("planning")


class TestTransitionSideEffects:
    def test_started_at_set_once(self):
        state = GoalState(description="test")
        state.transition_to("executing")
        first_start = state.started_at

        # Pause and resume
        state.transition_to("paused")
        state.transition_to("executing")

        # started_at should not change
        assert state.started_at == first_start

    def test_completed_at_set_on_terminal(self):
        state = GoalState(description="test", status="executing")
        assert state.completed_at is None
        state.transition_to("completed")
        assert state.completed_at is not None

    def test_failed_sets_completed_at(self):
        state = GoalState(description="test", status="executing")
        state.transition_to("failed")
        assert state.completed_at is not None

    def test_error_message_contains_goal_id(self):
        state = GoalState(description="test", status="completed")
        with pytest.raises(InvalidTransition, match=state.goal_id):
            state.transition_to("executing")

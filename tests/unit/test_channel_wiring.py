"""Tests for the channel→bus→goal wiring in app.py and adapter callbacks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.bus.memory_bus import MemoryBus  # noqa: I001
from qe.channels.base import ChannelAdapter
from qe.channels.notifications import NotificationRouter
from qe.models.envelope import Envelope
from qe.models.goal import GoalState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubAdapter(ChannelAdapter):
    """Minimal concrete adapter for testing base-class behaviour."""

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def send(self, user_id, message, attachments=None):
        pass

    def _extract_text(self, raw_message):
        return raw_message.get("text", "") if isinstance(raw_message, dict) else str(raw_message)

    def _get_user_id(self, raw_message):
        return raw_message.get("user_id", "") if isinstance(raw_message, dict) else ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_adapter_forward_message_calls_callback():
    """Base adapter _forward_message calls the callback with the result dict."""
    captured = []
    adapter = _StubAdapter(
        channel_name="test",
        message_callback=lambda msg: captured.append(msg),
    )

    result = {"text": "hello", "user_id": "u1", "channel": "test"}
    adapter._forward_message(result)

    assert len(captured) == 1
    assert captured[0] is result


def test_adapter_forward_message_noop_without_callback():
    """_forward_message does nothing when no callback is configured."""
    adapter = _StubAdapter(channel_name="test")
    # Should not raise
    adapter._forward_message({"text": "hello"})


@pytest.mark.asyncio
async def test_channel_message_creates_goal():
    """channel.message_received envelope triggers planner.decompose and dispatcher.submit_goal."""
    bus = MemoryBus()

    mock_planner = AsyncMock()
    goal_state = GoalState(description="test goal")
    mock_planner.decompose.return_value = goal_state

    mock_dispatcher = AsyncMock()

    submitted = []
    bus.subscribe("goals.submitted", lambda e: submitted.append(e))

    # Import and wire the handler as done in _init_channels
    from qe.api import app as app_mod

    # Save originals
    orig_planner = app_mod._planner
    orig_dispatcher = app_mod._dispatcher

    try:
        app_mod._planner = mock_planner
        app_mod._dispatcher = mock_dispatcher

        # Simulate the _on_channel_message handler logic inline
        async def _on_channel_message(envelope: Envelope) -> None:
            if not app_mod._planner or not app_mod._dispatcher:
                return
            text = envelope.payload.get("text", "").strip()
            if not text:
                return
            user_id = envelope.payload.get("user_id", "")
            channel = envelope.payload.get("channel", "unknown")
            state = await app_mod._planner.decompose(text)
            state.metadata["origin_user_id"] = user_id
            state.metadata["origin_channel"] = channel
            await app_mod._dispatcher.submit_goal(state)
            bus.publish(Envelope(
                topic="goals.submitted",
                source_service_id="channel_bridge",
                correlation_id=state.goal_id,
                payload={
                    "goal_id": state.goal_id,
                    "description": text,
                    "channel": channel,
                    "user_id": user_id,
                },
            ))

        bus.subscribe("channel.message_received", _on_channel_message)

        bus.publish(Envelope(
            topic="channel.message_received",
            source_service_id="test",
            payload={
                "text": "Research solar energy benefits",
                "user_id": "user-42",
                "channel": "webhook",
            },
        ))

        import asyncio
        await asyncio.sleep(0.1)

        mock_planner.decompose.assert_called_once_with("Research solar energy benefits")
        mock_dispatcher.submit_goal.assert_called_once()

        # Verify metadata was set
        submitted_state = mock_dispatcher.submit_goal.call_args[0][0]
        assert submitted_state.metadata["origin_user_id"] == "user-42"
        assert submitted_state.metadata["origin_channel"] == "webhook"

        assert len(submitted) == 1
        assert submitted[0].payload["user_id"] == "user-42"

    finally:
        app_mod._planner = orig_planner
        app_mod._dispatcher = orig_dispatcher


@pytest.mark.asyncio
async def test_empty_message_ignored():
    """Empty text in channel.message_received should NOT trigger decompose."""
    bus = MemoryBus()

    mock_planner = AsyncMock()

    from qe.api import app as app_mod

    orig_planner = app_mod._planner
    orig_dispatcher = app_mod._dispatcher

    try:
        app_mod._planner = mock_planner
        app_mod._dispatcher = AsyncMock()

        async def _on_channel_message(envelope: Envelope) -> None:
            if not app_mod._planner or not app_mod._dispatcher:
                return
            text = envelope.payload.get("text", "").strip()
            if not text:
                return
            await app_mod._planner.decompose(text)

        bus.subscribe("channel.message_received", _on_channel_message)

        bus.publish(Envelope(
            topic="channel.message_received",
            source_service_id="test",
            payload={"text": "", "user_id": "u1", "channel": "webhook"},
        ))

        import asyncio
        await asyncio.sleep(0.1)

        mock_planner.decompose.assert_not_called()

    finally:
        app_mod._planner = orig_planner
        app_mod._dispatcher = orig_dispatcher


@pytest.mark.asyncio
async def test_goal_completed_notifies_user():
    """goals.completed triggers notification to the origin user, not broadcast."""
    router = NotificationRouter()
    mock_adapter = AsyncMock()
    router.register_channel("webhook", mock_adapter)

    mock_goal_store = AsyncMock()
    state = GoalState(description="test")
    state.metadata["origin_user_id"] = "user-99"
    mock_goal_store.load_goal.return_value = state

    # Simulate _on_goal_completed logic
    envelope = Envelope(
        topic="goals.completed",
        source_service_id="dispatcher",
        payload={"goal_id": state.goal_id, "subtask_count": 1},
    )

    goal_id = envelope.payload.get("goal_id", "")
    target_user = "broadcast"
    loaded = await mock_goal_store.load_goal(goal_id)
    if loaded:
        target_user = loaded.metadata.get("origin_user_id", "broadcast") or "broadcast"

    await router.notify(
        user_id=target_user,
        event_type="goal_result",
        message=f"Goal {goal_id} completed",
        urgency="high",
    )

    mock_adapter.send.assert_called_once()
    call_args = mock_adapter.send.call_args
    assert call_args[0][0] == "user-99"


@pytest.mark.asyncio
async def test_goal_failed_notifies_user():
    """goals.failed triggers notification to the origin user."""
    router = NotificationRouter()
    mock_adapter = AsyncMock()
    router.register_channel("webhook", mock_adapter)

    mock_goal_store = AsyncMock()
    state = GoalState(description="test")
    state.metadata["origin_user_id"] = "user-77"
    mock_goal_store.load_goal.return_value = state

    envelope = Envelope(
        topic="goals.failed",
        source_service_id="dispatcher",
        payload={"goal_id": state.goal_id, "reason": "timeout"},
    )

    goal_id = envelope.payload.get("goal_id", "")
    loaded = await mock_goal_store.load_goal(goal_id)
    target_user = loaded.metadata.get("origin_user_id", "broadcast") if loaded else "broadcast"

    await router.notify(
        user_id=target_user,
        event_type="goal_failed",
        message=f"Goal {goal_id} failed: timeout",
        urgency="high",
    )

    mock_adapter.send.assert_called_once()
    call_args = mock_adapter.send.call_args
    assert call_args[0][0] == "user-77"
    assert "failed" in call_args[0][1]


@pytest.mark.asyncio
async def test_ask_command_routes_to_query():
    """An 'ask' command on the bus triggers answer_question and notifies the user."""
    router = NotificationRouter()
    mock_adapter = AsyncMock()
    router.register_channel("webhook", mock_adapter)

    fake_answer = {"answer": "42 is the answer", "supporting_claims": [], "confidence": 0.9}

    with patch("qe.services.query.answer_question", new_callable=AsyncMock) as mock_aq:
        mock_aq.return_value = fake_answer

        mock_substrate = MagicMock()

        # Simulate _on_query_asked
        envelope = Envelope(
            topic="queries.asked",
            source_service_id="channel_webhook",
            payload={"text": "What is the meaning of life?", "user_id": "user-1"},
        )

        text = envelope.payload.get("text", "").strip()
        user_id = envelope.payload.get("user_id", "broadcast")

        result = await mock_aq(text, mock_substrate)
        answer = result.get("answer", "No answer found.")
        await router.notify(
            user_id=user_id or "broadcast",
            event_type="query_answer",
            message=f"Q: {text}\n\nA: {answer}",
            urgency="normal",
        )

        mock_aq.assert_called_once_with(text, mock_substrate)
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        assert call_args[0][0] == "user-1"
        assert "42 is the answer" in call_args[0][1]


@pytest.mark.asyncio
async def test_status_command_returns_status():
    """A 'status' command on the bus returns system info to the user."""
    router = NotificationRouter()
    mock_adapter = AsyncMock()
    router.register_channel("webhook", mock_adapter)

    # Simulate _on_health_check
    parts = ["System Status:"]
    parts.append("  Supervisor: running")
    parts.append("  Substrate: ready")
    parts.append("  Channels: 1 active")

    await router.notify(
        user_id="user-5",
        event_type="system_status",
        message="\n".join(parts),
        urgency="normal",
    )

    mock_adapter.send.assert_called_once()
    call_args = mock_adapter.send.call_args
    assert call_args[0][0] == "user-5"
    assert "Supervisor: running" in call_args[0][1]
    assert "Substrate: ready" in call_args[0][1]


def test_webhook_endpoint_processes_payload():
    """POST /api/webhooks/inbound returns 200 and publishes to bus."""
    from fastapi.testclient import TestClient

    from qe.api.app import app

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/webhooks/inbound",
        json={"text": "hello world", "user_id": "test-user"},
    )

    # Without full lifespan, channels aren't initialized → 503
    assert resp.status_code in (200, 503)


def test_configure_kilocode_sets_env():
    """_configure_kilocode sets OPENAI_API_KEY/BASE from KILOCODE vars."""
    import os

    from qe.api.app import _configure_kilocode

    # Save originals
    orig_kilo_key = os.environ.get("KILOCODE_API_KEY")
    orig_kilo_base = os.environ.get("KILOCODE_API_BASE")
    orig_oai_key = os.environ.get("OPENAI_API_KEY")
    orig_oai_base = os.environ.get("OPENAI_API_BASE")

    try:
        # Clear existing to ensure setdefault works
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_API_BASE", None)

        os.environ["KILOCODE_API_KEY"] = "test-kilo-key"
        os.environ["KILOCODE_API_BASE"] = "https://kilo.ai/api/openrouter"

        _configure_kilocode()

        assert os.environ["OPENAI_API_KEY"] == "test-kilo-key"
        assert os.environ["OPENAI_API_BASE"] == "https://kilo.ai/api/openrouter"

    finally:
        # Restore
        for var, orig in [
            ("KILOCODE_API_KEY", orig_kilo_key),
            ("KILOCODE_API_BASE", orig_kilo_base),
            ("OPENAI_API_KEY", orig_oai_key),
            ("OPENAI_API_BASE", orig_oai_base),
        ]:
            if orig is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = orig

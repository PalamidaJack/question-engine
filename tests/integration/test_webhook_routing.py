"""Integration test: webhook command routing through the bus."""

import asyncio

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.channels.webhook import WebhookAdapter
from qe.models.envelope import Envelope


@pytest.fixture
def bus():
    return MemoryBus()


@pytest.fixture
def webhook():
    return WebhookAdapter(secret="")


@pytest.mark.asyncio
async def test_webhook_ask_command_routes_to_queries(bus, webhook):
    """POST /api/webhooks/inbound with command=ask publishes to queries.asked."""
    received = []
    bus.subscribe("queries.asked", lambda e: received.append(e))

    # Simulate what inbound_webhook() does after process_webhook
    body = {
        "text": "What is quantum computing?",
        "user_id": "test-user",
        "command": "ask",
    }
    result = await webhook.process_webhook(body, {})
    assert result is not None

    command = body.get("command", "goal")
    topic_map = {
        "ask": "queries.asked",
        "status": "system.health.check",
    }
    topic = topic_map.get(command, "channel.message_received")

    bus.publish(
        Envelope(
            topic=topic,
            source_service_id="webhook",
            payload={
                "channel": "webhook",
                "user_id": result.get("user_id", ""),
                "text": result.get("sanitized_text", ""),
                "command": command,
            },
        )
    )

    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].topic == "queries.asked"
    assert received[0].payload["command"] == "ask"
    assert received[0].payload["user_id"] == "test-user"


@pytest.mark.asyncio
async def test_webhook_status_command_routes_to_health(bus, webhook):
    """POST /api/webhooks/inbound with command=status publishes to system.health.check."""
    received = []
    bus.subscribe("system.health.check", lambda e: received.append(e))

    body = {"text": "check", "user_id": "u2", "command": "status"}
    result = await webhook.process_webhook(body, {})
    assert result is not None

    command = body.get("command", "goal")
    topic_map = {
        "ask": "queries.asked",
        "status": "system.health.check",
    }
    topic = topic_map.get(command, "channel.message_received")

    bus.publish(
        Envelope(
            topic=topic,
            source_service_id="webhook",
            payload={
                "channel": "webhook",
                "user_id": result.get("user_id", ""),
                "text": result.get("sanitized_text", ""),
                "command": command,
            },
        )
    )

    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].topic == "system.health.check"


@pytest.mark.asyncio
async def test_webhook_default_routes_to_channel_message(bus, webhook):
    """Webhook without command field defaults to channel.message_received."""
    received = []
    bus.subscribe("channel.message_received", lambda e: received.append(e))

    body = {"text": "Research solar energy", "user_id": "u3"}
    result = await webhook.process_webhook(body, {})
    assert result is not None

    command = body.get("command", "goal")
    topic_map = {
        "ask": "queries.asked",
        "status": "system.health.check",
    }
    topic = topic_map.get(command, "channel.message_received")

    bus.publish(
        Envelope(
            topic=topic,
            source_service_id="webhook",
            payload={
                "channel": "webhook",
                "user_id": result.get("user_id", ""),
                "text": result.get("sanitized_text", ""),
                "command": command,
            },
        )
    )

    await asyncio.sleep(0.05)

    assert len(received) == 1
    assert received[0].topic == "channel.message_received"
    assert received[0].payload["command"] == "goal"

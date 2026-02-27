import pytest
import asyncio

from qe.models.envelope import Envelope
from qe.bus.memory_bus import MemoryBus
from qe.bus.protocol import validate_envelope


@pytest.mark.asyncio
async def test_subscribe_and_publish():
    """Subscribe a handler to 'observations.structured', publish, assert handler called."""
    bus = MemoryBus()
    received_envelopes = []

    async def handler(envelope: Envelope) -> None:
        received_envelopes.append(envelope)

    bus.subscribe("observations.structured", handler)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test-service",
        payload={"text": "test observation"}
    )

    bus.publish(envelope)

    # Give async tasks time to complete
    await asyncio.sleep(0.01)

    assert len(received_envelopes) == 1
    assert received_envelopes[0].envelope_id == envelope.envelope_id
    assert received_envelopes[0].payload == {"text": "test observation"}


@pytest.mark.asyncio
async def test_publish_to_topic_with_no_subscribers():
    """Publish to a topic with no subscribers — assert no error."""
    bus = MemoryBus()

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test-service",
        payload={"text": "test"}
    )

    # Should not raise any exception
    bus.publish(envelope)
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_publish_invalid_topic_raises_value_error():
    """Publish an envelope with an invalid topic — assert ValueError is raised."""
    bus = MemoryBus()

    envelope = Envelope(
        topic="invalid.topic",
        source_service_id="test-service",
        payload={"text": "test"}
    )

    with pytest.raises(ValueError, match="Unknown topic"):
        bus.publish(envelope)


@pytest.mark.asyncio
async def test_two_handlers_same_topic_both_called():
    """Subscribe two handlers to same topic, publish once, assert both called."""
    bus = MemoryBus()
    received_1 = []
    received_2 = []

    async def handler_1(envelope: Envelope) -> None:
        received_1.append(envelope)

    async def handler_2(envelope: Envelope) -> None:
        received_2.append(envelope)

    bus.subscribe("observations.structured", handler_1)
    bus.subscribe("observations.structured", handler_2)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="test-service",
        payload={"text": "test"}
    )

    bus.publish(envelope)
    await asyncio.sleep(0.01)

    assert len(received_1) == 1
    assert len(received_2) == 1
    assert received_1[0].envelope_id == received_2[0].envelope_id


@pytest.mark.asyncio
async def test_unsubscribe_handler():
    """Test that unsubscribing a handler prevents it from receiving messages."""
    bus = MemoryBus()
    received = []

    async def handler(envelope: Envelope) -> None:
        received.append(envelope)

    bus.subscribe("observations.structured", handler)

    # First publish - should receive
    envelope1 = Envelope(
        topic="observations.structured",
        source_service_id="test-service",
        payload={"text": "first"}
    )
    bus.publish(envelope1)
    await asyncio.sleep(0.01)

    # Unsubscribe
    bus.unsubscribe("observations.structured", handler)

    # Second publish - should not receive
    envelope2 = Envelope(
        topic="observations.structured",
        source_service_id="test-service",
        payload={"text": "second"}
    )
    bus.publish(envelope2)
    await asyncio.sleep(0.01)

    assert len(received) == 1
    assert received[0].payload == {"text": "first"}

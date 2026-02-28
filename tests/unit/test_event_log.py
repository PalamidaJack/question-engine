"""Tests for the durable event log."""

import os
import tempfile

import pytest

from qe.bus.event_log import EventLog
from qe.models.envelope import Envelope


@pytest.fixture
async def event_log():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    log = EventLog(db_path=path)
    await log.initialize()
    yield log
    os.unlink(path)


@pytest.mark.asyncio
async def test_append_and_replay(event_log):
    envelope = Envelope(
        topic="claims.proposed",
        source_service_id="researcher_alpha",
        payload={"subject": "test", "predicate": "is", "object": "working"},
    )

    await event_log.append(envelope)
    events = await event_log.replay()

    assert len(events) == 1
    assert events[0]["envelope_id"] == envelope.envelope_id
    assert events[0]["topic"] == "claims.proposed"
    assert events[0]["payload"]["subject"] == "test"


@pytest.mark.asyncio
async def test_replay_with_topic_filter(event_log):
    for topic in ["claims.proposed", "claims.committed", "claims.proposed"]:
        await event_log.append(Envelope(
            topic=topic,
            source_service_id="test",
            payload={"data": topic},
        ))

    proposed = await event_log.replay(topic="claims.proposed")
    assert len(proposed) == 2

    committed = await event_log.replay(topic="claims.committed")
    assert len(committed) == 1


@pytest.mark.asyncio
async def test_replay_respects_limit(event_log):
    for i in range(10):
        await event_log.append(Envelope(
            topic="claims.proposed",
            source_service_id="test",
            payload={"index": i},
        ))

    events = await event_log.replay(limit=3)
    assert len(events) == 3

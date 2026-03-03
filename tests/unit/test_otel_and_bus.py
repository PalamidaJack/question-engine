import asyncio
from contextlib import contextmanager

import pytest


def test_otel_noop_start_span():
    """Ensure otel.start_span is callable and no-op when OTEL SDK missing."""
    import qe.runtime.otel as otel

    tracer = otel.get_tracer("test_tracer")
    # tracer should provide start_as_current_span (noop or real)
    assert hasattr(tracer, "start_as_current_span")

    # start_span should be usable as a context manager without error
    with otel.start_span("unit.test", {"k": "v"}):
        x = 1 + 1
    assert x == 2


@pytest.mark.asyncio
async def test_memory_bus_publish_traces_and_dispatch(monkeypatch):
    """MemoryBus.publish should call otel.start_span and dispatch handlers."""
    from qe.bus.memory_bus import MemoryBus
    from qe.models.envelope import Envelope

    calls = []

    @contextmanager
    def fake_start_span(name, attributes=None):
        calls.append((name, attributes))
        yield

    monkeypatch.setattr("qe.runtime.otel.start_span", fake_start_span)

    bus = MemoryBus()

    handled = []

    async def handler(env):
        handled.append(env.topic)

    topic_name = "memory.updated"
    bus.subscribe(topic_name, handler)

    env = Envelope(topic=topic_name, source_service_id="test", payload={"hello": "world"})
    tasks = bus.publish(env)
    # await created tasks
    if tasks:
        await asyncio.gather(*tasks)

    assert handled == [topic_name]
    # Ensure our fake start_span was invoked at least once
    assert any(name == "bus.publish" for name, _ in calls)

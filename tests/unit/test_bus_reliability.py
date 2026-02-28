"""Tests for bus reliability: DLQ, idempotency, backpressure, retry."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.envelope import Envelope


def _make_envelope(topic: str = "system.heartbeat", **kwargs) -> Envelope:
    return Envelope(
        topic=topic,
        source_service_id="test",
        payload={"msg": "test"},
        **kwargs,
    )


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_duplicate_envelope_dropped(self):
        bus = MemoryBus()
        handler = AsyncMock()
        bus.subscribe("system.heartbeat", handler)

        env = _make_envelope()
        bus.publish(env)
        bus.publish(env)  # same envelope_id

        # Let tasks run
        await asyncio.sleep(0.05)

        # Handler should only be called once
        assert handler.await_count == 1

    @pytest.mark.asyncio
    async def test_different_envelope_ids_both_processed(self):
        bus = MemoryBus()
        handler = AsyncMock()
        bus.subscribe("system.heartbeat", handler)

        bus.publish(_make_envelope())
        bus.publish(_make_envelope())  # new UUID each time

        await asyncio.sleep(0.05)
        assert handler.await_count == 2

    def test_dedup_eviction_on_overflow(self):
        bus = MemoryBus(dedup_ttl=0.001)
        bus._dedup_max_size = 5

        # Fill beyond max
        for _ in range(10):
            env = _make_envelope()
            bus._seen_ids[env.envelope_id] = 0.0  # old timestamp

        # Eviction should clear stale entries
        bus._evict_stale_ids(1000.0)
        assert len(bus._seen_ids) == 0


class TestRetryAndDLQ:
    @pytest.mark.asyncio
    async def test_handler_retried_on_failure(self):
        bus = MemoryBus(max_retries=2)
        call_count = 0

        async def flaky_handler(_env: Envelope) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")

        bus.subscribe("system.heartbeat", flaky_handler)
        bus.publish(_make_envelope())
        await asyncio.sleep(1.0)  # allow retries with backoff

        # Should have been called 3 times (1 initial + 2 retries), and succeeded
        assert call_count == 3
        assert bus.dlq_size() == 0

    @pytest.mark.asyncio
    async def test_permanent_failure_routes_to_dlq(self):
        bus = MemoryBus(max_retries=1)

        async def bad_handler(_env: Envelope) -> None:
            raise RuntimeError("permanent error")

        bus.subscribe("system.heartbeat", bad_handler)
        env = _make_envelope()
        bus.publish(env)
        await asyncio.sleep(0.5)

        assert bus.dlq_size() == 1
        entries = bus.dlq_list()
        assert entries[0]["envelope_id"] == env.envelope_id
        assert "permanent error" in entries[0]["error"]
        assert entries[0]["attempts"] == 2  # 1 initial + 1 retry

    @pytest.mark.asyncio
    async def test_zero_retries_immediate_dlq(self):
        bus = MemoryBus(max_retries=0)

        async def bad_handler(_env: Envelope) -> None:
            raise ValueError("fail")

        bus.subscribe("system.heartbeat", bad_handler)
        bus.publish(_make_envelope())
        await asyncio.sleep(0.1)

        assert bus.dlq_size() == 1

    @pytest.mark.asyncio
    async def test_successful_handler_no_dlq(self):
        bus = MemoryBus(max_retries=2)
        bus.subscribe("system.heartbeat", AsyncMock())
        bus.publish(_make_envelope())
        await asyncio.sleep(0.1)

        assert bus.dlq_size() == 0


class TestDLQOperations:
    @pytest.mark.asyncio
    async def test_dlq_replay(self):
        bus = MemoryBus(max_retries=0)
        handler = AsyncMock(side_effect=[RuntimeError("fail"), None])
        bus.subscribe("system.heartbeat", handler)

        env = _make_envelope()
        bus.publish(env)
        await asyncio.sleep(0.1)
        assert bus.dlq_size() == 1

        # Fix the handler (second call succeeds) and replay
        ok = await bus.dlq_replay(env.envelope_id)
        assert ok is True
        await asyncio.sleep(0.1)

        assert bus.dlq_size() == 0
        assert handler.await_count == 2

    @pytest.mark.asyncio
    async def test_dlq_replay_not_found(self):
        bus = MemoryBus()
        ok = await bus.dlq_replay("nonexistent")
        assert ok is False

    @pytest.mark.asyncio
    async def test_dlq_purge(self):
        bus = MemoryBus(max_retries=0)

        async def bad_handler(_env: Envelope) -> None:
            raise RuntimeError("fail")

        bus.subscribe("system.heartbeat", bad_handler)
        for _ in range(3):
            bus.publish(_make_envelope())
        await asyncio.sleep(0.2)

        assert bus.dlq_size() == 3
        count = await bus.dlq_purge()
        assert count == 3
        assert bus.dlq_size() == 0

    @pytest.mark.asyncio
    async def test_dlq_list_returns_recent_first(self):
        bus = MemoryBus(max_retries=0)

        async def bad_handler(_env: Envelope) -> None:
            raise RuntimeError("fail")

        bus.subscribe("system.heartbeat", bad_handler)
        envs = [_make_envelope() for _ in range(3)]
        for env in envs:
            bus.publish(env)
        await asyncio.sleep(0.2)

        entries = bus.dlq_list()
        # Most recent first
        assert entries[0]["envelope_id"] == envs[2].envelope_id


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_concurrent_handlers_limited(self):
        bus = MemoryBus(topic_concurrency=2)
        active = 0
        max_active = 0

        async def slow_handler(_env: Envelope) -> None:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.1)
            active -= 1

        bus.subscribe("system.heartbeat", slow_handler)

        # Publish 5 envelopes rapidly
        for _ in range(5):
            bus.publish(_make_envelope())
        await asyncio.sleep(1.0)

        # Should never exceed concurrency limit of 2
        assert max_active <= 2

    @pytest.mark.asyncio
    async def test_default_semaphore_allows_throughput(self):
        bus = MemoryBus(topic_concurrency=10)
        handler = AsyncMock()
        bus.subscribe("system.heartbeat", handler)

        for _ in range(5):
            bus.publish(_make_envelope())
        await asyncio.sleep(0.1)

        assert handler.await_count == 5


class TestDLQNotification:
    @pytest.mark.asyncio
    async def test_dlq_subscribers_notified(self):
        bus = MemoryBus(max_retries=0)
        dlq_handler = AsyncMock()
        bus.subscribe("system.dlq", dlq_handler)

        async def bad_handler(_env: Envelope) -> None:
            raise RuntimeError("fail")

        bus.subscribe("system.heartbeat", bad_handler)
        bus.publish(_make_envelope())
        await asyncio.sleep(0.2)

        assert dlq_handler.await_count == 1
        dlq_env = dlq_handler.call_args[0][0]
        assert dlq_env.topic == "system.dlq"
        assert "fail" in dlq_env.payload["error"]

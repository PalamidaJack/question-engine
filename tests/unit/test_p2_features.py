"""Comprehensive tests for all P2 features.

Covers: LLM response caching, provider rate limiting, secrets rotation,
service container/DI, saga compensation, structured concurrency.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from qe.runtime.container import ServiceContainer
from qe.runtime.llm_cache import CacheStats, LLMCache
from qe.runtime.rate_limiter import RateLimiter, TokenBucket
from qe.runtime.saga import (
    CompensationAction,
    CompensationExecutor,
    CompensationResult,
    NoopExecutor,
    SagaCoordinator,
    SagaState,
)
from qe.runtime.secrets import SecretsManager
from qe.runtime.task_scope import DaemonScope, TaskScope

# ── P2 #1: LLM Response Caching ──────────────────────────────────────────


class TestLLMCache:
    def test_put_and_get(self):
        cache = LLMCache()
        cache.put("key1", {"answer": "hello"}, "gpt-4o")
        result = cache.get("key1")
        assert result == {"answer": "hello"}

    def test_cache_miss(self):
        cache = LLMCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = LLMCache(default_ttl_seconds=0.01)
        cache.put("key1", "value", "gpt-4o")
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_cache_stats_hit(self):
        cache = LLMCache()
        cache.put("k", "v", "m")
        cache.get("k")
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_stats_miss(self):
        cache = LLMCache()
        cache.get("nonexistent")
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1

    def test_hit_rate(self):
        s = CacheStats(hits=3, misses=1)
        assert s.hit_rate == 0.75

    def test_hit_rate_zero_total(self):
        s = CacheStats()
        assert s.hit_rate == 0.0

    def test_max_entries_eviction(self):
        cache = LLMCache(max_entries=3)
        cache.put("k1", "v1", "m")
        cache.put("k2", "v2", "m")
        cache.put("k3", "v3", "m")
        cache.put("k4", "v4", "m")  # evicts oldest
        assert cache.get("k1") is None
        assert cache.get("k4") == "v4"
        stats = cache.stats()
        assert stats["entries"] == 3
        assert stats["evictions"] >= 1

    def test_clear(self):
        cache = LLMCache()
        cache.put("k1", "v1", "m")
        cache.put("k2", "v2", "m")
        count = cache.clear()
        assert count == 2
        assert cache.get("k1") is None
        assert cache.stats()["entries"] == 0

    def test_disabled_cache(self):
        cache = LLMCache(enabled=False)
        cache.put("k", "v", "m")
        assert cache.get("k") is None

    def test_enable_disable(self):
        cache = LLMCache()
        cache.put("k", "v", "m")
        assert cache.get("k") == "v"
        cache.enabled = False
        assert cache.get("k") is None

    def test_make_key_deterministic(self):
        k1 = LLMCache.make_key("gpt-4o", [{"role": "user", "content": "hi"}], "Schema")
        k2 = LLMCache.make_key("gpt-4o", [{"role": "user", "content": "hi"}], "Schema")
        assert k1 == k2

    def test_make_key_different_models(self):
        k1 = LLMCache.make_key("gpt-4o", [{"role": "user", "content": "hi"}], "Schema")
        k2 = LLMCache.make_key("gpt-4o-mini", [{"role": "user", "content": "hi"}], "Schema")
        assert k1 != k2

    def test_make_key_different_messages(self):
        k1 = LLMCache.make_key("gpt-4o", [{"role": "user", "content": "hi"}], "Schema")
        k2 = LLMCache.make_key("gpt-4o", [{"role": "user", "content": "bye"}], "Schema")
        assert k1 != k2

    def test_saved_calls_counter(self):
        cache = LLMCache()
        cache.put("k", "v", "m")
        cache.get("k")
        cache.get("k")
        assert cache.stats()["total_saved_calls"] == 2

    def test_expired_entry_evicts(self):
        cache = LLMCache(default_ttl_seconds=0.01)
        cache.put("k", "v", "m")
        time.sleep(0.02)
        cache.get("k")
        stats = cache.stats()
        assert stats["evictions"] >= 1


# ── P2 #2: Provider Rate Limiting ────────────────────────────────────────


class TestTokenBucket:
    def test_initial_tokens(self):
        bucket = TokenBucket(provider="gpt-", rpm=60)
        assert bucket.available_tokens == 60.0

    def test_try_acquire_consumes_token(self):
        bucket = TokenBucket(provider="gpt-", rpm=60)
        assert bucket.try_acquire() is True
        assert bucket.available_tokens < 60.0

    def test_try_acquire_when_empty(self):
        bucket = TokenBucket(provider="test", rpm=1)
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is False

    def test_refill_over_time(self):
        bucket = TokenBucket(provider="test", rpm=6000)  # 100/sec
        for _ in range(100):
            bucket.try_acquire()
        time.sleep(0.02)
        assert bucket.available_tokens > 0

    @pytest.mark.asyncio
    async def test_acquire_waits(self):
        bucket = TokenBucket(provider="test", rpm=600)  # 10/sec
        # Drain tokens
        while bucket.try_acquire():
            pass
        # Should wait and get a token
        result = await bucket.acquire(max_wait=5.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        bucket = TokenBucket(provider="test", rpm=1)  # 1/min
        bucket.try_acquire()  # drain
        result = await bucket.acquire(max_wait=0.01)
        assert result is False


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_creates_bucket(self):
        limiter = RateLimiter()
        result = await limiter.acquire("gpt-4o-mini")
        assert result is True
        stats = limiter.stats()
        assert len(stats["buckets"]) == 1

    @pytest.mark.asyncio
    async def test_disabled_always_passes(self):
        limiter = RateLimiter(enabled=False)
        result = await limiter.acquire("gpt-4o")
        assert result is True
        assert limiter.stats()["total_requests"] == 0

    @pytest.mark.asyncio
    async def test_different_providers_separate_buckets(self):
        limiter = RateLimiter()
        await limiter.acquire("gpt-4o")
        await limiter.acquire("claude-sonnet")
        stats = limiter.stats()
        assert len(stats["buckets"]) == 2

    @pytest.mark.asyncio
    async def test_same_provider_shared_bucket(self):
        limiter = RateLimiter()
        await limiter.acquire("gpt-4o")
        await limiter.acquire("gpt-4o-mini")
        stats = limiter.stats()
        # Both "gpt-" prefix, same bucket
        assert len(stats["buckets"]) == 1

    def test_try_acquire(self):
        limiter = RateLimiter()
        assert limiter.try_acquire("gpt-4o") is True

    def test_try_acquire_disabled(self):
        limiter = RateLimiter(enabled=False)
        assert limiter.try_acquire("gpt-4o") is True

    def test_set_rpm(self):
        limiter = RateLimiter()
        limiter.set_rpm("gpt-", 100)
        limiter.try_acquire("gpt-4o")
        stats = limiter.stats()
        assert stats["buckets"]["gpt-"]["rpm"] == 100

    def test_custom_limits(self):
        limiter = RateLimiter(custom_limits={"gpt-": 42})
        limiter.try_acquire("gpt-4o")
        stats = limiter.stats()
        assert stats["buckets"]["gpt-"]["rpm"] == 42

    @pytest.mark.asyncio
    async def test_stats_tracks_requests(self):
        limiter = RateLimiter()
        await limiter.acquire("gpt-4o")
        await limiter.acquire("gpt-4o")
        assert limiter.stats()["total_requests"] == 2

    def test_ollama_high_limit(self):
        limiter = RateLimiter()
        limiter.try_acquire("ollama/qwen3")
        stats = limiter.stats()
        assert stats["buckets"]["ollama/"]["rpm"] == 10000


# ── P2 #3: Secrets Rotation ─────────────────────────────────────────────


class TestSecretsManager:
    def test_load_initial(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("API_KEY=sk-test123\nSECRET=mysecret\n")
        mgr = SecretsManager(env_path=env_file)
        assert mgr.get("API_KEY") == "sk-test123"
        assert mgr.get("SECRET") == "mysecret"

    def test_load_missing_file(self, tmp_path):
        mgr = SecretsManager(env_path=tmp_path / "nonexistent.env")
        assert mgr.get("ANYTHING") is None

    def test_check_no_changes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=val\n")
        mgr = SecretsManager(env_path=env_file)
        assert mgr.check_for_changes() is None

    def test_detect_changes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=old\n")
        mgr = SecretsManager(env_path=env_file)

        # Modify file (ensure mtime changes)
        time.sleep(0.05)
        env_file.write_text("KEY=new\n")

        changed = mgr.check_for_changes()
        assert changed is not None
        assert changed["KEY"] == "new"
        assert mgr.get("KEY") == "new"

    def test_detect_new_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=val\n")
        mgr = SecretsManager(env_path=env_file)

        time.sleep(0.05)
        env_file.write_text("KEY=val\nNEW_KEY=added\n")

        changed = mgr.check_for_changes()
        assert "NEW_KEY" in changed

    def test_detect_removed_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=val\nOLD=remove_me\n")
        mgr = SecretsManager(env_path=env_file)

        time.sleep(0.05)
        env_file.write_text("KEY=val\n")

        changed = mgr.check_for_changes()
        assert "OLD" in changed
        assert changed["OLD"] is None

    def test_callback_invoked(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=old\n")
        mgr = SecretsManager(env_path=env_file)

        received = []
        mgr.on_rotate(lambda changes: received.append(changes))

        time.sleep(0.05)
        env_file.write_text("KEY=new\n")
        mgr.check_for_changes()

        assert len(received) == 1
        assert received[0]["KEY"] == "new"

    def test_rotation_history(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("A=1\n")
        mgr = SecretsManager(env_path=env_file)

        time.sleep(0.05)
        env_file.write_text("A=2\n")
        mgr.check_for_changes()

        history = mgr.rotation_history()
        assert len(history) == 1
        assert "A" in history[0]["changed_keys"]

    def test_force_reload(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("K=v1\n")
        mgr = SecretsManager(env_path=env_file)

        env_file.write_text("K=v2\n")
        changed = mgr.force_reload()
        assert changed.get("K") == "v2"

    def test_masked_values(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("API_KEY=sk-1234567890abcdef\n")
        mgr = SecretsManager(env_path=env_file)

        masked = mgr.masked_values()
        assert "API_KEY" in masked
        assert "sk-1" in masked["API_KEY"]
        assert "cdef" in masked["API_KEY"]
        assert "1234567890" not in masked["API_KEY"]

    def test_env_var_updated(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_SECRET_KEY=initial\n")
        mgr = SecretsManager(env_path=env_file)

        time.sleep(0.05)
        env_file.write_text("TEST_SECRET_KEY=rotated\n")
        mgr.check_for_changes()

        assert os.environ.get("TEST_SECRET_KEY") == "rotated"

        # Cleanup
        os.environ.pop("TEST_SECRET_KEY", None)


# ── P2 #4: Service Container ────────────────────────────────────────────


class TestServiceContainer:
    def test_register_and_resolve(self):
        c = ServiceContainer()
        c.register("bus", lambda: {"type": "memory_bus"})
        result = c.resolve("bus")
        assert result == {"type": "memory_bus"}

    def test_singleton_behavior(self):
        c = ServiceContainer()
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"instance": call_count}

        c.register("svc", factory)
        r1 = c.resolve("svc")
        r2 = c.resolve("svc")
        assert r1 is r2
        assert call_count == 1

    def test_register_instance(self):
        c = ServiceContainer()
        obj = object()
        c.register_instance("thing", obj)
        assert c.resolve("thing") is obj

    def test_override(self):
        c = ServiceContainer()
        c.register("bus", lambda: "real_bus")
        c.override("bus", "mock_bus")
        assert c.resolve("bus") == "mock_bus"

    def test_clear_override(self):
        c = ServiceContainer()
        c.register("bus", lambda: "real")
        c.override("bus", "mock")
        assert c.resolve("bus") == "mock"
        c.clear_override("bus")
        assert c.resolve("bus") == "real"

    def test_clear_all_overrides(self):
        c = ServiceContainer()
        c.register("a", lambda: "real_a")
        c.register("b", lambda: "real_b")
        c.override("a", "mock_a")
        c.override("b", "mock_b")
        c.clear_overrides()
        assert c.resolve("a") == "real_a"
        assert c.resolve("b") == "real_b"

    def test_resolve_unregistered_raises(self):
        c = ServiceContainer()
        with pytest.raises(KeyError, match="not registered"):
            c.resolve("nonexistent")

    def test_has(self):
        c = ServiceContainer()
        assert c.has("bus") is False
        c.register("bus", lambda: None)
        assert c.has("bus") is True

    def test_has_override(self):
        c = ServiceContainer()
        c.override("bus", "mock")
        assert c.has("bus") is True

    def test_registered_names(self):
        c = ServiceContainer()
        c.register("bus", lambda: None)
        c.register("auth", lambda: None)
        c.register_instance("metrics", object())
        names = c.registered_names()
        assert "auth" in names
        assert "bus" in names
        assert "metrics" in names

    def test_reset(self):
        c = ServiceContainer()
        c.register("bus", lambda: "bus")
        c.resolve("bus")
        c.override("auth", "mock")
        c.reset()
        assert c.has("bus") is False
        assert c.registered_names() == []

    def test_status(self):
        c = ServiceContainer()
        c.register("a", lambda: None)
        c.register("b", lambda: None)
        c.resolve("a")
        s = c.status()
        assert s["registered"] == 2
        assert s["instantiated"] == 1
        assert s["overrides"] == 0

    def test_re_register_replaces_factory(self):
        c = ServiceContainer()
        c.register("svc", lambda: "v1")
        c.resolve("svc")
        c.register("svc", lambda: "v2")
        # Cached instance from first factory still returned
        assert c.resolve("svc") == "v1"


# ── P2 #5: Saga Compensation ────────────────────────────────────────────


class TestSagaState:
    def test_register_compensation(self):
        saga = SagaState(goal_id="g1")
        action = saga.register("sub1", "retract_claims", {"claim_ids": ["c1"]})
        assert action.subtask_id == "sub1"
        assert action.action_type == "retract_claims"
        assert len(saga.actions) == 1

    def test_pending_compensations_reverse_order(self):
        saga = SagaState(goal_id="g1")
        saga.register("sub1", "retract_claims")
        saga.register("sub2", "delete_output")
        saga.register("sub3", "notify")
        pending = saga.pending_compensations
        assert [a.subtask_id for a in pending] == ["sub3", "sub2", "sub1"]

    def test_executed_excluded_from_pending(self):
        saga = SagaState(goal_id="g1")
        a1 = saga.register("sub1", "retract_claims")
        saga.register("sub2", "notify")
        a1.executed = True
        pending = saga.pending_compensations
        assert len(pending) == 1
        assert pending[0].subtask_id == "sub2"

    def test_to_dict(self):
        saga = SagaState(goal_id="g1")
        saga.register("sub1", "noop")
        d = saga.to_dict()
        assert d["goal_id"] == "g1"
        assert d["action_count"] == 1
        assert d["executed_count"] == 0


class TestSagaCoordinator:
    def test_create_saga(self):
        coord = SagaCoordinator()
        saga = coord.create_saga("g1")
        assert saga.goal_id == "g1"
        assert coord.get_saga("g1") is saga

    def test_register_compensation(self):
        coord = SagaCoordinator()
        action = coord.register_compensation("g1", "sub1", "retract_claims")
        assert action is not None
        assert action.subtask_id == "sub1"

    def test_register_auto_creates_saga(self):
        coord = SagaCoordinator()
        coord.register_compensation("g1", "sub1", "noop")
        assert coord.get_saga("g1") is not None

    @pytest.mark.asyncio
    async def test_compensate_noop(self):
        coord = SagaCoordinator()
        coord.register_compensation("g1", "sub1", "retract_claims")
        coord.register_compensation("g1", "sub2", "notify")

        result = await coord.compensate("g1")
        assert result.status == "compensated"
        assert result.actions_total == 2
        assert result.actions_executed == 2

    @pytest.mark.asyncio
    async def test_compensate_no_saga(self):
        coord = SagaCoordinator()
        result = await coord.compensate("nonexistent")
        assert result.status == "no_saga"

    @pytest.mark.asyncio
    async def test_compensate_with_error(self):
        coord = SagaCoordinator()
        coord.register_compensation("g1", "sub1", "retract_claims")

        class FailingExecutor(CompensationExecutor):
            async def execute(self, action):
                raise RuntimeError("compensation failed")

        result = await coord.compensate("g1", executor=FailingExecutor())
        assert result.status == "failed"
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_compensate_with_custom_executor(self):
        coord = SagaCoordinator()
        coord.register_compensation("g1", "sub1", "retract_claims", {"ids": ["c1"]})

        executed_actions = []

        class TrackingExecutor(CompensationExecutor):
            async def execute(self, action):
                executed_actions.append(action)

        result = await coord.compensate("g1", executor=TrackingExecutor())
        assert result.status == "compensated"
        assert len(executed_actions) == 1
        assert executed_actions[0].parameters == {"ids": ["c1"]}

    def test_remove_saga(self):
        coord = SagaCoordinator()
        coord.create_saga("g1")
        coord.remove_saga("g1")
        assert coord.get_saga("g1") is None

    def test_active_sagas(self):
        coord = SagaCoordinator()
        coord.create_saga("g1")
        coord.create_saga("g2")
        sagas = coord.active_sagas()
        assert len(sagas) == 2

    @pytest.mark.asyncio
    async def test_compensation_result_to_dict(self):
        result = CompensationResult(
            goal_id="g1",
            status="compensated",
            actions_total=3,
            actions_executed=3,
        )
        d = result.to_dict()
        assert d["goal_id"] == "g1"
        assert d["status"] == "compensated"


class TestCompensationAction:
    def test_defaults(self):
        action = CompensationAction(subtask_id="s1", action_type="noop")
        assert action.executed is False
        assert action.error is None
        assert action.parameters == {}


class TestNoopExecutor:
    @pytest.mark.asyncio
    async def test_noop_does_nothing(self):
        executor = NoopExecutor()
        action = CompensationAction(subtask_id="s1", action_type="test")
        await executor.execute(action)  # should not raise


# ── P2 #6: Structured Concurrency ───────────────────────────────────────


class TestTaskScope:
    @pytest.mark.asyncio
    async def test_basic_scope(self):
        results = []

        async def worker(n):
            results.append(n)

        async with TaskScope("test", fail_fast=False) as scope:
            scope.spawn(worker(1))
            scope.spawn(worker(2))
            await asyncio.sleep(0.01)

        assert sorted(results) == [1, 2]

    @pytest.mark.asyncio
    async def test_graceful_cleanup(self):
        cancelled = []

        async def long_task():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async with TaskScope("test", fail_fast=False) as scope:
            scope.spawn(long_task())
            await asyncio.sleep(0.01)
            # Exiting scope should cancel

        assert len(cancelled) == 1

    @pytest.mark.asyncio
    async def test_scope_errors_collected(self):
        async def failing():
            raise ValueError("boom")

        async with TaskScope("test", fail_fast=False) as scope:
            scope.spawn(failing())
            await asyncio.sleep(0.01)

        assert len(scope.errors) >= 1

    @pytest.mark.asyncio
    async def test_fail_fast_with_taskgroup(self):
        async def good_task():
            await asyncio.sleep(0.01)

        async def bad_task():
            raise ValueError("fail fast!")

        with pytest.raises(ExceptionGroup):
            async with TaskScope("test", fail_fast=True) as scope:
                scope.spawn(good_task())
                scope.spawn(bad_task())

    @pytest.mark.asyncio
    async def test_running_property(self):
        async with TaskScope("test", fail_fast=False) as scope:
            assert scope.running is True
        assert scope.running is False


class TestDaemonScope:
    @pytest.mark.asyncio
    async def test_register_and_status(self):
        ds = DaemonScope(name="test")
        ds.register("heartbeat", lambda: asyncio.sleep(0.01))
        status = ds.status()
        assert "heartbeat" in status["daemons"]
        assert status["running"] is False

    @pytest.mark.asyncio
    async def test_stop(self):
        ds = DaemonScope(name="test")
        ds.register("task", lambda: asyncio.sleep(100))

        run_task = asyncio.create_task(ds.run())
        await asyncio.sleep(0.05)
        await ds.stop()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

        assert ds._running is False

    @pytest.mark.asyncio
    async def test_daemon_restart_on_failure(self):
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient")
            await asyncio.sleep(100)

        ds = DaemonScope(name="test", max_restarts=3)
        ds.register("flaky", flaky)

        run_task = asyncio.create_task(ds.run())
        await asyncio.sleep(3.5)  # supervisor polls every 1s; need 2+ cycles
        await ds.stop()
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass

        assert call_count >= 2  # at least original + 1 restart

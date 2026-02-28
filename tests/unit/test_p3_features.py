"""Tests for P3 features.

P3 #1: Readiness probe
P3 #2: Graceful shutdown with drain
P3 #3: SQLite connection pooling
P3 #4: DLQ persistence
P3 #5: CLI admin commands (config-validate only, others need running server)
P3 #6: Outbound webhook notifications
P3 #7: General-purpose feature flags
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from qe.bus.persistent_dlq import PersistentDLQ
from qe.models.envelope import Envelope
from qe.runtime.connection_pool import ConnectionPool, PoolManager
from qe.runtime.feature_flags import FeatureFlagDef, FeatureFlagStore
from qe.runtime.readiness import ReadinessState
from qe.runtime.shutdown import ShutdownCoordinator
from qe.runtime.webhooks import WebhookDelivery, WebhookNotifier, WebhookTarget

# ═══════════════════════════════════════════════════════════════════════════
# P3 #1 — Readiness Probe
# ═══════════════════════════════════════════════════════════════════════════


class TestReadinessState:
    def test_initial_state_not_ready(self):
        state = ReadinessState()
        assert state.is_ready is False

    def test_all_phases_make_ready(self):
        state = ReadinessState()
        state.mark_ready("substrate_ready")
        assert state.is_ready is False
        state.mark_ready("event_log_ready")
        assert state.is_ready is False
        state.mark_ready("services_subscribed")
        assert state.is_ready is False
        state.mark_ready("supervisor_ready")
        assert state.is_ready is True

    def test_ready_at_set_once(self):
        state = ReadinessState()
        all_phases = [
            "substrate_ready", "event_log_ready",
            "services_subscribed", "supervisor_ready",
        ]
        for phase in all_phases:
            state.mark_ready(phase)
        first_ready = state.ready_at
        assert first_ready is not None
        # Re-marking shouldn't change ready_at
        state.mark_ready("supervisor_ready")
        assert state.ready_at == first_ready

    def test_to_dict(self):
        state = ReadinessState()
        state.mark_ready("substrate_ready")
        d = state.to_dict()
        assert d["ready"] is False
        assert d["phases"]["substrate_ready"] is True
        assert d["phases"]["event_log_ready"] is False
        assert "uptime_seconds" in d
        assert d["startup_duration"] is None

    def test_to_dict_when_ready(self):
        state = ReadinessState()
        all_phases = [
            "substrate_ready", "event_log_ready",
            "services_subscribed", "supervisor_ready",
        ]
        for phase in all_phases:
            state.mark_ready(phase)
        d = state.to_dict()
        assert d["ready"] is True
        assert d["startup_duration"] is not None

    def test_unknown_phase_ignored(self):
        state = ReadinessState()
        state.mark_ready("nonexistent_phase")
        assert state.is_ready is False

    def test_partial_readiness(self):
        state = ReadinessState()
        state.mark_ready("substrate_ready")
        state.mark_ready("event_log_ready")
        d = state.to_dict()
        assert d["phases"]["substrate_ready"] is True
        assert d["phases"]["event_log_ready"] is True
        assert d["phases"]["services_subscribed"] is False
        assert d["phases"]["supervisor_ready"] is False


# ═══════════════════════════════════════════════════════════════════════════
# P3 #2 — Graceful Shutdown
# ═══════════════════════════════════════════════════════════════════════════


class TestShutdownCoordinator:
    def test_initial_state(self):
        sc = ShutdownCoordinator(drain_timeout=10.0)
        assert sc.in_flight == 0
        assert sc.is_draining is False
        assert sc.shutdown_requested is False

    def test_enter_exit_handler(self):
        sc = ShutdownCoordinator()
        assert sc.enter_handler() is True
        assert sc.in_flight == 1
        sc.enter_handler()
        assert sc.in_flight == 2
        sc.exit_handler()
        assert sc.in_flight == 1
        sc.exit_handler()
        assert sc.in_flight == 0

    def test_enter_rejected_when_draining(self):
        sc = ShutdownCoordinator()
        sc._draining = True
        assert sc.enter_handler() is False
        assert sc.in_flight == 0

    def test_exit_handler_never_negative(self):
        sc = ShutdownCoordinator()
        sc.exit_handler()  # no enter
        assert sc.in_flight == 0

    @pytest.mark.asyncio
    async def test_drain_immediate_when_no_work(self):
        sc = ShutdownCoordinator(drain_timeout=1.0)
        result = await sc.drain()
        assert result is True
        assert sc.is_draining is True

    @pytest.mark.asyncio
    async def test_drain_waits_for_in_flight(self):
        sc = ShutdownCoordinator(drain_timeout=5.0)
        sc.enter_handler()

        async def finish_later():
            await asyncio.sleep(0.1)
            sc.exit_handler()

        asyncio.create_task(finish_later())
        result = await sc.drain()
        assert result is True
        assert sc.in_flight == 0

    @pytest.mark.asyncio
    async def test_drain_timeout(self):
        sc = ShutdownCoordinator(drain_timeout=0.1)
        sc.enter_handler()  # never exits
        result = await sc.drain()
        assert result is False
        assert sc.in_flight == 1
        # Cleanup
        sc.exit_handler()

    def test_on_shutdown_callback(self):
        sc = ShutdownCoordinator()
        called = []
        sc.on_shutdown(lambda: called.append(True))
        assert len(sc._shutdown_callbacks) == 1

    def test_status(self):
        sc = ShutdownCoordinator(drain_timeout=15.0)
        sc.enter_handler()
        s = sc.status()
        assert s["in_flight"] == 1
        assert s["draining"] is False
        assert s["drain_timeout"] == 15.0
        sc.exit_handler()


# ═══════════════════════════════════════════════════════════════════════════
# P3 #3 — Connection Pooling
# ═══════════════════════════════════════════════════════════════════════════


class TestConnectionPool:
    @pytest.mark.asyncio
    async def test_initialize_creates_connection(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path, max_size=3)
            await pool.initialize()
            assert pool._size == 1
            assert pool._pool.qsize() == 1
            await pool.close()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_acquire_and_execute(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path, max_size=3)
            await pool.initialize()
            async with pool.acquire() as conn:
                await conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
                await conn.execute("INSERT INTO test VALUES (1)")
                await conn.commit()
                cursor = await conn.execute("SELECT COUNT(*) FROM test")
                row = await cursor.fetchone()
                assert row[0] == 1
            await pool.close()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_connection_reuse(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path, max_size=3)
            await pool.initialize()
            # Use and return
            async with pool.acquire():
                pass
            assert pool._total_created == 1
            # Reuse same connection
            async with pool.acquire():
                pass
            assert pool._total_created == 1
            assert pool._total_acquired == 2
            await pool.close()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_pool_grows_under_pressure(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path, max_size=3)
            await pool.initialize()

            # Hold first connection, forcing pool to create a second
            async with pool.acquire():
                async with pool.acquire():
                    assert pool._total_created == 2
            await pool.close()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_close_pool(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path, max_size=2)
            await pool.initialize()
            await pool.close()
            assert pool._closed is True
            assert pool._size == 0
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_acquire_after_close_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path, max_size=2)
            await pool.initialize()
            await pool.close()
            with pytest.raises(RuntimeError, match="closed"):
                async with pool.acquire():
                    pass
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pool = ConnectionPool(db_path, max_size=5)
            await pool.initialize()
            s = pool.stats()
            assert s["max_size"] == 5
            assert s["current_size"] == 1
            assert s["available"] == 1
            assert s["closed"] is False
            await pool.close()
        finally:
            os.unlink(db_path)


class TestPoolManager:
    @pytest.mark.asyncio
    async def test_get_pool_creates_and_caches(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pm = PoolManager(default_max_size=3)
            pool1 = await pm.get_pool(db_path)
            pool2 = await pm.get_pool(db_path)
            assert pool1 is pool2
            await pm.close_all()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_acquire_shortcut(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pm = PoolManager()
            async with pm.acquire(db_path) as conn:
                await conn.execute("CREATE TABLE t (id INTEGER)")
                await conn.commit()
            await pm.close_all()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pm = PoolManager()
            await pm.get_pool(db_path)
            s = pm.stats()
            assert s["total_pools"] == 1
            assert db_path in s["pools"]
            await pm.close_all()
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_close_all(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            pm = PoolManager()
            await pm.get_pool(db_path)
            await pm.close_all()
            assert len(pm._pools) == 0
        finally:
            os.unlink(db_path)


# ═══════════════════════════════════════════════════════════════════════════
# P3 #4 — Persistent DLQ
# ═══════════════════════════════════════════════════════════════════════════


class TestPersistentDLQ:
    def _make_envelope(self, topic="test.topic") -> Envelope:
        return Envelope(
            topic=topic,
            source_service_id="test",
            payload={"data": "test"},
        )

    @pytest.mark.asyncio
    async def test_initialize(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            dlq = PersistentDLQ(db_path)
            await dlq.initialize()
            count = await dlq.size()
            assert count == 0
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_append_and_list(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            dlq = PersistentDLQ(db_path)
            await dlq.initialize()
            env = self._make_envelope()
            await dlq.append(env, "handler_a", "boom", 3)
            entries = await dlq.list_entries()
            assert len(entries) == 1
            assert entries[0]["envelope_id"] == env.envelope_id
            assert entries[0]["error"] == "boom"
            assert entries[0]["attempts"] == 3
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_get_entry(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            dlq = PersistentDLQ(db_path)
            await dlq.initialize()
            env = self._make_envelope()
            await dlq.append(env, "handler_b", "error", 2)
            entry = await dlq.get_entry(env.envelope_id)
            assert entry is not None
            assert entry["handler_name"] == "handler_b"
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_get_entry_not_found(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            dlq = PersistentDLQ(db_path)
            await dlq.initialize()
            entry = await dlq.get_entry("nonexistent")
            assert entry is None
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_remove(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            dlq = PersistentDLQ(db_path)
            await dlq.initialize()
            env = self._make_envelope()
            await dlq.append(env, "handler", "err", 1)
            assert await dlq.size() == 1
            removed = await dlq.remove(env.envelope_id)
            assert removed is True
            assert await dlq.size() == 0
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_remove_not_found(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            dlq = PersistentDLQ(db_path)
            await dlq.initialize()
            removed = await dlq.remove("nonexistent")
            assert removed is False
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_get_envelope(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            dlq = PersistentDLQ(db_path)
            await dlq.initialize()
            env = self._make_envelope()
            await dlq.append(env, "handler", "err", 1)
            restored = await dlq.get_envelope(env.envelope_id)
            assert restored is not None
            assert restored.envelope_id == env.envelope_id
            assert restored.topic == env.topic
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_purge(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            dlq = PersistentDLQ(db_path)
            await dlq.initialize()
            for _ in range(5):
                await dlq.append(self._make_envelope(), "h", "e", 1)
            assert await dlq.size() == 5
            count = await dlq.purge()
            assert count == 5
            assert await dlq.size() == 0
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_multiple_entries_ordering(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            dlq = PersistentDLQ(db_path)
            await dlq.initialize()
            envs = [self._make_envelope() for _ in range(3)]
            for i, env in enumerate(envs):
                await dlq.append(env, f"handler_{i}", f"error_{i}", i + 1)
            entries = await dlq.list_entries()
            assert len(entries) == 3
            # Most recent first
            assert entries[0]["handler_name"] == "handler_2"
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_list_respects_limit(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            dlq = PersistentDLQ(db_path)
            await dlq.initialize()
            for _ in range(10):
                await dlq.append(self._make_envelope(), "h", "e", 1)
            entries = await dlq.list_entries(limit=3)
            assert len(entries) == 3
        finally:
            os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_removed_entries_hidden_from_list(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            dlq = PersistentDLQ(db_path)
            await dlq.initialize()
            env = self._make_envelope()
            await dlq.append(env, "h", "e", 1)
            await dlq.remove(env.envelope_id)
            entries = await dlq.list_entries()
            assert len(entries) == 0
        finally:
            os.unlink(db_path)


# ═══════════════════════════════════════════════════════════════════════════
# P3 #5 — CLI Admin Commands (config validate)
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigValidate:
    def test_valid_config(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[budget]\nmonthly_limit_usd = 100.0\nalert_at_pct = 0.9\n'
            '[runtime]\nlog_level = "DEBUG"\n'
        )
        from qe.config import load_config

        cfg = load_config(config_file)
        assert cfg.budget.monthly_limit_usd == 100.0
        assert cfg.runtime.log_level == "DEBUG"

    def test_invalid_config(self, tmp_path):
        from pydantic import ValidationError

        from qe.config import load_config

        config_file = tmp_path / "config.toml"
        config_file.write_text('[budget]\nmonthly_limit_usd = -5\n')
        with pytest.raises(ValidationError):
            load_config(config_file)

    def test_missing_config_uses_defaults(self, tmp_path):
        from qe.config import load_config

        cfg = load_config(tmp_path / "nonexistent.toml")
        assert cfg.budget.monthly_limit_usd == 50.0
        assert cfg.runtime.log_level == "INFO"


# ═══════════════════════════════════════════════════════════════════════════
# P3 #6 — Outbound Webhooks
# ═══════════════════════════════════════════════════════════════════════════


class TestWebhookTarget:
    def test_defaults(self):
        t = WebhookTarget(url="https://example.com/hook")
        assert t.enabled is True
        assert "goal.completed" in t.events
        assert t.max_retries == 3
        assert t.secret is None

    def test_custom_events(self):
        t = WebhookTarget(url="https://x.com", events=["custom.event"])
        assert t.events == ["custom.event"]


class TestWebhookNotifier:
    def test_add_and_list_targets(self):
        notifier = WebhookNotifier()
        notifier.add_target(WebhookTarget(url="https://a.com"))
        notifier.add_target(WebhookTarget(url="https://b.com"))
        targets = notifier.list_targets()
        assert len(targets) == 2
        assert targets[0]["url"] == "https://a.com"

    def test_remove_target(self):
        notifier = WebhookNotifier()
        notifier.add_target(WebhookTarget(url="https://a.com"))
        assert notifier.remove_target("https://a.com") is True
        assert len(notifier.list_targets()) == 0

    def test_remove_nonexistent(self):
        notifier = WebhookNotifier()
        assert notifier.remove_target("https://nope.com") is False

    @pytest.mark.asyncio
    async def test_notify_skips_disabled(self):
        notifier = WebhookNotifier()
        notifier.add_target(WebhookTarget(url="https://a.com", enabled=False))
        deliveries = await notifier.notify("goal.completed", {"goal_id": "g-1"})
        assert len(deliveries) == 0

    @pytest.mark.asyncio
    async def test_notify_skips_non_matching_events(self):
        notifier = WebhookNotifier()
        notifier.add_target(WebhookTarget(url="https://a.com", events=["goal.completed"]))
        deliveries = await notifier.notify("goal.failed", {"goal_id": "g-1"})
        assert len(deliveries) == 0

    @pytest.mark.asyncio
    async def test_notify_delivers_matching(self):
        notifier = WebhookNotifier()
        notifier.add_target(WebhookTarget(
            url="https://httpbin.org/post",
            events=["goal.completed"],
            max_retries=1,
        ))
        # Mock httpx to avoid real network calls
        mock_response = AsyncMock()
        mock_response.status_code = 200
        with patch("qe.runtime.webhooks.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            deliveries = await notifier.notify(
                "goal.completed",
                {"goal_id": "g-123"},
            )
            assert len(deliveries) == 1
            assert deliveries[0].success is True

    @pytest.mark.asyncio
    async def test_notify_with_hmac_signing(self):
        notifier = WebhookNotifier()
        notifier.add_target(WebhookTarget(
            url="https://example.com/hook",
            secret="my-secret-key",
            events=["goal.completed"],
            max_retries=1,
        ))
        mock_response = AsyncMock()
        mock_response.status_code = 200

        captured_headers = {}

        async def capture_post(url, content, headers):
            captured_headers.update(headers)
            return mock_response

        with patch("qe.runtime.webhooks.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = capture_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await notifier.notify("goal.completed", {"goal_id": "g-1"})
            assert "X-QE-Signature" in captured_headers
            assert captured_headers["X-QE-Signature"].startswith("sha256=")

    @pytest.mark.asyncio
    async def test_notify_retries_on_failure(self):
        notifier = WebhookNotifier()
        notifier.add_target(WebhookTarget(
            url="https://example.com/hook",
            events=["goal.completed"],
            max_retries=3,
        ))

        call_count = 0

        async def failing_post(url, content, headers):
            nonlocal call_count
            call_count += 1
            resp = AsyncMock()
            resp.status_code = 500
            return resp

        with patch("qe.runtime.webhooks.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = failing_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            deliveries = await notifier.notify("goal.completed", {"goal_id": "g-1"})
            assert len(deliveries) == 1
            assert deliveries[0].success is False
            assert deliveries[0].attempts == 3
            assert call_count == 3

    def test_delivery_history(self):
        notifier = WebhookNotifier()
        # Manually add deliveries
        notifier._history.append(WebhookDelivery(
            target_url="https://a.com",
            event_type="goal.completed",
            payload={"x": 1},
            success=True,
        ))
        history = notifier.delivery_history()
        assert len(history) == 1
        assert history[0]["success"] is True

    def test_stats(self):
        notifier = WebhookNotifier()
        notifier.add_target(WebhookTarget(url="https://a.com"))
        notifier.add_target(WebhookTarget(url="https://b.com", enabled=False))
        s = notifier.stats()
        assert s["targets"] == 2
        assert s["enabled_targets"] == 1
        assert s["total_deliveries"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# P3 #7 — Feature Flags
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureFlagDef:
    def test_defaults(self):
        f = FeatureFlagDef(name="test_flag")
        assert f.enabled is False
        assert f.rollout_pct == 100.0
        assert f.targeting == {}

    def test_to_dict(self):
        f = FeatureFlagDef(name="test", enabled=True, description="A test flag")
        d = f.to_dict()
        assert d["name"] == "test"
        assert d["enabled"] is True
        assert d["description"] == "A test flag"


class TestFeatureFlagStore:
    def test_define_flag(self):
        store = FeatureFlagStore()
        flag = store.define("new_feature", enabled=True, description="Test")
        assert flag.name == "new_feature"
        assert flag.enabled is True

    def test_is_enabled_undefined_returns_default(self):
        store = FeatureFlagStore()
        assert store.is_enabled("nonexistent") is False
        assert store.is_enabled("nonexistent", default=True) is True

    def test_is_enabled_disabled_flag(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=False)
        assert store.is_enabled("flag_a") is False

    def test_is_enabled_enabled_flag(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True)
        assert store.is_enabled("flag_a") is True

    def test_enable_disable(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=False)
        store.enable("flag_a")
        assert store.is_enabled("flag_a") is True
        store.disable("flag_a")
        assert store.is_enabled("flag_a") is False

    def test_enable_nonexistent(self):
        store = FeatureFlagStore()
        assert store.enable("nope") is False

    def test_delete_flag(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True)
        assert store.delete("flag_a") is True
        assert store.get("flag_a") is None

    def test_delete_nonexistent(self):
        store = FeatureFlagStore()
        assert store.delete("nope") is False

    def test_targeting_match(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True)
        store.add_targeting("flag_a", "user_id", ["alice", "bob"])
        assert store.is_enabled("flag_a", {"user_id": "alice"}) is True
        assert store.is_enabled("flag_a", {"user_id": "charlie"}) is False

    def test_targeting_no_context_denies(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True)
        store.add_targeting("flag_a", "user_id", ["alice"])
        assert store.is_enabled("flag_a") is False

    def test_targeting_nonexistent_flag(self):
        store = FeatureFlagStore()
        assert store.add_targeting("nope", "key", ["val"]) is False

    def test_clear_targeting(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True)
        store.add_targeting("flag_a", "user_id", ["alice"])
        store.clear_targeting("flag_a")
        # Without targeting, flag should be enabled for anyone
        assert store.is_enabled("flag_a") is True

    def test_rollout_percentage(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True, rollout_pct=50)
        # Deterministic: with different user_ids, some should pass, some fail
        results = set()
        for i in range(100):
            results.add(store.is_enabled("flag_a", {"user_id": f"user_{i}"}))
        # With 50% rollout, we should have both True and False
        assert True in results
        assert False in results

    def test_rollout_zero_percent(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True, rollout_pct=0)
        # 0% rollout = nobody gets it
        for i in range(20):
            assert store.is_enabled("flag_a", {"user_id": f"user_{i}"}) is False

    def test_rollout_100_percent(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True, rollout_pct=100)
        for i in range(20):
            assert store.is_enabled("flag_a", {"user_id": f"user_{i}"}) is True

    def test_list_flags(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True)
        store.define("flag_b", enabled=False)
        flags = store.list_flags()
        assert len(flags) == 2
        names = {f["name"] for f in flags}
        assert names == {"flag_a", "flag_b"}

    def test_evaluation_log(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True)
        store.is_enabled("flag_a")
        log_entries = store.evaluation_log()
        assert len(log_entries) == 1
        assert log_entries[0]["flag"] == "flag_a"
        assert log_entries[0]["result"] is True

    def test_stats(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True)
        store.define("flag_b", enabled=False)
        store.define("flag_c", enabled=True)
        s = store.stats()
        assert s["total_flags"] == 3
        assert s["enabled_flags"] == 2
        assert s["disabled_flags"] == 1

    def test_update_existing_flag(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=False, description="first")
        store.define("flag_a", enabled=True, description="second")
        flag = store.get("flag_a")
        assert flag.enabled is True
        assert flag.description == "second"

    def test_get_flag(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True)
        flag = store.get("flag_a")
        assert flag is not None
        assert flag.name == "flag_a"

    def test_get_nonexistent(self):
        store = FeatureFlagStore()
        assert store.get("nope") is None

    def test_multiple_targeting_keys(self):
        store = FeatureFlagStore()
        store.define("flag_a", enabled=True)
        store.add_targeting("flag_a", "user_id", ["alice"])
        store.add_targeting("flag_a", "goal_id", ["g-42"])
        # Match on user_id
        assert store.is_enabled("flag_a", {"user_id": "alice"}) is True
        # Match on goal_id
        assert store.is_enabled("flag_a", {"goal_id": "g-42"}) is True
        # No match on either
        assert store.is_enabled("flag_a", {"user_id": "charlie", "goal_id": "g-99"}) is False

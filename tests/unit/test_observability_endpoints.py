"""Tests for Operational Observability endpoints — pool, strategy, flags, episodic."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from qe.runtime.episodic_memory import Episode, EpisodicMemory
from qe.runtime.feature_flags import get_flag_store, reset_flag_store

# ── Helpers ──────────────────────────────────────────────────────────────


@contextmanager
def skip_lifespan():
    """Patch out lifespan initialization for endpoint-only testing."""
    with (
        patch("qe.api.app.is_setup_complete", return_value=False),
        patch("qe.api.app.configure_from_config"),
    ):
        yield


def _make_client():
    """Return an AsyncClient against the app with lifespan skipped."""
    from qe.api.app import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


def _mock_pool():
    """Return a mock cognitive pool with pool_status."""
    pool = MagicMock()
    pool.pool_status.return_value = {
        "total_agents": 2,
        "active_agents": 1,
        "max_agents": 5,
        "agents": [
            {
                "agent_id": "agent_1",
                "specialization": "breadth_first",
                "status": "active",
                "strategy": "breadth_first",
                "model_tier": "balanced",
            },
            {
                "agent_id": "agent_2",
                "specialization": "depth_first",
                "status": "idle",
                "strategy": "depth_first",
                "model_tier": "fast",
            },
        ],
    }
    return pool


def _mock_evolver():
    """Return a mock strategy evolver with _current_strategy and get_snapshots."""
    evolver = MagicMock()
    evolver._current_strategy = "breadth_first"
    snapshot = MagicMock()
    snapshot.model_dump.return_value = {
        "strategy_name": "breadth_first",
        "alpha": 2.0,
        "beta": 1.0,
        "avg_cost": 0.05,
        "avg_duration": 3.2,
        "sample_count": 10,
    }
    evolver.get_snapshots.return_value = [snapshot]
    return evolver


def _mock_scaler():
    """Return a mock elastic scaler with current_profile_name."""
    scaler = MagicMock()
    scaler.current_profile_name.return_value = "balanced"
    return scaler


# ── Pool Endpoints ───────────────────────────────────────────────────────


class TestPoolEndpoints:
    @pytest.mark.asyncio
    async def test_returns_pool_dict(self):
        """GET /api/pool/status returns pool_status dict."""
        import qe.api.app as app_module

        old = app_module._cognitive_pool
        try:
            app_module._cognitive_pool = _mock_pool()
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/pool/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_agents"] == 2
            assert data["max_agents"] == 5
        finally:
            app_module._cognitive_pool = old

    @pytest.mark.asyncio
    async def test_503_when_uninitialized(self):
        """GET /api/pool/status returns 503 when pool is None."""
        import qe.api.app as app_module

        old = app_module._cognitive_pool
        try:
            app_module._cognitive_pool = None
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/pool/status")
            assert resp.status_code == 503
        finally:
            app_module._cognitive_pool = old

    @pytest.mark.asyncio
    async def test_agents_list_shape(self):
        """Agents list has expected keys."""
        import qe.api.app as app_module

        old = app_module._cognitive_pool
        try:
            app_module._cognitive_pool = _mock_pool()
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/pool/status")
            agents = resp.json()["agents"]
            assert len(agents) == 2
            for a in agents:
                assert "agent_id" in a
                assert "specialization" in a
                assert "status" in a
                assert "strategy" in a
                assert "model_tier" in a
        finally:
            app_module._cognitive_pool = old

    @pytest.mark.asyncio
    async def test_counts_match(self):
        """total_agents and active_agents match the mock data."""
        import qe.api.app as app_module

        old = app_module._cognitive_pool
        try:
            app_module._cognitive_pool = _mock_pool()
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/pool/status")
            data = resp.json()
            assert data["total_agents"] == 2
            assert data["active_agents"] == 1
        finally:
            app_module._cognitive_pool = old


# ── Strategy Endpoints ───────────────────────────────────────────────────


class TestStrategyEndpoints:
    @pytest.mark.asyncio
    async def test_returns_snapshots_list(self):
        """GET /api/strategy/snapshots returns snapshots list."""
        import qe.api.app as app_module

        old_e, old_s = app_module._strategy_evolver, app_module._elastic_scaler
        try:
            app_module._strategy_evolver = _mock_evolver()
            app_module._elastic_scaler = _mock_scaler()
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/strategy/snapshots")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data["snapshots"], list)
            assert len(data["snapshots"]) == 1
        finally:
            app_module._strategy_evolver = old_e
            app_module._elastic_scaler = old_s

    @pytest.mark.asyncio
    async def test_503_when_uninitialized(self):
        """GET /api/strategy/snapshots returns 503 when evolver is None."""
        import qe.api.app as app_module

        old = app_module._strategy_evolver
        try:
            app_module._strategy_evolver = None
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/strategy/snapshots")
            assert resp.status_code == 503
        finally:
            app_module._strategy_evolver = old

    @pytest.mark.asyncio
    async def test_includes_current_strategy(self):
        """Response includes current_strategy field."""
        import qe.api.app as app_module

        old_e, old_s = app_module._strategy_evolver, app_module._elastic_scaler
        try:
            app_module._strategy_evolver = _mock_evolver()
            app_module._elastic_scaler = _mock_scaler()
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/strategy/snapshots")
            assert resp.json()["current_strategy"] == "breadth_first"
        finally:
            app_module._strategy_evolver = old_e
            app_module._elastic_scaler = old_s

    @pytest.mark.asyncio
    async def test_includes_scaling_profile(self):
        """Response includes scaling_profile field."""
        import qe.api.app as app_module

        old_e, old_s = app_module._strategy_evolver, app_module._elastic_scaler
        try:
            app_module._strategy_evolver = _mock_evolver()
            app_module._elastic_scaler = _mock_scaler()
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/strategy/snapshots")
            assert resp.json()["scaling_profile"] == "balanced"
        finally:
            app_module._strategy_evolver = old_e
            app_module._elastic_scaler = old_s


# ── Flag Endpoints ───────────────────────────────────────────────────────


class TestFlagEndpoints:
    @pytest.fixture(autouse=True)
    def _setup_flags(self):
        """Reset flag store and define test flags."""
        reset_flag_store()
        store = get_flag_store()
        store.define("test_flag_a", enabled=True, description="A test flag")
        store.define("test_flag_b", enabled=False, description="Another flag")
        yield
        reset_flag_store()

    @pytest.mark.asyncio
    async def test_list_returns_all_flags(self):
        """GET /api/flags returns all defined flags."""
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.get("/api/flags")
        assert resp.status_code == 200
        data = resp.json()
        names = {f["name"] for f in data["flags"]}
        assert "test_flag_a" in names
        assert "test_flag_b" in names

    @pytest.mark.asyncio
    async def test_list_includes_stats(self):
        """GET /api/flags includes stats section."""
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.get("/api/flags")
        data = resp.json()
        assert "stats" in data
        assert "total_flags" in data["stats"]
        assert data["stats"]["total_flags"] >= 2

    @pytest.mark.asyncio
    async def test_get_existing_flag(self):
        """GET /api/flags/{flag_name} returns flag dict."""
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.get("/api/flags/test_flag_a")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test_flag_a"
        assert data["enabled"] is True

    @pytest.mark.asyncio
    async def test_get_nonexistent_404(self):
        """GET /api/flags/{flag_name} returns 404 for unknown flag."""
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.get("/api/flags/nonexistent_flag")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_enable_success(self):
        """POST /api/flags/{flag_name}/enable returns success."""
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.post("/api/flags/test_flag_b/enable")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "enabled"
        assert data["flag_name"] == "test_flag_b"
        assert get_flag_store().get("test_flag_b").enabled is True

    @pytest.mark.asyncio
    async def test_enable_nonexistent_404(self):
        """POST /api/flags/{flag_name}/enable returns 404 for unknown flag."""
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.post("/api/flags/nonexistent/enable")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_disable_success(self):
        """POST /api/flags/{flag_name}/disable returns success."""
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.post("/api/flags/test_flag_a/disable")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "disabled"
        assert data["flag_name"] == "test_flag_a"
        assert get_flag_store().get("test_flag_a").enabled is False

    @pytest.mark.asyncio
    async def test_disable_nonexistent_404(self):
        """POST /api/flags/{flag_name}/disable returns 404 for unknown flag."""
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.post("/api/flags/nonexistent/disable")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_enable_records_audit(self):
        """POST /api/flags/{flag_name}/enable records an audit entry."""
        mock_log = MagicMock()
        with (
            patch("qe.api.app.is_setup_complete", return_value=False),
            patch("qe.api.app.configure_from_config"),
            patch("qe.api.app.get_audit_log", return_value=mock_log),
        ):
            async with _make_client() as client:
                await client.post("/api/flags/test_flag_a/enable")
            mock_log.record.assert_called_once_with(
                "flag.enabled", resource="flag/test_flag_a"
            )

    @pytest.mark.asyncio
    async def test_evaluations_returns_log(self):
        """GET /api/flags/evaluations returns evaluation log."""
        store = get_flag_store()
        store.is_enabled("test_flag_a")
        store.is_enabled("test_flag_b")

        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.get("/api/flags/evaluations")
        assert resp.status_code == 200
        data = resp.json()
        assert "evaluations" in data
        assert "count" in data
        assert data["count"] >= 2


# ── Episodic Endpoints ───────────────────────────────────────────────────


class TestEpisodicEndpoints:
    @pytest.mark.asyncio
    async def test_status_returns_counts(self):
        """GET /api/episodic/status returns hot/warm counts."""
        import qe.api.app as app_module

        old = app_module._episodic_memory
        try:
            mem = EpisodicMemory(max_hot_entries=100)
            app_module._episodic_memory = mem
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/episodic/status")
            assert resp.status_code == 200
            data = resp.json()
            assert "hot_entries" in data
            assert "max_hot" in data
            assert "warm_entries" in data
            assert data["hot_entries"] == 0
            assert data["max_hot"] == 100
        finally:
            app_module._episodic_memory = old

    @pytest.mark.asyncio
    async def test_503_when_uninitialized(self):
        """GET /api/episodic/status returns 503 when memory is None."""
        import qe.api.app as app_module

        old = app_module._episodic_memory
        try:
            app_module._episodic_memory = None
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/episodic/status")
            assert resp.status_code == 503
        finally:
            app_module._episodic_memory = old

    @pytest.mark.asyncio
    async def test_search_returns_episodes(self):
        """GET /api/episodic/search returns matching episodes."""
        import qe.api.app as app_module

        old = app_module._episodic_memory
        try:
            mem = EpisodicMemory(max_hot_entries=100)
            await mem.store(Episode(
                episode_type="observation",
                summary="bitcoin price crash analysis",
                goal_id="g1",
            ))
            app_module._episodic_memory = mem
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/episodic/search", params={"query": "bitcoin"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] >= 1
            assert "episodes" in data
        finally:
            app_module._episodic_memory = old

    @pytest.mark.asyncio
    async def test_search_requires_query(self):
        """GET /api/episodic/search returns 400 if query is empty."""
        import qe.api.app as app_module

        old = app_module._episodic_memory
        try:
            app_module._episodic_memory = EpisodicMemory(max_hot_entries=100)
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/episodic/search", params={"query": ""})
            assert resp.status_code == 400
        finally:
            app_module._episodic_memory = old

    @pytest.mark.asyncio
    async def test_search_passes_filters(self):
        """GET /api/episodic/search passes goal_id and episode_type filters."""
        import qe.api.app as app_module

        old = app_module._episodic_memory
        try:
            mem = EpisodicMemory(max_hot_entries=100)
            await mem.store(Episode(
                episode_type="observation",
                summary="bitcoin observation",
                goal_id="g1",
            ))
            await mem.store(Episode(
                episode_type="synthesis",
                summary="bitcoin synthesis",
                goal_id="g2",
            ))
            app_module._episodic_memory = mem
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get(
                        "/api/episodic/search",
                        params={
                            "query": "bitcoin",
                            "goal_id": "g1",
                            "episode_type": "observation",
                        },
                    )
            data = resp.json()
            assert data["count"] == 1
            assert data["episodes"][0]["goal_id"] == "g1"
        finally:
            app_module._episodic_memory = old

    @pytest.mark.asyncio
    async def test_goal_returns_episodes(self):
        """GET /api/episodic/goal/{goal_id} returns episodes for the goal."""
        import qe.api.app as app_module

        old = app_module._episodic_memory
        try:
            mem = EpisodicMemory(max_hot_entries=100)
            await mem.store(Episode(
                episode_type="observation",
                summary="test",
                goal_id="g42",
            ))
            app_module._episodic_memory = mem
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/episodic/goal/g42")
            assert resp.status_code == 200
            data = resp.json()
            assert data["goal_id"] == "g42"
            assert data["count"] == 1
        finally:
            app_module._episodic_memory = old

    @pytest.mark.asyncio
    async def test_latest_returns_recent(self):
        """GET /api/episodic/latest returns recent episodes."""
        import qe.api.app as app_module

        old = app_module._episodic_memory
        try:
            mem = EpisodicMemory(max_hot_entries=100)
            for i in range(5):
                await mem.store(Episode(
                    episode_type="observation",
                    summary=f"ep_{i}",
                ))
            app_module._episodic_memory = mem
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/episodic/latest", params={"limit": 3})
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] == 3
        finally:
            app_module._episodic_memory = old

    @pytest.mark.asyncio
    async def test_latest_default_limit(self):
        """GET /api/episodic/latest uses default limit of 20."""
        import qe.api.app as app_module

        old = app_module._episodic_memory
        try:
            mem = EpisodicMemory(max_hot_entries=100)
            for i in range(25):
                await mem.store(Episode(
                    episode_type="observation",
                    summary=f"ep_{i}",
                ))
            app_module._episodic_memory = mem
            with skip_lifespan():
                async with _make_client() as client:
                    resp = await client.get("/api/episodic/latest")
            data = resp.json()
            assert data["count"] == 20
        finally:
            app_module._episodic_memory = old


# ── Status Enrichment ────────────────────────────────────────────────────


class TestStatusEnrichment:
    """Test that /api/status includes the new pool/strategy/flags/loop sections."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import qe.api.app as app_module

        self._app = app_module
        self._old_pool = app_module._cognitive_pool
        self._old_evolver = app_module._strategy_evolver
        self._old_scaler = app_module._elastic_scaler
        self._old_bridge = app_module._inquiry_bridge
        self._old_kl = app_module._knowledge_loop
        self._old_supervisor = app_module._supervisor

        # Set up mocks
        app_module._cognitive_pool = _mock_pool()
        app_module._strategy_evolver = _mock_evolver()
        app_module._elastic_scaler = _mock_scaler()

        bridge = MagicMock()
        bridge.status.return_value = {"running": True, "events_processed": 5}
        app_module._inquiry_bridge = bridge

        kl = MagicMock()
        kl.status.return_value = {"running": True, "cycles": 3}
        app_module._knowledge_loop = kl

        # Mock supervisor
        supervisor = MagicMock()
        supervisor.registry.all_services.return_value = []
        supervisor.budget_tracker.total_spend.return_value = 1.23
        supervisor.budget_tracker.remaining_pct.return_value = 0.87
        supervisor.budget_tracker.monthly_limit_usd = 100.0
        supervisor.budget_tracker.spend_by_model.return_value = {}
        supervisor._circuits = {}
        app_module._supervisor = supervisor

        yield

        app_module._cognitive_pool = self._old_pool
        app_module._strategy_evolver = self._old_evolver
        app_module._elastic_scaler = self._old_scaler
        app_module._inquiry_bridge = self._old_bridge
        app_module._knowledge_loop = self._old_kl
        app_module._supervisor = self._old_supervisor

    @pytest.mark.asyncio
    async def test_includes_pool_section(self):
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.get("/api/status")
        data = resp.json()
        assert "pool" in data
        assert data["pool"]["total_agents"] == 2

    @pytest.mark.asyncio
    async def test_includes_strategy_section(self):
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.get("/api/status")
        data = resp.json()
        assert "strategy" in data
        assert data["strategy"]["current_strategy"] == "breadth_first"
        assert data["strategy"]["scaling_profile"] == "balanced"
        assert len(data["strategy"]["snapshots"]) == 1

    @pytest.mark.asyncio
    async def test_includes_flags_section(self):
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.get("/api/status")
        data = resp.json()
        assert "flags" in data
        assert "total_flags" in data["flags"]

    @pytest.mark.asyncio
    async def test_includes_bridge_and_knowledge_loop(self):
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.get("/api/status")
        data = resp.json()
        assert "bridge" in data
        assert data["bridge"]["running"] is True
        assert "knowledge_loop" in data
        assert data["knowledge_loop"]["running"] is True


# ── get_latest() unit tests ──────────────────────────────────────────────


class TestGetLatest:
    @pytest.mark.asyncio
    async def test_returns_most_recent(self):
        """get_latest returns the most recent episodes."""
        mem = EpisodicMemory(max_hot_entries=100)
        for i in range(10):
            await mem.store(Episode(episode_type="observation", summary=f"ep_{i}"))
        result = mem.get_latest(limit=5)
        assert len(result) == 5
        assert result[0].summary == "ep_9"
        assert result[4].summary == "ep_5"

    @pytest.mark.asyncio
    async def test_empty_store_returns_empty(self):
        """get_latest on empty store returns []."""
        mem = EpisodicMemory(max_hot_entries=100)
        assert mem.get_latest() == []

    @pytest.mark.asyncio
    async def test_fewer_than_limit(self):
        """get_latest when fewer items than limit returns all."""
        mem = EpisodicMemory(max_hot_entries=100)
        for i in range(3):
            await mem.store(Episode(episode_type="observation", summary=f"ep_{i}"))
        result = mem.get_latest(limit=20)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_order_is_most_recent_first(self):
        """get_latest returns episodes in reverse insertion order (most recent first)."""
        mem = EpisodicMemory(max_hot_entries=100)
        await mem.store(Episode(episode_type="observation", summary="first"))
        await mem.store(Episode(episode_type="observation", summary="second"))
        await mem.store(Episode(episode_type="observation", summary="third"))
        result = mem.get_latest(limit=3)
        assert result[0].summary == "third"
        assert result[1].summary == "second"
        assert result[2].summary == "first"


# ── Global Promotion ─────────────────────────────────────────────────────


class TestGlobalPromotion:
    def test_elastic_scaler_global_declared(self):
        """Module-level _elastic_scaler exists."""
        import qe.api.app as app_module

        assert hasattr(app_module, "_elastic_scaler")

    def test_episodic_memory_global_declared(self):
        """Module-level _episodic_memory exists."""
        import qe.api.app as app_module

        assert hasattr(app_module, "_episodic_memory")

    def test_both_none_before_lifespan(self):
        """Both globals are None when lifespan hasn't run."""
        import qe.api.app as app_module

        assert hasattr(app_module, "_elastic_scaler")
        assert hasattr(app_module, "_episodic_memory")


# ── Flag Integration ─────────────────────────────────────────────────────


class TestFlagIntegration:
    @pytest.fixture(autouse=True)
    def _setup_flags(self):
        reset_flag_store()
        store = get_flag_store()
        store.define("inquiry_mode", enabled=False, description="Inquiry mode")
        store.define("multi_agent_mode", enabled=False, description="Multi-agent")
        store.define("prompt_evolution", enabled=False, description="Prompt evolution")
        store.define("knowledge_consolidation", enabled=False, description="Knowledge loop")
        yield
        reset_flag_store()

    @pytest.mark.asyncio
    async def test_enable_changes_is_enabled(self):
        """Enabling a flag via API actually changes is_enabled result."""
        store = get_flag_store()
        assert not store.is_enabled("inquiry_mode")
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.post("/api/flags/inquiry_mode/enable")
        assert resp.status_code == 200
        assert store.is_enabled("inquiry_mode")

    @pytest.mark.asyncio
    async def test_disable_changes_is_enabled(self):
        """Disabling a flag via API actually changes is_enabled result."""
        store = get_flag_store()
        store.enable("inquiry_mode")
        assert store.is_enabled("inquiry_mode")
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.post("/api/flags/inquiry_mode/disable")
        assert resp.status_code == 200
        assert not store.is_enabled("inquiry_mode")

    @pytest.mark.asyncio
    async def test_singleton_store_consistency(self):
        """Flag changes via API are visible through get_flag_store()."""
        with skip_lifespan():
            async with _make_client() as client:
                await client.post("/api/flags/multi_agent_mode/enable")
        assert get_flag_store().is_enabled("multi_agent_mode")

    @pytest.mark.asyncio
    async def test_all_4_defined_flags_visible(self):
        """All 4 standard flags are visible in GET /api/flags."""
        with skip_lifespan():
            async with _make_client() as client:
                resp = await client.get("/api/flags")
        names = {f["name"] for f in resp.json()["flags"]}
        assert "inquiry_mode" in names
        assert "multi_agent_mode" in names
        assert "prompt_evolution" in names
        assert "knowledge_consolidation" in names

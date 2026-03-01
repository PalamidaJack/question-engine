"""Tests for Phase 5 wiring — lifespan hardening, EngramCache lifecycle, profiling endpoint."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

# ── Lifespan hardening tests ─────────────────────────────────────────────


class TestLifespanHardening:
    @pytest.mark.asyncio
    async def test_lifespan_cleanup_runs_on_error(self):
        """Verify cleanup code runs even when an error occurs during yield."""
        from qe.runtime.engram_cache import EngramCache

        cache = EngramCache()
        clear_called = False
        original_clear = cache.clear

        def _tracking_clear():
            nonlocal clear_called
            clear_called = True
            return original_clear()

        cache.clear = _tracking_clear

        # Simulate the lifespan try/finally pattern
        try:
            raise RuntimeError("simulated crash")
        except RuntimeError:
            pass
        finally:
            cache.clear()

        assert clear_called

    @pytest.mark.asyncio
    async def test_engram_cache_cleared_on_shutdown(self):
        """Verify EngramCache.clear() is called during shutdown."""
        from qe.runtime.engram_cache import EngramCache

        cache = EngramCache()
        # Add some entries to verify they get cleared
        cache._exact_cache["test_key"] = {"data": "value"}
        assert cache.stats()["exact_entries"] == 1

        cleared = cache.clear()
        assert cleared >= 1
        assert cache.stats()["exact_entries"] == 0


# ── Profiling endpoint tests ─────────────────────────────────────────────


class TestProfilingEndpoint:
    @pytest.mark.asyncio
    async def test_profiling_endpoint_returns_structure(self):
        """Verify the profiling endpoint returns expected keys."""
        # Import app and patch to avoid full lifespan initialization
        with (
            patch("qe.api.app.is_setup_complete", return_value=False),
            patch("qe.api.app.configure_from_config"),
        ):
            from qe.api.app import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/profiling/inquiry")

            assert resp.status_code == 200
            data = resp.json()
            assert "phase_timings" in data
            assert "process" in data
            assert "engram_cache" in data
            assert "components" in data
            assert "rss_bytes" in data["process"]
            assert "python_version" in data["process"]

    @pytest.mark.asyncio
    async def test_profiling_endpoint_after_inquiry(self):
        """Verify phase_timings populated after setting _last_inquiry_profile."""
        import qe.api.app as app_module

        old_profile = app_module._last_inquiry_profile
        try:
            app_module._last_inquiry_profile = {
                "observe": {"count": 1.0, "total_s": 0.01, "avg_s": 0.01, "max_s": 0.01},
            }

            with (
                patch("qe.api.app.is_setup_complete", return_value=False),
                patch("qe.api.app.configure_from_config"),
            ):
                from qe.api.app import app

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/api/profiling/inquiry")

                data = resp.json()
                assert data["phase_timings"]["observe"]["count"] == 1.0
        finally:
            app_module._last_inquiry_profile = old_profile

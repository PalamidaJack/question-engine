"""Tests for InquiryProfilingStore — ring buffer, percentiles, endpoint."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from qe.api.profiling import InquiryProfilingStore


class TestProfilingStore:
    def test_profiling_store_record_and_last(self):
        """Record an entry and verify .last returns it."""
        store = InquiryProfilingStore(max_entries=10)
        assert store.last is None
        assert store.count == 0

        timings = {"observe": {"count": 1.0, "total_s": 0.01, "avg_s": 0.01, "max_s": 0.01}}
        store.record(timings, duration_s=0.5)

        assert store.count == 1
        assert store.last == timings

        # Second recording replaces last
        timings2 = {"orient": {"count": 1.0, "total_s": 0.02, "avg_s": 0.02, "max_s": 0.02}}
        store.record(timings2, duration_s=1.0)
        assert store.count == 2
        assert store.last == timings2

    def test_profiling_store_ring_buffer(self):
        """60 entries with max=50 should keep only 50."""
        store = InquiryProfilingStore(max_entries=50)

        for i in range(60):
            store.record(
                {"observe": {
                    "count": 1.0, "total_s": float(i),
                    "avg_s": float(i), "max_s": float(i),
                }},
                duration_s=float(i),
            )

        assert store.count == 50
        # Last entry should be i=59
        assert store.last["observe"]["avg_s"] == 59.0

    def test_profiling_store_percentiles(self):
        """Known values, verify p50/p95/p99."""
        store = InquiryProfilingStore(max_entries=200)

        # Insert 100 entries with avg_s = 1..100
        for i in range(1, 101):
            store.record(
                {"observe": {
                    "count": 1.0, "total_s": float(i),
                    "avg_s": float(i), "max_s": float(i),
                }},
                duration_s=float(i),
            )

        pcts = store.percentiles()
        assert "observe" in pcts
        obs = pcts["observe"]

        # p50 should be around 50
        assert 45.0 <= obs["p50"] <= 55.0
        # p95 should be around 95
        assert 90.0 <= obs["p95"] <= 100.0
        # p99 should be around 99
        assert 95.0 <= obs["p99"] <= 100.0

    @pytest.mark.asyncio
    async def test_profiling_endpoint_includes_history(self):
        """ASGI client test: endpoint returns history fields."""
        import qe.api.app as app_module

        old_store = app_module._inquiry_profiling_store
        old_profile = app_module._last_inquiry_profile

        try:
            test_store = InquiryProfilingStore(max_entries=10)
            test_store.record(
                {"observe": {"count": 1.0, "total_s": 0.01, "avg_s": 0.01, "max_s": 0.01}},
                duration_s=0.5,
            )
            app_module._inquiry_profiling_store = test_store
            app_module._last_inquiry_profile = {}

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
                assert "last_inquiry" in data
                assert "history_count" in data
                assert data["history_count"] == 1
                assert "percentiles" in data
                assert "observe" in data["percentiles"]
        finally:
            app_module._inquiry_profiling_store = old_store
            app_module._last_inquiry_profile = old_profile

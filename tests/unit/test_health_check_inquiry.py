"""Tests for health check inquiry tracking."""

from __future__ import annotations

import time

from qe.runtime.readiness import ReadinessState


class TestHealthCheckInquiry:
    def test_readiness_includes_inquiry_section(self):
        """to_dict() includes 'inquiry' section."""
        state = ReadinessState()
        d = state.to_dict()

        assert "inquiry" in d
        inquiry = d["inquiry"]
        assert "engine_ready" in inquiry
        assert "healthy" in inquiry
        assert "last_status" in inquiry
        assert "age_seconds" in inquiry
        assert "duration_s" in inquiry

    def test_inquiry_engine_ready_tracked(self):
        """inquiry_engine_ready starts False and can be set True."""
        state = ReadinessState()
        assert state.inquiry_engine_ready is False

        state.mark_ready("inquiry_engine_ready")
        assert state.inquiry_engine_ready is True

        d = state.to_dict()
        assert d["inquiry"]["engine_ready"] is True

    def test_inquiry_healthy_after_success(self):
        """inquiry_healthy returns True after a successful run."""
        state = ReadinessState()
        state.inquiry_engine_ready = True
        state.last_inquiry_status = "completed"
        state.last_inquiry_at = time.monotonic()
        state.last_inquiry_duration_s = 1.5

        assert state.inquiry_healthy is True
        d = state.to_dict()
        assert d["inquiry"]["healthy"] is True

    def test_inquiry_unhealthy_after_recent_failure(self):
        """inquiry_healthy returns False after a recent failure."""
        state = ReadinessState()
        state.inquiry_engine_ready = True
        state.last_inquiry_status = "failed"
        state.last_inquiry_at = time.monotonic()  # just now

        assert state.inquiry_healthy is False
        d = state.to_dict()
        assert d["inquiry"]["healthy"] is False

    def test_inquiry_healthy_after_old_failure(self):
        """inquiry_healthy returns True if failure was >300s ago."""
        state = ReadinessState()
        state.inquiry_engine_ready = True
        state.last_inquiry_status = "failed"
        state.last_inquiry_at = time.monotonic() - 400.0  # 400s ago

        assert state.inquiry_healthy is True
        d = state.to_dict()
        assert d["inquiry"]["healthy"] is True

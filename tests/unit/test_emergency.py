"""Tests for the EmergencyStop kill switch."""

from unittest.mock import MagicMock

import pytest

from qe.runtime.emergency import EmergencyStop


class TestEmergencyStop:
    def test_initially_inactive(self):
        es = EmergencyStop()
        assert es.is_active is False

    def test_activate(self):
        es = EmergencyStop()
        es.activate("test reason")
        assert es.is_active is True
        status = es.status()
        assert status["active"] is True
        assert status["reason"] == "test reason"
        assert status["activated_at"] is not None

    def test_deactivate(self):
        es = EmergencyStop()
        es.activate("test")
        es.deactivate()
        assert es.is_active is False
        status = es.status()
        assert status["reason"] == ""

    def test_double_activate_is_idempotent(self):
        es = EmergencyStop()
        es.activate("first")
        es.activate("second")
        assert es.status()["reason"] == "first"

    def test_double_deactivate_is_idempotent(self):
        es = EmergencyStop()
        es.deactivate()  # no-op
        assert es.is_active is False

    def test_check_or_raise_when_active(self):
        es = EmergencyStop()
        es.activate("halt")
        with pytest.raises(RuntimeError, match="halt"):
            es.check_or_raise()

    def test_check_or_raise_when_inactive(self):
        es = EmergencyStop()
        es.check_or_raise()  # should not raise

    def test_disables_flag_store_on_activate(self):
        mock_flags = MagicMock()
        es = EmergencyStop()
        es.configure(flag_store=mock_flags)
        es.activate("stop")
        mock_flags.disable.assert_called_once_with("llm_calls")

    def test_enables_flag_store_on_deactivate(self):
        mock_flags = MagicMock()
        es = EmergencyStop()
        es.configure(flag_store=mock_flags)
        es.activate("stop")
        es.deactivate()
        mock_flags.enable.assert_called_once_with("llm_calls")

    def test_status_fields(self):
        es = EmergencyStop()
        status = es.status()
        assert "active" in status
        assert "reason" in status
        assert "activated_at" in status

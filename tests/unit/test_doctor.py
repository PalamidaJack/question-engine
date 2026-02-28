"""Tests for the Doctor health monitoring service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.services.doctor.service import CheckStatus, DoctorService


class TestDoctorChecks:
    @pytest.fixture
    def mock_bus(self):
        bus = MagicMock()
        bus.publish = MagicMock()
        return bus

    @pytest.fixture
    def mock_substrate(self):
        sub = AsyncMock()
        sub.count_claims = AsyncMock(return_value=42)
        sub.embeddings = AsyncMock()
        sub.embeddings.count = AsyncMock(return_value=42)
        sub.embeddings.search = AsyncMock(return_value=[])
        return sub

    @pytest.fixture
    def mock_event_log(self):
        log = AsyncMock()
        log.replay = AsyncMock(return_value=[{"envelope_id": "e1"}])
        return log

    @pytest.fixture
    def mock_budget(self):
        bt = MagicMock()
        bt.remaining_pct.return_value = 0.75
        bt.total_spend.return_value = 12.50
        bt.monthly_limit_usd = 50.0
        return bt

    @pytest.mark.asyncio
    async def test_bus_check_passes(self, mock_bus):
        doc = DoctorService(bus=mock_bus)
        check = await doc._check_bus()
        assert check.status == CheckStatus.PASS
        assert "operational" in check.message

    @pytest.mark.asyncio
    async def test_substrate_check_passes(self, mock_bus, mock_substrate):
        doc = DoctorService(bus=mock_bus, substrate=mock_substrate)
        check = await doc._check_substrate()
        assert check.status == CheckStatus.PASS
        assert "42 claims" in check.message

    @pytest.mark.asyncio
    async def test_substrate_check_skips_when_missing(self, mock_bus):
        doc = DoctorService(bus=mock_bus, substrate=None)
        check = await doc._check_substrate()
        assert check.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_substrate_check_fails_on_error(self, mock_bus):
        bad_sub = AsyncMock()
        bad_sub.count_claims = AsyncMock(side_effect=RuntimeError("DB locked"))
        doc = DoctorService(bus=mock_bus, substrate=bad_sub)
        check = await doc._check_substrate()
        assert check.status == CheckStatus.FAIL
        assert "DB locked" in check.message

    @pytest.mark.asyncio
    async def test_event_log_check_passes(self, mock_bus, mock_event_log):
        doc = DoctorService(bus=mock_bus, event_log=mock_event_log)
        check = await doc._check_event_log()
        assert check.status == CheckStatus.PASS

    @pytest.mark.asyncio
    async def test_vector_check_passes(self, mock_bus, mock_substrate):
        doc = DoctorService(bus=mock_bus, substrate=mock_substrate)
        check = await doc._check_vectors()
        assert check.status == CheckStatus.PASS
        assert "healthy" in check.message.lower()

    @pytest.mark.asyncio
    async def test_vector_check_warns_on_empty_index_with_claims(self, mock_bus):
        sub = AsyncMock()
        sub.count_claims = AsyncMock(return_value=5)
        sub.embeddings = AsyncMock()
        sub.embeddings.count = AsyncMock(return_value=0)
        sub.embeddings.search = AsyncMock(return_value=[])
        doc = DoctorService(bus=mock_bus, substrate=sub)
        check = await doc._check_vectors()
        assert check.status == CheckStatus.WARN
        assert "empty" in check.message.lower()

    @pytest.mark.asyncio
    async def test_budget_check_healthy(self, mock_bus, mock_budget):
        doc = DoctorService(bus=mock_bus, budget_tracker=mock_budget)
        check = await doc._check_budget()
        assert check.status == CheckStatus.PASS
        assert "healthy" in check.message

    @pytest.mark.asyncio
    async def test_budget_check_warns_when_low(self, mock_bus):
        bt = MagicMock()
        bt.remaining_pct.return_value = 0.15
        bt.total_spend.return_value = 42.50
        bt.monthly_limit_usd = 50.0
        doc = DoctorService(bus=mock_bus, budget_tracker=bt)
        check = await doc._check_budget()
        assert check.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_budget_check_fails_when_exhausted(self, mock_bus):
        bt = MagicMock()
        bt.remaining_pct.return_value = 0.0
        bt.total_spend.return_value = 50.0
        bt.monthly_limit_usd = 50.0
        doc = DoctorService(bus=mock_bus, budget_tracker=bt)
        check = await doc._check_budget()
        assert check.status == CheckStatus.FAIL
        assert "EXHAUSTED" in check.message


class TestDoctorReport:
    @pytest.mark.asyncio
    async def test_run_all_checks_produces_report(self):
        bus = MagicMock()
        bus.publish = MagicMock()
        sub = AsyncMock()
        sub.count_claims = AsyncMock(return_value=10)
        sub.embeddings = AsyncMock()
        sub.embeddings.count = AsyncMock(return_value=10)
        sub.embeddings.search = AsyncMock(return_value=[])
        elog = AsyncMock()
        elog.replay = AsyncMock(return_value=[])
        bt = MagicMock()
        bt.remaining_pct.return_value = 0.80
        bt.total_spend.return_value = 10.0
        bt.monthly_limit_usd = 50.0

        doc = DoctorService(
            bus=bus,
            substrate=sub,
            event_log=elog,
            budget_tracker=bt,
        )
        report = await doc.run_all_checks()

        assert report.overall in (CheckStatus.PASS, CheckStatus.WARN)
        assert report.pass_count >= 1
        assert len(report.checks) >= 4
        assert report.checked_at != ""

    @pytest.mark.asyncio
    async def test_report_published_to_bus(self):
        bus = MagicMock()
        bus.publish = MagicMock()

        doc = DoctorService(bus=bus)
        report = await doc.run_all_checks()
        doc._publish_report(report)

        # Should publish at least the summary report
        assert bus.publish.called
        topics = [
            call.args[0].topic for call in bus.publish.call_args_list
        ]
        assert "system.health.report" in topics

"""Tests for P4 features.

P4 #1: Bus metrics per topic
P4 #2: API request middleware
P4 #3: Envelope TTL
P4 #4: Event replay CLI (endpoint tested via app)
P4 #5: Data export CLI (unit helpers)
P4 #6: Service topology API
P4 #7: Database backup and restore
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from qe.bus.bus_metrics import BusMetrics, TopicStats, get_bus_metrics
from qe.models.envelope import Envelope
from qe.runtime.db_backup import (
    backup_all,
    backup_database,
    list_backups,
    restore_database,
)

# ═══════════════════════════════════════════════════════════════════════════
# P4 #1 — Bus Metrics Per Topic
# ═══════════════════════════════════════════════════════════════════════════


class TestTopicStats:
    def test_defaults(self):
        stats = TopicStats()
        assert stats.publish_count == 0
        assert stats.handler_calls == 0
        assert stats.handler_errors == 0
        assert stats.dlq_count == 0
        assert stats.total_latency_ms == 0.0
        assert stats.max_latency_ms == 0.0
        assert stats.min_latency_ms == float("inf")

    def test_avg_latency_no_calls(self):
        stats = TopicStats()
        assert stats.avg_latency_ms == 0.0

    def test_avg_latency_with_calls(self):
        stats = TopicStats(handler_calls=4, total_latency_ms=100.0)
        assert stats.avg_latency_ms == 25.0

    def test_error_rate_no_calls(self):
        stats = TopicStats()
        assert stats.error_rate == 0.0

    def test_error_rate_with_errors(self):
        stats = TopicStats(handler_calls=10, handler_errors=3)
        assert stats.error_rate == pytest.approx(0.3)

    def test_to_dict(self):
        stats = TopicStats(
            publish_count=5,
            handler_calls=3,
            handler_errors=1,
            dlq_count=1,
            total_latency_ms=150.0,
            max_latency_ms=80.0,
            min_latency_ms=20.0,
        )
        d = stats.to_dict()
        assert d["publish_count"] == 5
        assert d["handler_calls"] == 3
        assert d["handler_errors"] == 1
        assert d["dlq_count"] == 1
        assert d["avg_latency_ms"] == 50.0
        assert d["max_latency_ms"] == 80.0
        assert d["min_latency_ms"] == 20.0
        assert d["error_rate"] == pytest.approx(0.3333, abs=0.001)

    def test_to_dict_min_latency_inf(self):
        stats = TopicStats()
        d = stats.to_dict()
        assert d["min_latency_ms"] == 0.0


class TestBusMetrics:
    def test_record_publish(self):
        m = BusMetrics()
        m.record_publish("test.topic")
        m.record_publish("test.topic")
        m.record_publish("other.topic")
        assert m._topics["test.topic"].publish_count == 2
        assert m._topics["other.topic"].publish_count == 1

    def test_record_handler_done(self):
        m = BusMetrics()
        m.record_handler_done("t", 10.0)
        m.record_handler_done("t", 30.0)
        stats = m._topics["t"]
        assert stats.handler_calls == 2
        assert stats.total_latency_ms == 40.0
        assert stats.max_latency_ms == 30.0
        assert stats.min_latency_ms == 10.0

    def test_record_handler_error(self):
        m = BusMetrics()
        m.record_handler_error("t")
        m.record_handler_error("t")
        assert m._topics["t"].handler_errors == 2

    def test_record_dlq(self):
        m = BusMetrics()
        m.record_dlq("t")
        assert m._topics["t"].dlq_count == 1

    def test_set_subscriber_count(self):
        m = BusMetrics()
        m.set_subscriber_count("t", 5)
        assert m._subscriber_counts["t"] == 5

    def test_get_topic_stats(self):
        m = BusMetrics()
        m.record_publish("t")
        d = m.get_topic_stats("t")
        assert d["publish_count"] == 1

    def test_snapshot(self):
        m = BusMetrics()
        m.record_publish("a")
        m.record_publish("a")
        m.record_publish("b")
        m.record_handler_done("a", 10.0)
        m.record_handler_error("b")
        m.record_dlq("b")
        m.set_subscriber_count("a", 3)

        snap = m.snapshot()
        assert snap["total_published"] == 3
        assert snap["total_errors"] == 1
        assert snap["total_dlq"] == 1
        assert snap["active_topics"] == 2
        assert "a" in snap["topics"]
        assert "b" in snap["topics"]
        assert snap["top_topics"][0]["topic"] == "a"
        assert snap["subscriber_counts"]["a"] == 3
        assert "uptime_seconds" in snap

    def test_snapshot_top_topics_limit(self):
        m = BusMetrics()
        for i in range(15):
            topic = f"topic.{i}"
            for _ in range(i + 1):
                m.record_publish(topic)
        snap = m.snapshot()
        assert len(snap["top_topics"]) == 10
        # Highest volume first
        assert snap["top_topics"][0]["publish_count"] == 15

    def test_reset(self):
        m = BusMetrics()
        m.record_publish("t")
        m.set_subscriber_count("t", 2)
        m.reset()
        assert len(m._topics) == 0
        assert len(m._subscriber_counts) == 0

    def test_singleton(self):
        m = get_bus_metrics()
        assert isinstance(m, BusMetrics)


# ═══════════════════════════════════════════════════════════════════════════
# P4 #2 — API Request Middleware
# ═══════════════════════════════════════════════════════════════════════════


class TestRequestTimingMiddleware:
    def test_middleware_import(self):
        from qe.api.middleware import RequestTimingMiddleware

        assert RequestTimingMiddleware is not None

    def test_middleware_is_starlette_middleware(self):
        from starlette.middleware.base import BaseHTTPMiddleware

        from qe.api.middleware import RequestTimingMiddleware

        assert issubclass(RequestTimingMiddleware, BaseHTTPMiddleware)


# ═══════════════════════════════════════════════════════════════════════════
# P4 #3 — Envelope TTL
# ═══════════════════════════════════════════════════════════════════════════


class TestEnvelopeTTL:
    def test_envelope_ttl_field_default_none(self):
        env = Envelope(
            topic="test",
            source_service_id="svc",
            payload={"k": "v"},
        )
        assert env.ttl_seconds is None

    def test_envelope_ttl_field_set(self):
        env = Envelope(
            topic="test",
            source_service_id="svc",
            payload={"k": "v"},
            ttl_seconds=60,
        )
        assert env.ttl_seconds == 60

    @pytest.mark.asyncio
    async def test_expired_envelope_routes_to_dlq(self):
        """An envelope with an expired TTL should be routed to DLQ."""
        from qe.bus.memory_bus import MemoryBus

        bus = MemoryBus()
        handler = AsyncMock()
        bus.subscribe("system.heartbeat", handler)

        # Create envelope timestamped 10s in the past with 5s TTL
        env = Envelope(
            topic="system.heartbeat",
            source_service_id="svc",
            payload={"k": "v"},
            ttl_seconds=5,
            timestamp=datetime.now(UTC) - timedelta(seconds=10),
        )

        bus.publish(env)
        await asyncio.sleep(0.1)

        # Handler should NOT have been called — envelope expired
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_ttl_envelope_delivered(self):
        """An envelope with a valid TTL should be delivered normally."""
        from qe.bus.memory_bus import MemoryBus

        bus = MemoryBus()
        handler = AsyncMock()
        bus.subscribe("system.heartbeat", handler)

        env = Envelope(
            topic="system.heartbeat",
            source_service_id="svc",
            payload={"k": "v"},
            ttl_seconds=3600,
            timestamp=datetime.now(UTC),
        )

        bus.publish(env)
        await asyncio.sleep(0.1)

        handler.assert_called_once_with(env)

    @pytest.mark.asyncio
    async def test_no_ttl_envelope_delivered(self):
        """An envelope without TTL should always be delivered."""
        from qe.bus.memory_bus import MemoryBus

        bus = MemoryBus()
        handler = AsyncMock()
        bus.subscribe("system.heartbeat", handler)

        env = Envelope(
            topic="system.heartbeat",
            source_service_id="svc",
            payload={"k": "v"},
        )

        bus.publish(env)
        await asyncio.sleep(0.1)

        handler.assert_called_once_with(env)


# ═══════════════════════════════════════════════════════════════════════════
# P4 #4 — Event Replay (endpoint logic)
# ═══════════════════════════════════════════════════════════════════════════


class TestEventReplayEndpoint:
    def test_replay_without_event_log_returns_503(self):
        """POST /api/events/replay returns 503 when event log not ready."""
        from unittest.mock import patch as _patch

        from fastapi.testclient import TestClient

        from qe.api.app import app

        with _patch("qe.api.app._event_log", None):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/events/replay", json={"limit": 10})
            assert resp.status_code == 503


# ═══════════════════════════════════════════════════════════════════════════
# P4 #5 — Data Export (helpers)
# ═══════════════════════════════════════════════════════════════════════════


class TestDataExport:
    def test_json_dumps_default_str(self):
        """Verify datetime objects serialize via default=str."""
        import json

        now = datetime.now(UTC)
        result = json.dumps({"ts": now}, default=str)
        assert str(now) in result

    def test_csv_round_trip(self):
        """Verify CSV export produces correct headers and rows."""
        import csv
        import io

        rows = [
            {"id": "1", "text": "hello", "score": 0.9},
            {"id": "2", "text": "world", "score": 0.8},
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        text = buf.getvalue()

        reader = csv.DictReader(io.StringIO(text))
        out = list(reader)
        assert len(out) == 2
        assert out[0]["id"] == "1"
        assert out[1]["text"] == "world"


# ═══════════════════════════════════════════════════════════════════════════
# P4 #6 — Service Topology API
# ═══════════════════════════════════════════════════════════════════════════


class TestTopologyEndpoint:
    def test_topology_without_supervisor_returns_503(self):
        """GET /api/topology returns 503 when engine not started."""
        from unittest.mock import patch as _patch

        from fastapi.testclient import TestClient

        from qe.api.app import app

        with _patch("qe.api.app._supervisor", None):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/topology")
            assert resp.status_code == 503

    def test_bus_stats_endpoint(self):
        """GET /api/bus/stats returns a metrics snapshot."""
        from fastapi.testclient import TestClient

        from qe.api.app import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/bus/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_published" in data
        assert "topics" in data


# ═══════════════════════════════════════════════════════════════════════════
# P4 #7 — Database Backup and Restore
# ═══════════════════════════════════════════════════════════════════════════


class TestBackupDatabase:
    def test_backup_missing_source(self, tmp_path):
        result = backup_database(
            str(tmp_path / "nonexistent.db"),
            str(tmp_path / "backup.db"),
        )
        assert result["status"] == "skipped"
        assert "not found" in result["reason"]

    def test_backup_success(self, tmp_path):
        # Create a real SQLite database
        source = tmp_path / "test.db"
        conn = sqlite3.connect(str(source))
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'hello')")
        conn.commit()
        conn.close()

        dest = tmp_path / "backup.db"
        result = backup_database(str(source), str(dest))

        assert result["status"] == "completed"
        assert result["size_bytes"] > 0
        assert result["elapsed_seconds"] >= 0
        assert dest.exists()

        # Verify the backup is a valid SQLite database with data
        bconn = sqlite3.connect(str(dest))
        row = bconn.execute("SELECT val FROM t WHERE id = 1").fetchone()
        bconn.close()
        assert row[0] == "hello"

    def test_backup_creates_parent_dirs(self, tmp_path):
        source = tmp_path / "test.db"
        conn = sqlite3.connect(str(source))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
        conn.close()

        dest = tmp_path / "deep" / "nested" / "backup.db"
        result = backup_database(str(source), str(dest))
        assert result["status"] == "completed"
        assert dest.exists()


class TestRestoreDatabase:
    def test_restore_missing_backup(self, tmp_path):
        result = restore_database(
            str(tmp_path / "nonexistent.db"),
            str(tmp_path / "restored.db"),
        )
        assert result["status"] == "failed"
        assert "not found" in result["reason"]

    def test_restore_success(self, tmp_path):
        # Create a backup file (real SQLite db)
        backup = tmp_path / "backup.db"
        conn = sqlite3.connect(str(backup))
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'restored_value')")
        conn.commit()
        conn.close()

        dest = tmp_path / "restored.db"
        result = restore_database(str(backup), str(dest))

        assert result["status"] == "restored"
        assert result["elapsed_seconds"] >= 0
        assert dest.exists()

        # Verify content
        rconn = sqlite3.connect(str(dest))
        row = rconn.execute("SELECT val FROM t WHERE id = 1").fetchone()
        rconn.close()
        assert row[0] == "restored_value"

    def test_restore_creates_parent_dirs(self, tmp_path):
        backup = tmp_path / "backup.db"
        conn = sqlite3.connect(str(backup))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
        conn.close()

        dest = tmp_path / "deep" / "nested" / "restored.db"
        result = restore_database(str(backup), str(dest))
        assert result["status"] == "restored"
        assert dest.exists()


class TestBackupAll:
    def test_backup_all_with_files(self, tmp_path):
        # Create two test databases
        db1 = tmp_path / "data" / "a.db"
        db2 = tmp_path / "data" / "b.db"
        db1.parent.mkdir(parents=True, exist_ok=True)

        for db_path in [db1, db2]:
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.commit()
            conn.close()

        dest_dir = str(tmp_path / "backups")
        result = backup_all(
            dest_dir=dest_dir,
            db_files=[str(db1), str(db2)],
        )

        assert result["completed"] == 2
        assert result["total"] == 2
        assert result["total_size_bytes"] > 0
        assert result["timestamp"]
        assert Path(result["backup_dir"]).exists()

    def test_backup_all_mixed_results(self, tmp_path):
        # One real, one missing
        db1 = tmp_path / "data" / "exists.db"
        db1.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db1))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.commit()
        conn.close()

        dest_dir = str(tmp_path / "backups")
        result = backup_all(
            dest_dir=dest_dir,
            db_files=[str(db1), str(tmp_path / "missing.db")],
        )

        assert result["completed"] == 1
        assert result["total"] == 2
        dbs = result["databases"]
        statuses = {r["status"] for r in dbs}
        assert "completed" in statuses
        assert "skipped" in statuses


class TestListBackups:
    def test_list_backups_no_dir(self, tmp_path):
        result = list_backups(str(tmp_path / "nonexistent"))
        assert result == []

    def test_list_backups_with_entries(self, tmp_path):
        backup_dir = tmp_path / "backups"

        # Create two backup directories
        for name in ["20260101_120000", "20260102_120000"]:
            d = backup_dir / name
            d.mkdir(parents=True)
            conn = sqlite3.connect(str(d / "qe.db"))
            conn.execute("CREATE TABLE t (id INTEGER)")
            conn.commit()
            conn.close()

        result = list_backups(str(backup_dir))
        assert len(result) == 2
        # Sorted reverse by name (most recent first)
        assert result[0]["name"] == "20260102_120000"
        assert "qe.db" in result[0]["databases"]
        assert result[0]["total_size_bytes"] > 0

    def test_list_backups_ignores_files(self, tmp_path):
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        # Create a file (not a directory)
        (backup_dir / "random.txt").write_text("not a backup")
        result = list_backups(str(backup_dir))
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# P4 — Bus Metrics Integration with MemoryBus
# ═══════════════════════════════════════════════════════════════════════════


class TestBusMetricsIntegration:
    def test_publish_increments_bus_metrics(self):
        """Publishing an envelope should record in bus metrics."""
        from qe.bus.memory_bus import MemoryBus

        bm = get_bus_metrics()
        bm.reset()

        bus = MemoryBus()
        env = Envelope(
            topic="system.heartbeat",
            source_service_id="svc",
            payload={"k": "v"},
        )
        bus.publish(env)

        stats = bm.get_topic_stats("system.heartbeat")
        assert stats["publish_count"] >= 1

    @pytest.mark.asyncio
    async def test_handler_done_records_latency(self):
        """Successful handler execution should record latency in bus metrics."""
        from qe.bus.memory_bus import MemoryBus

        bm = get_bus_metrics()
        bm.reset()

        bus = MemoryBus()

        async def slow_handler(env):
            await asyncio.sleep(0.01)

        bus.subscribe("claims.proposed", slow_handler)

        env = Envelope(
            topic="claims.proposed",
            source_service_id="svc",
            payload={},
        )
        bus.publish(env)
        await asyncio.sleep(0.2)

        stats = bm.get_topic_stats("claims.proposed")
        assert stats["handler_calls"] >= 1
        assert stats["avg_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_handler_error_records_in_metrics(self):
        """Handler that permanently fails should record error in bus metrics."""
        from qe.bus.memory_bus import MemoryBus

        bm = get_bus_metrics()
        bm.reset()

        bus = MemoryBus()
        bus._max_retries = 0  # No retries, fail immediately

        async def bad_handler(env):
            raise ValueError("boom")

        bus.subscribe("system.error", bad_handler)

        env = Envelope(
            topic="system.error",
            source_service_id="svc",
            payload={},
        )
        bus.publish(env)
        await asyncio.sleep(0.2)

        stats = bm.get_topic_stats("system.error")
        assert stats["handler_errors"] >= 1

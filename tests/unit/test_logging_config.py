"""Tests for the centralized logging configuration."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pytest

from qe.runtime.logging_config import (
    CorrelationFilter,
    HumanFormatter,
    JSONFormatter,
    configure_from_config,
    configure_logging,
    ctx_correlation_id,
    ctx_envelope_id,
    ctx_service_id,
    update_log_level,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset root logger after each test to avoid cross-contamination."""
    yield
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


class TestCorrelationFilter:
    def test_injects_context_vars(self):
        filt = CorrelationFilter()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "msg", (), None
        )

        tok_env = ctx_envelope_id.set("env_abc123")
        tok_cor = ctx_correlation_id.set("cor_xyz789")
        tok_svc = ctx_service_id.set("researcher_v1")
        try:
            filt.filter(record)
            assert record.envelope_id == "env_abc123"  # type: ignore[attr-defined]
            assert record.correlation_id == "cor_xyz789"  # type: ignore[attr-defined]
            assert record.service_id == "researcher_v1"  # type: ignore[attr-defined]
        finally:
            ctx_envelope_id.reset(tok_env)
            ctx_correlation_id.reset(tok_cor)
            ctx_service_id.reset(tok_svc)

    def test_defaults_to_empty_string(self):
        filt = CorrelationFilter()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "msg", (), None
        )
        filt.filter(record)
        assert record.envelope_id == ""  # type: ignore[attr-defined]
        assert record.correlation_id == ""  # type: ignore[attr-defined]
        assert record.service_id == ""  # type: ignore[attr-defined]

    def test_always_returns_true(self):
        filt = CorrelationFilter()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "msg", (), None
        )
        assert filt.filter(record) is True


class TestJSONFormatter:
    def test_basic_format(self):
        fmt = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord(
            "qe.runtime.service", logging.INFO, "", 0, "hello world", (), None
        )
        record.correlation_id = ""  # type: ignore[attr-defined]
        record.envelope_id = ""  # type: ignore[attr-defined]
        record.service_id = ""  # type: ignore[attr-defined]

        output = fmt.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "qe.runtime.service"
        assert data["msg"] == "hello world"
        assert "ts" in data

    def test_includes_correlation_context(self):
        fmt = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "msg", (), None
        )
        record.correlation_id = "cor_123"  # type: ignore[attr-defined]
        record.envelope_id = "env_456"  # type: ignore[attr-defined]
        record.service_id = "validator"  # type: ignore[attr-defined]

        data = json.loads(fmt.format(record))
        assert data["correlation_id"] == "cor_123"
        assert data["envelope_id"] == "env_456"
        assert data["service_id"] == "validator"

    def test_omits_empty_context(self):
        fmt = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "msg", (), None
        )
        record.correlation_id = ""  # type: ignore[attr-defined]
        record.envelope_id = ""  # type: ignore[attr-defined]
        record.service_id = ""  # type: ignore[attr-defined]

        data = json.loads(fmt.format(record))
        assert "correlation_id" not in data
        assert "envelope_id" not in data
        assert "service_id" not in data

    def test_includes_exception(self):
        fmt = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                "test", logging.ERROR, "", 0, "boom", (), sys.exc_info()
            )
            record.correlation_id = ""  # type: ignore[attr-defined]
            record.envelope_id = ""  # type: ignore[attr-defined]
            record.service_id = ""  # type: ignore[attr-defined]

        data = json.loads(fmt.format(record))
        assert "exception" in data
        assert "ValueError" in data["exception"]


class TestHumanFormatter:
    def test_no_context(self):
        fmt = HumanFormatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "hello", (), None
        )
        record.correlation_id = ""  # type: ignore[attr-defined]
        record.envelope_id = ""  # type: ignore[attr-defined]
        record.service_id = ""  # type: ignore[attr-defined]

        output = fmt.format(record)
        assert "hello" in output
        # No context suffix
        assert "[" not in output.split("hello")[1]

    def test_with_context(self):
        fmt = HumanFormatter(
            fmt="%(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "msg", (), None
        )
        record.service_id = "researcher"  # type: ignore[attr-defined]
        record.envelope_id = "env_abcdef123456"  # type: ignore[attr-defined]
        record.correlation_id = ""  # type: ignore[attr-defined]

        output = fmt.format(record)
        assert "[svc=researcher env=env_abcdef12]" in output

    def test_truncates_ids(self):
        fmt = HumanFormatter(fmt="%(message)s")
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "msg", (), None
        )
        record.service_id = ""  # type: ignore[attr-defined]
        record.envelope_id = "a" * 36  # type: ignore[attr-defined]
        record.correlation_id = "b" * 36  # type: ignore[attr-defined]

        output = fmt.format(record)
        assert f"env={'a' * 12}" in output
        assert f"cor={'b' * 12}" in output


class TestConfigureLogging:
    def test_sets_root_level(self):
        configure_logging(level="DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_creates_console_handler(self):
        configure_logging(level="INFO")
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)

    def test_json_output_uses_json_formatter(self):
        configure_logging(json_output=True)
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_human_output_uses_human_formatter(self):
        configure_logging(json_output=False)
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, HumanFormatter)

    def test_file_handler_with_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_logging(log_dir=tmpdir)
            root = logging.getLogger()
            assert len(root.handlers) == 2  # console + file
            file_handlers = [
                h
                for h in root.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            assert len(file_handlers) == 1
            assert file_handlers[0].baseFilename.endswith("qe.log")

    def test_file_handler_always_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_logging(log_dir=tmpdir, json_output=False)
            root = logging.getLogger()
            file_handlers = [
                h
                for h in root.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            assert isinstance(file_handlers[0].formatter, JSONFormatter)

    def test_tames_noisy_loggers(self):
        configure_logging()
        assert logging.getLogger("litellm").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("uvicorn.access").level == logging.WARNING
        assert logging.getLogger("instructor").level == logging.WARNING

    def test_module_level_overrides(self):
        configure_logging(
            level="INFO",
            module_levels={"qe.runtime.service": "DEBUG"},
        )
        assert logging.getLogger("qe.runtime.service").level == logging.DEBUG

    def test_reconfigure_clears_old_handlers(self):
        configure_logging(level="INFO")
        configure_logging(level="DEBUG")
        root = logging.getLogger()
        assert len(root.handlers) == 1  # old handler cleared


class TestConfigureFromConfig:
    def test_uses_config_level(self):
        configure_from_config({"runtime": {"log_level": "WARNING"}})
        assert logging.getLogger().level == logging.WARNING

    def test_verbose_overrides_config(self):
        configure_from_config(
            {"runtime": {"log_level": "WARNING"}}, verbose=True
        )
        assert logging.getLogger().level == logging.DEBUG

    def test_defaults_when_no_runtime_section(self):
        configure_from_config({})
        assert logging.getLogger().level == logging.INFO

    def test_log_dir_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_from_config({"runtime": {"log_dir": tmpdir}})
            root = logging.getLogger()
            file_handlers = [
                h
                for h in root.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            assert len(file_handlers) == 1


class TestUpdateLogLevel:
    def test_updates_root_level(self):
        configure_logging(level="INFO")
        assert logging.getLogger().level == logging.INFO

        update_log_level("DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_invalid_level_is_ignored(self):
        configure_logging(level="INFO")
        update_log_level("NONEXISTENT")
        # Level unchanged
        assert logging.getLogger().level == logging.INFO

    def test_preserves_handlers(self):
        configure_logging(level="INFO")
        handler_count = len(logging.getLogger().handlers)

        update_log_level("WARNING")
        assert len(logging.getLogger().handlers) == handler_count


class TestCorrelationFilterIntegration:
    def test_handler_receives_correlation(self):
        configure_logging(level="DEBUG")
        root = logging.getLogger()

        # Capture output
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        capture = Capture()
        capture.addFilter(CorrelationFilter())
        root.addHandler(capture)

        tok = ctx_service_id.set("test_svc")
        try:
            logging.getLogger("qe.test").info("test message")
        finally:
            ctx_service_id.reset(tok)

        assert len(records) >= 1
        assert records[-1].service_id == "test_svc"  # type: ignore[attr-defined]


class TestLogFileOutput:
    def test_writes_json_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            configure_logging(level="INFO", log_dir=tmpdir)
            log = logging.getLogger("qe.test.file")
            log.info("file test message")

            # Flush handlers
            for h in logging.getLogger().handlers:
                h.flush()

            log_file = Path(tmpdir) / "qe.log"
            assert log_file.exists()
            content = log_file.read_text(encoding="utf-8")
            assert content.strip()
            data = json.loads(content.strip().split("\n")[-1])
            assert data["msg"] == "file test message"
            assert data["logger"] == "qe.test.file"

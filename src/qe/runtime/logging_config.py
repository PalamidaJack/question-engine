"""Centralized logging configuration for Question Engine.

Provides:
- Structured JSON formatter for production / machine parsing
- Human-readable formatter for development
- Correlation context via contextvars (envelope_id, correlation_id, service_id)
- Rotating file handler for log persistence
- Third-party logger noise control
- Hot-updatable log level without reconfiguring handlers
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from contextvars import ContextVar
from pathlib import Path

# ── Correlation context propagated across async tasks ──────────────────────
ctx_correlation_id: ContextVar[str] = ContextVar("ctx_correlation_id", default="")
ctx_envelope_id: ContextVar[str] = ContextVar("ctx_envelope_id", default="")
ctx_service_id: ContextVar[str] = ContextVar("ctx_service_id", default="")


class CorrelationFilter(logging.Filter):
    """Inject correlation context vars into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = ctx_correlation_id.get("")  # type: ignore[attr-defined]
        record.envelope_id = ctx_envelope_id.get("")  # type: ignore[attr-defined]
        record.service_id = ctx_service_id.get("")  # type: ignore[attr-defined]
        return True


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production / log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Correlation context
        for attr in ("correlation_id", "envelope_id", "service_id"):
            val = getattr(record, attr, "")
            if val:
                entry[attr] = val
        # Exception info
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable formatter with optional correlation context suffix."""

    def format(self, record: logging.LogRecord) -> str:
        ctx_parts: list[str] = []
        svc = getattr(record, "service_id", "")
        if svc:
            ctx_parts.append(f"svc={svc}")
        env = getattr(record, "envelope_id", "")
        if env:
            ctx_parts.append(f"env={env[:12]}")
        cor = getattr(record, "correlation_id", "")
        if cor:
            ctx_parts.append(f"cor={cor[:12]}")

        base = super().format(record)
        if ctx_parts:
            return f"{base} [{' '.join(ctx_parts)}]"
        return base


# Third-party loggers to quiet down
_NOISY_LOGGERS: dict[str, int] = {
    "litellm": logging.WARNING,
    "litellm.utils": logging.WARNING,
    "litellm.litellm_core_utils": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "uvicorn": logging.INFO,
    "uvicorn.access": logging.WARNING,
    "uvicorn.error": logging.INFO,
    "watchdog": logging.WARNING,
    "watchfiles": logging.WARNING,
    "aiosqlite": logging.WARNING,
    "instructor": logging.WARNING,
}


def configure_logging(
    *,
    level: str = "INFO",
    json_output: bool = False,
    log_dir: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per file
    backup_count: int = 5,
    module_levels: dict[str, str] | None = None,
) -> None:
    """Configure logging for the entire QE process.

    Args:
        level: Root log level (DEBUG, INFO, WARNING, ERROR).
        json_output: Use JSON formatter for console; else human-readable.
        log_dir: Directory for rotating log files. None = stderr only.
        max_bytes: Max bytes per log file before rotation.
        backup_count: Number of rotated backup files to keep.
        module_levels: Per-logger level overrides, e.g. {"qe.runtime.service": "DEBUG"}.
    """
    root = logging.getLogger()
    root.handlers.clear()

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    corr_filter = CorrelationFilter()

    # ── Console handler ──
    if json_output:
        formatter: logging.Formatter = JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
    else:
        formatter = HumanFormatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.addFilter(corr_filter)
    root.addHandler(console)

    # ── Rotating file handler (always JSON for machine parsing) ──
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "qe.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
        file_handler.addFilter(corr_filter)
        root.addHandler(file_handler)

    # ── Third-party noise control ──
    for logger_name, logger_level in _NOISY_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(logger_level)

    # ── Per-module overrides ──
    if module_levels:
        for mod, mod_level in module_levels.items():
            logging.getLogger(mod).setLevel(
                getattr(logging, mod_level.upper(), numeric_level)
            )


def configure_from_config(config: dict, *, verbose: bool = False) -> None:
    """Configure logging from config.toml runtime settings.

    Args:
        config: Settings dict with optional 'runtime' section containing
                log_level, log_json, log_dir, module_levels.
        verbose: CLI --verbose flag overrides config level to DEBUG.
    """
    runtime = config.get("runtime", {})

    level = "DEBUG" if verbose else runtime.get("log_level", "INFO")
    json_output = runtime.get("log_json", False)
    log_dir = runtime.get("log_dir", None)
    module_levels = runtime.get("module_levels", None)

    configure_logging(
        level=level,
        json_output=json_output,
        log_dir=log_dir,
        module_levels=module_levels,
    )


def update_log_level(level: str) -> None:
    """Hot-update root log level without reconfiguring handlers."""
    numeric = getattr(logging, level.upper(), None)
    if numeric is not None:
        logging.getLogger().setLevel(numeric)
        logging.getLogger(__name__).info("Log level changed to %s", level)

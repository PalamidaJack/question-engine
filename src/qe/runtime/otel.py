"""Lightweight optional OpenTelemetry helpers.

This module provides a no-op fallback when opentelemetry packages are not
installed so the rest of the codebase can call `otel.init_tracing()` and
`otel.start_span()` without introducing a hard dependency during local dev
or in test environments.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _OTEL_AVAILABLE = False


def init_tracing(
    service_name: str = "question-engine",
    exporter: str = "console",
    otlp_endpoint: str | None = None,
) -> None:
    """Initialize tracing provider if OpenTelemetry SDK is available.

    exporter: "console" or "otlp"
    """
    if not _OTEL_AVAILABLE:
        log.debug("otel.init_tracing: opentelemetry not installed; no-op")
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    trace.set_tracer_provider(provider)

    if exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter_inst = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter_inst))
        except Exception:
            log.exception(
                "otel.init_tracing: failed to configure OTLP exporter;"
                " falling back to console",
            )
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))


def get_tracer(name: str = "qe") -> Any:
    if not _OTEL_AVAILABLE:
        return _NoopTracer()
    return trace.get_tracer(name)


class _NoopSpan:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None


class _NoopTracer:
    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoopSpan:
        return _NoopSpan()


@contextmanager
def start_span(name: str, attributes: dict | None = None):
    """Context manager for a span. No-op if OpenTelemetry is unavailable.

    Usage:
        with otel.start_span("operation", {"key": "value"}):
            do_work()
    """
    if not _OTEL_AVAILABLE:
        yield
        return

    tracer = get_tracer()
    span_ctx = tracer.start_as_current_span(name, attributes=(attributes or {}))
    try:
        with span_ctx:
            yield span_ctx
    finally:
        # span exit happens via context manager
        pass

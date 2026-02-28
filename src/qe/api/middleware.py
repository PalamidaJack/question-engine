"""API request middleware: timing, structured logging, error wrapping.

Records request latency in MetricsCollector and emits structured
log entries for every HTTP request.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from qe.runtime.metrics import get_metrics

log = logging.getLogger(__name__)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Middleware that times requests, logs them, and wraps errors.

    For every request:
    1. Records start time
    2. Calls the route handler
    3. Logs method, path, status, and duration
    4. Records latency in MetricsCollector histogram
    5. Catches unhandled exceptions and returns consistent JSON errors
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        start = time.monotonic()
        method = request.method
        path = request.url.path

        metrics = get_metrics()
        metrics.counter("api_requests_total").inc()

        try:
            response = await call_next(request)
            elapsed_ms = (time.monotonic() - start) * 1000
            status = response.status_code

            metrics.histogram("api_latency_ms").observe(elapsed_ms)

            if status >= 400:
                metrics.counter("api_errors_total").inc()

            log.info(
                "api.request method=%s path=%s status=%d duration_ms=%.1f",
                method,
                path,
                status,
                elapsed_ms,
            )

            return response

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            metrics.counter("api_errors_total").inc()
            metrics.histogram("api_latency_ms").observe(elapsed_ms)

            log.error(
                "api.unhandled_error method=%s path=%s "
                "error=%s duration_ms=%.1f",
                method,
                path,
                str(exc),
                elapsed_ms,
            )

            return JSONResponse(
                {"error": "Internal server error", "detail": str(exc)},
                status_code=500,
            )

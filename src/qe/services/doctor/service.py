"""Doctor service: continuous health monitoring and system diagnostics.

Runs periodic health probes against all QE subsystems and publishes
``system.health.check`` events per check and ``system.health.report``
summary events.  Does NOT call LLMs — all checks are deterministic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from qe.models.envelope import Envelope

log = logging.getLogger(__name__)


class CheckStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class HealthCheck(BaseModel):
    """Result of a single health check."""

    name: str
    status: CheckStatus = CheckStatus.SKIP
    message: str = ""
    duration_ms: float = 0.0
    checked_at: str = ""
    fix_hint: str = ""


class HealthReport(BaseModel):
    """Aggregated health report from all checks."""

    overall: CheckStatus = CheckStatus.PASS
    checks: list[HealthCheck] = Field(default_factory=list)
    checked_at: str = ""
    pass_count: int = 0
    warn_count: int = 0
    fail_count: int = 0


class DoctorService:
    """Continuous health monitoring service.

    Unlike other services, Doctor does NOT extend BaseService — it has
    no LLM calls, no genome, and runs its own scheduling loop.  It's
    instantiated directly by the Supervisor or API lifespan.
    """

    def __init__(
        self,
        bus: Any,
        substrate: Any | None = None,
        supervisor: Any | None = None,
        event_log: Any | None = None,
        budget_tracker: Any | None = None,
        *,
        check_interval_seconds: int = 60,
    ) -> None:
        self._bus = bus
        self._substrate = substrate
        self._supervisor = supervisor
        self._event_log = event_log
        self._budget_tracker = budget_tracker
        self._interval = check_interval_seconds
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_report: HealthReport | None = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        log.info("Doctor service started (interval=%ds)", self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def last_report(self) -> HealthReport | None:
        return self._last_report

    # ── Main Loop ────────────────────────────────────────────────────

    async def _check_loop(self) -> None:
        # Short initial delay to let system boot
        await asyncio.sleep(5)
        while self._running:
            try:
                report = await self.run_all_checks()
                self._last_report = report
                self._publish_report(report)
            except Exception:
                log.exception("Doctor check cycle failed")
            await asyncio.sleep(self._interval)

    # ── Public: On-Demand Check ──────────────────────────────────────

    async def run_all_checks(self) -> HealthReport:
        """Run all health checks and return aggregated report."""
        checks: list[HealthCheck] = []

        # Run all checks concurrently
        check_coros = [
            self._check_bus(),
            self._check_substrate(),
            self._check_vectors(),
            self._check_event_log(),
            self._check_budget(),
            self._check_services(),
            self._check_circuit_breakers(),
            self._check_heartbeats(),
        ]

        results = await asyncio.gather(*check_coros, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                checks.append(HealthCheck(
                    name="unknown",
                    status=CheckStatus.FAIL,
                    message=str(result),
                ))
            elif isinstance(result, list):
                checks.extend(result)
            else:
                checks.append(result)

        # Compute overall status
        statuses = [c.status for c in checks]
        if CheckStatus.FAIL in statuses:
            overall = CheckStatus.FAIL
        elif CheckStatus.WARN in statuses:
            overall = CheckStatus.WARN
        else:
            overall = CheckStatus.PASS

        report = HealthReport(
            overall=overall,
            checks=checks,
            checked_at=datetime.now(UTC).isoformat(),
            pass_count=statuses.count(CheckStatus.PASS),
            warn_count=statuses.count(CheckStatus.WARN),
            fail_count=statuses.count(CheckStatus.FAIL),
        )

        log.debug(
            "doctor.report overall=%s pass=%d warn=%d fail=%d",
            overall, report.pass_count, report.warn_count, report.fail_count,
        )
        return report

    # ── Individual Checks ────────────────────────────────────────────

    async def _check_bus(self) -> HealthCheck:
        """Verify the memory bus is operational."""
        start = time.monotonic()
        try:
            # Publish a test heartbeat and verify no exception
            self._bus.publish(Envelope(
                topic="system.heartbeat",
                source_service_id="doctor",
                payload={"probe": True},
            ))
            return HealthCheck(
                name="bus",
                status=CheckStatus.PASS,
                message="Memory bus operational",
                duration_ms=(time.monotonic() - start) * 1000,
                checked_at=datetime.now(UTC).isoformat(),
            )
        except Exception as e:
            return HealthCheck(
                name="bus",
                status=CheckStatus.FAIL,
                message=f"Bus publish failed: {e}",
                duration_ms=(time.monotonic() - start) * 1000,
                checked_at=datetime.now(UTC).isoformat(),
                fix_hint="Check bus initialization in app lifespan",
            )

    async def _check_substrate(self) -> HealthCheck:
        """Verify substrate DB is accessible."""
        start = time.monotonic()
        if not self._substrate:
            return HealthCheck(
                name="substrate",
                status=CheckStatus.SKIP,
                message="Substrate not configured",
                checked_at=datetime.now(UTC).isoformat(),
            )
        try:
            count = await self._substrate.count_claims()
            return HealthCheck(
                name="substrate",
                status=CheckStatus.PASS,
                message=f"Belief ledger accessible ({count} claims)",
                duration_ms=(time.monotonic() - start) * 1000,
                checked_at=datetime.now(UTC).isoformat(),
            )
        except Exception as e:
            return HealthCheck(
                name="substrate",
                status=CheckStatus.FAIL,
                message=f"Substrate query failed: {e}",
                duration_ms=(time.monotonic() - start) * 1000,
                checked_at=datetime.now(UTC).isoformat(),
                fix_hint="Check SQLite DB path and migrations",
            )

    async def _check_event_log(self) -> HealthCheck:
        """Verify event log is writable and queryable."""
        start = time.monotonic()
        if not self._event_log:
            return HealthCheck(
                name="event_log",
                status=CheckStatus.SKIP,
                message="Event log not configured",
                checked_at=datetime.now(UTC).isoformat(),
            )
        try:
            events = await self._event_log.replay(limit=1)
            return HealthCheck(
                name="event_log",
                status=CheckStatus.PASS,
                message=f"Event log queryable (latest: {len(events)} events)",
                duration_ms=(time.monotonic() - start) * 1000,
                checked_at=datetime.now(UTC).isoformat(),
            )
        except Exception as e:
            return HealthCheck(
                name="event_log",
                status=CheckStatus.FAIL,
                message=f"Event log query failed: {e}",
                duration_ms=(time.monotonic() - start) * 1000,
                checked_at=datetime.now(UTC).isoformat(),
                fix_hint="Check event_log.db path and initialization",
            )

    async def _check_vectors(self) -> HealthCheck:
        """Verify vector index availability and probe semantic search latency."""
        start = time.monotonic()
        if not self._substrate or not hasattr(self._substrate, "embeddings"):
            return HealthCheck(
                name="vectors",
                status=CheckStatus.SKIP,
                message="Vector store not configured",
                checked_at=datetime.now(UTC).isoformat(),
            )

        try:
            from qe.runtime.metrics import get_metrics

            metrics = get_metrics()
            embeddings = self._substrate.embeddings
            count = await embeddings.count()
            claim_count = await self._substrate.count_claims()
            metrics.gauge("vector_index_size").set(float(count))
            metrics.gauge("vector_hnsw_enabled").set(
                1.0 if getattr(embeddings, "_hnsw_index", None) is not None else 0.0
            )

            # Probe semantic retrieval only when vectors exist.
            probe_ms = 0.0
            if count > 0:
                probe_start = time.monotonic()
                await embeddings.search("health probe", top_k=1, min_similarity=0.0)
                probe_ms = (time.monotonic() - probe_start) * 1000
                metrics.histogram("vector_query_latency_ms").observe(probe_ms)

            # If we have claims but no vectors, this indicates indexing drift.
            if claim_count > 0 and count == 0:
                return HealthCheck(
                    name="vectors",
                    status=CheckStatus.WARN,
                    message=(
                        f"Vector index empty while ledger has {claim_count} claims"
                    ),
                    duration_ms=(time.monotonic() - start) * 1000,
                    checked_at=datetime.now(UTC).isoformat(),
                    fix_hint="Run embedding reindex to restore semantic retrieval",
                )

            # Latency guardrail for semantic probe.
            if probe_ms > 500.0:
                return HealthCheck(
                    name="vectors",
                    status=CheckStatus.WARN,
                    message=f"Vector probe slow ({probe_ms:.1f}ms, size={count})",
                    duration_ms=(time.monotonic() - start) * 1000,
                    checked_at=datetime.now(UTC).isoformat(),
                    fix_hint="Check HNSW index health and embedding table size",
                )

            return HealthCheck(
                name="vectors",
                status=CheckStatus.PASS,
                message=f"Vector store healthy (size={count})",
                duration_ms=(time.monotonic() - start) * 1000,
                checked_at=datetime.now(UTC).isoformat(),
            )
        except Exception as e:
            return HealthCheck(
                name="vectors",
                status=CheckStatus.FAIL,
                message=f"Vector check failed: {e}",
                duration_ms=(time.monotonic() - start) * 1000,
                checked_at=datetime.now(UTC).isoformat(),
                fix_hint="Check embeddings table and vector backend configuration",
            )

    async def _check_budget(self) -> HealthCheck:
        """Check budget utilization thresholds."""
        start = time.monotonic()
        if not self._budget_tracker:
            return HealthCheck(
                name="budget",
                status=CheckStatus.SKIP,
                message="Budget tracker not configured",
                checked_at=datetime.now(UTC).isoformat(),
            )

        remaining = self._budget_tracker.remaining_pct()
        spend = self._budget_tracker.total_spend()
        limit = self._budget_tracker.monthly_limit_usd

        if remaining <= 0:
            status = CheckStatus.FAIL
            msg = f"Budget EXHAUSTED: ${spend:.2f} / ${limit:.2f}"
            hint = "Increase monthly_limit_usd in settings or wait for month reset"
        elif remaining < 0.10:
            status = CheckStatus.WARN
            msg = f"Budget critical: {remaining:.0%} remaining (${spend:.2f} / ${limit:.2f})"
            hint = "Consider increasing budget limit"
        elif remaining < 0.20:
            status = CheckStatus.WARN
            msg = f"Budget low: {remaining:.0%} remaining (${spend:.2f} / ${limit:.2f})"
            hint = ""
        else:
            status = CheckStatus.PASS
            msg = f"Budget healthy: {remaining:.0%} remaining (${spend:.2f} / ${limit:.2f})"
            hint = ""

        return HealthCheck(
            name="budget",
            status=status,
            message=msg,
            duration_ms=(time.monotonic() - start) * 1000,
            checked_at=datetime.now(UTC).isoformat(),
            fix_hint=hint,
        )

    async def _check_services(self) -> HealthCheck:
        """Check that all registered services are running."""
        start = time.monotonic()
        if not self._supervisor:
            return HealthCheck(
                name="services",
                status=CheckStatus.SKIP,
                message="Supervisor not configured",
                checked_at=datetime.now(UTC).isoformat(),
            )

        services = list(self._supervisor.registry.all_services())
        running = sum(1 for s in services if s._running)
        stopped = len(services) - running

        if stopped > 0:
            return HealthCheck(
                name="services",
                status=CheckStatus.WARN,
                message=f"{stopped}/{len(services)} services not running",
                duration_ms=(time.monotonic() - start) * 1000,
                checked_at=datetime.now(UTC).isoformat(),
                fix_hint="Check service logs for startup errors",
            )

        return HealthCheck(
            name="services",
            status=CheckStatus.PASS,
            message=f"All {len(services)} services running",
            duration_ms=(time.monotonic() - start) * 1000,
            checked_at=datetime.now(UTC).isoformat(),
        )

    async def _check_circuit_breakers(self) -> list[HealthCheck]:
        """Check for circuit-broken services."""
        if not self._supervisor:
            return []

        checks: list[HealthCheck] = []
        broken = self._supervisor._circuit_broken

        if broken:
            for sid in broken:
                checks.append(HealthCheck(
                    name=f"circuit:{sid}",
                    status=CheckStatus.FAIL,
                    message=f"Service '{sid}' is circuit-broken",
                    checked_at=datetime.now(UTC).isoformat(),
                    fix_hint=f"POST /api/services/{sid}/reset-circuit to re-enable",
                ))
        else:
            checks.append(HealthCheck(
                name="circuit_breakers",
                status=CheckStatus.PASS,
                message="No circuit-broken services",
                checked_at=datetime.now(UTC).isoformat(),
            ))

        return checks

    async def _check_heartbeats(self) -> list[HealthCheck]:
        """Check for services with stale heartbeats."""
        if not self._supervisor:
            return []

        checks: list[HealthCheck] = []
        now = datetime.now(UTC)
        stale_threshold = timedelta(seconds=90)

        for service in self._supervisor.registry.all_services():
            sid = service.blueprint.service_id
            last_hb = self._supervisor._last_heartbeat.get(sid)

            if last_hb is None:
                # No heartbeat yet — might be newly started
                checks.append(HealthCheck(
                    name=f"heartbeat:{sid}",
                    status=CheckStatus.WARN,
                    message=f"No heartbeat received from '{sid}'",
                    checked_at=now.isoformat(),
                    fix_hint="Service may still be starting up",
                ))
            elif (now - last_hb) > stale_threshold:
                checks.append(HealthCheck(
                    name=f"heartbeat:{sid}",
                    status=CheckStatus.WARN,
                    message=(
                        f"Stale heartbeat from '{sid}': "
                        f"{(now - last_hb).total_seconds():.0f}s ago"
                    ),
                    checked_at=now.isoformat(),
                    fix_hint="Service may be stuck or overloaded",
                ))

        if not checks:
            checks.append(HealthCheck(
                name="heartbeats",
                status=CheckStatus.PASS,
                message="All service heartbeats current",
                checked_at=now.isoformat(),
            ))

        return checks

    # ── Publishing ───────────────────────────────────────────────────

    def _publish_report(self, report: HealthReport) -> None:
        """Publish health report to the bus."""
        # Individual check events
        for check in report.checks:
            if check.status in (CheckStatus.WARN, CheckStatus.FAIL):
                self._bus.publish(Envelope(
                    topic="system.health.check",
                    source_service_id="doctor",
                    payload=check.model_dump(mode="json"),
                ))

        # Summary report
        self._bus.publish(Envelope(
            topic="system.health.report",
            source_service_id="doctor",
            payload=report.model_dump(mode="json"),
        ))

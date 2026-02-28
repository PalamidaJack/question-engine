from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from qe.bus import get_bus
from qe.kernel.blueprint import load_blueprint
from qe.kernel.registry import ServiceRegistry
from qe.models.envelope import Envelope
from qe.runtime.agent_pool import AgentPerformanceTracker, AgentPool
from qe.runtime.budget import BudgetTracker
from qe.runtime.coordination import CoordinationProtocol
from qe.runtime.service import BaseService
from qe.runtime.tool_bootstrap import create_default_gate, create_default_registry
from qe.runtime.working_memory import WorkingMemory
from qe.services.hil import HILService
from qe.services.researcher import ResearcherService
from qe.services.validator import ClaimValidatorService

log = logging.getLogger(__name__)

# Loop detection defaults
_LOOP_WINDOW_SECONDS = 60
_LOOP_THRESHOLD = 5  # identical publications within window triggers circuit break
_CIRCUIT_COOLDOWN_SECONDS = 60


@dataclass
class CircuitState:
    status: str  # "closed" | "open" | "half_open"
    opened_at: datetime | None = None
    probe_envelope_id: str | None = None

    def should_probe(self, now: datetime) -> bool:
        return (
            self.status == "open"
            and self.opened_at is not None
            and (now - self.opened_at).total_seconds() >= _CIRCUIT_COOLDOWN_SECONDS
        )


class _ConfigWatcher(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop, callback: Any) -> None:
        super().__init__()
        self._loop = loop
        self._callback = callback

    def on_modified(self, event: Any) -> None:
        if event.is_directory:
            return
        if Path(event.src_path).name == "config.toml":
            asyncio.run_coroutine_threadsafe(self._callback(), self._loop)


class Supervisor:
    def __init__(
        self,
        bus: Any | None = None,
        substrate: Any | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.bus = bus or get_bus()
        self.substrate = substrate
        self.registry = ServiceRegistry()
        self.config_path = config_path or Path("config.toml")
        self._observer: Observer | None = None
        self._running = False
        self._daemon_tasks: list[asyncio.Task] = []
        # Budget tracker shared across all services
        db_path = None
        if self.substrate and hasattr(self.substrate, "belief_ledger"):
            db_path = self.substrate.belief_ledger._db_path
        self.budget_tracker = BudgetTracker(db_path=db_path)
        BaseService.set_budget_tracker(self.budget_tracker)
        # Tool registry and security gate
        self.tool_registry = create_default_registry()
        self.tool_gate = create_default_gate()
        BaseService.set_tool_registry(self.tool_registry)
        BaseService.set_tool_gate(self.tool_gate)
        # Multi-agent orchestration
        db_path_str = str(db_path) if db_path else None
        self.agent_performance_tracker = AgentPerformanceTracker(db_path=db_path_str)
        self.agent_pool = AgentPool()
        self.agent_pool.set_tracker(self.agent_performance_tracker)
        self.working_memory = WorkingMemory()
        self.coordination = CoordinationProtocol(self.bus)
        # Loop detection: service_id -> deque of (timestamp, payload_hash)
        self._pub_history: dict[str, deque] = {}
        # Circuit-broken services (half-open circuit breaker)
        self._circuits: dict[str, CircuitState] = {}
        # Last heartbeat per service
        self._last_heartbeat: dict[str, datetime] = {}

    @property
    def _circuit_broken(self) -> set[str]:
        """Backward-compatible view of circuit-broken service IDs."""
        return set(self._circuits.keys())

    async def start(self, genome_paths: list[Path]) -> None:
        self._running = True
        for genome_path in genome_paths:
            blueprint = load_blueprint(genome_path)
            try:
                service = self._instantiate_service(blueprint)
            except Exception:
                log.warning("Skipping genome %s: instantiation failed", genome_path.name)
                continue
            if not isinstance(service, BaseService):
                log.warning(
                    "Skipping genome %s: %s does not extend BaseService",
                    genome_path.name,
                    type(service).__name__,
                )
                continue
            service._handle_envelope = self._wrap_service_handler(service._handle_envelope, service)  # type: ignore[method-assign]
            self.registry.register(blueprint, service)
            await service.start()
            log.info(
                "[%s] subscribed to: %s",
                blueprint.service_id,
                blueprint.capabilities.bus_topics_subscribe,
            )

        # Load persisted budget state
        await self.budget_tracker.load()

        # Start multi-agent coordination
        await self.coordination.start()

        self._start_watchdog(asyncio.get_running_loop())
        self._start_daemons()
        # Subscribe to heartbeats for monitoring
        self.bus.subscribe("system.heartbeat", self._on_heartbeat)
        # Observe outbound publications for loop detection
        self.bus.add_publish_listener(self._on_publish)
        log.info("[READY]")

        shutdown_event = asyncio.Event()
        self._shutdown_event = shutdown_event
        await shutdown_event.wait()

    async def stop(self) -> None:
        self._running = False
        if hasattr(self, "_shutdown_event"):
            self._shutdown_event.set()
        for task in self._daemon_tasks:
            task.cancel()
        for task in self._daemon_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._daemon_tasks.clear()
        for service in self.registry.all_services():
            await service.stop()
        # Flush budget state before shutdown
        await self.budget_tracker.save()
        # Stop multi-agent coordination and flush agent metrics
        await self.coordination.stop()
        await self.agent_performance_tracker.flush()
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None

    async def _reconfigure_all(self) -> None:
        config = self._load_config()
        for service in self.registry.all_services():
            await service.reconfigure(config)

    def _load_config(self) -> dict[str, Any]:
        # Phase 0 minimal hot-reload payload
        return {"config_path": str(self.config_path)}

    def _start_watchdog(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self.config_path.exists():
            return
        watcher = _ConfigWatcher(loop, self._reconfigure_all)
        observer = Observer()
        observer.schedule(watcher, str(self.config_path.parent), recursive=False)
        observer.daemon = True
        observer.start()
        self._observer = observer

    def _instantiate_service(self, blueprint: Any) -> Any:
        # Dynamic import via service_class entrypoint
        if blueprint.service_class:
            module_path, class_name = blueprint.service_class.rsplit(":", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls(blueprint, self.bus, self.substrate)

        # Backwards-compatible prefix matching
        if blueprint.service_id.startswith("researcher"):
            return ResearcherService(blueprint, self.bus, self.substrate)
        if blueprint.service_id.startswith("validator"):
            return ClaimValidatorService(blueprint, self.bus, self.substrate)
        if blueprint.service_id.startswith("hil"):
            return HILService(blueprint, self.bus, self.substrate)
        raise ValueError(f"No service class dispatch for service_id={blueprint.service_id}")

    def _wrap_service_handler(self, handler: Any, service: Any) -> Any:
        sid = service.blueprint.service_id

        async def wrapped(envelope: Envelope) -> None:
            circuit = self._circuits.get(sid)

            if circuit is not None:
                now = datetime.now(UTC)
                if circuit.should_probe(now):
                    # Transition to half-open — allow one probe envelope
                    circuit.status = "half_open"
                    circuit.probe_envelope_id = envelope.envelope_id
                    log.info(
                        "circuit.half_open service=%s probe=%s",
                        sid,
                        envelope.envelope_id,
                    )
                elif circuit.status != "half_open":
                    log.warning(
                        "Dropping envelope for circuit-broken service %s", sid
                    )
                    return

            try:
                await handler(envelope)
                # If we were probing and succeeded, close the circuit
                if circuit is not None and circuit.status == "half_open":
                    log.info("circuit.closed service=%s (probe succeeded)", sid)
                    del self._circuits[sid]
            except Exception:
                # If probe failed, reopen the circuit
                if circuit is not None and circuit.status == "half_open":
                    circuit.status = "open"
                    circuit.opened_at = datetime.now(UTC)
                    circuit.probe_envelope_id = None
                    log.warning("circuit.reopened service=%s (probe failed)", sid)

                tb = traceback.format_exc()
                log.error(
                    "Unhandled service exception service=%s "
                    "envelope_id=%s\n%s",
                    sid,
                    envelope.envelope_id,
                    tb,
                )
                self.bus.publish(
                    Envelope(
                        topic="system.error",
                        source_service_id=sid,
                        correlation_id=envelope.envelope_id,
                        causation_id=envelope.envelope_id,
                        payload={
                            "envelope_id": envelope.envelope_id,
                            "topic": envelope.topic,
                            "error": tb,
                        },
                    )
                )

        return wrapped

    def _on_publish(self, envelope: Envelope) -> None:
        """Observe every outbound publication for loop detection."""
        sid = envelope.source_service_id
        if sid and sid != "supervisor":
            self._check_loop(sid, envelope)

    # ------------------------------------------------------------------
    # Loop Detection
    # ------------------------------------------------------------------

    def _check_loop(self, service_id: str, envelope: Envelope) -> None:
        """Detect repeated identical publications from a service."""
        now = datetime.now(UTC)
        payload_hash = hashlib.md5(  # noqa: S324
            json.dumps(envelope.payload, sort_keys=True).encode()
        ).hexdigest()
        key = f"{envelope.topic}:{payload_hash}"

        if service_id not in self._pub_history:
            self._pub_history[service_id] = deque(maxlen=100)

        history = self._pub_history[service_id]
        history.append((now, key))

        # Count identical keys within window
        cutoff = now - timedelta(seconds=_LOOP_WINDOW_SECONDS)
        recent = [ts for ts, k in history if k == key and ts > cutoff]

        if len(recent) >= _LOOP_THRESHOLD:
            log.error(
                "Loop detected for service %s: %d identical publications "
                "in %ds — triggering circuit break",
                service_id,
                len(recent),
                _LOOP_WINDOW_SECONDS,
            )
            self._circuits[service_id] = CircuitState(
                status="open", opened_at=datetime.now(UTC)
            )
            self.bus.publish(
                Envelope(
                    topic="system.circuit_break",
                    source_service_id="supervisor",
                    payload={
                        "service_id": service_id,
                        "reason": "loop_detected",
                        "count": len(recent),
                        "window_seconds": _LOOP_WINDOW_SECONDS,
                    },
                )
            )

    # ------------------------------------------------------------------
    # Background Daemons
    # ------------------------------------------------------------------

    def _start_daemons(self) -> None:
        """Launch background daemon tasks."""
        self._daemon_tasks.append(
            asyncio.create_task(self._heartbeat_monitor())
        )
        self._daemon_tasks.append(
            asyncio.create_task(self._stall_detector())
        )
        self._daemon_tasks.append(
            asyncio.create_task(self._budget_flush_loop())
        )
        self._daemon_tasks.append(
            asyncio.create_task(self._agent_metrics_flush_loop())
        )

    async def _on_heartbeat(self, envelope: Envelope) -> None:
        """Record heartbeat timestamp for a service."""
        sid = envelope.source_service_id
        self._last_heartbeat[sid] = datetime.now(UTC)

    async def _heartbeat_monitor(self) -> None:
        """Check for services that stopped sending heartbeats."""
        while self._running:
            await asyncio.sleep(60)
            now = datetime.now(UTC)
            for service in self.registry.all_services():
                sid = service.blueprint.service_id
                last = self._last_heartbeat.get(sid)
                if last and (now - last) > timedelta(seconds=90):
                    log.warning(
                        "Service %s missed heartbeat (last: %s)",
                        sid,
                        last.isoformat(),
                    )
                    self.bus.publish(
                        Envelope(
                            topic="system.service_stalled",
                            source_service_id="supervisor",
                            payload={
                                "service_id": sid,
                                "last_heartbeat": last.isoformat(),
                                "seconds_since": (now - last).total_seconds(),
                            },
                        )
                    )

    async def _stall_detector(self) -> None:
        """Monitor for services with no bus activity for extended periods."""
        while self._running:
            await asyncio.sleep(300)  # Check every 5 minutes
            now = datetime.now(UTC)
            for service in self.registry.all_services():
                sid = service.blueprint.service_id
                if sid in self._circuit_broken:
                    continue
                last = self._last_heartbeat.get(sid)
                if last and (now - last) > timedelta(minutes=10):
                    log.error(
                        "Service %s appears stalled — no activity for 10+ "
                        "minutes",
                        sid,
                    )
                    self.bus.publish(
                        Envelope(
                            topic="system.service_stalled",
                            source_service_id="supervisor",
                            payload={
                                "service_id": sid,
                                "reason": "no_activity",
                                "last_heartbeat": (
                                    last.isoformat() if last else None
                                ),
                            },
                        )
                    )

    async def _budget_flush_loop(self) -> None:
        """Periodically flush budget records to SQLite."""
        while self._running:
            await asyncio.sleep(30)
            try:
                await self.budget_tracker.save()
            except Exception:
                log.exception("Failed to flush budget records")

    async def _agent_metrics_flush_loop(self) -> None:
        """Periodically flush agent performance metrics to SQLite."""
        while self._running:
            await asyncio.sleep(30)
            try:
                await self.agent_performance_tracker.flush()
            except Exception:
                log.exception("Failed to flush agent metrics")

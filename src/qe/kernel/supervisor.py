from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import traceback
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from qe.bus import get_bus
from qe.kernel.blueprint import load_blueprint
from qe.kernel.registry import ServiceRegistry
from qe.models.envelope import Envelope
from qe.runtime.budget import BudgetTracker
from qe.runtime.service import BaseService
from qe.services.hil import HILService
from qe.services.researcher import ResearcherService

log = logging.getLogger(__name__)

# Loop detection defaults
_LOOP_WINDOW_SECONDS = 60
_LOOP_THRESHOLD = 5  # identical publications within window triggers circuit break


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
        self.budget_tracker = BudgetTracker()
        BaseService.set_budget_tracker(self.budget_tracker)
        # Loop detection: service_id -> deque of (timestamp, payload_hash)
        self._pub_history: dict[str, deque] = {}
        # Circuit-broken services
        self._circuit_broken: set[str] = set()
        # Last heartbeat per service
        self._last_heartbeat: dict[str, datetime] = {}

    async def start(self, genome_paths: list[Path]) -> None:
        self._running = True
        for genome_path in genome_paths:
            blueprint = load_blueprint(genome_path)
            service = self._instantiate_service(blueprint)
            service._handle_envelope = self._wrap_service_handler(service._handle_envelope, service)  # type: ignore[method-assign]
            self.registry.register(blueprint, service)
            await service.start()
            log.info(
                "[%s] subscribed to: %s",
                blueprint.service_id,
                blueprint.capabilities.bus_topics_subscribe,
            )

        self._start_watchdog(asyncio.get_running_loop())
        self._start_daemons()
        # Subscribe to heartbeats for monitoring
        self.bus.subscribe("system.heartbeat", self._on_heartbeat)
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
        if blueprint.service_id.startswith("researcher"):
            return ResearcherService(blueprint, self.bus, self.substrate)
        if blueprint.service_id.startswith("hil"):
            return HILService(blueprint, self.bus, self.substrate)
        raise ValueError(f"No service class dispatch for service_id={blueprint.service_id}")

    def _wrap_service_handler(self, handler: Any, service: Any) -> Any:
        sid = service.blueprint.service_id

        async def wrapped(envelope: Envelope) -> None:
            # Circuit breaker check
            if sid in self._circuit_broken:
                log.warning(
                    "Dropping envelope for circuit-broken service %s", sid
                )
                return

            try:
                await handler(envelope)
            except Exception:
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

            # Loop detection: check publications from this service
            self._check_loop(sid, envelope)

        return wrapped

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
            self._circuit_broken.add(service_id)
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

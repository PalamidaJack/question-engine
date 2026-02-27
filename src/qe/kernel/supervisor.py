import asyncio
import sys
import traceback
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from qe.bus import get_bus
from qe.models.envelope import Envelope
from qe.kernel.blueprint import load_blueprint
from qe.kernel.registry import ServiceRegistry
from qe.services.hil import HILService
from qe.services.researcher import ResearcherService


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
    def __init__(self, bus: Any | None = None, substrate: Any | None = None, config_path: Path | None = None) -> None:
        self.bus = bus or get_bus()
        self.substrate = substrate
        self.registry = ServiceRegistry()
        self.config_path = config_path or Path("config.toml")
        self._observer: Observer | None = None
        self._running = False

    async def start(self, genome_paths: list[Path]) -> None:
        self._running = True
        for genome_path in genome_paths:
            blueprint = load_blueprint(genome_path)
            service = self._instantiate_service(blueprint)
            service._handle_envelope = self._wrap_service_handler(service._handle_envelope, service)  # type: ignore[method-assign]
            self.registry.register(blueprint, service)
            await service.start()
            print(f"[{blueprint.service_id}] subscribed to: {blueprint.capabilities.bus_topics_subscribe}")

        self._start_watchdog(asyncio.get_running_loop())
        print("[READY]")

        while self._running:
            await asyncio.sleep(3600)

    async def stop(self) -> None:
        self._running = False
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
        async def wrapped(envelope: Envelope) -> None:
            try:
                await handler(envelope)
            except Exception:
                tb = traceback.format_exc()
                print(
                    f"Unhandled service exception service={service.blueprint.service_id} "
                    f"envelope_id={envelope.envelope_id}\n{tb}",
                    file=sys.stderr,
                )
                self.bus.publish(
                    Envelope(
                        topic="system.error",
                        source_service_id=service.blueprint.service_id,
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

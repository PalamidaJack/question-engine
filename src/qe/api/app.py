"""FastAPI application for Question Engine OS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from qe.api.middleware import AuthMiddleware, RateLimitMiddleware, RequestTimingMiddleware
from qe.api.endpoints.goals import register_goal_routes
from qe.api.endpoints.memory import register_memory_routes
from qe.api.setup import (
    PROVIDERS,
    get_configured_providers,
    get_current_tiers,
    get_settings,
    is_setup_complete,
    save_settings,
    save_setup,
)
from qe.api.ws import ConnectionManager
from qe.audit import get_audit_log
from qe.bus import get_bus
from qe.bus.bus_metrics import get_bus_metrics
from qe.bus.event_log import EventLog
from qe.kernel.supervisor import Supervisor
from qe.models.envelope import Envelope
from qe.runtime.logging_config import configure_from_config, update_log_level
from qe.runtime.metrics import get_metrics
from qe.runtime.readiness import get_readiness
from qe.services.chat import ChatService
from qe.services.dispatcher import Dispatcher
from qe.services.doctor import DoctorService
from qe.services.executor import ExecutorService
from qe.services.planner import PlannerService
from qe.services.query import answer_question
from qe.substrate import Substrate
from qe.substrate.goal_store import GoalStore
from qe.substrate.memory_store import MemoryStore

load_dotenv()

log = logging.getLogger(__name__)

ws_manager = ConnectionManager()

# Global references set during lifespan
_supervisor: Supervisor | None = None
_substrate: Substrate | None = None
_supervisor_task: asyncio.Task | None = None
_event_log: EventLog | None = None
_chat_service: ChatService | None = None
_planner: PlannerService | None = None
_dispatcher: Dispatcher | None = None
_executor: ExecutorService | None = None
_goal_store: GoalStore | None = None
_doctor: DoctorService | None = None
_memory_store: MemoryStore | None = None
_extra_routes_registered = False

INBOX_DIR = Path("data/runtime_inbox")


def _genome_paths() -> list[Path]:
    genomes_dir = Path("genomes")
    if not genomes_dir.exists():
        return []
    return sorted(genomes_dir.glob("*.toml"))


async def _inbox_relay_loop() -> None:
    """Relay cross-process submissions into the in-memory bus."""
    bus = get_bus()
    INBOX_DIR.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

    while True:
        for item in sorted(INBOX_DIR.glob("*.json")):  # noqa: ASYNC240
            try:
                payload = json.loads(item.read_text(encoding="utf-8"))
                env = Envelope.model_validate(payload)
                bus.publish(env)
            finally:
                item.unlink(missing_ok=True)
        await asyncio.sleep(0.5)


def _bus_to_ws_bridge() -> None:
    """Subscribe to all bus topics and forward events to WebSocket clients."""
    from qe.bus.protocol import TOPICS

    bus = get_bus()

    async def _forward(envelope: Envelope) -> None:
        event = {
            "type": "bus_event",
            "envelope_id": envelope.envelope_id,
            "topic": envelope.topic,
            "source_service_id": envelope.source_service_id,
            "timestamp": envelope.timestamp.isoformat(),
            "payload": envelope.payload,
        }
        await ws_manager.broadcast(json.dumps(event))

    for topic in TOPICS:
        bus.subscribe(topic, _forward)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the QE engine on app startup, shut down on teardown."""
    global _supervisor, _substrate, _supervisor_task, _event_log, _chat_service
    global _planner, _dispatcher, _executor, _goal_store, _doctor
    global _memory_store, _extra_routes_registered

    configure_from_config(get_settings())

    bus = get_bus()

    # Initialize durable event log
    readiness = get_readiness()

    _event_log = EventLog()
    await _event_log.initialize()
    bus.set_event_log(_event_log)
    readiness.mark_ready("event_log_ready")

    _substrate = Substrate()
    await _substrate.initialize()
    _memory_store = MemoryStore(_substrate.belief_ledger._db_path)
    _substrate.set_memory_store(_memory_store)

    if not _extra_routes_registered:
        register_goal_routes(
            app=app,
            planner=_planner,
            dispatcher=_dispatcher,
            goal_store=_goal_store,
        )
        register_memory_routes(
            app=app,
            memory_store=_memory_store,
        )
        _extra_routes_registered = True

    readiness.mark_ready("substrate_ready")

    relay_task: asyncio.Task | None = None

    if is_setup_complete():
        _supervisor = Supervisor(
            bus=bus, substrate=_substrate, config_path=Path("config.toml")
        )

        _bus_to_ws_bridge()

        relay_task = asyncio.create_task(_inbox_relay_loop())
        _supervisor_task = asyncio.create_task(
            _supervisor.start(_genome_paths())
        )
        balanced_model = get_current_tiers().get("balanced", "gpt-4o")
        _chat_service = ChatService(
            substrate=_substrate,
            bus=bus,
            budget_tracker=_supervisor.budget_tracker,
            model=balanced_model,
        )
        _goal_store = GoalStore(_substrate.belief_ledger._db_path)
        _planner = PlannerService(
            bus=bus,
            substrate=_substrate,
            budget_tracker=_supervisor.budget_tracker,
            model=balanced_model,
        )
        _dispatcher = Dispatcher(bus=bus, goal_store=_goal_store)

        # Wire dispatcher to receive subtask completion/failure events
        async def _on_task_result(envelope: Envelope) -> None:
            from qe.models.goal import SubtaskResult as _SR

            result = _SR.model_validate(envelope.payload)
            await _dispatcher.handle_subtask_completed(result.goal_id, result)

        bus.subscribe("tasks.completed", _on_task_result)
        bus.subscribe("tasks.failed", _on_task_result)

        # Start the executor service that processes dispatched tasks
        _executor = ExecutorService(
            bus=bus,
            substrate=_substrate,
            budget_tracker=_supervisor.budget_tracker,
            model=balanced_model,
        )
        await _executor.start()

        # Wire event log into substrate for MAGMA temporal/causal queries
        _substrate.set_event_log(_event_log)
        # Start Doctor health monitoring service
        _doctor = DoctorService(
            bus=bus,
            substrate=_substrate,
            supervisor=_supervisor,
            event_log=_event_log,
            budget_tracker=_supervisor.budget_tracker,
        )
        await _doctor.start()
        # Reconcile in-flight goals from previous run
        await _dispatcher.reconcile()

        readiness.mark_ready("services_subscribed")
        readiness.mark_ready("supervisor_ready")
        log.info("QE API server started (engine running)")
    else:
        log.info("QE API server started (setup required — no API keys configured)")

    yield

    # Shutdown
    if _executor:
        await _executor.stop()
    if _doctor:
        await _doctor.stop()
    if _supervisor:
        await _supervisor.stop()
    if relay_task:
        relay_task.cancel()
    if _supervisor_task:
        _supervisor_task.cancel()
    log.info("QE API server stopped")


app = FastAPI(
    title="Question Engine OS",
    version="0.1.0",
    lifespan=lifespan,
)

_cors_origins = (
    [o.strip() for o in os.environ["QE_CORS_ORIGINS"].split(",")]
    if os.environ.get("QE_CORS_ORIGINS")
    else ["http://localhost:8000", "http://127.0.0.1:8000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(RateLimitMiddleware, rpm=120, burst=20)
app.add_middleware(AuthMiddleware)
try:
    from qe.api.endpoints.ingest import router as ingest_router

    app.include_router(ingest_router)
except RuntimeError as exc:
    # FastAPI raises RuntimeError when python-multipart is missing.
    log.warning("Ingest routes disabled: %s", exc)

# Serve dashboard
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ── Dashboard ───────────────────────────────────────────────────────────────


@app.get("/")
async def dashboard():
    """Serve the dashboard UI."""
    return FileResponse(str(_static_dir / "index.html"))


# ── Setup Endpoints ─────────────────────────────────────────────────────────


@app.get("/api/setup/status")
async def setup_status():
    """Return setup status: whether complete, configured providers, tier mapping."""
    return {
        "complete": is_setup_complete(),
        "providers": get_configured_providers(),
        "tiers": get_current_tiers(),
    }


@app.get("/api/setup/providers")
async def setup_providers():
    """Return the static list of supported providers."""
    return {
        "providers": [
            {
                "name": p["name"],
                "env_var": p["env_var"],
                "example_models": p["example_models"],
                "tier_defaults": p["tier_defaults"],
            }
            for p in PROVIDERS
        ],
    }


@app.post("/api/setup/save")
async def setup_save(body: dict[str, Any]):
    """Save provider API keys and tier assignments.

    Expects:
        {
            "providers": {"OPENAI_API_KEY": "sk-...", ...},
            "tiers": {
                "fast": {"provider": "OpenAI", "model": "gpt-4o-mini"},
                "balanced": {"provider": "OpenAI", "model": "gpt-4o"},
                "powerful": {"provider": "Anthropic", "model": "claude-sonnet-4-20250514"}
            }
        }
    """
    providers = body.get("providers", {})
    tiers = body.get("tiers", {})

    if not providers and not tiers:
        return JSONResponse(
            {"error": "providers or tiers required"}, status_code=400
        )

    save_setup(providers=providers, tier_config=tiers)
    return {"status": "saved", "complete": is_setup_complete()}


# ── Settings ─────────────────────────────────────────────────────────────


@app.get("/api/settings")
async def get_settings_endpoint():
    """Return runtime settings from config.toml."""
    return get_settings()


@app.post("/api/settings")
async def save_settings_endpoint(body: dict[str, Any]):
    """Save runtime settings and apply budget changes at runtime."""
    save_settings(body)

    # Apply budget limits at runtime if supervisor is running
    budget_vals = body.get("budget")
    if budget_vals and isinstance(budget_vals, dict) and _supervisor:
        _supervisor.budget_tracker.update_limits(
            monthly_limit_usd=budget_vals.get("monthly_limit_usd"),
            alert_at_pct=budget_vals.get("alert_at_pct"),
        )

    # Apply log level at runtime without restart
    runtime_vals = body.get("runtime")
    if runtime_vals and isinstance(runtime_vals, dict):
        if "log_level" in runtime_vals:
            update_log_level(runtime_vals["log_level"])

    get_audit_log().record("settings.update", resource="config", detail=body)
    return {"status": "saved"}


@app.post("/api/optimize/{genome_id}")
async def optimize_genome(genome_id: str):
    """Run DSPy prompt optimization on a genome using calibration data."""
    if not _supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    # Find the genome path
    genome_path = None
    for p in _genome_paths():
        import tomllib
        with p.open("rb") as f:
            g = tomllib.load(f)
        if g.get("service_id") == genome_id:
            genome_path = p
            break

    if not genome_path:
        return JSONResponse(
            {"error": f"Genome '{genome_id}' not found"}, status_code=404
        )

    from qe.optimization.prompt_tuner import PromptTuner
    from qe.runtime.calibration import CalibrationTracker

    # Get or create calibration tracker
    cal = CalibrationTracker(
        db_path=_substrate.belief_ledger._db_path if _substrate else None
    )

    tuner = PromptTuner(cal, _substrate)
    result = await tuner.optimize_genome(genome_id, genome_path)
    return result.to_dict()


@app.post("/api/services/{service_id}/reset-circuit")
async def reset_circuit_breaker(service_id: str):
    """Reset a circuit-broken service so it can resume processing."""
    if not _supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    # Check service exists
    found = any(
        s.blueprint.service_id == service_id
        for s in _supervisor.registry.all_services()
    )
    if not found:
        return JSONResponse(
            {"error": f"Service '{service_id}' not found"}, status_code=404
        )

    _supervisor._circuits.pop(service_id, None)
    _supervisor._pub_history.pop(service_id, None)
    get_audit_log().record(
        "circuit.reset", resource=f"service/{service_id}"
    )
    return {"status": "reset", "service_id": service_id}


# ── Dead Letter Queue ──────────────────────────────────────────────────────


@app.get("/api/dlq")
async def list_dlq(limit: int = 100):
    """List dead-letter queue entries."""
    bus = get_bus()
    return {
        "entries": bus.dlq_list(limit=limit),
        "count": bus.dlq_size(),
    }


@app.post("/api/dlq/{envelope_id}/replay")
async def replay_dlq(envelope_id: str):
    """Replay a dead-lettered envelope back into the bus."""
    bus = get_bus()
    ok = await bus.dlq_replay(envelope_id)
    if not ok:
        return JSONResponse(
            {"error": f"Envelope '{envelope_id}' not found in DLQ"},
            status_code=404,
        )
    return {"status": "replayed", "envelope_id": envelope_id}


@app.delete("/api/dlq")
async def purge_dlq():
    """Purge all entries from the dead-letter queue."""
    bus = get_bus()
    count = await bus.dlq_purge()
    return {"status": "purged", "count": count}


# ── Audit Trail ────────────────────────────────────────────────────────────


@app.get("/api/audit")
async def list_audit(
    action: str | None = None,
    actor: str | None = None,
    limit: int = 100,
):
    """Query the admin audit trail."""
    entries = get_audit_log().query(action=action, actor=actor, limit=limit)
    return {"entries": entries, "count": len(entries)}


# ── Metrics ───────────────────────────────────────────────────────────────


@app.get("/api/metrics")
async def metrics_snapshot():
    """Return full metrics snapshot: counters, histograms, gauges, SLOs."""
    return get_metrics().snapshot()


# ── Bus Stats ─────────────────────────────────────────────────────────────


@app.get("/api/bus/stats")
async def bus_stats():
    """Return per-topic bus metrics: publish counts, latency, errors."""
    return get_bus_metrics().snapshot()


# ── Topology ──────────────────────────────────────────────────────────────


@app.get("/api/topology")
async def topology():
    """Return service dependency graph from blueprint declarations."""
    if not _supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    services = []
    all_topics: set[str] = set()

    for svc in _supervisor.registry.all_services():
        bp = svc.blueprint
        subs = bp.capabilities.bus_topics_subscribe
        pubs = bp.capabilities.bus_topics_publish
        all_topics.update(subs)
        all_topics.update(pubs)
        services.append({
            "service_id": bp.service_id,
            "display_name": bp.display_name,
            "subscribes": subs,
            "publishes": pubs,
        })

    return {
        "services": services,
        "topics": sorted(all_topics),
        "service_count": len(services),
        "topic_count": len(all_topics),
    }


# ── Event Replay ──────────────────────────────────────────────────────────


@app.post("/api/events/replay")
async def replay_events(body: dict[str, Any]):
    """Bulk replay historical events from the event log back into the bus.

    Accepts: {"topic": "...", "since": "ISO8601", "limit": 100}
    """
    if not _event_log:
        return JSONResponse(
            {"error": "Event log not ready"}, status_code=503
        )

    topic = body.get("topic")
    since_str = body.get("since")
    limit = body.get("limit", 100)

    since = None
    if since_str:
        since = datetime.fromisoformat(since_str)

    events = await _event_log.replay(since=since, topic=topic, limit=limit)

    bus = get_bus()
    replayed = 0
    for event in events:
        env = Envelope(
            envelope_id=event["envelope_id"],
            schema_version=event.get("schema_version") or "1.0",
            topic=event["topic"],
            source_service_id=event["source_service_id"],
            correlation_id=event.get("correlation_id"),
            causation_id=event.get("causation_id"),
            timestamp=datetime.fromisoformat(event["timestamp"]),
            payload=event["payload"],
            ttl_seconds=event.get("ttl_seconds"),
        )
        bus.publish(env)
        replayed += 1

    get_audit_log().record(
        "events.replayed",
        detail={"topic": topic, "since": since_str, "count": replayed},
    )
    return {"status": "replayed", "count": replayed}


# ── REST Endpoints ──────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/api/health/ready")
async def health_ready():
    """Readiness probe: returns 200 when fully initialized, 503 during startup."""
    readiness = get_readiness()
    status_data = readiness.to_dict()
    if readiness.is_ready:
        return status_data
    return JSONResponse(status_data, status_code=503)


@app.get("/api/health/live")
async def health_live():
    """Live health report from the Doctor service."""
    if not _doctor:
        return JSONResponse({"error": "Doctor service not running"}, status_code=503)

    report = _doctor.last_report
    if report is None:
        # First check hasn't run yet — run on demand
        report = await _doctor.run_all_checks()

    return report.model_dump(mode="json")


@app.get("/api/status")
async def status():
    """Overall engine status: services, budget, circuit breakers."""
    if not _supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    services = []
    for svc in _supervisor.registry.all_services():
        sid = svc.blueprint.service_id
        services.append({
            "service_id": sid,
            "display_name": svc.blueprint.display_name,
            "status": "alive" if svc._running else "stopped",
            "turn_count": svc._turn_count,
            "circuit_broken": sid in _supervisor._circuits,
        })

    return {
        "services": services,
        "budget": {
            "total_spend": _supervisor.budget_tracker.total_spend(),
            "remaining_pct": _supervisor.budget_tracker.remaining_pct(),
            "limit_usd": _supervisor.budget_tracker.monthly_limit_usd,
            "by_model": _supervisor.budget_tracker.spend_by_model(),
        },
    }


@app.post("/api/submit")
async def submit(body: dict[str, Any]):
    """Submit an observation to the engine."""
    text = body.get("text", "")
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    envelope = Envelope(
        topic="observations.structured",
        source_service_id="api",
        payload={"text": text},
    )

    get_bus().publish(envelope)

    # Also write to inbox for cross-process relay
    INBOX_DIR.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    inbox_file = INBOX_DIR / f"{envelope.envelope_id}.json"
    inbox_file.write_text(envelope.model_dump_json(), encoding="utf-8")

    return {"envelope_id": envelope.envelope_id, "status": "submitted"}


@app.post("/api/ask")
async def ask(body: dict[str, Any]):
    """Ask a question against the belief ledger."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    question = body.get("question", "")
    if not question:
        return JSONResponse(
            {"error": "question is required"}, status_code=400
        )

    result = await answer_question(question, _substrate)
    return result


@app.get("/api/claims")
async def list_claims(
    subject: str | None = None,
    include_superseded: bool = False,
):
    """List claims from the belief ledger."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    claims = await _substrate.get_claims(
        subject_entity_id=subject,
        include_superseded=include_superseded,
    )
    return {
        "claims": [c.model_dump(mode="json") for c in claims],
        "count": len(claims),
    }


@app.get("/api/claims/{claim_id}")
async def get_claim(claim_id: str):
    """Get a specific claim by ID."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    claim = await _substrate.get_claim_by_id(claim_id)
    if not claim:
        return JSONResponse({"error": "Claim not found"}, status_code=404)
    return claim.model_dump(mode="json")


@app.delete("/api/claims/{claim_id}")
async def retract_claim(claim_id: str):
    """Soft-retract a claim (mark as superseded by 'retracted')."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    retracted = await _substrate.retract_claim(claim_id)
    if not retracted:
        return JSONResponse({"error": "Claim not found"}, status_code=404)
    return {"status": "retracted", "claim_id": claim_id}


@app.get("/api/entities")
async def list_entities():
    """List all entities with claim counts."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    entities = await _substrate.entity_resolver.list_entities()
    # Enrich with claim counts
    for ent in entities:
        claims = await _substrate.get_claims(
            subject_entity_id=ent["canonical_name"]
        )
        ent["claim_count"] = len(claims)
    return {"entities": entities, "count": len(entities)}


@app.get("/api/entities/{name}")
async def get_entity(name: str):
    """Get claims for a specific entity."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    canonical = await _substrate.entity_resolver.resolve(name)
    claims = await _substrate.get_claims(subject_entity_id=canonical)
    return {
        "canonical_name": canonical,
        "claims": [c.model_dump(mode="json") for c in claims],
        "count": len(claims),
    }


@app.post("/api/entities/{name}/alias")
async def add_entity_alias(name: str, body: dict[str, Any]):
    """Add an alias for an entity."""
    if not _substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    alias = body.get("alias", "")
    if not alias:
        return JSONResponse({"error": "alias is required"}, status_code=400)

    await _substrate.entity_resolver.add_alias(name, alias)
    return {"status": "alias_added", "canonical_name": name, "alias": alias}


@app.get("/api/events")
async def list_events(
    topic: str | None = None,
    limit: int = 100,
):
    """Query the durable event log."""
    if not _event_log:
        return JSONResponse({"error": "Event log not ready"}, status_code=503)

    events = await _event_log.replay(topic=topic, limit=limit)
    return {"events": events, "count": len(events)}


@app.get("/api/hil/pending")
async def hil_pending():
    """List pending HIL approval requests."""
    pending_dir = Path("data/hil_queue/pending")
    pending_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240

    items = []
    for file in sorted(pending_dir.glob("*.json")):  # noqa: ASYNC240
        payload = json.loads(file.read_text(encoding="utf-8"))
        items.append(payload)
    return {"pending": items, "count": len(items)}


@app.post("/api/hil/{envelope_id}/approve")
async def hil_approve(envelope_id: str):
    """Approve a pending HIL request."""
    completed_dir = Path("data/hil_queue/completed")
    completed_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    decision_file = completed_dir / f"{envelope_id}.json"
    decision_file.write_text(
        json.dumps({
            "decision": "approved",
            "decided_at": datetime.now(UTC).isoformat(),
        }, indent=2),
        encoding="utf-8",
    )
    get_audit_log().record(
        "hil.approve", resource=f"envelope/{envelope_id}"
    )
    return {"status": "approved", "envelope_id": envelope_id}


@app.post("/api/hil/{envelope_id}/reject")
async def hil_reject(envelope_id: str, body: dict[str, Any] | None = None):
    """Reject a pending HIL request."""
    reason = (body or {}).get("reason", "rejected")
    completed_dir = Path("data/hil_queue/completed")
    completed_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    decision_file = completed_dir / f"{envelope_id}.json"
    decision_file.write_text(
        json.dumps({
            "decision": "rejected",
            "reason": reason,
            "decided_at": datetime.now(UTC).isoformat(),
        }, indent=2),
        encoding="utf-8",
    )
    get_audit_log().record(
        "hil.reject",
        resource=f"envelope/{envelope_id}",
        detail={"reason": reason},
    )
    return {"status": "rejected", "envelope_id": envelope_id}


# ── Goals ───────────────────────────────────────────────────────────────────


@app.post("/api/goals")
async def submit_goal(body: dict[str, Any]):
    """Submit a new goal for decomposition and execution."""
    if not _planner or not _dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    description = body.get("description", "").strip()
    if not description:
        return JSONResponse(
            {"error": "description is required"}, status_code=400
        )

    state = await _planner.decompose(description)
    await _dispatcher.submit_goal(state)

    get_bus().publish(
        Envelope(
            topic="goals.submitted",
            source_service_id="api",
            correlation_id=state.goal_id,
            payload={
                "goal_id": state.goal_id,
                "description": description,
            },
        )
    )

    return {
        "goal_id": state.goal_id,
        "status": state.status,
        "subtask_count": len(state.subtask_states),
        "strategy": (
            state.decomposition.strategy if state.decomposition else ""
        ),
    }


@app.get("/api/goals")
async def list_goals(status: str | None = None):
    """List all goals with status."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    goals = await _goal_store.list_goals(status=status)
    return {
        "goals": [
            {
                "goal_id": g.goal_id,
                "description": g.description,
                "status": g.status,
                "subtask_count": len(g.subtask_states),
                "created_at": g.created_at.isoformat(),
                "completed_at": (
                    g.completed_at.isoformat()
                    if g.completed_at
                    else None
                ),
            }
            for g in goals
        ],
        "count": len(goals),
    }


@app.get("/api/goals/{goal_id}")
async def get_goal(goal_id: str):
    """Get goal detail with DAG and subtask states."""
    if not _dispatcher or not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    # Try in-memory first, then store
    state = _dispatcher.get_goal_state(goal_id)
    if not state:
        state = await _goal_store.load_goal(goal_id)
    if not state:
        return JSONResponse({"error": "Goal not found"}, status_code=404)

    return {
        "goal_id": state.goal_id,
        "description": state.description,
        "status": state.status,
        "subtask_states": state.subtask_states,
        "created_at": state.created_at.isoformat(),
        "completed_at": (
            state.completed_at.isoformat()
            if state.completed_at
            else None
        ),
        "decomposition": (
            state.decomposition.model_dump(mode="json")
            if state.decomposition
            else None
        ),
    }


@app.post("/api/goals/{goal_id}/pause")
async def pause_goal(goal_id: str):
    """Pause execution of a goal."""
    if not _dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)
    ok = await _dispatcher.pause_goal(goal_id)
    if not ok:
        return JSONResponse(
            {"error": "Goal not found or not running"}, status_code=404
        )
    return {"status": "paused", "goal_id": goal_id}


@app.post("/api/goals/{goal_id}/resume")
async def resume_goal(goal_id: str):
    """Resume a paused goal."""
    if not _dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)
    ok = await _dispatcher.resume_goal(goal_id)
    if not ok:
        return JSONResponse(
            {"error": "Goal not found or not paused"}, status_code=404
        )
    return {"status": "resumed", "goal_id": goal_id}


@app.post("/api/goals/{goal_id}/assign")
async def assign_goal_to_project(goal_id: str, body: dict[str, Any]):
    """Assign a goal to a project."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    project_id = body.get("project_id")
    if not project_id:
        return JSONResponse(
            {"error": "project_id is required"}, status_code=400
        )

    state = await _goal_store.load_goal(goal_id)
    if not state:
        return JSONResponse({"error": "Goal not found"}, status_code=404)

    state.project_id = project_id
    await _goal_store.save_goal(state)
    return {"status": "assigned", "goal_id": goal_id, "project_id": project_id}


@app.post("/api/goals/{goal_id}/cancel")
async def cancel_goal(goal_id: str):
    """Cancel a running goal."""
    if not _dispatcher:
        return JSONResponse({"error": "Engine not started"}, status_code=503)
    ok = await _dispatcher.cancel_goal(goal_id)
    if not ok:
        return JSONResponse(
            {"error": "Goal not found"}, status_code=404
        )
    return {"status": "cancelled", "goal_id": goal_id}


# ── Projects ─────────────────────────────────────────────────────────────────


@app.get("/api/projects")
async def list_projects(status: str | None = None):
    """List all projects with goal counts."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    projects = await _goal_store.list_projects(status=status)
    result = []
    for p in projects:
        goals = await _goal_store.get_project_goals(p.project_id)
        result.append({
            **p.model_dump(mode="json"),
            "goal_count": len(goals),
            "completed_goals": sum(1 for g in goals if g.status == "completed"),
        })
    return {"projects": result, "count": len(result)}


@app.post("/api/projects")
async def create_project(body: dict[str, Any]):
    """Create a new project."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    from qe.models.goal import Project

    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    project = Project(
        name=name,
        description=body.get("description", ""),
        owner=body.get("owner", ""),
        tags=body.get("tags", []),
    )
    await _goal_store.save_project(project)
    return project.model_dump(mode="json")


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get project detail with goals and metrics."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    project = await _goal_store.get_project(project_id)
    if not project:
        return JSONResponse({"error": "Project not found"}, status_code=404)

    goals = await _goal_store.get_project_goals(project_id)
    metrics = await _goal_store.get_project_metrics(project_id)

    return {
        **project.model_dump(mode="json"),
        "goals": [
            {
                "goal_id": g.goal_id,
                "description": g.description,
                "status": g.status,
                "created_at": g.created_at.isoformat(),
                "completed_at": (
                    g.completed_at.isoformat() if g.completed_at else None
                ),
            }
            for g in goals
        ],
        "metrics": metrics,
    }


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, body: dict[str, Any]):
    """Update a project's fields."""
    if not _goal_store:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    project = await _goal_store.get_project(project_id)
    if not project:
        return JSONResponse({"error": "Project not found"}, status_code=404)

    from datetime import UTC, datetime

    if "name" in body:
        project.name = body["name"]
    if "description" in body:
        project.description = body["description"]
    if "owner" in body:
        project.owner = body["owner"]
    if "status" in body:
        project.status = body["status"]
    if "tags" in body:
        project.tags = body["tags"]
    project.updated_at = datetime.now(UTC)

    await _goal_store.save_project(project)
    return project.model_dump(mode="json")


# ── Chat ────────────────────────────────────────────────────────────────────


@app.post("/api/chat")
async def chat_rest(body: dict[str, Any]):
    """REST endpoint for chat (non-streaming fallback)."""
    if not _chat_service:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    message = body.get("message", "").strip()
    session_id = body.get("session_id")

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    if not session_id:
        session_id = str(uuid.uuid4())

    response = await _chat_service.handle_message(session_id, message)
    return {
        "session_id": session_id,
        **response.model_dump(mode="json"),
    }


# ── WebSocket ───────────────────────────────────────────────────────────────


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            log.debug("WS received: %s", data)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """Per-session chat WebSocket with pipeline progress events."""
    await websocket.accept()
    session_id = str(uuid.uuid4())

    if not _chat_service:
        await websocket.send_json({
            "type": "error",
            "error": "Engine not started. Please complete setup first.",
        })
        await websocket.close()
        return

    await websocket.send_json({
        "type": "session_init",
        "session_id": session_id,
    })

    # Track envelopes for pipeline progress forwarding
    tracked_envelopes: set[str] = set()

    async def _pipeline_forwarder(envelope: Envelope) -> None:
        """Forward pipeline events for envelopes this session is tracking."""
        correlation = envelope.correlation_id or envelope.causation_id
        if correlation in tracked_envelopes:
            try:
                await websocket.send_json({
                    "type": "pipeline_event",
                    "topic": envelope.topic,
                    "envelope_id": envelope.envelope_id,
                    "correlation_id": correlation,
                    "payload": envelope.payload,
                    "timestamp": envelope.timestamp.isoformat(),
                })
            except Exception:
                pass

    pipeline_topics = [
        "claims.proposed",
        "claims.committed",
        "claims.contradiction_detected",
    ]
    bus = get_bus()
    for topic in pipeline_topics:
        bus.subscribe(topic, _pipeline_forwarder)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "message":
                user_text = data.get("content", "").strip()
                if not user_text:
                    continue

                await websocket.send_json({"type": "typing", "active": True})

                response = await _chat_service.handle_message(
                    session_id, user_text
                )

                if response.tracking_envelope_id:
                    tracked_envelopes.add(response.tracking_envelope_id)

                await websocket.send_json({"type": "typing", "active": False})
                await websocket.send_json({
                    "type": "chat_response",
                    **response.model_dump(mode="json"),
                })

            elif msg_type == "track_envelope":
                envelope_id = data.get("envelope_id")
                if envelope_id:
                    tracked_envelopes.add(envelope_id)

    except WebSocketDisconnect:
        pass
    finally:
        for topic in pipeline_topics:
            bus.unsubscribe(topic, _pipeline_forwarder)

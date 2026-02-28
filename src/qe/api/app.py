"""FastAPI application for Question Engine OS."""

from __future__ import annotations

import asyncio
import json
import logging
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

from qe.api.setup import (
    PROVIDERS,
    get_configured_providers,
    get_current_tiers,
    is_setup_complete,
    save_setup,
)
from qe.api.ws import ConnectionManager
from qe.bus import get_bus
from qe.bus.event_log import EventLog
from qe.kernel.supervisor import Supervisor
from qe.models.envelope import Envelope
from qe.services.chat import ChatService
from qe.services.dispatcher import Dispatcher
from qe.services.planner import PlannerService
from qe.services.query import answer_question
from qe.substrate import Substrate
from qe.substrate.goal_store import GoalStore

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
_goal_store: GoalStore | None = None

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
    global _planner, _dispatcher, _goal_store

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    bus = get_bus()

    # Initialize durable event log
    _event_log = EventLog()
    await _event_log.initialize()
    bus.set_event_log(_event_log)

    _substrate = Substrate()
    await _substrate.initialize()

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
        log.info("QE API server started (engine running)")
    else:
        log.info("QE API server started (setup required — no API keys configured)")

    yield

    # Shutdown
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# ── REST Endpoints ──────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


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
            "circuit_broken": sid in _supervisor._circuit_broken,
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

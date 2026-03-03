"""System API endpoints (Health, Status, Topology, Dashboard) extracted from app.py."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from qe.runtime.readiness import get_readiness

router = APIRouter(tags=["System"])

@router.get("/")
async def dashboard(request: Request):
    """Serve the dashboard UI."""
    index = request.app.state.static_dir / "index.html"
    if not index.exists():
        return JSONResponse(
            {"status": "ok", "message": "Question Engine is running. No dashboard UI found."},
        )
    return FileResponse(str(index))



# Routers registered via APIRouter
from qe.api.endpoints.setup import router as setup_router


# Routers registered via APIRouter
from qe.api.endpoints.webhooks import router as webhooks_router
from qe.api.endpoints.scout import router as scout_router

# ── Audit Trail ────────────────────────────────────────────────────────────


@router.get("/api/audit")
async def list_audit(request: Request, 
    action: str | None = None,
    actor: str | None = None,
    limit: int = 100,
):
    """Query the admin audit trail."""
    entries = get_audit_log().query(action=action, actor=actor, limit=limit)
    return {"entries": entries, "count": len(entries)}


# ── Metrics ───────────────────────────────────────────────────────────────


@router.get("/api/metrics")
async def metrics_snapshot(request: Request):
    """Return full metrics snapshot: counters, histograms, gauges, SLOs."""
    return get_metrics().snapshot()


# ── Profiling ─────────────────────────────────────────────────────────────


@router.get("/api/profiling/inquiry")
async def profiling_inquiry(request: Request):
    """Return profiling data for inquiry runs and system resources."""
    import resource
    import sys

    from qe.runtime.engram_cache import get_engram_cache as _get_cache

    rusage = resource.getrusage(resource.RUSAGE_SELF)

    # Engram cache stats
    try:
        cache_stats = _get_cache().stats()
    except Exception:
        cache_stats = {}

    # Episodic memory hot store size
    episodic_hot_size = 0
    if BaseService._shared_episodic_memory is not None:
        try:
            episodic_hot_size = len(BaseService._shared_episodic_memory._hot_store)
        except Exception:
            pass

    # Belief store status
    belief_status = "unavailable"
    if BaseService._shared_bayesian_belief is not None:
        belief_status = "available"

    # Build response with profiling store data
    store_data = _inquiry_profiling_store.to_dict()

    return {
        "phase_timings": _last_inquiry_profile,
        "last_inquiry": store_data["last_inquiry"],
        "history_count": store_data["history_count"],
        "percentiles": store_data["percentiles"],
        "process": {
            "rss_bytes": rusage.ru_maxrss,
            "python_version": sys.version,
        },
        "engram_cache": cache_stats,
        "components": {
            "episodic_hot_store_size": episodic_hot_size,
            "belief_store_status": belief_status,
        },
    }


# ── Prompt Evolution ───────────────────────────────────────────────────────


@router.get("/api/prompts/stats")
async def prompt_stats(request: Request):
    """Return prompt evolution registry status and per-slot stats."""
    if _prompt_registry is None:
        return {"enabled": False, "slots": 0}
    return _prompt_registry.status()


@router.get("/api/knowledge/status")
async def knowledge_loop_status(request: Request):
    """Return knowledge loop status."""
    if _knowledge_loop is None:
        return {"running": False}
    return _knowledge_loop.status()


@router.get("/api/bridge/status")
async def inquiry_bridge_status(request: Request):
    """Return inquiry bridge status."""
    if _inquiry_bridge is None:
        return {"running": False}
    return _inquiry_bridge.status()



# Mass Intelligence and Harvest routers registered
from qe.api.endpoints.mass_intelligence import router as mass_intel_router
from qe.api.endpoints.harvest import router as harvest_router

# Telemetry router registered
from qe.api.endpoints.telemetry import router as telemetry_router
from qe.api.endpoints.goals_v2 import router as goals_v2_router
from qe.api.endpoints.knowledge import router as knowledge_router
# ── Topology ──────────────────────────────────────────────────────────────


@router.get("/api/topology")
async def topology(request: Request):
    """Return service dependency graph from blueprint declarations."""
    if not request.app.state.supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    services = []
    all_topics: set[str] = set()

    for svc in request.app.state.supervisor.registry.all_services():
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


@router.post("/api/events/replay")
async def replay_events(request: Request, body: dict[str, Any]):
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


@router.get("/api/health")
async def health(request: Request):
    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


@router.get("/api/health/ready")
async def health_ready(request: Request):
    """Readiness probe: returns 200 when fully initialized, 503 during startup."""
    readiness = get_readiness()
    status_data = readiness.to_dict()
    if readiness.is_ready:
        return status_data
    return JSONResponse(status_data, status_code=503)


@router.get("/api/health/live")
async def health_live(request: Request):
    """Live health report from the Doctor service."""
    if not _doctor:
        return JSONResponse({"error": "Doctor service not running"}, status_code=503)

    report = _doctor.last_report
    if report is None:
        # First check hasn't run yet — run on demand
        report = await _doctor.run_all_checks()

    return report.model_dump(mode="json")


@router.get("/api/status")
async def status(request: Request):
    """Overall engine status: services, budget, circuit breakers."""
    if not request.app.state.supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    services = []
    for svc in request.app.state.supervisor.registry.all_services():
        sid = svc.blueprint.service_id
        services.append({
            "service_id": sid,
            "display_name": svc.blueprint.display_name,
            "status": "alive" if svc._running else "stopped",
            "turn_count": svc._turn_count,
            "circuit_broken": sid in request.app.state.supervisor._circuits,
        })

    return {
        "services": services,
        "budget": {
            "total_spend": request.app.state.supervisor.budget_tracker.total_spend(),
            "remaining_pct": request.app.state.supervisor.budget_tracker.remaining_pct(),
            "limit_usd": request.app.state.supervisor.budget_tracker.monthly_limit_usd,
            "by_model": request.app.state.supervisor.budget_tracker.spend_by_model(),
        },
        "pool": request.app.state.cognitive_pool.pool_status() if request.app.state.cognitive_pool else None,
        "strategy": {
            "current_strategy": (
                request.app.state.strategy_evolver._current_strategy if request.app.state.strategy_evolver else None
            ),
            "scaling_profile": (
                request.app.state.elastic_scaler.current_profile_name() if request.app.state.elastic_scaler else None
            ),
            "snapshots": (
                [s.model_dump() for s in request.app.state.strategy_evolver.get_snapshots()]
                if request.app.state.strategy_evolver else []
            ),
        },
        "flags": get_flag_store().stats(),
        "arena": request.app.state.competitive_arena.status() if request.app.state.competitive_arena else None,
        "bridge": request.app.state.inquiry_bridge.status() if request.app.state.inquiry_bridge else None,
        "knowledge_loop": request.app.state.knowledge_loop.status() if request.app.state.knowledge_loop else None,
        "scout": request.app.state.scout_service.status() if request.app.state.scout_service else None,
        "harvest": request.app.state.harvest_service.status() if request.app.state.harvest_service else None,
    }


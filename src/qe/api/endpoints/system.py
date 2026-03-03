"""System API endpoints (Health, Status, Topology, Dashboard)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from qe.audit import get_audit_log
from qe.bus import get_bus
from qe.models.envelope import Envelope
from qe.runtime.feature_flags import get_flag_store
from qe.runtime.metrics import get_metrics
from qe.runtime.readiness import get_readiness
from qe.runtime.service import BaseService

router = APIRouter(tags=["System"])
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def get_app_globals():
    import qe.api.app as app_mod

    return app_mod


def _state_attr(request: Request, key: str, default=None):
    return getattr(request.app.state, key, default)


@router.get("/")
async def dashboard(request: Request):
    """Serve the dashboard UI."""
    static_dir = _state_attr(request, "static_dir", STATIC_DIR)
    index = static_dir / "index.html"
    if not index.exists():
        return JSONResponse(
            {
                "status": "ok",
                "message": "Question Engine is running. No dashboard UI found.",
            }
        )
    return FileResponse(str(index))


@router.get("/api/audit")
async def list_audit(
    request: Request,
    action: str | None = None,
    actor: str | None = None,
    limit: int = 100,
):
    """Query the admin audit trail."""
    entries = get_audit_log().query(action=action, actor=actor, limit=limit)
    return {"entries": entries, "count": len(entries)}


@router.get("/api/metrics")
async def metrics_snapshot(request: Request):
    """Return full metrics snapshot: counters, histograms, gauges, SLOs."""
    return get_metrics().snapshot()


@router.get("/api/profiling/inquiry")
async def profiling_inquiry(request: Request):
    """Return profiling data for inquiry runs and system resources."""
    import resource
    import sys

    from qe.runtime.engram_cache import get_engram_cache as _get_cache

    app_mod = get_app_globals()
    rusage = resource.getrusage(resource.RUSAGE_SELF)

    try:
        cache_stats = _get_cache().stats()
    except Exception:
        cache_stats = {}

    episodic_hot_size = 0
    if BaseService._shared_episodic_memory is not None:
        try:
            episodic_hot_size = len(BaseService._shared_episodic_memory._hot_store)
        except Exception:
            pass

    belief_status = (
        "available" if BaseService._shared_bayesian_belief is not None else "unavailable"
    )

    store_data = app_mod._inquiry_profiling_store.to_dict()

    return {
        "phase_timings": app_mod._last_inquiry_profile,
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


@router.get("/api/prompts/stats")
async def prompt_stats(request: Request):
    """Return prompt evolution registry status and per-slot stats."""
    app_mod = get_app_globals()
    if app_mod._prompt_registry is None:
        return {"enabled": False, "slots": 0}
    return app_mod._prompt_registry.status()


@router.get("/api/knowledge/status")
async def knowledge_loop_status(request: Request):
    """Return knowledge loop status."""
    app_mod = get_app_globals()
    if app_mod._knowledge_loop is None:
        return {"running": False}
    return app_mod._knowledge_loop.status()


@router.get("/api/bridge/status")
async def inquiry_bridge_status(request: Request):
    """Return inquiry bridge status."""
    app_mod = get_app_globals()
    if app_mod._inquiry_bridge is None:
        return {"running": False}
    return app_mod._inquiry_bridge.status()


@router.get("/api/topology")
async def topology(request: Request):
    """Return service dependency graph from blueprint declarations."""
    supervisor = _state_attr(request, "supervisor")
    if not supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    services = []
    all_topics: set[str] = set()

    for svc in supervisor.registry.all_services():
        bp = svc.blueprint
        subs = bp.capabilities.bus_topics_subscribe
        pubs = bp.capabilities.bus_topics_publish
        all_topics.update(subs)
        all_topics.update(pubs)
        services.append(
            {
                "service_id": bp.service_id,
                "display_name": bp.display_name,
                "subscribes": subs,
                "publishes": pubs,
            }
        )

    return {
        "services": services,
        "topics": sorted(all_topics),
        "service_count": len(services),
        "topic_count": len(all_topics),
    }


@router.post("/api/events/replay")
async def replay_events(request: Request, body: dict[str, Any]):
    """Bulk replay historical events from the event log back into the bus."""
    event_log = _state_attr(request, "event_log")
    if not event_log:
        return JSONResponse({"error": "Event log not ready"}, status_code=503)

    topic = body.get("topic")
    since_str = body.get("since")
    limit = body.get("limit", 100)

    since = datetime.fromisoformat(since_str) if since_str else None
    events = await event_log.replay(since=since, topic=topic, limit=limit)

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
    app_mod = get_app_globals()
    doctor = app_mod._doctor
    if not doctor:
        return JSONResponse({"error": "Doctor service not running"}, status_code=503)

    report = doctor.last_report
    if report is None:
        report = await doctor.run_all_checks()

    return report.model_dump(mode="json")


@router.get("/api/status")
async def status(request: Request):
    """Overall engine status: services, budget, circuit breakers."""
    supervisor = _state_attr(request, "supervisor")
    if not supervisor:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    services = []
    for svc in supervisor.registry.all_services():
        sid = svc.blueprint.service_id
        services.append(
            {
                "service_id": sid,
                "display_name": svc.blueprint.display_name,
                "status": "alive" if svc._running else "stopped",
                "turn_count": svc._turn_count,
                "circuit_broken": sid in supervisor._circuits,
            }
        )

    cognitive_pool = _state_attr(request, "cognitive_pool")
    strategy_evolver = _state_attr(request, "strategy_evolver")
    elastic_scaler = _state_attr(request, "elastic_scaler")
    competitive_arena = _state_attr(request, "competitive_arena")
    inquiry_bridge = _state_attr(request, "inquiry_bridge")
    knowledge_loop = _state_attr(request, "knowledge_loop")
    scout_service = _state_attr(request, "scout_service")
    harvest_service = _state_attr(request, "harvest_service")

    return {
        "services": services,
        "budget": {
            "total_spend": supervisor.budget_tracker.total_spend(),
            "remaining_pct": supervisor.budget_tracker.remaining_pct(),
            "limit_usd": supervisor.budget_tracker.monthly_limit_usd,
            "by_model": supervisor.budget_tracker.spend_by_model(),
        },
        "pool": cognitive_pool.pool_status() if cognitive_pool else None,
        "strategy": {
            "current_strategy": strategy_evolver._current_strategy if strategy_evolver else None,
            "scaling_profile": elastic_scaler.current_profile_name() if elastic_scaler else None,
            "snapshots": [s.model_dump() for s in strategy_evolver.get_snapshots()]
            if strategy_evolver
            else [],
        },
        "flags": get_flag_store().stats(),
        "arena": competitive_arena.status() if competitive_arena else None,
        "bridge": inquiry_bridge.status() if inquiry_bridge else None,
        "knowledge_loop": knowledge_loop.status() if knowledge_loop else None,
        "scout": scout_service.status() if scout_service else None,
        "harvest": harvest_service.status() if harvest_service else None,
    }

"""Webhook and DLQ API endpoints extracted from app.py."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from qe.bus import get_bus
from qe.models.envelope import Envelope

router = APIRouter(prefix="/api", tags=["Webhooks"])


# Access globals from app via lazy imports to avoid circularity
def get_app_globals():
    import qe.api.app as app_mod
    return app_mod




@router.post("/webhooks/inbound")
async def inbound_webhook(request: Request):
    """Receive inbound webhook payloads from external systems."""
    if request.app.state.notification_router is None:
        return JSONResponse({"error": "Channels not initialized"}, status_code=503)

    body = await request.json()
    headers = dict(request.headers)

    # Find the webhook adapter from active adapters
    from qe.channels.webhook import WebhookAdapter

    webhook = None
    for adapter in request.app.state.active_adapters:
        if isinstance(adapter, WebhookAdapter):
            webhook = adapter
            break

    if webhook is None:
        return JSONResponse({"error": "Webhook adapter not available"}, status_code=503)

    result = await webhook.process_webhook(body, headers)
    if result is None:
        return JSONResponse({"error": "Invalid signature or rejected"}, status_code=403)

    # Route based on command field in the original payload
    command = body.get("command", "goal")
    topic_map = {
        "ask": "queries.asked",
        "status": "system.health.check",
    }
    topic = topic_map.get(command, "channel.message_received")

    get_bus().publish(
        Envelope(
            topic=topic,
            source_service_id="webhook",
            payload={
                "channel": "webhook",
                "user_id": result.get("user_id", ""),
                "text": result.get("sanitized_text", ""),
                "command": command,
            },
        )
    )

    return {"status": "received", "user_id": result.get("user_id", "")}


# ── Settings ─────────────────────────────────────────────────────────────


@router.get("/settings")
async def get_settings_endpoint(request: Request):
    """Return runtime settings from config.toml."""
    return get_settings()


@router.post("/settings")
async def save_settings_endpoint(body: dict[str, Any]):
    """Save runtime settings and apply budget changes at runtime."""
    agent_access = body.get("agent_access")
    if agent_access and isinstance(agent_access, dict):
        mode = agent_access.get("mode")
        if mode is not None and mode not in _AGENT_ACCESS_MODES:
            return JSONResponse(
                {"error": f"Invalid agent_access.mode: {mode}"},
                status_code=400,
            )

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

    # Apply agent access mode at runtime
    if agent_access and isinstance(agent_access, dict):
        mode = _resolve_agent_access_mode({"agent_access": agent_access})
        from qe.tools.file_ops import set_workspace_root

        workspace_root = _workspace_root_for_mode(mode)
        workspace_root.mkdir(parents=True, exist_ok=True)
        set_workspace_root(workspace_root)

        if _chat_service is not None:
            _chat_service.set_access_mode(mode)

        log.info(
            "agent_access.updated mode=%s workspace_root=%s",
            mode,
            workspace_root,
        )

    get_audit_log().record("settings.update", resource="config", detail=body)
    return {"status": "saved"}


@router.post("/optimize/{genome_id}")
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


@router.post("/services/{service_id}/reset-circuit")
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


@router.get("/dlq")
async def list_dlq(limit: int = 100):
    """List dead-letter queue entries."""
    bus = get_bus()
    return {
        "entries": bus.dlq_list(limit=limit),
        "count": bus.dlq_size(),
    }


@router.post("/dlq/{envelope_id}/replay")
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


@router.delete("/dlq")
async def purge_dlq(request: Request):
    """Purge all entries from the dead-letter queue."""
    bus = get_bus()
    count = await bus.dlq_purge()
    return {"status": "purged", "count": count}



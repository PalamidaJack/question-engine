"""Harvest API endpoints extracted from app.py."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/harvest", tags=["Harvest"])


def get_app_globals():
    import qe.api.app as app_mod

    return app_mod




@router.get("/status")
async def harvest_status(request: Request):
    """Return harvest service status."""
    if request.app.state.harvest_service is None:
        return {"running": False}
    return request.app.state.harvest_service.status()


@router.post("/scout/proposals/{proposal_id}/approve")
async def scout_approve(request: Request, proposal_id: str):
    """Approve a scout proposal — writes HIL decision file."""
    app_mod = get_app_globals()
    _scout_store = app_mod._scout_store
    if _scout_store is None:
        return JSONResponse({"error": "Scout not initialized"}, status_code=503)
    proposal = await _scout_store.get_proposal(proposal_id)
    if proposal is None:
        return JSONResponse({"error": "Proposal not found"}, status_code=404)
    if proposal.status != "pending_review":
        return JSONResponse(
            {"error": f"Cannot approve proposal in status '{proposal.status}'"},
            status_code=400,
        )

    # Write HIL decision file
    hil_dir = Path("data/hil_queue/completed")
    await asyncio.to_thread(hil_dir.mkdir, parents=True, exist_ok=True)
    if proposal.hil_envelope_id:
        safe_name = Path(proposal.hil_envelope_id).name
        if not safe_name or safe_name == ".":
            return JSONResponse({"error": "Invalid envelope id"}, status_code=400)
        decision_file = hil_dir / f"{safe_name}.json"
        content = json.dumps({
            "decision": "approved",
            "decided_at": datetime.now(UTC).isoformat(),
        }, indent=2)
        await asyncio.to_thread(decision_file.write_text, content, "utf-8")

    return {"status": "approved", "proposal_id": proposal_id}


@router.post("/scout/proposals/{proposal_id}/reject")
async def scout_reject(request: Request, proposal_id: str, body: dict[str, Any] | None = None):
    """Reject a scout proposal with optional reason."""
    app_mod = get_app_globals()
    _scout_store = app_mod._scout_store
    if _scout_store is None:
        return JSONResponse({"error": "Scout not initialized"}, status_code=503)
    proposal = await _scout_store.get_proposal(proposal_id)
    if proposal is None:
        return JSONResponse({"error": "Proposal not found"}, status_code=404)
    if proposal.status != "pending_review":
        return JSONResponse(
            {"error": f"Cannot reject proposal in status '{proposal.status}'"},
            status_code=400,
        )

    reason = ""
    if body:
        reason = body.get("reason", "")

    # Write HIL decision file
    hil_dir = Path("data/hil_queue/completed")
    await asyncio.to_thread(hil_dir.mkdir, parents=True, exist_ok=True)
    if proposal.hil_envelope_id:
        safe_name = Path(proposal.hil_envelope_id).name
        if not safe_name or safe_name == ".":
            return JSONResponse({"error": "Invalid envelope id"}, status_code=400)
        decision_file = hil_dir / f"{safe_name}.json"
        content = json.dumps({
            "decision": "rejected",
            "reason": reason,
            "decided_at": datetime.now(UTC).isoformat(),
        }, indent=2)
        await asyncio.to_thread(decision_file.write_text, content, "utf-8")

    return {"status": "rejected", "proposal_id": proposal_id, "reason": reason}


@router.get("/scout/learning")
async def scout_learning(request: Request):
    """Return feedback stats (approval rate by category/source)."""
    _scout_store = get_app_globals()._scout_store
    if _scout_store is None:
        return {"total": 0, "approved": 0, "rejected": 0, "approval_rate": 0.0}
    return await _scout_store.get_feedback_stats()


@router.get("/prompts/mutator/status")
async def prompt_mutator_status(request: Request):
    """Return prompt mutator status."""
    _prompt_mutator = get_app_globals()._prompt_mutator
    if _prompt_mutator is None:
        return {"running": False}
    return _prompt_mutator.status()


@router.get("/prompts/slots/{slot_key}")
async def prompt_slot_detail(request: Request, slot_key: str):
    """Return detailed stats for a specific prompt slot."""
    _prompt_registry = get_app_globals()._prompt_registry
    if _prompt_registry is None:
        return {"error": "prompt registry not initialized"}
    stats = _prompt_registry.get_slot_stats(slot_key)
    if not stats:
        return {"error": f"slot '{slot_key}' not found", "variants": []}
    return {"slot_key": slot_key, "variants": stats}



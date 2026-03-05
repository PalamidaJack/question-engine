"""Scout API endpoints extracted from app.py."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/scout", tags=["Scout"])




@router.get("/status")
async def scout_status(request: Request):
    """Return scout service status."""
    if request.app.state.scout_service is None:
        return {"running": False}
    return request.app.state.scout_service.status()


@router.get("/proposals")
async def scout_proposals(request: Request, status: str | None = None, limit: int = 50):
    """List scout proposals with optional status filter."""
    if request.app.state.scout_store is None:
        return {"proposals": [], "count": 0}
    proposals = await request.app.state.scout_store.list_proposals(status=status, limit=limit)
    return {
        "proposals": [p.model_dump(mode="json") for p in proposals],
        "count": len(proposals),
    }


@router.get("/proposals/{proposal_id}")
async def scout_proposal_detail(request: Request, proposal_id: str):
    """Return full proposal detail with diffs and test results."""
    if request.app.state.scout_store is None:
        return JSONResponse({"error": "Scout not initialized"}, status_code=503)
    proposal = await request.app.state.scout_store.get_proposal(proposal_id)
    if proposal is None:
        return JSONResponse({"error": "Proposal not found"}, status_code=404)
    return proposal.model_dump(mode="json")



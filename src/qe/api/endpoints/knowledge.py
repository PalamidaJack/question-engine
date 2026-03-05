"""Knowledge API endpoints extracted from app.py."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from qe.bus import get_bus
from qe.models.envelope import Envelope
from qe.services.query import answer_question

router = APIRouter(prefix="/api", tags=["Knowledge"])


def _substrate_from_request(request: Request):
    return getattr(request.app.state, "substrate", None)


def _inbox_dir():
    import qe.api.app as app_mod

    return app_mod.INBOX_DIR

@router.post("/submit")
async def submit(request: Request, body: dict[str, Any]):
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
    inbox_dir = _inbox_dir()
    await asyncio.to_thread(inbox_dir.mkdir, parents=True, exist_ok=True)
    inbox_file = inbox_dir / f"{envelope.envelope_id}.json"
    content = envelope.model_dump_json()
    await asyncio.to_thread(inbox_file.write_text, content, "utf-8")

    return {"envelope_id": envelope.envelope_id, "status": "submitted"}


@router.post("/ask")
async def ask(request: Request, body: dict[str, Any]):
    """Ask a question against the belief ledger."""
    substrate = _substrate_from_request(request)
    if not substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    question = body.get("question", "")
    if not question:
        return JSONResponse(
            {"error": "question is required"}, status_code=400
        )

    result = await answer_question(question, substrate)
    return result


@router.get("/claims")
async def list_claims(request: Request,
    subject: str | None = None,
    include_superseded: bool = False,
):
    """List claims from the belief ledger."""
    substrate = _substrate_from_request(request)
    if not substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    claims = await substrate.get_claims(
        subject_entity_id=subject,
        include_superseded=include_superseded,
    )
    return {
        "claims": [c.model_dump(mode="json") for c in claims],
        "count": len(claims),
    }


@router.get("/claims/{claim_id}")
async def get_claim(request: Request, claim_id: str):
    """Get a specific claim by ID."""
    substrate = _substrate_from_request(request)
    if not substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    claim = await substrate.get_claim_by_id(claim_id)
    if not claim:
        return JSONResponse({"error": "Claim not found"}, status_code=404)
    return claim.model_dump(mode="json")


@router.delete("/claims/{claim_id}")
async def retract_claim(request: Request, claim_id: str):
    """Soft-retract a claim (mark as superseded by 'retracted')."""
    substrate = _substrate_from_request(request)
    if not substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    retracted = await substrate.retract_claim(claim_id)
    if not retracted:
        return JSONResponse({"error": "Claim not found"}, status_code=404)
    return {"status": "retracted", "claim_id": claim_id}


@router.get("/entities")
async def list_entities(request: Request):
    """List all entities with claim counts."""
    substrate = _substrate_from_request(request)
    if not substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    entities = await substrate.entity_resolver.list_entities()
    # Enrich with claim counts
    for ent in entities:
        claims = await substrate.get_claims(
            subject_entity_id=ent["canonical_name"]
        )
        ent["claim_count"] = len(claims)
    return {"entities": entities, "count": len(entities)}


@router.get("/entities/{name}")
async def get_entity(request: Request, name: str):
    """Get claims for a specific entity."""
    substrate = _substrate_from_request(request)
    if not substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    canonical = await substrate.entity_resolver.resolve(name)
    claims = await substrate.get_claims(subject_entity_id=canonical)
    return {
        "canonical_name": canonical,
        "claims": [c.model_dump(mode="json") for c in claims],
        "count": len(claims),
    }


@router.post("/entities/{name}/alias")
async def add_entity_alias(request: Request, name: str, body: dict[str, Any]):
    """Add an alias for an entity."""
    substrate = _substrate_from_request(request)
    if not substrate:
        return JSONResponse({"error": "Substrate not ready"}, status_code=503)

    alias = body.get("alias", "")
    if not alias:
        return JSONResponse({"error": "alias is required"}, status_code=400)

    await substrate.entity_resolver.add_alias(name, alias)
    return {"status": "added", "entity": name, "alias": alias}

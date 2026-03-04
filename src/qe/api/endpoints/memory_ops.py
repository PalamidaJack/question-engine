"""Memory operations HTTP endpoints (Phase 1).

Provides cross-tier search, tier status, procedural templates, working memory
inspection, context-curator drift reporting, consolidation history, and import/
export helpers.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request

router = APIRouter(prefix="/api/memory", tags=["Memory"])


def register_memory_ops_routes(app: FastAPI, memory_store: Any | None = None) -> None:
    """Register memory ops routes onto the provided FastAPI app.

    Handlers resolve runtime components lazily from `qe.api.app` so the
    registration can occur early in the startup lifespan.
    """
    app.include_router(router)


@router.get("/search")
async def memory_search(
    request: Request,
    query: str = "",
    tiers: str = "episodic,belief,procedural",
    top_k: int = 10,
    goal_id: str | None = None,
    min_confidence: float = 0.0,
) -> dict[str, Any]:
    """Unified cross-tier search.

    Dispatches to EpisodicMemory.recall, Substrate.hybrid_search, and
    ProceduralMemory pattern matching in parallel and returns fused results.
    """
    tiers_set = {t.strip() for t in tiers.split(",") if t.strip()}

    # resolve components lazily to avoid import cycles during app startup
    from qe.api import app as _app_module  # type: ignore

    episodic = getattr(_app_module, "_episodic_memory", None)
    substrate = getattr(_app_module, "_substrate", None)
    procedural = getattr(_app_module, "_procedural_memory", None)

    async def _episodic_search():
        if episodic is None:
            return []
        return [
            ep.model_dump()
            for ep in await episodic.recall(
                query=query, top_k=top_k, goal_id=goal_id,
            )
        ]

    async def _belief_search():
        if substrate is None:
            return []
        claims = await substrate.hybrid_search(query, fts_top_k=top_k, semantic_top_k=top_k)
        # Normalize to dicts
        return [c.model_dump() if hasattr(c, "model_dump") else c for c in claims]

    async def _procedural_search():
        if procedural is None:
            return []
        templates = await procedural.get_best_templates(top_k=top_k)
        sequences = await procedural.get_best_sequences(top_k=top_k)
        return {
            "templates": [t.model_dump() for t in templates],
            "sequences": [s.model_dump() for s in sequences],
        }

    tasks = []
    results: dict[str, Any] = {}
    if "episodic" in tiers_set:
        tasks.append(asyncio.create_task(_episodic_search()))
    if "belief" in tiers_set:
        tasks.append(asyncio.create_task(_belief_search()))
    if "procedural" in tiers_set:
        tasks.append(asyncio.create_task(_procedural_search()))

    if tasks:
        done = await asyncio.gather(*tasks)
    else:
        done = []

    idx = 0
    if "episodic" in tiers_set:
        results["episodic"] = done[idx]
        idx += 1
    if "belief" in tiers_set:
        results["beliefs"] = done[idx]
        idx += 1
    if "procedural" in tiers_set:
        results["procedural"] = done[idx]

    return {"query": query, "results": results}


@router.get("/tiers")
async def tiers_status() -> dict[str, Any]:
    """Return status of all four tiers (working, episodic, semantic, procedural).

    Counts and simple metrics are returned when available.
    """
    from qe.api import app as _app_module  # type: ignore

    ctx = getattr(_app_module, "_context_curator", None)
    episodic = getattr(_app_module, "_episodic_memory", None)
    substrate = getattr(_app_module, "_substrate", None)
    procedural = getattr(_app_module, "_procedural_memory", None)

    resp: dict[str, Any] = {}

    # Working (Tier 0)
    resp["working"] = ctx.status() if ctx is not None else {}

    # Episodic (Tier 1)
    if episodic is not None:
        warm = await episodic.warm_count()
        resp["episodic"] = {
            "hot_entries": episodic.hot_count(),
            "warm_entries": warm,
        }
    else:
        resp["episodic"] = {}

    # Semantic / Belief (Tier 2)
    if substrate is not None:
        try:
            count = await substrate.count_claims()
        except Exception:
            count = None
        resp["beliefs"] = {"claim_count": count}
    else:
        resp["beliefs"] = {}

    # Procedural (Tier 3)
    if procedural is not None:
        resp["procedural"] = {
            "templates": len(getattr(procedural, "_templates", {})),
            "sequences": len(getattr(procedural, "_sequences", {})),
        }
    else:
        resp["procedural"] = {}

    return resp


@router.get("/procedural")
async def procedural_best(domain: str = "general", top_k: int = 5) -> dict[str, Any]:
    from qe.api import app as _app_module  # type: ignore
    procedural = getattr(_app_module, "_procedural_memory", None)
    if procedural is None:
        raise HTTPException(status_code=503, detail="procedural memory not available")

    templates = await procedural.get_best_templates(domain=domain, top_k=top_k)
    sequences = await procedural.get_best_sequences(domain=domain, top_k=top_k)
    return {
        "templates": [t.model_dump() for t in templates],
        "sequences": [s.model_dump() for s in sequences],
    }


@router.get("/working/{goal_id}")
async def working_memory(goal_id: str) -> dict[str, Any]:
    from qe.api import app as _app_module  # type: ignore
    curator = getattr(_app_module, "_context_curator", None)
    if curator is None:
        raise HTTPException(status_code=503, detail="context curator not initialized")
    state = curator._states.get(goal_id)
    if state is None:
        raise HTTPException(status_code=404, detail="working memory not found for goal")
    return state.model_dump()


@router.get("/context-curator/{goal_id}")
async def context_curator_drift(goal_id: str) -> dict[str, Any]:
    from qe.api import app as _app_module  # type: ignore
    curator = getattr(_app_module, "_context_curator", None)
    if curator is None:
        raise HTTPException(status_code=503, detail="context curator not initialized")
    drift = await curator.detect_drift(goal_id)
    return drift.model_dump()


@router.get("/consolidation/history")
async def consolidation_history(limit: int = 20) -> dict[str, Any]:
    from qe.api import app as _app_module  # type: ignore
    kl = getattr(_app_module, "_knowledge_loop", None)
    if kl is None:
        raise HTTPException(status_code=503, detail="knowledge loop not available")
    try:
        hist = kl.get_history(limit=limit)
        return {"history": [h.model_dump() for h in hist]}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="failed to retrieve consolidation history",
        ) from exc


@router.post("/export")
async def memory_export() -> dict[str, Any]:
    from qe.api import app as _app_module  # type: ignore
    episodic = getattr(_app_module, "_episodic_memory", None)
    substrate = getattr(_app_module, "_substrate", None)
    procedural = getattr(_app_module, "_procedural_memory", None)

    out: dict[str, Any] = {"episodic": [], "claims": [], "procedural": {}}

    if episodic is not None:
        out["episodic"] = [e.model_dump() for e in episodic.get_latest(200)]

    if substrate is not None:
        try:
            claims = await substrate.get_claims(limit=500)
            out["claims"] = [c.model_dump() if hasattr(c, "model_dump") else c for c in claims]
        except Exception:
            out["claims"] = []

    if procedural is not None:
        templates = await procedural.get_best_templates(top_k=500)
        sequences = await procedural.get_best_sequences(top_k=500)
        out["procedural"] = {
            "templates": [t.model_dump() for t in templates],
            "sequences": [s.model_dump() for s in sequences],
        }

    return out


@router.post("/import")
async def memory_import(payload: dict[str, Any]) -> dict[str, Any]:
    """Import episodic + claims + procedural from exported JSON structure.

    This endpoint is best-effort and will not overwrite existing IDs.
    """
    from qe.api import app as _app_module  # type: ignore
    episodic = getattr(_app_module, "_episodic_memory", None)
    substrate = getattr(_app_module, "_substrate", None)
    procedural = getattr(_app_module, "_procedural_memory", None)

    results = {"episodes_imported": 0, "claims_imported": 0, "procedural_imported": 0}

    eps = payload.get("episodic") or []
    for ep in eps:
        if episodic is None:
            break
        try:
            from qe.runtime.episodic_memory import Episode

            episode = Episode(**ep)
            await episodic.store(episode)
            results["episodes_imported"] += 1
        except Exception:
            continue

    cls = payload.get("claims") or []
    for c in cls:
        if substrate is None:
            break
        try:
            # best-effort: substrate.commit_claim expects a Claim model
            from qe.models.claim import Claim

            claim = Claim(**c)
            await substrate.commit_claim(claim)
            results["claims_imported"] += 1
        except Exception:
            continue

    proc = payload.get("procedural") or {}
    for t in proc.get("templates", []):
        if procedural is None:
            break
        try:
            await procedural.record_template_outcome(
                template_id=t.get("template_id"),
                pattern=t.get("pattern", ""),
                question_type=t.get("question_type", "factual"),
                success=bool(t.get("success_count", 0) > t.get("failure_count", 0)),
                info_gain=float(t.get("avg_info_gain", 0.0)),
                domain=t.get("domain", "general"),
            )
            results["procedural_imported"] += 1
        except Exception:
            continue

    return results

"""Question answering: search belief ledger, build context, synthesize answer."""

from __future__ import annotations

import logging
from typing import Any

import instructor
import litellm
from dotenv import load_dotenv

from qe.runtime.metrics import get_metrics
from qe.services.query.schemas import AnswerResponse
from qe.substrate import Substrate

load_dotenv()

log = logging.getLogger(__name__)

_MAX_CONTEXT_CLAIMS = 20


def _get_retrieval_settings() -> dict[str, float | int]:
    """Load retrieval settings from config.toml with safe defaults."""
    defaults: dict[str, float | int] = {
        "fts_top_k": _MAX_CONTEXT_CLAIMS,
        "semantic_top_k": _MAX_CONTEXT_CLAIMS,
        "semantic_min_similarity": 0.3,
        "fts_weight": 0.6,
        "semantic_weight": 0.4,
        "rrf_k": 60,
    }
    try:
        from qe.api.setup import get_settings

        retrieval = get_settings().get("retrieval", {})
        if isinstance(retrieval, dict):
            merged = defaults.copy()
            merged.update({
                k: retrieval.get(k, defaults[k]) for k in defaults
            })
            return merged
    except Exception:
        log.debug("query.retrieval_settings_defaulted", exc_info=True)
    return defaults


async def answer_question(
    question: str,
    substrate: Substrate,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Search the belief ledger and synthesize an answer.

    Returns dict with answer, confidence, reasoning, and supporting_claims.
    """
    # 1. Hybrid retrieval (FTS + semantic)
    get_metrics().counter("retrieval_queries_total").inc()
    retrieval = _get_retrieval_settings()
    claims = await substrate.hybrid_search(
        question,
        fts_top_k=int(retrieval["fts_top_k"]),
        semantic_top_k=int(retrieval["semantic_top_k"]),
        semantic_min_similarity=float(retrieval["semantic_min_similarity"]),
        fts_weight=float(retrieval["fts_weight"]),
        semantic_weight=float(retrieval["semantic_weight"]),
        rrf_k=int(retrieval["rrf_k"]),
    )

    # 2. If retrieval returns nothing, try keyword-based entity scan
    if not claims:
        all_claims = await substrate.get_claims()
        q_lower = question.lower()
        claims = [
            c for c in all_claims
            if q_lower in c.subject_entity_id.lower()
            or q_lower in c.object_value.lower()
            or any(word in c.object_value.lower() for word in q_lower.split() if len(word) > 3)
        ][:_MAX_CONTEXT_CLAIMS]

    if not claims:
        return {
            "answer": (
                "I don't have enough information in my "
                "belief ledger to answer this question."
            ),
            "confidence": 0.0,
            "reasoning": "No relevant claims found in the belief ledger.",
            "supporting_claims": [],
        }

    # 3. Build context from claims
    claim_context = "\n".join(
        f"- [{c.confidence:.0%} confidence] {c.subject_entity_id} {c.predicate} {c.object_value}"
        for c in claims
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Question Engine that answers questions based on a belief ledger. "
                "Only use the provided claims as evidence. Be honest about uncertainty. "
                "If the evidence is insufficient, say so."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Based on these claims from the belief ledger:\n\n{claim_context}\n\n"
                f"Question: {question}"
            ),
        },
    ]

    # 4. LLM synthesizes answer
    client = instructor.from_litellm(litellm.acompletion)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=AnswerResponse,
    )

    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "reasoning": response.reasoning,
        "supporting_claims": [c.model_dump(mode="json") for c in claims],
    }

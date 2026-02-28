"""Question answering: search belief ledger, build context, synthesize answer."""

from __future__ import annotations

import logging
from typing import Any

import instructor
import litellm
from dotenv import load_dotenv

from qe.services.query.schemas import AnswerResponse
from qe.substrate import Substrate

load_dotenv()

log = logging.getLogger(__name__)

_MAX_CONTEXT_CLAIMS = 20


async def answer_question(
    question: str,
    substrate: Substrate,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Search the belief ledger and synthesize an answer.

    Returns dict with answer, confidence, reasoning, and supporting_claims.
    """
    # 1. Search via FTS5
    claims = await substrate.search_full_text(question, limit=_MAX_CONTEXT_CLAIMS)

    # 2. If FTS5 returns nothing, try keyword-based entity search
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

"""Conversation history endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _get_chat_store():
    import qe.api.app as app_mod

    store = getattr(app_mod, "_chat_store", None)
    if store is None:
        raise HTTPException(503, "Chat store not initialized")
    return store


# ── Request / Response Models ───────────────────────────────────────


class UpdateConversationBody(BaseModel):
    title: str | None = None
    metadata: dict[str, Any] | None = None


class AddMessageBody(BaseModel):
    role: str
    content: str
    metadata: dict[str, Any] | None = None


class PreferResponseBody(BaseModel):
    reason: str | None = None


# ── Endpoints ───────────────────────────────────────────────────────
# NOTE: /stats is placed before /{conv_id} to avoid route shadowing.


@router.get("/stats")
async def conversation_stats():
    """Return aggregate conversation statistics."""
    store = _get_chat_store()
    stats = await store.get_stats()
    return stats


@router.get("/")
async def list_conversations(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    model: str | None = None,
):
    """List conversations with optional model filter."""
    store = _get_chat_store()
    convs = await store.list_conversations(
        limit=limit, offset=offset, model=model,
    )
    return {
        "conversations": convs,
        "count": len(convs),
        "limit": limit,
        "offset": offset,
    }


@router.get("/search")
async def search_conversations(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=200),
):
    """Full-text search across conversation messages."""
    store = _get_chat_store()
    results = await store.search(q, limit=limit)
    return {"results": results, "count": len(results), "query": q}


@router.get("/{conv_id}")
async def get_conversation(conv_id: str):
    """Get a single conversation with its messages."""
    store = _get_chat_store()
    conv = await store.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, f"Conversation {conv_id} not found")
    return conv


@router.patch("/{conv_id}")
async def update_conversation(conv_id: str, body: UpdateConversationBody):
    """Update conversation title or metadata."""
    store = _get_chat_store()
    conv = await store.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, f"Conversation {conv_id} not found")
    updated = await store.update_conversation(
        conv_id,
        title=body.title,
        metadata=body.metadata,
    )
    return updated


@router.delete("/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation and its messages."""
    store = _get_chat_store()
    conv = await store.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, f"Conversation {conv_id} not found")
    await store.delete_conversation(conv_id)
    return {"status": "deleted", "conversation_id": conv_id}


@router.get("/{conv_id}/messages")
async def get_messages(
    conv_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Get messages for a conversation with pagination."""
    store = _get_chat_store()
    conv = await store.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, f"Conversation {conv_id} not found")
    messages = await store.get_messages(
        conv_id, limit=limit, offset=offset,
    )
    return {
        "messages": messages,
        "count": len(messages),
        "conversation_id": conv_id,
    }


@router.post("/{conv_id}/messages")
async def add_message(conv_id: str, body: AddMessageBody):
    """Manually add a message to a conversation."""
    store = _get_chat_store()
    conv = await store.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, f"Conversation {conv_id} not found")
    msg = await store.add_message(
        conv_id,
        role=body.role,
        content=body.content,
        metadata=body.metadata,
    )
    return msg


@router.get("/{conv_id}/multi")
async def get_multi_model_responses(conv_id: str):
    """Get multi-model response variants for a conversation."""
    store = _get_chat_store()
    conv = await store.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, f"Conversation {conv_id} not found")
    responses = await store.get_multi_responses(conv_id)
    return {
        "responses": responses,
        "count": len(responses),
        "conversation_id": conv_id,
    }


@router.post("/{conv_id}/multi/{response_id}/prefer")
async def set_preferred_response(
    conv_id: str,
    response_id: str,
    body: PreferResponseBody | None = None,
):
    """Mark a multi-model response as the preferred one."""
    store = _get_chat_store()
    conv = await store.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, f"Conversation {conv_id} not found")
    reason = body.reason if body else None
    result = await store.set_preferred(
        conv_id, response_id, reason=reason,
    )
    return result

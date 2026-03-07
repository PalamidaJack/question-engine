"""Communication channel management endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/communications", tags=["communications"],
)


def _get_notification_router():
    import qe.api.app as app_mod

    return getattr(app_mod, "_notification_router", None)


def _get_webhook_notifier():
    import qe.api.app as app_mod

    return getattr(app_mod, "_webhook_notifier", None)


def _get_active_adapters():
    import qe.api.app as app_mod

    return getattr(app_mod, "_active_adapters", [])


# ── Request / Response Models ───────────────────────────────────────


class UpdatePreferencesBody(BaseModel):
    preferences: dict[str, Any]


class AddWebhookBody(BaseModel):
    url: str
    secret: str | None = None
    events: list[str] = []
    enabled: bool = True


class UpdateWebhookBody(BaseModel):
    secret: str | None = None
    events: list[str] | None = None
    enabled: bool | None = None


class TestMessageBody(BaseModel):
    message: str = "Test notification from Question Engine"


# ── Endpoints ───────────────────────────────────────────────────────


@router.get("/channels")
async def list_channels():
    """List communication channels with their status."""
    adapters = _get_active_adapters()
    channels = []
    for adapter in adapters:
        name = getattr(adapter, "name", type(adapter).__name__)
        running = getattr(adapter, "running", None)
        if running is None:
            running = getattr(adapter, "_running", None)
        channels.append({
            "name": name,
            "type": type(adapter).__name__,
            "active": bool(running) if running is not None else True,
        })
    return {"channels": channels, "count": len(channels)}


@router.post("/channels/{name}/test")
async def test_channel(name: str, body: TestMessageBody | None = None):
    """Send a test message through a named channel."""
    adapters = _get_active_adapters()
    target = None
    for adapter in adapters:
        adapter_name = getattr(
            adapter, "name", type(adapter).__name__,
        )
        if adapter_name == name:
            target = adapter
            break
    if target is None:
        raise HTTPException(404, f"Channel '{name}' not found")
    msg = body.message if body else "Test notification"
    try:
        await target.send(msg)
    except Exception as exc:
        log.warning("Test send to %s failed: %s", name, exc)
        return {
            "status": "error",
            "channel": name,
            "error": str(exc),
        }
    return {"status": "sent", "channel": name}


@router.get("/preferences")
async def get_preferences():
    """Get notification routing preferences."""
    nr = _get_notification_router()
    if nr is None:
        raise HTTPException(
            503, "Notification router not initialized",
        )
    prefs = await nr.get_preferences()
    return prefs


@router.put("/preferences")
async def update_preferences(body: UpdatePreferencesBody):
    """Update notification routing preferences."""
    nr = _get_notification_router()
    if nr is None:
        raise HTTPException(
            503, "Notification router not initialized",
        )
    await nr.set_preferences(body.preferences)
    return {"status": "updated"}


@router.get("/webhooks")
async def list_webhooks():
    """List outbound webhook targets."""
    wh = _get_webhook_notifier()
    if wh is None:
        raise HTTPException(
            503, "Webhook notifier not initialized",
        )
    targets = await wh.list_targets()
    return {"webhooks": targets, "count": len(targets)}


@router.post("/webhooks")
async def add_webhook(body: AddWebhookBody):
    """Register a new outbound webhook target."""
    wh = _get_webhook_notifier()
    if wh is None:
        raise HTTPException(
            503, "Webhook notifier not initialized",
        )
    target = await wh.add_target(
        url=body.url,
        secret=body.secret,
        events=body.events,
        enabled=body.enabled,
    )
    return target


@router.put("/webhooks/{url:path}")
async def update_webhook(url: str, body: UpdateWebhookBody):
    """Update an existing webhook target."""
    wh = _get_webhook_notifier()
    if wh is None:
        raise HTTPException(
            503, "Webhook notifier not initialized",
        )
    existing = await wh.get_target(url)
    if existing is None:
        raise HTTPException(404, f"Webhook '{url}' not found")
    updated = await wh.update_target(
        url=url,
        secret=body.secret,
        events=body.events,
        enabled=body.enabled,
    )
    return updated


@router.delete("/webhooks/{url:path}")
async def delete_webhook(url: str):
    """Remove an outbound webhook target."""
    wh = _get_webhook_notifier()
    if wh is None:
        raise HTTPException(
            503, "Webhook notifier not initialized",
        )
    existing = await wh.get_target(url)
    if existing is None:
        raise HTTPException(404, f"Webhook '{url}' not found")
    await wh.remove_target(url)
    return {"status": "deleted", "url": url}


@router.get("/history")
async def notification_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Get notification delivery history."""
    nr = _get_notification_router()
    if nr is None:
        raise HTTPException(
            503, "Notification router not initialized",
        )
    history = await nr.get_history(limit=limit, offset=offset)
    return {
        "history": history,
        "count": len(history),
        "limit": limit,
        "offset": offset,
    }


@router.get("/stats")
async def communication_stats():
    """Return aggregate communication channel statistics."""
    nr = _get_notification_router()
    wh = _get_webhook_notifier()
    adapters = _get_active_adapters()

    stats: dict[str, Any] = {
        "active_channels": len(adapters),
    }

    if nr is not None:
        try:
            stats["routing"] = await nr.get_stats()
        except Exception:
            stats["routing"] = None

    if wh is not None:
        try:
            stats["webhooks"] = await wh.get_stats()
        except Exception:
            stats["webhooks"] = None

    return stats

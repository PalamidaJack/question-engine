"""Chat API endpoints extracted from app.py."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from qe.api.auth import get_auth_provider
from qe.bus import get_bus
from qe.models.envelope import Envelope
from qe.services.chat.schemas import AgentPermissions, PermissionScope

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


def get_app_globals():
    import qe.api.app as app_mod

    return app_mod

@router.post("")
async def chat_rest(request: Request, body: dict[str, Any]):
    """REST endpoint for chat (non-streaming fallback)."""
    if not request.app.state.chat_service:
        return JSONResponse({"error": "Engine not started"}, status_code=503)

    message = body.get("message", "").strip()
    session_id = body.get("session_id")

    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    if not session_id:
        session_id = str(uuid.uuid4())

    response = await request.app.state.chat_service.handle_message(session_id, message)
    return {
        "session_id": session_id,
        **response.model_dump(mode="json"),
    }


# ── WebSocket ───────────────────────────────────────────────────────────────


async def _ws_authenticate(websocket: WebSocket) -> bool:
    """Check API key from query params for WebSocket connections.

    Returns True if auth passes (or auth is disabled). Closes the
    WebSocket with code 4008 and returns False if auth fails.
    """
    provider = get_auth_provider()
    if not provider.enabled:
        return True
    api_key = websocket.query_params.get("api_key", "")
    if not api_key:
        await websocket.close(code=4008, reason="X-API-Key required")
        return False
    ctx = provider.validate_key(api_key)
    if ctx is None:
        await websocket.close(code=4008, reason="Invalid API key")
        return False
    return True


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not await _ws_authenticate(websocket):
        return
    app_mod = get_app_globals()
    await app_mod.ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            log.debug("WS received: %s", data)
    except WebSocketDisconnect:
        app_mod.ws_manager.disconnect(websocket)


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """Per-session chat WebSocket with pipeline progress events."""
    if not await _ws_authenticate(websocket):
        return
    await websocket.accept()
    session_id = str(uuid.uuid4())

    if not websocket.app.state.chat_service:
        await websocket.send_json({
            "type": "error",
            "error": "Engine not started. Please complete setup first.",
        })
        await websocket.close()
        return

    chat_svc = websocket.app.state.chat_service
    default_perms = AgentPermissions.from_access_mode(chat_svc._access_mode)
    await websocket.send_json({
        "type": "session_init",
        "session_id": session_id,
        "permissions": {
            "scopes": {s.value: v for s, v in default_perms.scopes.items()},
            "budget_cap_usd": default_perms.budget_cap_usd,
        },
    })

    # Track envelopes for pipeline progress forwarding
    tracked_envelopes: set[str] = set()

    async def _pipeline_forwarder(envelope: Envelope) -> None:
        """Forward pipeline events for envelopes this session is tracking."""
        correlation = envelope.correlation_id or envelope.causation_id
        if correlation in tracked_envelopes:
            try:
                await websocket.send_json({
                    "type": "pipeline_event",
                    "topic": envelope.topic,
                    "envelope_id": envelope.envelope_id,
                    "correlation_id": correlation,
                    "payload": envelope.payload,
                    "timestamp": envelope.timestamp.isoformat(),
                })
            except Exception:
                pass

    pipeline_topics = [
        "claims.proposed",
        "claims.committed",
        "claims.contradiction_detected",
    ]
    bus = get_bus()
    for topic in pipeline_topics:
        bus.subscribe(topic, _pipeline_forwarder)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "message")

            if msg_type == "message":
                user_text = data.get("content", "").strip()
                if not user_text:
                    continue

                await websocket.send_json({"type": "typing", "active": True})

                progress_queue: asyncio.Queue[dict] = asyncio.Queue()
                interjection_queue: asyncio.Queue[str] = asyncio.Queue()
                _mid_execution = [True]

                async def _forward_progress(_q=progress_queue):
                    while True:
                        event = await _q.get()
                        if event is None:
                            break
                        try:
                            await websocket.send_json(event)
                        except Exception:
                            break

                async def _receive_interjections(
                    _iq=interjection_queue, _flag=_mid_execution,
                ):
                    """Read messages during execution, route as interjections."""
                    try:
                        while _flag[0]:
                            inj_data = await asyncio.wait_for(
                                websocket.receive_json(), timeout=1.0,
                            )
                            inj_type = inj_data.get("type", "message")
                            if inj_type in ("interjection", "message"):
                                text = inj_data.get("content", "").strip()
                                if text:
                                    await _iq.put(text)
                    except TimeoutError:
                        pass
                    except Exception:
                        pass

                forwarder_task = asyncio.create_task(_forward_progress())
                receiver_task = asyncio.create_task(_receive_interjections())
                try:
                    response = await websocket.app.state.chat_service.handle_message(
                        session_id, user_text,
                        progress_queue=progress_queue,
                        interjection_queue=interjection_queue,
                    )
                finally:
                    _mid_execution[0] = False
                    receiver_task.cancel()
                    await progress_queue.put(None)
                    await forwarder_task

                if response.tracking_envelope_id:
                    tracked_envelopes.add(response.tracking_envelope_id)

                await websocket.send_json({"type": "typing", "active": False})
                await websocket.send_json({
                    "type": "chat_response",
                    **response.model_dump(mode="json"),
                })

            elif msg_type == "track_envelope":
                envelope_id = data.get("envelope_id")
                if envelope_id:
                    tracked_envelopes.add(envelope_id)

            elif msg_type == "set_permissions":
                try:
                    preset = data.get("preset")
                    if preset:
                        perms = AgentPermissions.from_preset(preset)
                    else:
                        raw_scopes = data.get("scopes", {})
                        scope_dict = {
                            PermissionScope(k): bool(v)
                            for k, v in raw_scopes.items()
                            if k in PermissionScope.__members__.values()
                              or k in [s.value for s in PermissionScope]
                        }
                        perms = AgentPermissions(scopes=scope_dict)
                    budget = data.get("budget_cap_usd")
                    if budget is not None:
                        perms.budget_cap_usd = float(budget)
                    chat_svc.set_session_permissions(session_id, perms)
                    await websocket.send_json({
                        "type": "permissions_updated",
                        "scopes": {s.value: v for s, v in perms.scopes.items()},
                        "budget_cap_usd": perms.budget_cap_usd,
                    })
                except Exception as exc:
                    log.warning("set_permissions failed: %s", exc)
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Failed to set permissions: {exc}",
                    })

    except WebSocketDisconnect:
        pass
    finally:
        for topic in pipeline_topics:
            bus.unsubscribe(topic, _pipeline_forwarder)


@router.get("/stream")
async def chat_stream(request: Request, message: str = "", session_id: str | None = None):
    """SSE chat stream endpoint. Spawns `ChatService.handle_message` and
    streams typed progress events as Server-Sent Events.
    """

    if request.app.state.chat_service is None:
        return JSONResponse({"error": "chat service not available"}, status_code=503)

    progress_queue: asyncio.Queue[dict] = asyncio.Queue()

    async def event_generator():
        # Start the chat handler in background
        task = asyncio.create_task(
            request.app.state.chat_service.handle_message(
                session_id or None, message, progress_queue=progress_queue,
            )
        )
        try:
            while True:
                ev = await progress_queue.get()
                if ev is None:
                    break
                # Ensure we send valid SSE data lines
                try:
                    data = json.dumps(ev)
                except Exception:
                    data = json.dumps({"type": "error", "message": "serialization_failed"})
                yield f"data: {data}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


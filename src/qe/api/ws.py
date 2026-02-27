"""WebSocket connection manager for real-time bus event streaming."""

from __future__ import annotations

import logging

from fastapi import WebSocket

log = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts bus events."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        log.info("WebSocket client connected (%d total)", len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.remove(websocket)
        log.info("WebSocket client disconnected (%d total)", len(self._connections))

    async def broadcast(self, message: str) -> None:
        """Send a message to all connected clients."""
        disconnected = []
        for conn in self._connections:
            try:
                await conn.send_text(message)
            except Exception:
                disconnected.append(conn)
        for conn in disconnected:
            self._connections.remove(conn)

    @property
    def active_count(self) -> int:
        return len(self._connections)

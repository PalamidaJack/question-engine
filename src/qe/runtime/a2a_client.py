"""A2A client for outbound agent-to-agent delegation (Phase 5).

Provides a small async client wrapper around HTTP endpoints exposed by
other A2A-compatible agents. Uses `httpx.AsyncClient` with sensible timeouts.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

import httpx


class A2AClient:
    def __init__(self, base_url: str, timeout: float = 30.0, auth: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._auth = auth

    async def discover(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self.base_url}/.well-known/agent.json")
            r.raise_for_status()
            return r.json()

    async def send_task(self, description: str, task_id: str | None = None) -> dict[str, Any]:
        payload = {"description": description}
        if task_id:
            payload["id"] = task_id
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self.base_url}/api/a2a/tasks", json=payload)
            r.raise_for_status()
            return r.json()

    async def get_task_status(self, task_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self.base_url}/api/a2a/tasks/{task_id}")
            r.raise_for_status()
            return r.json()

    async def send_message(self, task_id: str, message: dict[str, Any]) -> None:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self.base_url}/api/a2a/tasks/{task_id}/messages", json=message)
            r.raise_for_status()

    async def stream_task(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=None) as c:
            url = f"{self.base_url}/api/a2a/tasks/{task_id}/stream"
            async with c.stream("GET", url) as stream:
                async for chunk in stream.aiter_text():
                    if not chunk:
                        continue
                    # SSE style may contain 'data: ...' lines
                    for line in chunk.splitlines():
                        if line.startswith("data:"):
                            payload = line.split("data:", 1)[1].strip()
                            try:
                                yield httpx.Response(200, content=payload).json()
                            except Exception:
                                yield {"raw": payload}

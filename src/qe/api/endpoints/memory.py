"""Memory API endpoints."""

from __future__ import annotations

from typing import Any


def register_memory_routes(
    app: Any,
    memory_store: Any,
) -> None:
    """Register memory-related API routes."""
    from fastapi import HTTPException

    @app.get("/api/memory")
    async def list_memories():
        if not memory_store:
            raise HTTPException(
                503, "Memory store not initialized"
            )
        entries = await memory_store.get_all_active()
        return {
            "memories": [
                {
                    "memory_id": e.memory_id,
                    "category": e.category,
                    "key": e.key,
                    "value": e.value,
                    "confidence": e.confidence,
                    "source": e.source,
                }
                for e in entries
            ]
        }

    @app.post("/api/memory/preferences")
    async def set_preference(body: dict):
        if not memory_store:
            raise HTTPException(
                503, "Memory store not initialized"
            )
        key = body.get("key", "")
        value = body.get("value", "")
        if not key or not value:
            raise HTTPException(
                400, "key and value required"
            )
        entry = await memory_store.set_preference(key, value)
        return {
            "memory_id": entry.memory_id,
            "key": entry.key,
            "value": entry.value,
        }

    @app.delete("/api/memory/{memory_id}")
    async def delete_memory(memory_id: str):
        if not memory_store:
            raise HTTPException(
                503, "Memory store not initialized"
            )
        deleted = await memory_store.delete(memory_id)
        if not deleted:
            raise HTTPException(404, "Memory not found")
        return {"deleted": True}

    @app.get("/api/projects")
    async def list_projects():
        if not memory_store:
            raise HTTPException(
                503, "Memory store not initialized"
            )
        projects = await memory_store.list_projects()
        return {"projects": projects}

    @app.post("/api/projects")
    async def create_project(body: dict):
        if not memory_store:
            raise HTTPException(
                503, "Memory store not initialized"
            )
        name = body.get("name", "")
        if not name:
            raise HTTPException(400, "name required")
        desc = body.get("description", "")
        project = await memory_store.create_project(
            name, desc
        )
        return project

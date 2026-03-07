"""Agent profile management endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/profiles", tags=["profiles"])


def _get_profile_loader():
    import qe.api.app as app_mod

    loader = getattr(app_mod, "_profile_loader", None)
    if loader is None:
        raise HTTPException(503, "Profile system not initialized")
    return loader


# ── Request / Response Models ───────────────────────────────────────


class SwitchProfileBody(BaseModel):
    profile: str


class SaveFileBody(BaseModel):
    path: str
    content: str


class ImportProfileBody(BaseModel):
    name: str
    files: dict[str, str]


# ── Endpoints ───────────────────────────────────────────────────────


@router.get("/")
async def list_profiles():
    """List available profiles."""
    loader = _get_profile_loader()
    profiles = await loader.list_profiles()
    return {"profiles": profiles, "count": len(profiles)}


@router.get("/active")
async def get_active_profile():
    """Get active profile name and manifest."""
    loader = _get_profile_loader()
    name = loader.active_profile
    manifest = await loader.get_manifest()
    return {"profile": name, "manifest": manifest}


@router.post("/switch")
async def switch_profile(body: SwitchProfileBody):
    """Switch the active profile."""
    loader = _get_profile_loader()
    try:
        await loader.switch(body.profile)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"status": "switched", "profile": body.profile}


@router.get("/files")
async def list_profile_files():
    """List all files in the active profile."""
    loader = _get_profile_loader()
    files = await loader.list_files()
    return {"files": files, "count": len(files)}


@router.get("/file")
async def get_file_content(path: str = Query(...)):
    """Get a single file's content from the active profile."""
    loader = _get_profile_loader()
    try:
        content = await loader.read_file(path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"path": path, "content": content}


@router.put("/file")
async def save_file_content(body: SaveFileBody):
    """Save (create or update) a file in the active profile."""
    loader = _get_profile_loader()
    await loader.write_file(body.path, body.content)
    return {"status": "saved", "path": body.path}


@router.delete("/file")
async def delete_file(path: str = Query(...)):
    """Delete a file from the active profile."""
    loader = _get_profile_loader()
    try:
        await loader.delete_file(path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"status": "deleted", "path": path}


@router.get("/playbooks")
async def list_playbooks():
    """List playbooks in the active profile."""
    loader = _get_profile_loader()
    playbooks = await loader.list_playbooks()
    return {"playbooks": playbooks, "count": len(playbooks)}


@router.get("/playbook/{name}")
async def get_playbook(name: str):
    """Get a single playbook's content."""
    loader = _get_profile_loader()
    try:
        content = await loader.read_playbook(name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"name": name, "content": content}


@router.post("/export")
async def export_profile():
    """Export the active profile as a JSON dict of files."""
    loader = _get_profile_loader()
    data = await loader.export_profile()
    return data


@router.post("/import")
async def import_profile(body: ImportProfileBody):
    """Import a profile from a dict of files."""
    loader = _get_profile_loader()
    try:
        await loader.import_profile(body.name, body.files)
    except (ValueError, FileExistsError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"status": "imported", "profile": body.name}

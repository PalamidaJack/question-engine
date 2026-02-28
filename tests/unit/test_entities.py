"""Tests for entity resolution: normalization, resolve, and alias management."""

import os
import tempfile

import pytest

from qe.substrate.entities import EntityResolver, normalize_entity


def test_normalize_lowercase():
    assert normalize_entity("SpaceX") == "spacex"


def test_normalize_strip_suffix():
    assert normalize_entity("Apple Inc.") == "apple"
    assert normalize_entity("Siemens AG") == "siemens"
    assert normalize_entity("Shell PLC") == "shell"


def test_normalize_collapse_whitespace():
    assert normalize_entity("Space  X") == "space x"
    assert normalize_entity("  Tesla  Motors  ") == "tesla motors"


def test_normalize_combined():
    assert normalize_entity("  SpaceX  Inc. ") == "spacex"


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
async def resolver(db_path):
    """Create resolver with initialized entities table."""
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                canonical_name TEXT PRIMARY KEY,
                aliases TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await db.commit()

    return EntityResolver(db_path)


@pytest.mark.asyncio
async def test_resolve_unknown_returns_normalized(resolver):
    result = await resolver.resolve("SpaceX Inc.")
    assert result == "spacex"


@pytest.mark.asyncio
async def test_resolve_via_alias(resolver):
    await resolver.add_alias("spacex", "space exploration technologies")
    result = await resolver.resolve("Space Exploration Technologies")
    assert result == "spacex"


@pytest.mark.asyncio
async def test_ensure_entity_creates_record(resolver):
    canonical = await resolver.ensure_entity("Tesla Motors Inc.")
    assert canonical == "tesla motors"

    entities = await resolver.list_entities()
    names = [e["canonical_name"] for e in entities]
    assert "tesla motors" in names


@pytest.mark.asyncio
async def test_add_alias_idempotent(resolver):
    await resolver.add_alias("google", "alphabet")
    await resolver.add_alias("google", "alphabet")

    entities = await resolver.list_entities()
    google = [e for e in entities if e["canonical_name"] == "google"][0]
    assert google["aliases"].count("alphabet") == 1

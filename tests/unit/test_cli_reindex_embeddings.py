from __future__ import annotations

import aiosqlite
import pytest
from unittest.mock import AsyncMock

from qe.cli.main import _reindex_claim_embeddings
from qe.models.claim import Claim
from qe.substrate import Substrate


@pytest.mark.asyncio
async def test_reindex_embeddings_rebuilds_claim_vectors_and_keeps_non_claim(tmp_path):
    substrate = Substrate(
        db_path=str(tmp_path / "reindex.db"),
        cold_path=str(tmp_path / "cold"),
    )
    await substrate.initialize()

    # Deterministic local embeddings for test isolation.
    substrate.embeddings.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])

    c1 = await substrate.commit_claim(
        Claim(
            subject_entity_id="acme",
            predicate="acquired",
            object_value="beta",
            confidence=0.9,
            source_service_id="test",
            source_envelope_ids=["env-1"],
        )
    )
    c2 = await substrate.commit_claim(
        Claim(
            subject_entity_id="globex",
            predicate="released",
            object_value="product-x",
            confidence=0.8,
            source_service_id="test",
            source_envelope_ids=["env-2"],
        )
    )

    # Add one stale claim embedding + one non-claim embedding.
    await substrate.embeddings.store(
        id="claim:stale",
        text="stale claim",
        metadata={"kind": "claim", "claim_id": "stale"},
    )
    await substrate.embeddings.store(
        id="goal_pattern:abc",
        text="pattern",
        metadata={"kind": "pattern"},
    )

    stats = await _reindex_claim_embeddings(substrate, batch_size=1)
    assert stats["indexed"] == 2
    assert stats["total"] == 2
    assert stats["deleted"] >= 1

    async with aiosqlite.connect(substrate.belief_ledger._db_path) as db:
        cursor = await db.execute("SELECT id FROM embeddings ORDER BY id")
        ids = [row[0] for row in await cursor.fetchall()]

    assert f"claim:{c1.claim_id}" in ids
    assert f"claim:{c2.claim_id}" in ids
    assert "claim:stale" not in ids
    assert "goal_pattern:abc" in ids


@pytest.mark.asyncio
async def test_reindex_embeddings_dry_run_makes_no_changes(tmp_path):
    substrate = Substrate(
        db_path=str(tmp_path / "reindex_dry.db"),
        cold_path=str(tmp_path / "cold"),
    )
    await substrate.initialize()
    substrate.embeddings.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])

    claim = await substrate.commit_claim(
        Claim(
            subject_entity_id="acme",
            predicate="acquired",
            object_value="beta",
            confidence=0.9,
            source_service_id="test",
            source_envelope_ids=["env-1"],
        )
    )
    await substrate.embeddings.store(
        id="claim:stale",
        text="stale claim",
        metadata={"kind": "claim", "claim_id": "stale"},
    )

    async with aiosqlite.connect(substrate.belief_ledger._db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM embeddings WHERE id LIKE 'claim:%'")
        before = (await cursor.fetchone())[0]

    stats = await _reindex_claim_embeddings(substrate, dry_run=True, batch_size=2)
    assert stats["dry_run"] == 1
    assert stats["indexed"] == 1
    assert stats["deleted"] == before

    async with aiosqlite.connect(substrate.belief_ledger._db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM embeddings WHERE id LIKE 'claim:%'")
        after = (await cursor.fetchone())[0]
        cursor = await db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE id = ?",
            (f"claim:{claim.claim_id}",),
        )
        present_claim = (await cursor.fetchone())[0]
        cursor = await db.execute(
            "SELECT COUNT(*) FROM embeddings WHERE id = 'claim:stale'"
        )
        present_stale = (await cursor.fetchone())[0]

    assert after == before
    assert present_claim == 1
    assert present_stale == 1

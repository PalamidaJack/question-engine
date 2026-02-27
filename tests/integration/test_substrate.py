import shutil
import tempfile
from pathlib import Path

import pytest

from qe.models.claim import Claim
from qe.models.envelope import Envelope
from qe.substrate.belief_ledger import BeliefLedger
from qe.substrate.cold_storage import ColdStorage


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    import os
    os.close(fd)
    yield path
    import os
    os.unlink(path)


@pytest.fixture
async def ledger(temp_db_path):
    """Create a BeliefLedger with a temporary database."""
    # Find the migrations directory relative to this file
    migrations_dir = Path(__file__).parent.parent.parent / "src" / "qe" / "substrate" / "migrations"

    ledger = BeliefLedger(temp_db_path, migrations_dir)
    await ledger.initialize()
    yield ledger


@pytest.fixture
def temp_cold_path():
    """Create a temporary cold storage directory."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def cold_storage(temp_cold_path):
    """Create a ColdStorage with a temporary directory."""
    return ColdStorage(temp_cold_path)


@pytest.mark.asyncio
async def test_commit_and_get_claim(ledger):
    """Commit a claim, read it back with get_claims, assert all fields match."""
    claim = Claim(
        subject_entity_id="entity-123",
        predicate="has-property",
        object_value="value-456",
        confidence=0.9,
        source_service_id="test-service",
        source_envelope_ids=["env-123"]
    )

    await ledger.commit_claim(claim)

    claims = await ledger.get_claims(
        subject_entity_id="entity-123",
        predicate="has-property"
    )

    assert len(claims) == 1
    retrieved = claims[0]
    assert retrieved.claim_id == claim.claim_id
    assert retrieved.subject_entity_id == claim.subject_entity_id
    assert retrieved.predicate == claim.predicate
    assert retrieved.object_value == claim.object_value
    assert retrieved.confidence == claim.confidence
    assert retrieved.source_service_id == claim.source_service_id
    assert retrieved.source_envelope_ids == claim.source_envelope_ids
    assert retrieved.superseded_by is None


@pytest.mark.asyncio
async def test_supersession_logic_higher_confidence(ledger):
    """
    Commit two claims with same (subject_entity_id, predicate) where second has
    higher confidence — assert first is superseded and second is current.
    """
    claim1 = Claim(
        subject_entity_id="entity-123",
        predicate="has-property",
        object_value="value-old",
        confidence=0.5,
        source_service_id="test-service",
        source_envelope_ids=["env-1"]
    )

    claim2 = Claim(
        subject_entity_id="entity-123",
        predicate="has-property",
        object_value="value-new",
        confidence=0.8,
        source_service_id="test-service",
        source_envelope_ids=["env-2"]
    )

    await ledger.commit_claim(claim1)
    await ledger.commit_claim(claim2)

    # Get current claims (not superseded)
    current_claims = await ledger.get_claims(
        subject_entity_id="entity-123",
        predicate="has-property",
        include_superseded=False
    )

    assert len(current_claims) == 1
    assert current_claims[0].claim_id == claim2.claim_id
    assert current_claims[0].superseded_by is None

    # Get all claims (including superseded)
    all_claims = await ledger.get_claims(
        subject_entity_id="entity-123",
        predicate="has-property",
        include_superseded=True
    )

    assert len(all_claims) == 2
    superseded = [c for c in all_claims if c.claim_id == claim1.claim_id]
    assert len(superseded) == 1
    assert superseded[0].superseded_by == claim2.claim_id


@pytest.mark.asyncio
async def test_alternative_claim_lower_confidence(ledger):
    """
    Commit two claims with same subject/predicate where second has lower
    confidence — assert neither is superseded.
    """
    claim1 = Claim(
        subject_entity_id="entity-456",
        predicate="has-property",
        object_value="value-high",
        confidence=0.9,
        source_service_id="test-service",
        source_envelope_ids=["env-1"]
    )

    claim2 = Claim(
        subject_entity_id="entity-456",
        predicate="has-property",
        object_value="value-low",
        confidence=0.4,
        source_service_id="test-service",
        source_envelope_ids=["env-2"]
    )

    await ledger.commit_claim(claim1)
    await ledger.commit_claim(claim2)

    # Both should be current (not superseded)
    current_claims = await ledger.get_claims(
        subject_entity_id="entity-456",
        predicate="has-property",
        include_superseded=False
    )

    assert len(current_claims) == 2
    for claim in current_claims:
        assert claim.superseded_by is None


@pytest.mark.asyncio
async def test_search_full_text_not_implemented(ledger):
    """
    search_full_text on a term present in a committed claim raises NotImplementedError.
    """
    claim = Claim(
        subject_entity_id="entity-789",
        predicate="contains-term",
        object_value="special-term-value",
        confidence=0.95,
        source_service_id="test-service",
        source_envelope_ids=["env-123"]
    )

    await ledger.commit_claim(claim)

    with pytest.raises(NotImplementedError):
        await ledger.search_full_text("special-term")


def test_cold_storage_append_and_read(cold_storage):
    """
    ColdStorage: append an envelope, read it back by ID, assert equality.
    """
    envelope = Envelope(
        topic="observations.structured",
        source_service_id="scanner",
        payload={"text": "test observation"}
    )

    written_path = cold_storage.append(envelope)

    # Verify file was created
    assert written_path.exists()

    # Read it back
    year = envelope.timestamp.year
    month = envelope.timestamp.month
    retrieved = cold_storage.read(envelope.envelope_id, year, month)

    assert retrieved is not None
    assert retrieved.envelope_id == envelope.envelope_id
    assert retrieved.topic == envelope.topic
    assert retrieved.source_service_id == envelope.source_service_id
    assert retrieved.payload == envelope.payload


def test_cold_storage_read_nonexistent(cold_storage):
    """ColdStorage: reading a nonexistent envelope returns None."""
    retrieved = cold_storage.read("nonexistent-id", 2026, 2)
    assert retrieved is None

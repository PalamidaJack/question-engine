"""Tests for ClaimValidatorService: duplicate rejection, contradiction detection."""

import asyncio
from pathlib import Path

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.claim import Claim
from qe.models.envelope import Envelope
from qe.models.genome import Blueprint
from qe.services.validator.service import ClaimValidatorService
from qe.substrate import Substrate


def _make_blueprint() -> Blueprint:
    return Blueprint.model_validate({
        "service_id": "validator_alpha",
        "display_name": "Claim Validator",
        "version": "1.0",
        "system_prompt": "Validate claims.",
        "model_preference": {"tier": "balanced"},
        "capabilities": {
            "bus_topics_subscribe": ["claims.proposed"],
            "bus_topics_publish": [
                "claims.committed",
                "claims.contradiction_detected",
            ],
        },
    })


@pytest.fixture
async def setup(tmp_path: Path):
    bus = MemoryBus()
    substrate = Substrate(
        db_path=str(tmp_path / "test.db"),
        cold_path=str(tmp_path / "cold"),
    )
    await substrate.initialize()

    validator = ClaimValidatorService(_make_blueprint(), bus, substrate)
    await validator.start()
    return bus, substrate, validator


@pytest.mark.asyncio
async def test_valid_claim_committed(setup):
    bus, substrate, validator = setup

    committed = []
    bus.subscribe("claims.committed", lambda e: committed.append(e))

    claim = Claim(
        subject_entity_id="spacex",
        predicate="launched",
        object_value="Falcon 9",
        confidence=0.9,
        source_service_id="researcher_alpha",
        source_envelope_ids=["env-1"],
    )

    bus.publish(Envelope(
        topic="claims.proposed",
        source_service_id="researcher_alpha",
        payload=claim.model_dump(mode="json"),
    ))
    await asyncio.sleep(0.1)

    assert len(committed) == 1
    claims = await substrate.get_claims()
    assert len(claims) == 1
    await validator.stop()


@pytest.mark.asyncio
async def test_duplicate_claim_dropped(setup):
    bus, substrate, validator = setup

    committed = []
    bus.subscribe("claims.committed", lambda e: committed.append(e))

    claim = Claim(
        subject_entity_id="spacex",
        predicate="launched",
        object_value="Falcon 9",
        confidence=0.9,
        source_service_id="researcher_alpha",
        source_envelope_ids=["env-1"],
    )

    # Publish same claim twice
    for _ in range(2):
        bus.publish(Envelope(
            topic="claims.proposed",
            source_service_id="researcher_alpha",
            payload=claim.model_dump(mode="json"),
        ))
        await asyncio.sleep(0.1)

    # Only one should be committed (second is duplicate)
    assert len(committed) == 1
    await validator.stop()


@pytest.mark.asyncio
async def test_contradiction_detected(setup):
    bus, substrate, validator = setup

    contradictions = []
    bus.subscribe(
        "claims.contradiction_detected",
        lambda e: contradictions.append(e),
    )

    claim1 = Claim(
        subject_entity_id="earth",
        predicate="shape",
        object_value="round",
        confidence=0.95,
        source_service_id="researcher_alpha",
        source_envelope_ids=["env-1"],
    )
    claim2 = Claim(
        subject_entity_id="earth",
        predicate="shape",
        object_value="flat",
        confidence=0.1,
        source_service_id="researcher_alpha",
        source_envelope_ids=["env-2"],
    )

    bus.publish(Envelope(
        topic="claims.proposed",
        source_service_id="researcher_alpha",
        payload=claim1.model_dump(mode="json"),
    ))
    await asyncio.sleep(0.1)

    bus.publish(Envelope(
        topic="claims.proposed",
        source_service_id="researcher_alpha",
        payload=claim2.model_dump(mode="json"),
    ))
    await asyncio.sleep(0.1)

    assert len(contradictions) >= 1
    assert contradictions[0].payload["subject"] == "earth"
    assert contradictions[0].payload["predicate"] == "shape"
    await validator.stop()

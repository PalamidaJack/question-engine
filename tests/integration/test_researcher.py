import asyncio
from pathlib import Path

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.envelope import Envelope
from qe.models.genome import Blueprint
from qe.services.researcher.service import ResearcherService
from qe.services.validator.service import ClaimValidatorService
from qe.substrate import Substrate


@pytest.mark.integration
@pytest.mark.asyncio
async def test_researcher_observation_to_claim_pipeline(tmp_path: Path, mock_llm):
    bus = MemoryBus()
    substrate = Substrate(
        db_path=str(tmp_path / "belief.db"),
        cold_path=str(tmp_path / "cold"),
    )
    await substrate.initialize()

    researcher_bp = Blueprint.model_validate({
        "service_id": "researcher_alpha",
        "display_name": "Researcher Alpha",
        "version": "1.0",
        "system_prompt": "Extract claims.",
        "model_preference": {"tier": "balanced"},
        "capabilities": {
            "bus_topics_subscribe": ["observations.structured"],
            "bus_topics_publish": ["claims.proposed"],
        },
    })

    validator_bp = Blueprint.model_validate({
        "service_id": "validator_alpha",
        "display_name": "Claim Validator",
        "version": "1.0",
        "system_prompt": "Validate claims.",
        "model_preference": {"tier": "balanced"},
        "capabilities": {
            "bus_topics_subscribe": ["claims.proposed"],
            "bus_topics_publish": ["claims.committed", "claims.contradiction_detected"],
        },
    })

    researcher = ResearcherService(researcher_bp, bus, substrate)
    validator = ClaimValidatorService(validator_bp, bus, substrate)

    proposed: list[Envelope] = []
    committed: list[Envelope] = []

    async def on_proposed(env: Envelope) -> None:
        proposed.append(env)

    async def on_committed(env: Envelope) -> None:
        committed.append(env)

    bus.subscribe("claims.proposed", on_proposed)
    bus.subscribe("claims.committed", on_committed)

    await researcher.start()
    await validator.start()

    bus.publish(
        Envelope(
            topic="observations.structured",
            source_service_id="cli",
            payload={"text": "Water vapor was found in K2-18b atmosphere."},
        )
    )

    # Two async hops: observation→researcher→proposed, proposed→validator→committed
    await asyncio.sleep(0.5)

    assert len(proposed) >= 1
    claims = await substrate.get_claims()
    assert len(claims) >= 1
    assert len(committed) >= 1

    cold_files = list((tmp_path / "cold").glob("**/*.json"))
    assert cold_files, "Expected observation envelope in cold storage"

    await validator.stop()
    await researcher.stop()

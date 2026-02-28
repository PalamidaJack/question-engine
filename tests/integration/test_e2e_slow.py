"""End-to-end test: submit observation -> extract claims -> validate -> commit -> query.

Requires a real LLM API key. Skip in CI with: pytest -m "not slow"
"""

import asyncio
import os
from pathlib import Path

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.envelope import Envelope
from qe.models.genome import Blueprint
from qe.services.query import answer_question
from qe.services.researcher.service import ResearcherService
from qe.services.validator.service import ClaimValidatorService
from qe.substrate import Substrate


def _has_any_llm_key() -> bool:
    """Check if any LLM API key is available (including from .env)."""
    from dotenv import load_dotenv

    load_dotenv()

    return bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("KILOCODE_API_KEY")
    )


_has_llm_key = _has_any_llm_key()


@pytest.fixture(autouse=False)
def _kilo_code_env():
    """If only KILOCODE_API_KEY is set, temporarily configure litellm env vars."""
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        yield
        return

    kilo_key = os.environ.get("KILOCODE_API_KEY", "")
    if not kilo_key:
        yield
        return

    kilo_base = os.environ.get(
        "KILOCODE_API_BASE", "https://kilo.ai/api/openrouter"
    )
    os.environ["OPENAI_API_KEY"] = kilo_key
    os.environ["OPENAI_API_BASE"] = kilo_base
    yield
    # Clean up to avoid polluting other tests
    if os.environ.get("OPENAI_API_KEY") == kilo_key:
        del os.environ["OPENAI_API_KEY"]
    if os.environ.get("OPENAI_API_BASE") == kilo_base:
        del os.environ["OPENAI_API_BASE"]


@pytest.mark.slow
@pytest.mark.skipif(not _has_llm_key, reason="No LLM API key configured")
@pytest.mark.usefixtures("_kilo_code_env")
@pytest.mark.asyncio
async def test_full_pipeline_observation_to_answer(tmp_path: Path):
    """Full pipeline: observation -> researcher -> validator -> commit -> query."""
    bus = MemoryBus()
    substrate = Substrate(
        db_path=str(tmp_path / "e2e.db"),
        cold_path=str(tmp_path / "cold"),
    )
    await substrate.initialize()

    researcher_bp = Blueprint.model_validate({
        "service_id": "researcher_alpha",
        "display_name": "Researcher Alpha",
        "version": "1.0",
        "system_prompt": (
            "You are a research analyst. Extract falsifiable claims from observations. "
            "Express uncertainty as a confidence score 0.0-1.0."
        ),
        "model_preference": {"tier": "fast"},
        "capabilities": {
            "bus_topics_subscribe": ["observations.structured"],
            "bus_topics_publish": ["claims.proposed"],
        },
    })

    validator_bp = Blueprint.model_validate({
        "service_id": "validator_alpha",
        "display_name": "Claim Validator",
        "version": "1.0",
        "system_prompt": "Validate incoming claims.",
        "model_preference": {"tier": "fast"},
        "capabilities": {
            "bus_topics_subscribe": ["claims.proposed"],
            "bus_topics_publish": ["claims.committed", "claims.contradiction_detected"],
        },
    })

    researcher = ResearcherService(researcher_bp, bus, substrate)
    validator = ClaimValidatorService(validator_bp, bus, substrate)

    committed = []
    bus.subscribe("claims.committed", lambda e: committed.append(e))

    await researcher.start()
    await validator.start()

    # 1. Submit observation
    bus.publish(Envelope(
        topic="observations.structured",
        source_service_id="e2e-test",
        payload={
            "text": "Water vapor was detected in the atmosphere of "
            "exoplanet K2-18b by the James Webb Space Telescope in 2023."
        },
    ))

    # Wait for the full async pipeline
    for _ in range(20):
        await asyncio.sleep(0.5)
        if committed:
            break

    # 2. Verify claims were committed
    claims = await substrate.get_claims()
    assert len(claims) >= 1, f"Expected claims to be committed, got {len(claims)}"

    # 3. Query the belief ledger
    result = await answer_question("What was found on K2-18b?", substrate)

    assert result["answer"], "Expected a non-empty answer"
    assert result["supporting_claims"], "Expected supporting claims"
    assert result["confidence"] > 0, "Expected non-zero confidence"

    await validator.stop()
    await researcher.stop()

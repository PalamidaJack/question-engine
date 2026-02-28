"""Tests for the question answering service."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from qe.models.claim import Claim
from qe.services.query.schemas import AnswerResponse
from qe.services.query.service import answer_question
from qe.substrate import Substrate


@pytest.fixture
async def substrate(tmp_path: Path):
    s = Substrate(
        db_path=str(tmp_path / "test.db"),
        cold_path=str(tmp_path / "cold"),
    )
    await s.initialize()
    return s


@pytest.mark.asyncio
async def test_answer_empty_ledger(substrate):
    """Returns low-confidence 'no info' answer when ledger is empty."""
    result = await answer_question("What is SpaceX?", substrate)
    assert result["confidence"] == 0.0
    assert result["supporting_claims"] == []
    assert "don't have enough" in result["answer"].lower() or "no" in result["answer"].lower()


@pytest.mark.asyncio
async def test_answer_with_claims(substrate):
    """With claims in ledger, synthesizes answer using LLM (mocked)."""
    await substrate.commit_claim(Claim(
        subject_entity_id="spacex",
        predicate="launched",
        object_value="Falcon 9 rocket from Cape Canaveral",
        confidence=0.9,
        source_service_id="test",
        source_envelope_ids=["env-1"],
    ))

    mock_response = AnswerResponse(
        answer="SpaceX launched a Falcon 9 rocket from Cape Canaveral.",
        confidence=0.85,
        reasoning="Based on a claim with 90% confidence.",
    )

    with patch("qe.services.query.service.instructor") as mock_instructor:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_instructor.from_litellm.return_value = mock_client

        result = await answer_question("What did SpaceX launch?", substrate)

    assert result["confidence"] == 0.85
    assert "Falcon 9" in result["answer"]
    assert len(result["supporting_claims"]) >= 1


def test_ask_api_returns_503_when_not_started():
    """POST /api/ask returns 503 when engine not started."""
    from fastapi.testclient import TestClient

    from qe.api.app import app

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/ask", json={"question": "What is SpaceX?"})
    assert resp.status_code == 503

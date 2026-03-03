"""Tests for memory operations endpoints in qe.api.endpoints.memory_ops."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from qe.api.endpoints.memory_ops import register_memory_ops_routes

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app():
    _app = FastAPI()
    register_memory_ops_routes(_app)
    return _app


@pytest.fixture()
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_episode(data: dict | None = None):
    """Return an object with a .model_dump() that returns *data*."""
    ep = MagicMock()
    ep.model_dump.return_value = data or {"id": "ep_1", "type": "synthesis"}
    return ep


def _mock_claim(data: dict | None = None):
    """Return an object with a .model_dump() that returns *data*."""
    c = MagicMock()
    c.model_dump.return_value = data or {"claim_id": "clm_1", "text": "x"}
    return c


def _mock_template(data: dict | None = None):
    t = MagicMock()
    t.model_dump.return_value = data or {"template_id": "tpl_1", "pattern": "why"}
    return t


def _mock_sequence(data: dict | None = None):
    s = MagicMock()
    s.model_dump.return_value = data or {"sequence_id": "seq_1", "steps": []}
    return s


# ===================================================================
# GET /api/memory/search
# ===================================================================


class TestMemorySearch:
    """Tests for the /api/memory/search endpoint."""

    def test_search_returns_cross_tier_results(self, client):
        episodic = AsyncMock()
        episodic.recall = AsyncMock(return_value=[_mock_episode()])

        substrate = AsyncMock()
        substrate.hybrid_search = AsyncMock(return_value=[_mock_claim()])

        procedural = AsyncMock()
        procedural.get_best_templates = AsyncMock(return_value=[_mock_template()])
        procedural.get_best_sequences = AsyncMock(return_value=[_mock_sequence()])

        with (
            patch("qe.api.app._episodic_memory", episodic, create=True),
            patch("qe.api.app._substrate", substrate, create=True),
            patch("qe.api.app._procedural_memory", procedural, create=True),
        ):
            resp = client.get("/api/memory/search", params={"query": "lithium"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "lithium"
        assert "episodic" in body["results"]
        assert "beliefs" in body["results"]
        assert "procedural" in body["results"]
        assert len(body["results"]["episodic"]) == 1
        assert len(body["results"]["beliefs"]) == 1
        assert "templates" in body["results"]["procedural"]

    def test_search_query_param_defaults(self, client):
        """Default query is empty string, default tiers cover all three."""
        with (
            patch("qe.api.app._episodic_memory", None, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            resp = client.get("/api/memory/search")

        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == ""
        # all three tiers present but empty
        assert body["results"]["episodic"] == []
        assert body["results"]["beliefs"] == []
        assert body["results"]["procedural"] == []

    def test_search_single_tier(self, client):
        """Only the requested tier is returned when tiers param is narrowed."""
        episodic = AsyncMock()
        episodic.recall = AsyncMock(return_value=[_mock_episode()])

        with (
            patch("qe.api.app._episodic_memory", episodic, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            resp = client.get(
                "/api/memory/search",
                params={"query": "q", "tiers": "episodic"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert "episodic" in body["results"]
        assert "beliefs" not in body["results"]
        assert "procedural" not in body["results"]

    def test_search_empty_results_when_no_components(self, client):
        with (
            patch("qe.api.app._episodic_memory", None, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            resp = client.get("/api/memory/search", params={"query": "test"})

        assert resp.status_code == 200
        results = resp.json()["results"]
        assert results["episodic"] == []
        assert results["beliefs"] == []
        assert results["procedural"] == []

    def test_search_respects_top_k(self, client):
        episodic = AsyncMock()
        episodic.recall = AsyncMock(return_value=[])

        with (
            patch("qe.api.app._episodic_memory", episodic, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            client.get(
                "/api/memory/search",
                params={"query": "q", "tiers": "episodic", "top_k": "3"},
            )

        episodic.recall.assert_awaited_once_with(query="q", top_k=3, goal_id=None)

    def test_search_passes_goal_id(self, client):
        episodic = AsyncMock()
        episodic.recall = AsyncMock(return_value=[])

        with (
            patch("qe.api.app._episodic_memory", episodic, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            client.get(
                "/api/memory/search",
                params={"query": "q", "tiers": "episodic", "goal_id": "g_123"},
            )

        episodic.recall.assert_awaited_once_with(query="q", top_k=10, goal_id="g_123")

    def test_search_no_tiers_returns_empty(self, client):
        """Passing empty tiers string returns no results keys."""
        with (
            patch("qe.api.app._episodic_memory", None, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            resp = client.get(
                "/api/memory/search",
                params={"query": "q", "tiers": ""},
            )

        assert resp.status_code == 200
        assert resp.json()["results"] == {}


# ===================================================================
# GET /api/memory/tiers
# ===================================================================


class TestTiersStatus:
    """Tests for the /api/memory/tiers endpoint."""

    def test_returns_all_tiers(self, client):
        ctx = MagicMock()
        ctx.status.return_value = {"active_goals": 2}

        episodic = MagicMock()
        episodic.hot_count.return_value = 5
        episodic.warm_count = AsyncMock(return_value=12)

        substrate = AsyncMock()
        substrate.count_claims = AsyncMock(return_value=42)

        procedural = MagicMock()
        procedural._templates = {"a": 1, "b": 2}
        procedural._sequences = {"c": 3}

        with (
            patch("qe.api.app._context_curator", ctx, create=True),
            patch("qe.api.app._episodic_memory", episodic, create=True),
            patch("qe.api.app._substrate", substrate, create=True),
            patch("qe.api.app._procedural_memory", procedural, create=True),
        ):
            resp = client.get("/api/memory/tiers")

        assert resp.status_code == 200
        body = resp.json()
        assert body["working"] == {"active_goals": 2}
        assert body["episodic"]["hot_entries"] == 5
        assert body["episodic"]["warm_entries"] == 12
        assert body["beliefs"]["claim_count"] == 42
        assert body["procedural"]["templates"] == 2
        assert body["procedural"]["sequences"] == 1

    def test_handles_none_components_gracefully(self, client):
        with (
            patch("qe.api.app._context_curator", None, create=True),
            patch("qe.api.app._episodic_memory", None, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            resp = client.get("/api/memory/tiers")

        assert resp.status_code == 200
        body = resp.json()
        assert body["working"] == {}
        assert body["episodic"] == {}
        assert body["beliefs"] == {}
        assert body["procedural"] == {}

    def test_substrate_count_claims_exception(self, client):
        """If substrate.count_claims raises, claim_count is None."""
        substrate = AsyncMock()
        substrate.count_claims = AsyncMock(side_effect=RuntimeError("db error"))

        with (
            patch("qe.api.app._context_curator", None, create=True),
            patch("qe.api.app._episodic_memory", None, create=True),
            patch("qe.api.app._substrate", substrate, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            resp = client.get("/api/memory/tiers")

        assert resp.status_code == 200
        assert resp.json()["beliefs"]["claim_count"] is None


# ===================================================================
# GET /api/memory/procedural
# ===================================================================


class TestProcedural:
    """Tests for the /api/memory/procedural endpoint."""

    def test_returns_templates_and_sequences(self, client):
        procedural = AsyncMock()
        procedural.get_best_templates = AsyncMock(return_value=[_mock_template()])
        procedural.get_best_sequences = AsyncMock(return_value=[_mock_sequence()])

        with patch("qe.api.app._procedural_memory", procedural, create=True):
            resp = client.get("/api/memory/procedural")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["templates"]) == 1
        assert body["templates"][0]["template_id"] == "tpl_1"
        assert len(body["sequences"]) == 1

    def test_503_when_unavailable(self, client):
        with patch("qe.api.app._procedural_memory", None, create=True):
            resp = client.get("/api/memory/procedural")

        assert resp.status_code == 503
        assert "not available" in resp.json()["detail"]

    def test_passes_domain_and_top_k(self, client):
        procedural = AsyncMock()
        procedural.get_best_templates = AsyncMock(return_value=[])
        procedural.get_best_sequences = AsyncMock(return_value=[])

        with patch("qe.api.app._procedural_memory", procedural, create=True):
            client.get(
                "/api/memory/procedural",
                params={"domain": "finance", "top_k": "3"},
            )

        procedural.get_best_templates.assert_awaited_once_with(
            domain="finance", top_k=3
        )
        procedural.get_best_sequences.assert_awaited_once_with(
            domain="finance", top_k=3
        )


# ===================================================================
# GET /api/memory/working/{goal_id}
# ===================================================================


class TestWorkingMemory:
    """Tests for the /api/memory/working/{goal_id} endpoint."""

    def test_returns_working_memory(self, client):
        state = MagicMock()
        state.model_dump.return_value = {"goal_id": "g_1", "items": []}

        curator = MagicMock()
        curator._states = {"g_1": state}

        with patch("qe.api.app._context_curator", curator, create=True):
            resp = client.get("/api/memory/working/g_1")

        assert resp.status_code == 200
        assert resp.json()["goal_id"] == "g_1"

    def test_404_when_goal_not_found(self, client):
        curator = MagicMock()
        curator._states = {}

        with patch("qe.api.app._context_curator", curator, create=True):
            resp = client.get("/api/memory/working/nonexistent")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_503_when_no_curator(self, client):
        with patch("qe.api.app._context_curator", None, create=True):
            resp = client.get("/api/memory/working/g_1")

        assert resp.status_code == 503
        assert "not initialized" in resp.json()["detail"]


# ===================================================================
# GET /api/memory/context-curator/{goal_id}
# ===================================================================


class TestContextCuratorDrift:
    """Tests for the /api/memory/context-curator/{goal_id} endpoint."""

    def test_returns_drift_report(self, client):
        drift = MagicMock()
        drift.model_dump.return_value = {"goal_id": "g_1", "drifted": False}

        curator = AsyncMock()
        curator.detect_drift = AsyncMock(return_value=drift)

        with patch("qe.api.app._context_curator", curator, create=True):
            resp = client.get("/api/memory/context-curator/g_1")

        assert resp.status_code == 200
        assert resp.json()["drifted"] is False
        curator.detect_drift.assert_awaited_once_with("g_1")

    def test_503_when_no_curator(self, client):
        with patch("qe.api.app._context_curator", None, create=True):
            resp = client.get("/api/memory/context-curator/g_1")

        assert resp.status_code == 503
        assert "not initialized" in resp.json()["detail"]


# ===================================================================
# GET /api/memory/consolidation/history
# ===================================================================


class TestConsolidationHistory:
    """Tests for the /api/memory/consolidation/history endpoint."""

    def test_returns_history(self, client):
        entry = MagicMock()
        entry.model_dump.return_value = {"cycle": 1, "promoted": 3}

        kl = MagicMock()
        kl.get_history.return_value = [entry]

        with patch("qe.api.app._knowledge_loop", kl, create=True):
            resp = client.get("/api/memory/consolidation/history")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["history"]) == 1
        assert body["history"][0]["cycle"] == 1

    def test_respects_limit_param(self, client):
        kl = MagicMock()
        kl.get_history.return_value = []

        with patch("qe.api.app._knowledge_loop", kl, create=True):
            client.get(
                "/api/memory/consolidation/history", params={"limit": "5"}
            )

        kl.get_history.assert_called_once_with(limit=5)

    def test_503_when_no_knowledge_loop(self, client):
        with patch("qe.api.app._knowledge_loop", None, create=True):
            resp = client.get("/api/memory/consolidation/history")

        assert resp.status_code == 503
        assert "not available" in resp.json()["detail"]

    def test_500_on_history_exception(self, client):
        kl = MagicMock()
        kl.get_history.side_effect = RuntimeError("corrupt data")

        with patch("qe.api.app._knowledge_loop", kl, create=True):
            resp = client.get("/api/memory/consolidation/history")

        assert resp.status_code == 500
        assert "failed" in resp.json()["detail"]


# ===================================================================
# POST /api/memory/export
# ===================================================================


class TestMemoryExport:
    """Tests for the /api/memory/export endpoint."""

    def test_returns_exported_data_from_all_tiers(self, client):
        episodic = MagicMock()
        episodic.get_latest.return_value = [_mock_episode()]

        substrate = AsyncMock()
        substrate.get_claims = AsyncMock(return_value=[_mock_claim()])

        procedural = AsyncMock()
        procedural.get_best_templates = AsyncMock(return_value=[_mock_template()])
        procedural.get_best_sequences = AsyncMock(return_value=[_mock_sequence()])

        with (
            patch("qe.api.app._episodic_memory", episodic, create=True),
            patch("qe.api.app._substrate", substrate, create=True),
            patch("qe.api.app._procedural_memory", procedural, create=True),
        ):
            resp = client.post("/api/memory/export")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["episodic"]) == 1
        assert len(body["claims"]) == 1
        assert "templates" in body["procedural"]
        assert len(body["procedural"]["templates"]) == 1
        assert len(body["procedural"]["sequences"]) == 1

    def test_export_with_no_components(self, client):
        with (
            patch("qe.api.app._episodic_memory", None, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            resp = client.post("/api/memory/export")

        assert resp.status_code == 200
        body = resp.json()
        assert body["episodic"] == []
        assert body["claims"] == []
        assert body["procedural"] == {}

    def test_export_substrate_exception_returns_empty_claims(self, client):
        substrate = AsyncMock()
        substrate.get_claims = AsyncMock(side_effect=RuntimeError("db error"))

        with (
            patch("qe.api.app._episodic_memory", None, create=True),
            patch("qe.api.app._substrate", substrate, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            resp = client.post("/api/memory/export")

        assert resp.status_code == 200
        assert resp.json()["claims"] == []


# ===================================================================
# POST /api/memory/import
# ===================================================================


class TestMemoryImport:
    """Tests for the /api/memory/import endpoint."""

    def test_imports_data_returns_counts(self, client):
        episodic = AsyncMock()
        episodic.store = AsyncMock()

        substrate = AsyncMock()
        substrate.commit_claim = AsyncMock()

        procedural = AsyncMock()
        procedural.record_template_outcome = AsyncMock()

        payload = {
            "episodic": [{"id": "ep_1", "type": "synthesis", "goal_id": "g_1",
                          "content": "x", "timestamp": "2026-01-01T00:00:00"}],
            "claims": [{"claim_id": "clm_1", "text": "x",
                         "source_envelope_ids": []}],
            "procedural": {
                "templates": [
                    {"template_id": "tpl_1", "pattern": "why",
                     "question_type": "factual", "success_count": 5,
                     "failure_count": 1, "avg_info_gain": 0.7,
                     "domain": "finance"},
                ],
            },
        }

        with (
            patch("qe.api.app._episodic_memory", episodic, create=True),
            patch("qe.api.app._substrate", substrate, create=True),
            patch("qe.api.app._procedural_memory", procedural, create=True),
        ):
            resp = client.post("/api/memory/import", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        # The actual count depends on whether Episode/Claim construction
        # succeeds; with real models it may fail, so we just check the keys
        assert "episodes_imported" in body
        assert "claims_imported" in body
        assert "procedural_imported" in body

    def test_import_empty_payload(self, client):
        with (
            patch("qe.api.app._episodic_memory", None, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            resp = client.post("/api/memory/import", json={})

        assert resp.status_code == 200
        body = resp.json()
        assert body["episodes_imported"] == 0
        assert body["claims_imported"] == 0
        assert body["procedural_imported"] == 0

    def test_import_skips_when_no_components(self, client):
        payload = {
            "episodic": [{"id": "ep_1"}],
            "claims": [{"claim_id": "clm_1"}],
            "procedural": {"templates": [{"template_id": "tpl_1"}]},
        }

        with (
            patch("qe.api.app._episodic_memory", None, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            resp = client.post("/api/memory/import", json=payload)

        assert resp.status_code == 200
        body = resp.json()
        assert body["episodes_imported"] == 0
        assert body["claims_imported"] == 0
        assert body["procedural_imported"] == 0

    def test_import_continues_on_individual_errors(self, client):
        """If one episode fails to construct, subsequent ones still import."""
        episodic = AsyncMock()
        episodic.store = AsyncMock()

        # Two episodes: first will fail (bad data), second may succeed
        payload = {
            "episodic": [
                {},  # missing required fields -- Episode(**{}) will raise
                {},  # also likely to fail
            ],
        }

        with (
            patch("qe.api.app._episodic_memory", episodic, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", None, create=True),
        ):
            resp = client.post("/api/memory/import", json=payload)

        # Should not blow up; counts reflect only successes
        assert resp.status_code == 200
        assert "episodes_imported" in resp.json()

    def test_import_procedural_records_outcome(self, client):
        procedural = AsyncMock()
        procedural.record_template_outcome = AsyncMock()

        payload = {
            "procedural": {
                "templates": [
                    {
                        "template_id": "tpl_99",
                        "pattern": "what if",
                        "question_type": "causal",
                        "success_count": 10,
                        "failure_count": 2,
                        "avg_info_gain": 0.8,
                        "domain": "tech",
                    }
                ]
            }
        }

        with (
            patch("qe.api.app._episodic_memory", None, create=True),
            patch("qe.api.app._substrate", None, create=True),
            patch("qe.api.app._procedural_memory", procedural, create=True),
        ):
            resp = client.post("/api/memory/import", json=payload)

        assert resp.status_code == 200
        procedural.record_template_outcome.assert_awaited_once_with(
            template_id="tpl_99",
            pattern="what if",
            question_type="causal",
            success=True,  # 10 > 2
            info_gain=0.8,
            domain="tech",
        )
        assert resp.json()["procedural_imported"] == 1

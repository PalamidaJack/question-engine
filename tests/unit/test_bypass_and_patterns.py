"""Tests for LLM bypass (item 3) and execution pattern memory (item 4)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from qe.models.claim import Claim
from qe.services.planner.schemas import (
    DecompositionOutput,
    ProblemRepresentation,
    SubtaskPlan,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_claim(
    subject: str = "test_entity",
    predicate: str = "has_property",
    object_value: str = "some value",
    confidence: float = 0.9,
) -> Claim:
    return Claim(
        subject_entity_id=subject,
        predicate=predicate,
        object_value=object_value,
        confidence=confidence,
        source_service_id="test",
        source_envelope_ids=[],
    )


def _make_decomposition_output() -> DecompositionOutput:
    return DecompositionOutput(
        representation=ProblemRepresentation(
            core_problem="Test the system",
            actual_need="Run unit tests",
            constraints=[],
            success_criteria=["all pass"],
            problem_type="well_defined",
        ),
        strategy="single research subtask",
        subtasks=[
            SubtaskPlan(
                description="Research the topic",
                task_type="research",
                depends_on_indices=[],
                model_tier="fast",
            ),
        ],
        assumptions=[],
        estimated_time_seconds=60,
    )


# ── Item 3: LLM Bypass Tests ────────────────────────────────────────────────


class TestTryDeterministic:
    """ExecutorService._try_deterministic skips LLM when data exists."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_substrate(self):
        from qe.services.executor import ExecutorService

        executor = ExecutorService(
            bus=MagicMock(), substrate=None, model="gpt-4o-mini",
        )
        result = await executor._try_deterministic(
            "fact_check", "Is the sky blue?", {},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_non_fact_check(self):
        from qe.services.executor import ExecutorService

        executor = ExecutorService(
            bus=MagicMock(), substrate=MagicMock(), model="gpt-4o-mini",
        )
        result = await executor._try_deterministic(
            "research", "Investigate topic X", {},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_high_conf_claims(self):
        from qe.services.executor import ExecutorService

        substrate = MagicMock()
        substrate.search_full_text = AsyncMock(return_value=[
            _make_claim(confidence=0.3),
            _make_claim(confidence=0.5),
        ])

        executor = ExecutorService(
            bus=MagicMock(), substrate=substrate, model="gpt-4o-mini",
        )
        result = await executor._try_deterministic(
            "fact_check", "Check claims about X", {},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_bypasses_llm_with_high_conf_claims(self):
        from qe.services.executor import ExecutorService

        substrate = MagicMock()
        substrate.search_full_text = AsyncMock(return_value=[
            _make_claim(
                subject="sky",
                predicate="color",
                object_value="blue",
                confidence=0.95,
            ),
        ])

        executor = ExecutorService(
            bus=MagicMock(), substrate=substrate, model="gpt-4o-mini",
        )
        result = await executor._try_deterministic(
            "fact_check", "Is the sky blue?", {},
        )
        assert result is not None
        assert result["bypassed_llm"] is True
        assert result["task_type"] == "fact_check"
        assert "sky" in result["content"]
        assert "blue" in result["content"]

    @pytest.mark.asyncio
    async def test_bypass_tolerates_substrate_error(self):
        from qe.services.executor import ExecutorService

        substrate = MagicMock()
        substrate.search_full_text = AsyncMock(
            side_effect=RuntimeError("DB gone"),
        )

        executor = ExecutorService(
            bus=MagicMock(), substrate=substrate, model="gpt-4o-mini",
        )
        result = await executor._try_deterministic(
            "fact_check", "Check something", {},
        )
        assert result is None  # graceful fallback, no crash


class TestExecuteTaskBypass:
    """_execute_task honours the deterministic fast-path."""

    @pytest.mark.asyncio
    async def test_execute_task_uses_bypass_when_available(self):
        from qe.services.executor import ExecutorService

        substrate = MagicMock()
        substrate.search_full_text = AsyncMock(return_value=[
            _make_claim(confidence=0.9),
        ])

        executor = ExecutorService(
            bus=MagicMock(), substrate=substrate, model="gpt-4o-mini",
        )
        result = await executor._execute_task(
            "fact_check", "Verify entity property", {},
        )
        assert result["bypassed_llm"] is True

    @pytest.mark.asyncio
    async def test_execute_task_falls_through_to_llm(self):
        """When no bypass matches, the LLM path is used."""
        from qe.services.executor import ExecutorService

        substrate = MagicMock()
        substrate.search_full_text = AsyncMock(return_value=[])

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "LLM answer"

        acompletion_path = (
            "qe.services.executor.service.litellm.acompletion"
        )
        limiter_path = (
            "qe.services.executor.service.get_rate_limiter"
        )
        with (
            patch(acompletion_path, new_callable=AsyncMock,
                  return_value=mock_response),
            patch(limiter_path) as mock_rl,
        ):
            mock_rl.return_value.acquire = AsyncMock()
            executor = ExecutorService(
                bus=MagicMock(), substrate=substrate, model="gpt-4o-mini",
            )
            result = await executor._execute_task(
                "fact_check", "Something new", {},
            )
        assert result["content"] == "LLM answer"
        assert "bypassed_llm" not in result


# ── Item 4: Pattern Memory Tests ────────────────────────────────────────────


@pytest.fixture
async def embedding_store(tmp_path):
    """Create a real EmbeddingStore backed by a temp SQLite DB."""
    from qe.substrate.embeddings import EmbeddingStore

    db_path = str(tmp_path / "pattern_test.db")
    migration = (
        Path(__file__).parent.parent.parent
        / "src/qe/substrate/migrations/0005_embeddings.sql"
    )
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(migration.read_text())
        await db.commit()
    return EmbeddingStore(db_path, model="test-model")


class TestStorePattern:
    """PlannerService._store_pattern persists decompositions."""

    @pytest.mark.asyncio
    async def test_pattern_stored_in_embeddings(self, embedding_store):
        from qe.services.planner import PlannerService

        # Mock embed_text so we don't need a real LLM
        embedding_store.embed_text = AsyncMock(return_value=[1.0, 0.0, 0.0])

        substrate = MagicMock()
        substrate.embeddings = embedding_store

        planner = PlannerService(
            bus=MagicMock(), substrate=substrate, model="test",
        )

        output = _make_decomposition_output()
        await planner._store_pattern(
            "Analyse quarterly earnings", output,
        )

        # Verify it's in the DB
        count = await embedding_store.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_pattern_stored_with_correct_prefix(self, embedding_store):
        from qe.services.planner import PlannerService

        embedding_store.embed_text = AsyncMock(return_value=[0.0, 1.0, 0.0])

        substrate = MagicMock()
        substrate.embeddings = embedding_store

        planner = PlannerService(
            bus=MagicMock(), substrate=substrate, model="test",
        )

        output = _make_decomposition_output()
        await planner._store_pattern(
            "Investigate supply chain risks", output,
        )

        # Query raw DB to check the ID
        async with aiosqlite.connect(embedding_store._db_path) as db:
            cursor = await db.execute("SELECT id FROM embeddings")
            rows = await cursor.fetchall()
        assert len(rows) == 1
        assert rows[0][0].startswith("goal_pattern:")

    @pytest.mark.asyncio
    async def test_store_pattern_tolerates_no_substrate(self):
        from qe.services.planner import PlannerService

        planner = PlannerService(
            bus=MagicMock(), substrate=None, model="test",
        )
        # Should not raise
        await planner._store_pattern(
            "Some goal", _make_decomposition_output(),
        )


class TestFindCachedPattern:
    """PlannerService._find_cached_pattern retrieves past decompositions."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_patterns(self, embedding_store):
        from qe.services.planner import PlannerService

        substrate = MagicMock()
        substrate.embeddings = embedding_store

        planner = PlannerService(
            bus=MagicMock(), substrate=substrate, model="test",
        )
        result = await planner._find_cached_pattern("Brand new goal")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_cached_pattern_on_exact_match(
        self, embedding_store,
    ):
        from qe.services.planner import PlannerService

        substrate = MagicMock()
        substrate.embeddings = embedding_store

        planner = PlannerService(
            bus=MagicMock(), substrate=substrate, model="test",
        )

        output = _make_decomposition_output()
        desc = "Analyse quarterly earnings for ACME Corp"

        # Store with explicit embedding so search can match without LLM
        emb = [1.0, 0.0, 0.0]
        import hashlib
        key = hashlib.sha256(desc.encode()).hexdigest()[:16]
        pattern_id = f"goal_pattern:{key}"
        await embedding_store.store(
            id=pattern_id,
            text=desc,
            metadata={"decomposition": output.model_dump_json()},
            embedding=emb,
        )

        # Search with same embedding → exact match (sim=1.0 > 0.85)
        result = await planner._find_cached_pattern(
            "Analyse quarterly earnings for ACME Corp",
        )
        # This will call embed_text which needs LLM, so let's patch search
        # Instead, let's mock the search call
        substrate.embeddings = MagicMock()

        from qe.substrate.embeddings import SearchResult

        substrate.embeddings.search = AsyncMock(return_value=[
            SearchResult(
                id=pattern_id,
                text=desc,
                similarity=0.95,
                metadata={"decomposition": output.model_dump_json()},
            ),
        ])
        planner.substrate = substrate

        result = await planner._find_cached_pattern(desc)
        assert result is not None
        assert result.strategy == "single research subtask"
        assert len(result.subtasks) == 1

    @pytest.mark.asyncio
    async def test_ignores_non_pattern_results(self):
        from qe.services.planner import PlannerService
        from qe.substrate.embeddings import SearchResult

        substrate = MagicMock()
        substrate.embeddings = MagicMock()
        substrate.embeddings.search = AsyncMock(return_value=[
            SearchResult(
                id="clm_abc123",  # not a pattern
                text="some claim",
                similarity=0.99,
                metadata={},
            ),
        ])

        planner = PlannerService(
            bus=MagicMock(), substrate=substrate, model="test",
        )
        result = await planner._find_cached_pattern("Some goal")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_substrate(self):
        from qe.services.planner import PlannerService

        planner = PlannerService(
            bus=MagicMock(), substrate=None, model="test",
        )
        result = await planner._find_cached_pattern("Any goal")
        assert result is None

    @pytest.mark.asyncio
    async def test_tolerates_search_error(self):
        from qe.services.planner import PlannerService

        substrate = MagicMock()
        substrate.embeddings = MagicMock()
        substrate.embeddings.search = AsyncMock(
            side_effect=RuntimeError("search broken"),
        )

        planner = PlannerService(
            bus=MagicMock(), substrate=substrate, model="test",
        )
        result = await planner._find_cached_pattern("Goal X")
        assert result is None  # graceful fallback


class TestDecomposePatternIntegration:
    """decompose() uses pattern memory before calling the LLM."""

    @pytest.mark.asyncio
    async def test_decompose_uses_cached_pattern(self):
        from qe.services.planner import PlannerService
        from qe.substrate.embeddings import SearchResult

        output = _make_decomposition_output()

        substrate = MagicMock()
        substrate.embeddings = MagicMock()
        substrate.embeddings.search = AsyncMock(return_value=[
            SearchResult(
                id="goal_pattern:abc123",
                text="Research AI safety",
                similarity=0.92,
                metadata={"decomposition": output.model_dump_json()},
            ),
        ])

        planner = PlannerService(
            bus=MagicMock(), substrate=substrate, model="test",
        )

        # Should NOT call the LLM at all
        with patch("qe.services.planner.service.instructor") as mock_instr:
            state = await planner.decompose("Research AI safety")

        mock_instr.from_litellm.assert_not_called()
        assert state.status == "executing"
        assert len(state.subtask_states) == 1

    @pytest.mark.asyncio
    async def test_decompose_stores_pattern_on_llm_call(self):
        from qe.services.planner import PlannerService

        output = _make_decomposition_output()

        substrate = MagicMock()
        # No cached pattern
        substrate.embeddings = MagicMock()
        substrate.embeddings.search = AsyncMock(return_value=[])
        substrate.embeddings.store = AsyncMock()

        planner = PlannerService(
            bus=MagicMock(), substrate=substrate, model="test",
        )

        # Mock the LLM call
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=output)

        with patch(
            "qe.services.planner.service.instructor"
        ) as mock_instr:
            mock_instr.from_litellm.return_value = mock_client
            await planner.decompose("Brand new goal")

        # LLM was called
        mock_client.chat.completions.create.assert_awaited_once()

        # Pattern was stored
        substrate.embeddings.store.assert_awaited_once()
        call_kwargs = substrate.embeddings.store.call_args
        assert call_kwargs[1]["id"].startswith("goal_pattern:")
        assert "decomposition" in call_kwargs[1]["metadata"]

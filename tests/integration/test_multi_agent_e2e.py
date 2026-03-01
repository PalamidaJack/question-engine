"""E2E multi-agent parallel inquiry walkthrough with mock LLM.

Tests CognitiveAgentPool fan-out, result merging, strategy assignment,
elastic scaling, and partial failure handling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.cognition import (
    ApproachAssessment,
    CrystallizedInsight,
    DialecticReport,
    MechanismExplanation,
    NoveltyAssessment,
    ProvenanceChain,
    SurpriseDetection,
    UncertaintyAssessment,
)
from qe.models.cognition import EpistemicState as CogEpistemicState
from qe.runtime.cognitive_agent_pool import CognitiveAgentPool
from qe.runtime.strategy_evolver import ElasticScaler, StrategyEvolver
from qe.runtime.strategy_models import (
    DEFAULT_STRATEGIES,
    StrategyOutcome,
)
from qe.services.inquiry.engine import InquiryEngine
from qe.services.inquiry.schemas import InquiryConfig, InquiryResult, Question


def _make_engine(
    cognitive_stack: dict[str, Any],
    bus: MemoryBus,
) -> InquiryEngine:
    """Create a fully-wired InquiryEngine with independent mock components."""
    # Independent mocks per engine (avoid shared state)
    meta = MagicMock()
    meta.suggest_next_approach = AsyncMock(
        return_value=ApproachAssessment(
            recommended_approach="Investigate market data",
            reasoning="Promising approach",
        )
    )

    er = MagicMock()
    er.assess_uncertainty = AsyncMock(
        return_value=UncertaintyAssessment(
            finding_summary="test", confidence_level="moderate",
        )
    )
    er.detect_surprise = AsyncMock(
        return_value=SurpriseDetection(finding="test", surprise_magnitude=0.3)
    )
    er.get_epistemic_state = MagicMock(
        return_value=CogEpistemicState(goal_id="test", overall_confidence="moderate")
    )
    er.get_blind_spot_warning = MagicMock(return_value=None)

    de = MagicMock()
    de.full_dialectic = AsyncMock(
        return_value=DialecticReport(
            original_conclusion="test", revised_confidence=0.72,
        )
    )

    pe = MagicMock()

    ic = MagicMock()
    ic.crystallize = AsyncMock(
        return_value=CrystallizedInsight(
            headline="Test insight from agent",
            mechanism=MechanismExplanation(
                what_happens="test", why_it_happens="test", how_it_works="test",
            ),
            novelty=NoveltyAssessment(finding="test", is_novel=True),
            provenance=ProvenanceChain(original_question="test"),
            actionability_score=0.7,
            confidence=0.7,
        )
    )

    qg = MagicMock()

    async def _generate(**kwargs: Any) -> list[Question]:
        return [
            Question(
                text="What are battery storage costs?",
                question_type="factual",
                expected_info_gain=0.8,
                relevance_to_goal=0.9,
                novelty_score=0.6,
            ),
        ]

    qg.generate = _generate

    async def _prioritize(goal: str, questions: list[Question]) -> list[Question]:
        return sorted(questions, key=lambda q: q.expected_info_gain, reverse=True)

    qg.prioritize = _prioritize

    hm = MagicMock()
    hm.get_active_hypotheses = AsyncMock(return_value=[])
    hm.generate_hypotheses = AsyncMock(return_value=[])
    hm.create_falsification_questions = MagicMock(return_value=[])

    return InquiryEngine(
        episodic_memory=cognitive_stack["episodic_memory"],
        context_curator=cognitive_stack["context_curator"],
        metacognitor=meta,
        epistemic_reasoner=er,
        dialectic_engine=de,
        persistence_engine=pe,
        insight_crystallizer=ic,
        question_generator=qg,
        hypothesis_manager=hm,
        bus=bus,
        config=InquiryConfig(max_iterations=1, questions_per_iteration=1),
    )


@pytest.fixture
def engine_factory(
    cognitive_stack: dict[str, Any],
    mock_bus: MemoryBus,
) -> Callable[[], InquiryEngine]:
    """Factory that returns independent InquiryEngine instances."""

    def _create() -> InquiryEngine:
        return _make_engine(cognitive_stack, mock_bus)

    return _create


@pytest.fixture
async def pool(
    engine_factory: Callable[[], InquiryEngine],
    mock_bus: MemoryBus,
) -> CognitiveAgentPool:
    """Pre-configured CognitiveAgentPool with engine factory."""
    return CognitiveAgentPool(
        bus=mock_bus,
        max_agents=5,
        engine_factory=engine_factory,
    )


GOAL_ID = "goal_energy_multi"
GOAL_DESC = "Evaluate renewable energy storage as an investment opportunity"


@pytest.mark.asyncio
class TestMultiAgentE2E:
    """Multi-agent parallel inquiry execution tests."""

    async def test_spawn_three_agents(self, pool: CognitiveAgentPool):
        strategies = list(DEFAULT_STRATEGIES.values())[:3]
        agents = []
        for i, strat in enumerate(strategies):
            agent = await pool.spawn_agent(
                specialization=f"specialist_{i}",
                strategy=strat,
            )
            agents.append(agent)

        assert len(agents) == 3
        status = pool.pool_status()
        assert status["total_agents"] == 3

    async def test_parallel_inquiry_returns_results(
        self, pool: CognitiveAgentPool,
    ):
        for i in range(3):
            await pool.spawn_agent(specialization=f"agent_{i}")

        results = await pool.run_parallel_inquiry(
            goal_id=GOAL_ID,
            goal_description=GOAL_DESC,
            config=InquiryConfig(max_iterations=1, questions_per_iteration=1),
        )
        assert len(results) == 3
        for r in results:
            assert isinstance(r, InquiryResult)
            assert r.status == "completed"

    async def test_results_merged_correctly(
        self, pool: CognitiveAgentPool,
    ):
        for i in range(3):
            await pool.spawn_agent(specialization=f"agent_{i}")

        results = await pool.run_parallel_inquiry(
            goal_id=GOAL_ID,
            goal_description=GOAL_DESC,
            config=InquiryConfig(max_iterations=1, questions_per_iteration=1),
        )
        merged = await pool.merge_results(results)

        assert merged.status == "completed"
        assert merged.inquiry_id.startswith("merged_")
        assert merged.total_cost_usd >= 0
        assert merged.total_questions_generated >= results[0].total_questions_generated

    async def test_different_strategies_assigned(
        self, pool: CognitiveAgentPool,
    ):
        strategies = list(DEFAULT_STRATEGIES.values())[:3]
        for i, strat in enumerate(strategies):
            await pool.spawn_agent(
                specialization=f"agent_{i}",
                strategy=strat,
            )

        status = pool.pool_status()
        strategy_names = {a["strategy"] for a in status["agents"]}
        assert len(strategy_names) == 3

    async def test_strategy_outcomes_recorded(
        self, pool: CognitiveAgentPool, mock_bus: MemoryBus,
    ):
        evolver = StrategyEvolver(agent_pool=pool, bus=mock_bus)
        evolver.record_outcome(StrategyOutcome(
            strategy_name="breadth_first",
            success=True,
            cost_usd=0.05,
            duration_s=1.0,
        ))
        snapshots = evolver.get_snapshots()
        bf_snap = next(s for s in snapshots if s.strategy_name == "breadth_first")
        assert bf_snap.sample_count == 1
        assert bf_snap.alpha == 2.0  # 1 + 1 success

    async def test_elastic_scaler_recommends_profile(
        self, pool: CognitiveAgentPool,
    ):
        scaler = ElasticScaler(agent_pool=pool)
        profile = scaler.recommend_profile(
            pool_stats=pool.pool_status(),
            budget_pct=0.5,
        )
        assert profile.name in {"minimal", "balanced", "aggressive"}

    async def test_pool_status_during_execution(
        self, pool: CognitiveAgentPool,
    ):
        for i in range(2):
            await pool.spawn_agent(specialization=f"agent_{i}")

        status_before = pool.pool_status()
        assert status_before["total_agents"] == 2
        assert status_before["active_agents"] == 0

        await pool.run_parallel_inquiry(
            goal_id=GOAL_ID,
            goal_description=GOAL_DESC,
            config=InquiryConfig(max_iterations=1, questions_per_iteration=1),
        )

        status_after = pool.pool_status()
        assert status_after["active_agents"] == 0

    async def test_agents_return_to_idle(
        self, pool: CognitiveAgentPool,
    ):
        agents = []
        for i in range(2):
            agents.append(await pool.spawn_agent(specialization=f"agent_{i}"))

        await pool.run_parallel_inquiry(
            goal_id=GOAL_ID,
            goal_description=GOAL_DESC,
            config=InquiryConfig(max_iterations=1, questions_per_iteration=1),
        )

        for agent in agents:
            assert agent.status == "idle"
            assert agent.active_inquiry_id is None

    async def test_retire_after_inquiry(
        self, pool: CognitiveAgentPool,
    ):
        agents = []
        for i in range(2):
            agents.append(await pool.spawn_agent(specialization=f"agent_{i}"))

        await pool.run_parallel_inquiry(
            goal_id=GOAL_ID,
            goal_description=GOAL_DESC,
            config=InquiryConfig(max_iterations=1, questions_per_iteration=1),
        )

        for agent in agents:
            ok = await pool.retire_agent(agent.agent_id)
            assert ok

        assert pool.pool_status()["total_agents"] == 0

    async def test_merge_handles_failed_agent(
        self,
        cognitive_stack: dict[str, Any],
        mock_bus: MemoryBus,
    ):
        """One engine raises -> other results still merged."""
        call_count = 0

        def _mixed_factory() -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                engine = MagicMock()
                engine.run_inquiry = AsyncMock(
                    side_effect=RuntimeError("LLM failed")
                )
                return engine
            return _make_engine(cognitive_stack, mock_bus)

        pool = CognitiveAgentPool(
            bus=mock_bus,
            max_agents=3,
            engine_factory=_mixed_factory,
        )
        for i in range(3):
            await pool.spawn_agent(specialization=f"agent_{i}")

        results = await pool.run_parallel_inquiry(
            goal_id=GOAL_ID,
            goal_description=GOAL_DESC,
            config=InquiryConfig(max_iterations=1, questions_per_iteration=1),
        )
        assert len(results) == 2
        merged = await pool.merge_results(results)
        assert merged.status == "completed"

    async def test_strategy_selection_after_outcomes(
        self, pool: CognitiveAgentPool, mock_bus: MemoryBus,
    ):
        evolver = StrategyEvolver(agent_pool=pool, bus=mock_bus)

        for _ in range(20):
            evolver.record_outcome(StrategyOutcome(
                strategy_name="breadth_first", success=True, cost_usd=0.01,
            ))
        for _ in range(20):
            evolver.record_outcome(StrategyOutcome(
                strategy_name="depth_first", success=False, cost_usd=0.01,
            ))

        selections = [evolver.select_strategy().name for _ in range(50)]
        bf_count = selections.count("breadth_first")
        df_count = selections.count("depth_first")
        assert bf_count > df_count

    async def test_bus_events_from_all_agents(
        self, pool: CognitiveAgentPool, mock_bus: MemoryBus,
    ):
        for i in range(2):
            await pool.spawn_agent(specialization=f"agent_{i}")

        await pool.run_parallel_inquiry(
            goal_id=GOAL_ID,
            goal_description=GOAL_DESC,
            config=InquiryConfig(max_iterations=1, questions_per_iteration=1),
        )

        events = mock_bus._collected_events  # type: ignore[attr-defined]
        started_events = [e for e in events if e.topic == "inquiry.started"]
        assert len(started_events) >= 2

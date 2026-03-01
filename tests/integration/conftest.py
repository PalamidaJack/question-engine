"""Shared fixtures for Phase 5 integration tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from qe.bus.memory_bus import MemoryBus
from qe.models.cognition import (
    CrystallizedInsight,
    DialecticReport,
    MechanismExplanation,
    NoveltyAssessment,
    ProvenanceChain,
)
from qe.models.envelope import Envelope
from qe.runtime.context_curator import ContextCurator
from qe.runtime.episodic_memory import EpisodicMemory
from qe.services.inquiry.engine import InquiryEngine
from qe.services.inquiry.schemas import InquiryConfig, Question
from qe.substrate.bayesian_belief import BayesianBeliefStore

# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------


def _make_mock_metacognitor() -> MagicMock:
    """Metacognitor mock that returns a canned approach assessment."""
    from qe.models.cognition import ApproachAssessment

    meta = MagicMock()
    meta.suggest_next_approach = AsyncMock(
        return_value=ApproachAssessment(
            recommended_approach="Investigate renewable energy storage market data",
            reasoning="This is the most promising initial approach",
            alternative_approaches=["Survey academic literature"],
            tools_needed=["web_search"],
            estimated_success_probability=0.7,
        )
    )
    return meta


def _make_mock_epistemic_reasoner() -> MagicMock:
    """EpistemicReasoner mock with canned uncertainty and surprise detection."""
    from qe.models.cognition import (
        EpistemicState,
        SurpriseDetection,
        UncertaintyAssessment,
    )

    er = MagicMock()
    er.assess_uncertainty = AsyncMock(
        return_value=UncertaintyAssessment(
            finding_summary="Energy storage costs declining",
            confidence_level="moderate",
        )
    )
    er.detect_surprise = AsyncMock(
        return_value=SurpriseDetection(
            finding="Battery costs dropped 90% in a decade",
            surprise_magnitude=0.3,
        )
    )
    er.get_epistemic_state = MagicMock(
        return_value=EpistemicState(goal_id="test", overall_confidence="moderate")
    )
    er.get_blind_spot_warning = MagicMock(return_value=None)
    return er


def _make_mock_dialectic_engine() -> MagicMock:
    """DialecticEngine mock returning a canned dialectic report."""
    de = MagicMock()
    de.full_dialectic = AsyncMock(
        return_value=DialecticReport(
            original_conclusion="Energy storage is a good investment",
            revised_confidence=0.72,
            synthesis="After critique, the conclusion holds with moderate confidence",
        )
    )
    return de


def _make_mock_persistence_engine() -> MagicMock:
    """PersistenceEngine mock (not directly called in inquiry loop)."""
    pe = MagicMock()
    pe.analyze_root_cause = AsyncMock(return_value=None)
    return pe


def _make_mock_crystallizer() -> MagicMock:
    """InsightCrystallizer mock returning a canned CrystallizedInsight."""
    ic = MagicMock()
    ic.crystallize = AsyncMock(
        return_value=CrystallizedInsight(
            headline="Battery storage costs have declined 90% enabling grid-scale viability",
            mechanism=MechanismExplanation(
                what_happens="Lithium-ion manufacturing scale drives cost down",
                why_it_happens="Economies of scale in battery manufacturing",
                how_it_works="Gigafactory production reduces per-unit cost",
                key_causal_links=["demand growth", "manufacturing scale"],
                confidence_in_mechanism=0.7,
            ),
            novelty=NoveltyAssessment(
                finding="Battery costs dropped 90%",
                is_novel=True,
                novelty_type="unexpected_magnitude",
            ),
            provenance=ProvenanceChain(
                original_question="What are the investment prospects for energy storage?",
                evidence_items=["BNEF report 2024"],
                insight="Battery storage is now cost-competitive",
            ),
            actionability_score=0.8,
            cross_domain_connections=["Solar PV had similar cost curve"],
            dialectic_survivor=True,
            confidence=0.7,
        )
    )
    return ic


def _make_mock_question_generator() -> MagicMock:
    """QuestionGenerator returning canned energy-storage questions."""
    qg = MagicMock()

    _call_count = 0

    async def _generate(**kwargs: Any) -> list[Question]:
        nonlocal _call_count
        _call_count += 1
        return [
            Question(
                text="What are the current costs of lithium-ion battery storage per kWh?",
                question_type="factual",
                expected_info_gain=0.8,
                relevance_to_goal=0.9,
                novelty_score=0.6,
                iteration_generated=kwargs.get("iteration", 0),
            ),
            Question(
                text="How do flow batteries compare to lithium-ion for grid-scale storage?",
                question_type="comparative",
                expected_info_gain=0.7,
                relevance_to_goal=0.8,
                novelty_score=0.7,
                iteration_generated=kwargs.get("iteration", 0),
            ),
        ]

    qg.generate = _generate

    async def _prioritize(goal: str, questions: list[Question]) -> list[Question]:
        return sorted(
            questions,
            key=lambda q: q.expected_info_gain * 0.4
            + q.relevance_to_goal * 0.35
            + q.novelty_score * 0.25,
            reverse=True,
        )

    qg.prioritize = _prioritize
    return qg


def _make_mock_hypothesis_manager() -> MagicMock:
    """HypothesisManager mock with canned behavior."""
    from qe.substrate.bayesian_belief import Hypothesis

    hm = MagicMock()
    hm.get_active_hypotheses = AsyncMock(return_value=[])
    hm.generate_hypotheses = AsyncMock(
        return_value=[
            Hypothesis(
                hypothesis_id="hyp_test1",
                statement="Battery storage will reach $50/kWh by 2030",
                current_probability=0.6,
            )
        ]
    )
    hm.create_falsification_questions = MagicMock(return_value=[])
    return hm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def mock_bus() -> MemoryBus:
    """MemoryBus with event collector using synchronous publish listener."""
    bus = MemoryBus()
    bus._collected_events: list[Envelope] = []  # type: ignore[attr-defined]

    # Use add_publish_listener for synchronous, inline collection.
    # This avoids async task race conditions — events are captured
    # immediately during publish() rather than via async handlers.
    def _collector(envelope: Envelope) -> None:
        bus._collected_events.append(envelope)  # type: ignore[attr-defined]

    bus.add_publish_listener(_collector)
    return bus


@pytest.fixture
async def cognitive_stack(
    tmp_path: Any,
) -> dict[str, Any]:
    """Full cognitive stack with mock LLM components and real memory stores."""
    db_path = str(tmp_path / "phase5_test.db")

    episodic = EpisodicMemory(db_path=db_path)
    await episodic.initialize()

    belief_store = BayesianBeliefStore(db_path=db_path)
    context_curator = ContextCurator()

    return {
        "episodic_memory": episodic,
        "belief_store": belief_store,
        "context_curator": context_curator,
        "metacognitor": _make_mock_metacognitor(),
        "epistemic_reasoner": _make_mock_epistemic_reasoner(),
        "dialectic_engine": _make_mock_dialectic_engine(),
        "persistence_engine": _make_mock_persistence_engine(),
        "insight_crystallizer": _make_mock_crystallizer(),
        "question_generator": _make_mock_question_generator(),
        "hypothesis_manager": _make_mock_hypothesis_manager(),
        "db_path": db_path,
    }


@pytest.fixture
def inquiry_engine_factory(
    cognitive_stack: dict[str, Any],
    mock_bus: MemoryBus,
) -> Callable[..., InquiryEngine]:
    """Factory that creates InquiryEngine instances wired to the cognitive stack."""

    def _create(config: InquiryConfig | None = None) -> InquiryEngine:
        return InquiryEngine(
            episodic_memory=cognitive_stack["episodic_memory"],
            context_curator=cognitive_stack["context_curator"],
            metacognitor=cognitive_stack["metacognitor"],
            epistemic_reasoner=cognitive_stack["epistemic_reasoner"],
            dialectic_engine=cognitive_stack["dialectic_engine"],
            persistence_engine=cognitive_stack["persistence_engine"],
            insight_crystallizer=cognitive_stack["insight_crystallizer"],
            question_generator=cognitive_stack["question_generator"],
            hypothesis_manager=cognitive_stack["hypothesis_manager"],
            bus=mock_bus,
            config=config or InquiryConfig(max_iterations=2, questions_per_iteration=2),
        )

    return _create

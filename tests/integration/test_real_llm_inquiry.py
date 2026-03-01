"""Integration tests that call real LLMs through Kilo Code.

These tests verify that the instructor + litellm structured output pipeline
works end-to-end and that Pydantic response models are properly populated.

Requires KILOCODE_API_KEY in .env. Skipped in CI when key is absent.
All tests use openai/anthropic/claude-3.5-haiku (~$0.01 total per run).
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment setup — must happen before any litellm / instructor import
# ---------------------------------------------------------------------------

load_dotenv()

_kilo_key = os.environ.get("KILOCODE_API_KEY", "")
_kilo_base = os.environ.get("KILOCODE_API_BASE", "https://kilo.ai/api/openrouter")
_has_llm_key = bool(_kilo_key)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _has_llm_key, reason="KILOCODE_API_KEY not set"),
]

# The cheap/fast model used for all tests
_MODEL = "openai/anthropic/claude-3.5-haiku"


# ---------------------------------------------------------------------------
# Session-scoped fixture: configure Kilo Code once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _kilo_code_env() -> None:
    """Configure OPENAI_API_KEY/BASE from KILOCODE vars and register models."""
    if not _has_llm_key:
        return

    import litellm

    os.environ.setdefault("OPENAI_API_KEY", _kilo_key)
    os.environ.setdefault("OPENAI_API_BASE", _kilo_base)

    litellm.register_model({
        "openai/anthropic/claude-sonnet-4": {
            "max_tokens": 8192,
            "max_input_tokens": 200000,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
            "litellm_provider": "openai",
        },
        "openai/anthropic/claude-3.5-haiku": {
            "max_tokens": 8192,
            "max_input_tokens": 1048576,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.0000001,
            "output_cost_per_token": 0.0000004,
            "litellm_provider": "openai",
        },
    })


# ===================================================================
# Layer 1 — Individual component tests (1 LLM call each, 30s timeout)
# ===================================================================


@pytest.mark.timeout(30)
async def test_question_generator_produces_valid_questions() -> None:
    """QuestionGenerator.generate() returns valid Question objects via real LLM."""
    from qe.services.inquiry.question_generator import QuestionGenerator
    from qe.services.inquiry.schemas import Question

    qg = QuestionGenerator(model=_MODEL)
    questions = await qg.generate(
        goal="Evaluate the investment potential of solid-state batteries",
        findings_summary="Early-stage technology with high energy density potential.",
        n_questions=2,
    )

    assert isinstance(questions, list)
    assert len(questions) >= 1
    for q in questions:
        assert isinstance(q, Question)
        # Substantive text (not empty or just whitespace)
        assert len(q.text.strip()) > 10, f"Question text too short: {q.text!r}"
        # Valid question type
        assert q.question_type in {
            "factual", "comparative", "causal", "counterfactual",
            "evaluative", "methodological",
        }
        # Scores in [0, 1]
        assert 0.0 <= q.expected_info_gain <= 1.0
        assert 0.0 <= q.relevance_to_goal <= 1.0
        assert 0.0 <= q.novelty_score <= 1.0


@pytest.mark.timeout(30)
async def test_metacognitor_suggests_approach() -> None:
    """Metacognitor.suggest_next_approach() returns valid ApproachAssessment."""
    from qe.models.cognition import ApproachAssessment
    from qe.runtime.metacognitor import Metacognitor

    mc = Metacognitor(model=_MODEL)
    assessment = await mc.suggest_next_approach(
        goal_id="goal_test_meta",
        goal_description=(
            "Determine whether hydrogen fuel cells can compete"
            " with BEVs for long-haul trucking"
        ),
    )

    assert isinstance(assessment, ApproachAssessment)
    assert len(assessment.recommended_approach.strip()) > 10
    assert len(assessment.reasoning.strip()) > 10
    assert 0.0 <= assessment.estimated_success_probability <= 1.0


@pytest.mark.timeout(30)
async def test_dialectic_engine_challenges_conclusion() -> None:
    """DialecticEngine.challenge() returns Counterarguments with valid strength."""
    from qe.models.cognition import Counterargument
    from qe.services.inquiry.dialectic import DialecticEngine

    de = DialecticEngine(model=_MODEL)
    counterargs = await de.challenge(
        goal_id="goal_test_dial",
        conclusion="Solid-state batteries will dominate the EV market by 2030",
        evidence="Lab prototypes show 2x energy density vs lithium-ion",
    )

    assert isinstance(counterargs, list)
    assert len(counterargs) >= 1
    for ca in counterargs:
        assert isinstance(ca, Counterargument)
        assert ca.strength in {"weak", "moderate", "strong", "decisive"}
        assert len(ca.counterargument.strip()) > 10


@pytest.mark.timeout(30)
async def test_hypothesis_manager_generates_hypotheses() -> None:
    """HypothesisManager.generate_hypotheses() returns Hypothesis objects."""
    from qe.services.inquiry.hypothesis import HypothesisManager
    from qe.substrate.bayesian_belief import Hypothesis

    hm = HypothesisManager(model=_MODEL)
    hypotheses = await hm.generate_hypotheses(
        goal="Assess viability of perovskite solar cells for residential use",
        contradictions=["Lab efficiency of 30% but commercial modules only reach 15%"],
    )

    assert isinstance(hypotheses, list)
    assert len(hypotheses) >= 1
    for h in hypotheses:
        assert isinstance(h, Hypothesis)
        assert len(h.statement.strip()) > 10
        assert len(h.falsification_criteria) >= 1, "Expected at least one falsification criterion"
        assert 0.0 <= h.prior_probability <= 1.0
        assert 0.0 <= h.current_probability <= 1.0


@pytest.mark.timeout(30)
async def test_insight_crystallizer_assesses_novelty() -> None:
    """InsightCrystallizer.assess_novelty() returns valid NoveltyAssessment."""
    from qe.models.cognition import NoveltyAssessment
    from qe.services.inquiry.insight import InsightCrystallizer

    ic = InsightCrystallizer(model=_MODEL)
    assessment = await ic.assess_novelty(
        finding="Sodium-ion batteries achieve 90% of lithium-ion energy density at 40% lower cost",
        domain="energy_storage",
    )

    assert isinstance(assessment, NoveltyAssessment)
    assert assessment.novelty_type in {
        "contradicts_consensus",
        "new_connection",
        "unexpected_magnitude",
        "temporal_anomaly",
        "structural_analogy",
        "absence_significant",
        "not_novel",
    }
    assert isinstance(assessment.is_novel, bool)
    assert len(assessment.finding.strip()) > 0


# ===================================================================
# Layer 2 — Full engine test (~9-13 LLM calls, 120s timeout)
# ===================================================================


@pytest.mark.timeout(120)
async def test_full_inquiry_engine_one_iteration(tmp_path: object) -> None:
    """Full 7-phase InquiryEngine with real cognitive components, 1 iteration."""
    from qe.bus.memory_bus import MemoryBus
    from qe.models.envelope import Envelope
    from qe.runtime.metacognitor import Metacognitor
    from qe.services.inquiry.dialectic import DialecticEngine
    from qe.services.inquiry.engine import InquiryEngine
    from qe.services.inquiry.hypothesis import HypothesisManager
    from qe.services.inquiry.insight import InsightCrystallizer
    from qe.services.inquiry.question_generator import QuestionGenerator
    from qe.services.inquiry.schemas import InquiryConfig, InquiryResult
    from qe.substrate.question_store import QuestionStore

    # --- Bus with event collection ---
    bus = MemoryBus()
    collected: list[Envelope] = []
    bus.add_publish_listener(lambda env: collected.append(env))

    # --- QuestionStore (SQLite in tmp_path) ---
    db_path = str(tmp_path / "real_llm_test.db")  # type: ignore[operator]
    qs = QuestionStore(db_path=db_path)
    await qs.initialize()

    # --- All real cognitive components (cheap model) ---
    question_gen = QuestionGenerator(model=_MODEL)
    metacognitor = Metacognitor(model=_MODEL)
    dialectic = DialecticEngine(model=_MODEL)
    crystallizer = InsightCrystallizer(model=_MODEL)
    hypothesis_mgr = HypothesisManager(model=_MODEL)

    config = InquiryConfig(
        max_iterations=1,
        questions_per_iteration=2,
        confidence_threshold=0.95,  # High threshold so we don't terminate early
        domain="energy_technology",
        model_fast=_MODEL,
        model_balanced=_MODEL,
    )

    engine = InquiryEngine(
        metacognitor=metacognitor,
        dialectic_engine=dialectic,
        insight_crystallizer=crystallizer,
        question_generator=question_gen,
        hypothesis_manager=hypothesis_mgr,
        question_store=qs,
        bus=bus,
        config=config,
    )

    # --- Run ---
    result = await engine.run_inquiry(
        goal_id="goal_real_llm_test",
        goal_description=(
            "Evaluate whether solid-state batteries are a"
            " viable investment for 2027-2030"
        ),
        config=config,
    )

    # --- Assertions ---

    # 1. Result structure
    assert isinstance(result, InquiryResult)
    assert result.goal_id == "goal_real_llm_test"
    assert result.status == "completed"
    assert result.iterations_completed >= 1

    # 2. LLM-generated questions
    assert result.total_questions_generated >= 1
    for q in result.question_tree:
        assert len(q.text.strip()) > 10, f"Shallow question: {q.text!r}"

    # 3. "No tool registry configured" findings (we passed no tool_registry)
    assert "No tool registry configured" in result.findings_summary

    # 4. All 7 phases timed
    expected_phases = {
        "observe", "orient", "question", "prioritize",
        "investigate", "synthesize", "reflect",
    }
    timed_phases = set(result.phase_timings.keys())
    assert expected_phases == timed_phases, f"Missing phases: {expected_phases - timed_phases}"

    for _phase, stats in result.phase_timings.items():
        assert stats["count"] >= 1.0
        assert stats["total_s"] > 0.0

    # 5. Bus event ordering: started < completed
    started_events = [e for e in collected if e.topic == "inquiry.started"]
    completed_events = [e for e in collected if e.topic == "inquiry.completed"]
    assert len(started_events) == 1
    assert len(completed_events) == 1

    # started is published before completed (check by list index)
    started_idx = collected.index(started_events[0])
    completed_idx = collected.index(completed_events[0])
    assert started_idx < completed_idx

    # 6. Question persistence in QuestionStore
    stored = await qs.get_question_tree(result.inquiry_id)
    assert len(stored) >= 1, "Questions should be persisted in QuestionStore"

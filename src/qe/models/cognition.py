"""Pydantic models for Cognitive Layer structured reasoning outputs.

These models are the shared vocabulary for all five cognitive components:
Metacognitor, Epistemic Reasoner, Dialectic Engine, Persistence Engine,
and Insight Crystallizer.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared Types
# ---------------------------------------------------------------------------

CognitivePhase = Literal[
    "metacognition", "epistemic", "dialectic", "persistence", "crystallization"
]

ConfidenceLevel = Literal["very_low", "low", "moderate", "high", "very_high"]


class ReasoningTrace(BaseModel):
    """Base for all cognitive reasoning steps — stored in EpisodicMemory."""

    trace_id: str = Field(default_factory=lambda: f"trc_{uuid.uuid4().hex[:12]}")
    phase: CognitivePhase
    goal_id: str
    inquiry_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    input_summary: str = ""
    output_summary: str = ""
    reasoning_steps: list[str] = Field(default_factory=list)
    token_cost: int = 0
    model_used: str = ""


# ---------------------------------------------------------------------------
# Metacognitor Models
# ---------------------------------------------------------------------------


class CapabilityProfile(BaseModel):
    """What a tool/capability can do."""

    tool_name: str
    description: str
    domains: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence_in_tool: float = 0.5


class CapabilityGap(BaseModel):
    """A recognized gap in the system's abilities."""

    gap_id: str = Field(default_factory=lambda: f"gap_{uuid.uuid4().hex[:12]}")
    description: str
    workaround: str = ""
    severity: Literal["blocking", "degraded", "cosmetic"] = "degraded"


class ApproachNode(BaseModel):
    """A single node in the approach tree."""

    node_id: str = Field(default_factory=lambda: f"apn_{uuid.uuid4().hex[:12]}")
    parent_id: str | None = None
    approach_description: str
    status: Literal[
        "untried", "in_progress", "succeeded", "failed", "abandoned"
    ] = "untried"
    failure_reason: str = ""
    children: list[str] = Field(default_factory=list)


class ApproachAssessment(BaseModel):
    """LLM-generated assessment of which approach to try next."""

    recommended_approach: str
    reasoning: str
    alternative_approaches: list[str] = Field(default_factory=list)
    tools_needed: list[str] = Field(default_factory=list)
    estimated_success_probability: float = 0.5
    reframe_suggestion: str = ""


class ToolCombinationSuggestion(BaseModel):
    """A creative combination of tools to solve a problem."""

    description: str
    tool_sequence: list[str]
    reasoning: str
    novelty_score: float = 0.5


# ---------------------------------------------------------------------------
# Epistemic Reasoner Models
# ---------------------------------------------------------------------------


class AbsenceDetection(BaseModel):
    """What expected data was NOT found."""

    absence_id: str = Field(default_factory=lambda: f"abs_{uuid.uuid4().hex[:12]}")
    expected_data: str
    why_expected: str
    search_scope: str = ""
    significance: Literal["low", "medium", "high", "critical"] = "medium"
    possible_explanations: list[str] = Field(default_factory=list)


class UncertaintyAssessment(BaseModel):
    """Structured assessment of what we do and don't know about a finding."""

    finding_summary: str
    confidence_level: ConfidenceLevel = "moderate"
    evidence_quality: Literal[
        "primary", "secondary", "hearsay", "inferred"
    ] = "secondary"
    potential_biases: list[str] = Field(default_factory=list)
    information_gaps: list[str] = Field(default_factory=list)
    could_be_wrong_because: list[str] = Field(default_factory=list)


class KnownUnknown(BaseModel):
    """An explicit question the system cannot currently answer."""

    unknown_id: str = Field(default_factory=lambda: f"unk_{uuid.uuid4().hex[:12]}")
    question: str
    why_unknown: str
    importance: Literal["low", "medium", "high", "critical"] = "medium"
    potential_resolution_approaches: list[str] = Field(default_factory=list)


class SurpriseDetection(BaseModel):
    """A finding that contradicts prior expectations or beliefs."""

    surprise_id: str = Field(default_factory=lambda: f"sur_{uuid.uuid4().hex[:12]}")
    finding: str
    expected_instead: str = ""
    surprise_magnitude: float = 0.5  # 0 = mildly unexpected, 1 = shocking
    implications: list[str] = Field(default_factory=list)
    related_belief_ids: list[str] = Field(default_factory=list)


class EpistemicState(BaseModel):
    """Full snapshot of the system's epistemic state for a goal."""

    goal_id: str
    known_facts: list[UncertaintyAssessment] = Field(default_factory=list)
    known_unknowns: list[KnownUnknown] = Field(default_factory=list)
    absences: list[AbsenceDetection] = Field(default_factory=list)
    surprises: list[SurpriseDetection] = Field(default_factory=list)
    overall_confidence: ConfidenceLevel = "moderate"
    blind_spot_warning: str = ""


# ---------------------------------------------------------------------------
# Dialectic Engine Models
# ---------------------------------------------------------------------------


class Counterargument(BaseModel):
    """A structured counterargument to a conclusion."""

    target_claim: str
    counterargument: str
    strength: Literal["weak", "moderate", "strong", "decisive"] = "moderate"
    evidence_needed_to_resolve: str = ""
    concession: str = ""


class PerspectiveAnalysis(BaseModel):
    """Analysis from a specific perspective/viewpoint."""

    perspective_name: str = ""
    key_observations: list[str] = Field(default_factory=list)
    risks_highlighted: list[str] = Field(default_factory=list)
    opportunities_highlighted: list[str] = Field(default_factory=list)
    overall_assessment: str = ""


class AssumptionChallenge(BaseModel):
    """A surfaced and challenged assumption."""

    assumption: str
    is_explicit: bool = True
    challenge: str
    what_if_wrong: str = ""
    testable: bool = True
    test_method: str = ""


class DialecticReport(BaseModel):
    """Full dialectic analysis output."""

    original_conclusion: str
    counterarguments: list[Counterargument] = Field(default_factory=list)
    perspectives: list[PerspectiveAnalysis] = Field(default_factory=list)
    assumptions_challenged: list[AssumptionChallenge] = Field(
        default_factory=list
    )
    revised_confidence: float = 0.5
    should_investigate_further: bool = False
    investigation_questions: list[str] = Field(default_factory=list)
    synthesis: str = ""


# ---------------------------------------------------------------------------
# Persistence Engine Models
# ---------------------------------------------------------------------------


class RootCauseLink(BaseModel):
    """A single link in a Why-Why-Why chain."""

    level: int  # 1 = surface, 2 = deeper, 3+ = root
    question: str
    answer: str
    confidence: float = 0.5
    actionable: bool = False


class RootCauseAnalysis(BaseModel):
    """Complete Why-Why-Why analysis."""

    failure_summary: str
    chain: list[RootCauseLink] = Field(default_factory=list)
    root_cause: str = ""
    lesson_learned: str = ""
    prevention_strategy: str = ""


class ReframingResult(BaseModel):
    """A reframed version of a stuck problem."""

    original_framing: str
    reframing_strategy: Literal[
        "inversion",
        "implication",
        "proxy",
        "decompose_differently",
        "change_domain",
        "stakeholder_shift",
        "temporal_shift",
    ]
    reframed_question: str
    reasoning: str
    estimated_tractability: float = 0.5


# ---------------------------------------------------------------------------
# Insight Crystallizer Models
# ---------------------------------------------------------------------------


class NoveltyAssessment(BaseModel):
    """Assessment of whether a finding is genuinely novel."""

    finding: str
    is_novel: bool = False
    novelty_type: Literal[
        "contradicts_consensus",
        "new_connection",
        "unexpected_magnitude",
        "temporal_anomaly",
        "structural_analogy",
        "absence_significant",
        "not_novel",
    ] = "not_novel"
    why_novel: str = ""
    who_would_find_this_surprising: str = ""
    comparable_known_fact: str = ""


class MechanismExplanation(BaseModel):
    """Causal mechanism behind a finding."""

    what_happens: str
    why_it_happens: str
    how_it_works: str
    key_causal_links: list[str] = Field(default_factory=list)
    confidence_in_mechanism: float = 0.5


class ProvenanceChain(BaseModel):
    """Full chain from question to insight."""

    original_question: str
    evidence_items: list[str] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    assumptions_made: list[str] = Field(default_factory=list)
    insight: str = ""
    confidence: float = 0.5


class ActionabilityResult(BaseModel):
    """How actionable an insight is."""

    score: float = 0.0  # 0 = purely informational, 1 = immediately actionable
    who_can_act: str = ""
    what_action: str = ""
    time_horizon: str = ""


class CrystallizedInsight(BaseModel):
    """A fully formed insight ready for reporting."""

    insight_id: str = Field(
        default_factory=lambda: f"ins_{uuid.uuid4().hex[:12]}"
    )
    headline: str
    mechanism: MechanismExplanation
    novelty: NoveltyAssessment
    provenance: ProvenanceChain
    actionability_score: float = 0.0
    actionability_description: str = ""
    cross_domain_connections: list[str] = Field(default_factory=list)
    dialectic_survivor: bool = False
    confidence: float = 0.5
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

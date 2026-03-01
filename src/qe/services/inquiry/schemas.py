"""Inquiry Loop schemas — pure Pydantic models for the 7-phase inquiry engine."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums (as Literals)
# ---------------------------------------------------------------------------

QuestionType = Literal[
    "factual", "causal", "comparative", "hypothetical",
    "clarifying", "falsification", "meta",
]

QuestionStatus = Literal["pending", "investigating", "answered", "abandoned"]

InquiryPhase = Literal[
    "observe", "orient", "question", "prioritize",
    "investigate", "synthesize", "reflect",
]

TerminationReason = Literal[
    "max_iterations", "budget_exhausted", "confidence_met",
    "all_questions_answered", "approaches_exhausted", "user_cancelled",
]


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


class Question(BaseModel):
    """A single question in the inquiry tree."""

    question_id: str = Field(default_factory=lambda: f"q_{uuid.uuid4().hex[:12]}")
    parent_id: str | None = None
    text: str
    question_type: QuestionType = "factual"
    expected_info_gain: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_to_goal: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    status: QuestionStatus = "pending"
    answer: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence_in_answer: float = Field(default=0.0, ge=0.0, le=1.0)
    hypothesis_id: str | None = None
    iteration_generated: int = 0


class QuestionPriority(BaseModel):
    """Priority assignment for a question."""

    question_id: str
    priority_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class InvestigationResult(BaseModel):
    """Result of investigating a single question."""

    question_id: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    raw_findings: str = ""
    structured_findings: dict[str, Any] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    tokens_used: int = 0
    cost_usd: float = 0.0


class Reflection(BaseModel):
    """Outcome of the Reflect phase."""

    iteration: int
    drift_score: float = Field(default=0.0, ge=0.0, le=1.0)
    on_track: bool = True
    confidence_in_progress: float = Field(default=0.5, ge=0.0, le=1.0)
    decision: Literal["continue", "refocus", "terminate"] = "continue"
    reasoning: str = ""
    questions_answered: int = 0


class InquiryConfig(BaseModel):
    """Configuration for an inquiry run."""

    max_iterations: int = 10
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    budget_hard_stop_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    questions_per_iteration: int = 3
    max_tool_calls_per_question: int = 5
    model_balanced: str = "openai/anthropic/claude-sonnet-4"
    model_fast: str = "openai/google/gemini-2.0-flash"
    domain: str = "general"


class InquiryState(BaseModel):
    """Mutable state for a running inquiry."""

    inquiry_id: str = Field(default_factory=lambda: f"inq_{uuid.uuid4().hex[:12]}")
    goal_id: str = ""
    goal_description: str = ""
    config: InquiryConfig = Field(default_factory=InquiryConfig)
    current_iteration: int = 0
    current_phase: InquiryPhase = "observe"
    questions: list[Question] = Field(default_factory=list)
    investigations: list[InvestigationResult] = Field(default_factory=list)
    reflections: list[Reflection] = Field(default_factory=list)
    findings_summary: str = ""
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    hypotheses_tested: int = 0


class InquiryResult(BaseModel):
    """Final result of a completed inquiry."""

    inquiry_id: str
    goal_id: str
    status: Literal["completed", "failed"] = "completed"
    termination_reason: TerminationReason = "max_iterations"
    iterations_completed: int = 0
    total_questions_generated: int = 0
    total_questions_answered: int = 0
    findings_summary: str = ""
    insights: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses_tested: int = 0
    total_cost_usd: float = 0.0
    duration_seconds: float = 0.0
    question_tree: list[Question] = Field(default_factory=list)

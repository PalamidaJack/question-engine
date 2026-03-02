"""Pydantic models for the Competitive Agent Arena.

Defines tournament-style verification competition models including
Elo ratings, cross-examination, match judgment, divergence checking,
and full arena tournament orchestration.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Elo Rating
# ---------------------------------------------------------------------------


class AgentEloRating(BaseModel):
    """Persistent Elo rating for a cognitive agent in the arena."""

    agent_id: str
    elo: float = 1200.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_matches(self) -> int:
        return self.wins + self.losses + self.draws


# ---------------------------------------------------------------------------
# Cross-Examination
# ---------------------------------------------------------------------------


class CrossExamination(BaseModel):
    """Agent A's structured challenge of Agent B's findings."""

    examination_id: str = Field(
        default_factory=lambda: f"xex_{uuid.uuid4().hex[:12]}"
    )
    challenger_id: str
    defender_id: str
    challenges: list[str] = Field(default_factory=list)
    weaknesses_identified: list[str] = Field(default_factory=list)
    questions_raised: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Defense Response
# ---------------------------------------------------------------------------


class DefenseResponse(BaseModel):
    """Agent B's defense against a cross-examination."""

    defender_id: str
    examination_id: str
    rebuttals: list[str] = Field(default_factory=list)
    concessions: list[str] = Field(default_factory=list)
    additional_evidence: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Match Judgment
# ---------------------------------------------------------------------------


class MatchJudgment(BaseModel):
    """Independent judge scoring of a head-to-head match."""

    judgment_id: str = Field(
        default_factory=lambda: f"jdg_{uuid.uuid4().hex[:12]}"
    )
    agent_a_id: str
    agent_b_id: str
    agent_a_score: float = Field(default=0.0, ge=0.0, le=10.0)
    agent_b_score: float = Field(default=0.0, ge=0.0, le=10.0)
    factual_accuracy_a: float = Field(default=0.0, ge=0.0, le=10.0)
    factual_accuracy_b: float = Field(default=0.0, ge=0.0, le=10.0)
    evidence_quality_a: float = Field(default=0.0, ge=0.0, le=10.0)
    evidence_quality_b: float = Field(default=0.0, ge=0.0, le=10.0)
    novelty_a: float = Field(default=0.0, ge=0.0, le=10.0)
    novelty_b: float = Field(default=0.0, ge=0.0, le=10.0)
    winner: Literal["agent_a", "agent_b", "draw"] = "draw"
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Divergence Check
# ---------------------------------------------------------------------------


class DivergenceCheck(BaseModel):
    """Anti-sycophancy assessment of agent output similarity."""

    divergence_id: str = Field(
        default_factory=lambda: f"div_{uuid.uuid4().hex[:12]}"
    )
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    shared_claims: list[str] = Field(default_factory=list)
    divergent_claims: list[str] = Field(default_factory=list)
    sycophancy_risk: bool = False


# ---------------------------------------------------------------------------
# Arena Match
# ---------------------------------------------------------------------------


class ArenaMatch(BaseModel):
    """One head-to-head match between two agents."""

    match_id: str = Field(
        default_factory=lambda: f"mtch_{uuid.uuid4().hex[:12]}"
    )
    agent_a_id: str
    agent_b_id: str
    examinations: list[CrossExamination] = Field(default_factory=list)
    defenses: list[DefenseResponse] = Field(default_factory=list)
    judgment: MatchJudgment | None = None
    winner: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Arena Config
# ---------------------------------------------------------------------------


class ArenaConfig(BaseModel):
    """Tournament configuration for the competitive arena."""

    enabled: bool = False
    max_rounds: int = 2
    divergence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    budget_limit_usd: float = 0.50
    tournament_style: Literal["round_robin", "single_elimination"] = "round_robin"
    judge_model: str = "openai/google/gemini-2.0-flash"
    examination_model: str = "openai/google/gemini-2.0-flash"


# ---------------------------------------------------------------------------
# Arena Result
# ---------------------------------------------------------------------------


class ArenaResult(BaseModel):
    """Full tournament output from the competitive arena."""

    arena_id: str = Field(
        default_factory=lambda: f"arena_{uuid.uuid4().hex[:12]}"
    )
    goal_id: str = ""
    matches: list[ArenaMatch] = Field(default_factory=list)
    divergence: DivergenceCheck | None = None
    winner_id: str | None = None
    rankings: list[AgentEloRating] = Field(default_factory=list)
    sycophancy_detected: bool = False
    total_cost_usd: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

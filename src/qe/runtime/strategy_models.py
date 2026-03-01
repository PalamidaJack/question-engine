"""Strategy models and schemas for Phase 4 elastic scaling.

Defines StrategyConfig, ScaleProfile, StrategyOutcome, StrategySnapshot,
and predefined defaults used by StrategyEvolver and ElasticScaler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from qe.services.inquiry.schemas import InquiryConfig


class StrategyConfig(BaseModel):
    """Defines an inquiry strategy for the strategy loop."""

    name: str
    description: str = ""
    question_batch_size: int = 3
    max_depth: int = 5
    exploration_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    preferred_model_tier: str = "balanced"


class ScaleProfile(BaseModel):
    """Defines a scaling configuration for the agent pool."""

    name: str
    min_agents: int = 1
    max_agents: int = 3
    model_tier: str = "balanced"
    target_success_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    max_cost_per_goal_usd: float = 1.0


class StrategyOutcome(BaseModel):
    """Records the result of a strategy execution."""

    strategy_name: str
    goal_id: str = ""
    agent_id: str = ""
    success: bool = False
    duration_s: float = 0.0
    cost_usd: float = 0.0
    insights_count: int = 0


class StrategySnapshot(BaseModel):
    """Thompson arm state for persistence/monitoring."""

    strategy_name: str
    alpha: float = 1.0
    beta: float = 1.0
    avg_cost: float = 0.0
    avg_duration: float = 0.0
    sample_count: int = 0


# ── Predefined strategies ────────────────────────────────────────────────

DEFAULT_STRATEGIES: dict[str, StrategyConfig] = {
    "breadth_first": StrategyConfig(
        name="breadth_first",
        description="Explore many questions shallowly before drilling down",
        question_batch_size=5,
        max_depth=3,
        exploration_rate=0.3,
        preferred_model_tier="fast",
    ),
    "depth_first": StrategyConfig(
        name="depth_first",
        description="Focus on one question thread and follow it deeply",
        question_batch_size=1,
        max_depth=10,
        exploration_rate=0.1,
        preferred_model_tier="balanced",
    ),
    "hypothesis_driven": StrategyConfig(
        name="hypothesis_driven",
        description="Generate hypotheses first, then seek evidence to test them",
        question_batch_size=3,
        max_depth=7,
        exploration_rate=0.2,
        preferred_model_tier="balanced",
    ),
    "iterative_refinement": StrategyConfig(
        name="iterative_refinement",
        description="Broad initial pass, then iteratively refine promising leads",
        question_batch_size=4,
        max_depth=5,
        exploration_rate=0.15,
        preferred_model_tier="fast",
    ),
}


def strategy_to_inquiry_config(strategy: StrategyConfig) -> InquiryConfig:
    """Map a StrategyConfig to an InquiryConfig for the InquiryEngine."""
    from qe.services.inquiry.schemas import InquiryConfig as _InquiryConfig

    model_map = {
        "fast": "openai/google/gemini-2.0-flash",
        "balanced": "openai/anthropic/claude-sonnet-4",
    }
    model = model_map.get(strategy.preferred_model_tier, model_map["balanced"])

    return _InquiryConfig(
        questions_per_iteration=strategy.question_batch_size,
        max_iterations=strategy.max_depth,
        confidence_threshold=max(0.1, min(1.0, 1.0 - strategy.exploration_rate)),
        model_balanced=model,
        model_fast="openai/google/gemini-2.0-flash",
    )


# ── Predefined scale profiles ────────────────────────────────────────────

DEFAULT_PROFILES: dict[str, ScaleProfile] = {
    "minimal": ScaleProfile(
        name="minimal",
        min_agents=1,
        max_agents=1,
        model_tier="fast",
        target_success_rate=0.5,
        max_cost_per_goal_usd=0.5,
    ),
    "balanced": ScaleProfile(
        name="balanced",
        min_agents=1,
        max_agents=3,
        model_tier="balanced",
        target_success_rate=0.7,
        max_cost_per_goal_usd=1.0,
    ),
    "aggressive": ScaleProfile(
        name="aggressive",
        min_agents=2,
        max_agents=5,
        model_tier="balanced",
        target_success_rate=0.85,
        max_cost_per_goal_usd=3.0,
    ),
}

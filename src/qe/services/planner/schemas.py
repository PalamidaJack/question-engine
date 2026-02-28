"""Pydantic schemas for the planner's LLM output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SubtaskPlan(BaseModel):
    """A single subtask in the decomposition."""

    description: str = Field(description="What this subtask should accomplish")
    task_type: Literal[
        "research",
        "analysis",
        "fact_check",
        "synthesis",
        "code_execution",
        "web_search",
        "document_generation",
    ] = Field(description="The type of work required")
    depends_on_indices: list[int] = Field(
        default_factory=list,
        description="Indices (0-based) of subtasks this depends on",
    )
    model_tier: Literal["fast", "balanced", "powerful", "local"] = Field(
        default="balanced",
        description="Model capability needed for this subtask",
    )


class ProblemRepresentation(BaseModel):
    """Stage 1: Problem analysis before decomposition."""

    core_problem: str = Field(description="The core problem in one sentence")
    actual_need: str = Field(
        description="What actually needs to happen (may differ from literal request)"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Hard limits and constraints"
    )
    success_criteria: list[str] = Field(
        default_factory=list, description="What a correct solution looks like"
    )
    problem_type: Literal[
        "well_defined", "ill_defined", "wicked"
    ] = Field(
        default="well_defined",
        description="Classification of the problem type",
    )


class DecompositionOutput(BaseModel):
    """Complete LLM output for goal decomposition."""

    representation: ProblemRepresentation = Field(
        description="Problem analysis (Stage 1)"
    )
    strategy: str = Field(
        description="Human-readable explanation of the approach"
    )
    subtasks: list[SubtaskPlan] = Field(
        description="Ordered list of subtasks (Stage 2)"
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions that could invalidate this plan",
    )
    estimated_time_seconds: int = Field(
        default=300,
        description="Estimated total execution time",
    )

"""Pydantic data models for the Innovation Scout meta-agent."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ScoutFinding(BaseModel):
    """A web resource discovered by the scout."""

    finding_id: str = Field(
        default_factory=lambda: f"fnd_{uuid.uuid4().hex[:12]}"
    )
    url: str
    title: str
    snippet: str
    full_content: str = ""
    source_type: Literal[
        "github", "hackernews", "reddit", "blog", "arxiv", "forum"
    ]
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    tags: list[str] = Field(default_factory=list)


class ImprovementIdea(BaseModel):
    """An actionable improvement idea derived from a finding."""

    idea_id: str = Field(
        default_factory=lambda: f"idea_{uuid.uuid4().hex[:12]}"
    )
    finding_id: str
    title: str
    description: str
    category: Literal[
        "performance", "feature", "refactor", "testing",
        "security", "dependency", "pattern", "model", "other",
    ]
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    feasibility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    impact_score: float = Field(default=0.0, ge=0.0, le=1.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_url: str = ""
    rationale: str = ""
    affected_files: list[str] = Field(default_factory=list)


class CodeChange(BaseModel):
    """A single file-level change in a proposal."""

    file_path: str
    change_type: Literal["create", "modify", "delete"]
    diff: str = ""


class TestResult(BaseModel):
    """Result of running tests in the sandbox."""

    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


class ImprovementProposal(BaseModel):
    """A complete improvement proposal with code, tests, and review status."""

    proposal_id: str = Field(
        default_factory=lambda: f"prop_{uuid.uuid4().hex[:12]}"
    )
    idea: ImprovementIdea
    status: Literal[
        "draft", "testing", "test_passed", "test_failed",
        "pending_review", "approved", "rejected", "applied", "reverted",
    ] = "draft"
    changes: list[CodeChange] = Field(default_factory=list)
    test_result: TestResult | None = None
    impact_assessment: str = ""
    risk_assessment: str = ""
    rollback_plan: str = ""
    branch_name: str = ""
    worktree_path: str = ""
    hil_envelope_id: str | None = None
    reviewer_feedback: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    decided_at: datetime | None = None
    applied_at: datetime | None = None


class ScoutFeedbackRecord(BaseModel):
    """Human feedback on a proposal for learning."""

    record_id: str = Field(
        default_factory=lambda: f"sfb_{uuid.uuid4().hex[:12]}"
    )
    proposal_id: str
    decision: Literal["approved", "rejected"]
    feedback: str = ""
    category: str = ""
    source_type: str = ""

"""Event payload schemas for typed bus communication.

Provides Pydantic models for critical topic payloads, enabling
validation on publish/subscribe and schema versioning.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ── Payload schemas per topic ──────────────────────────────────────────────


class ClaimProposedPayload(BaseModel):
    """Payload for claims.proposed topic."""

    claim_id: str = ""
    schema_version: str = "1.0"
    subject_entity_id: str
    predicate: str
    object_value: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_service_id: str = ""
    source_envelope_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimCommittedPayload(ClaimProposedPayload):
    """Payload for claims.committed topic."""

    created_at: str = ""


class GoalSubmittedPayload(BaseModel):
    """Payload for goals.submitted topic."""

    goal_id: str
    description: str


class GoalCompletedPayload(BaseModel):
    """Payload for goals.completed topic."""

    goal_id: str
    subtask_count: int = 0


class GoalFailedPayload(BaseModel):
    """Payload for goals.failed topic."""

    goal_id: str
    reason: str = ""


class TaskDispatchedPayload(BaseModel):
    """Payload for tasks.dispatched topic."""

    goal_id: str
    subtask_id: str
    description: str = ""
    task_type: str = ""
    model_tier: str = "balanced"
    depends_on: list[str] = Field(default_factory=list)
    contract: dict[str, Any] = Field(default_factory=dict)


class SystemErrorPayload(BaseModel):
    """Payload for system.error topic."""

    envelope_id: str = ""
    topic: str = ""
    error: str = ""


class SystemCircuitBreakPayload(BaseModel):
    """Payload for system.circuit_break topic."""

    service_id: str
    reason: str = ""
    count: int = 0
    window_seconds: int = 60


class HeartbeatPayload(BaseModel):
    """Payload for system.heartbeat topic."""

    turn_count: int = 0
    status: str = "alive"


class DLQPayload(BaseModel):
    """Payload for system.dlq topic."""

    envelope_id: str
    topic: str
    source_service_id: str = ""
    handler_name: str = ""
    error: str = ""
    attempts: int = 0
    failed_at: float = 0.0
    payload: dict[str, Any] = Field(default_factory=dict)


class GateDeniedPayload(BaseModel):
    """Payload for system.gate_denied topic."""

    reason: str = ""
    risk_score: float = 0.0
    patterns: list[str] = Field(default_factory=list)


# ── Multi-agent coordination payloads ──────────────────────────────────────


class AgentRegisteredPayload(BaseModel):
    """Payload for agents.registered topic."""

    agent_id: str
    service_id: str = ""
    capabilities: list[str] = Field(default_factory=list)
    task_types: list[str] = Field(default_factory=list)
    model_tier: str = "balanced"
    max_concurrency: int = 5


class VoteRequestPayload(BaseModel):
    """Payload for coordination.vote_request topic."""

    vote_id: str
    question: str
    options: list[str]
    goal_id: str = ""
    timeout_seconds: float = 10.0
    min_voters: int = 1


class VoteResponsePayload(BaseModel):
    """Payload for coordination.vote_response topic."""

    vote_id: str
    agent_id: str
    choice: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    reasoning: str = ""


class TaskDelegatedPayload(BaseModel):
    """Payload for tasks.delegated topic."""

    goal_id: str
    subtask_id: str
    description: str = ""
    task_type: str = ""
    assigned_agent_id: str = ""
    tools_required: list[str] = Field(default_factory=list)
    dependency_context: dict[str, Any] = Field(default_factory=dict)


# ── Inquiry Loop payloads ──────────────────────────────────────────────────


class InquiryStartedPayload(BaseModel):
    """Payload for inquiry.started topic."""

    inquiry_id: str
    goal_id: str
    goal: str = ""


class InquiryPhaseCompletedPayload(BaseModel):
    """Payload for inquiry.phase_completed topic."""

    inquiry_id: str
    goal_id: str
    phase: Literal[
        "observe", "orient", "question", "prioritize",
        "investigate", "synthesize", "reflect",
    ]
    iteration: int = 0
    decision: str = ""


class InquiryCompletedPayload(BaseModel):
    """Payload for inquiry.completed topic."""

    inquiry_id: str
    goal_id: str
    status: str = "completed"
    iterations: int = 0
    insights: int = 0
    questions_answered: int = 0
    duration_s: float = 0.0
    cost_usd: float = 0.0


class InquiryQuestionGeneratedPayload(BaseModel):
    """Payload for inquiry.question_generated topic."""

    inquiry_id: str
    question_id: str
    text: str = ""


class InquiryInvestigationCompletedPayload(BaseModel):
    """Payload for inquiry.investigation_completed topic."""

    inquiry_id: str
    question_id: str


class InquiryHypothesisGeneratedPayload(BaseModel):
    """Payload for inquiry.hypothesis_generated topic."""

    inquiry_id: str
    hypothesis_id: str
    statement: str = ""


class InquiryHypothesisUpdatedPayload(BaseModel):
    """Payload for inquiry.hypothesis_updated topic."""

    inquiry_id: str
    hypothesis_id: str
    probability: float = Field(default=0.5, ge=0.0, le=1.0)


class InquiryInsightGeneratedPayload(BaseModel):
    """Payload for inquiry.insight_generated topic."""

    inquiry_id: str
    insight_id: str
    headline: str = ""


class InquiryFailedPayload(BaseModel):
    """Payload for inquiry.failed topic."""

    inquiry_id: str
    iteration: int = 0


class InquiryBudgetWarningPayload(BaseModel):
    """Payload for inquiry.budget_warning topic."""

    inquiry_id: str
    iteration: int = 0


# ── Strategy Loop payloads (Phase 4) ──────────────────────────────────────


class StrategySelectedPayload(BaseModel):
    """Payload for strategy.selected topic."""

    strategy_name: str
    agent_id: str = ""
    reason: str = ""


class StrategySwitchRequestedPayload(BaseModel):
    """Payload for strategy.switch_requested topic."""

    agent_id: str = ""
    from_strategy: str
    to_strategy: str
    reason: str = ""


class StrategyEvaluatedPayload(BaseModel):
    """Payload for strategy.evaluated topic."""

    strategy_name: str
    alpha: float = 1.0
    beta: float = 1.0
    sample_count: int = 0


class PoolScaleRecommendedPayload(BaseModel):
    """Payload for pool.scale_recommended topic."""

    profile_name: str
    agents_count: int = 0
    model_tier: str = "balanced"
    reasoning: str = ""


class PoolScaleExecutedPayload(BaseModel):
    """Payload for pool.scale_executed topic."""

    profile_name: str
    agents_before: int = 0
    agents_after: int = 0


class PoolHealthCheckPayload(BaseModel):
    """Payload for pool.health_check topic."""

    total_agents: int = 0
    active_agents: int = 0
    avg_success_rate: float = 0.0
    avg_load_pct: float = 0.0


# ── Prompt Evolution payloads ─────────────────────────────────────────────


class PromptVariantSelectedPayload(BaseModel):
    """Payload for prompt.variant_selected topic."""

    slot_key: str
    variant_id: str
    is_baseline: bool = False


class PromptOutcomeRecordedPayload(BaseModel):
    """Payload for prompt.outcome_recorded topic."""

    slot_key: str
    variant_id: str
    success: bool
    quality_score: float = 0.0


class PromptVariantCreatedPayload(BaseModel):
    """Payload for prompt.variant_created topic."""

    slot_key: str
    variant_id: str
    parent_variant_id: str = ""
    strategy: str = "manual"


class PromptVariantDeactivatedPayload(BaseModel):
    """Payload for prompt.variant_deactivated topic."""

    slot_key: str
    variant_id: str
    reason: str = ""


class PromptMutationCyclePayload(BaseModel):
    """Payload for prompt.mutation_cycle_completed topic."""

    slots_evaluated: int = 0
    variants_created: int = 0
    variants_rolled_back: int = 0
    variants_promoted: int = 0


class PromptVariantPromotedPayload(BaseModel):
    """Payload for prompt.variant_promoted topic."""

    slot_key: str
    variant_id: str
    old_rollout_pct: float = 10.0
    new_rollout_pct: float = 50.0


# ── Knowledge Loop payloads ──────────────────────────────────────────────


class KnowledgeConsolidationCompletedPayload(BaseModel):
    """Payload for knowledge.consolidation_completed topic."""

    episodes_scanned: int = 0
    patterns_detected: int = 0
    beliefs_promoted: int = 0
    contradictions_found: int = 0
    hypotheses_reviewed: int = 0


class KnowledgeBeliefPromotedPayload(BaseModel):
    """Payload for knowledge.belief_promoted topic."""

    subject_entity_id: str
    predicate: str
    object_value: str
    confidence: float = 0.5
    evidence_count: int = 0


class KnowledgeHypothesisUpdatedPayload(BaseModel):
    """Payload for knowledge.hypothesis_updated topic."""

    hypothesis_id: str
    old_status: str = "active"
    new_status: str = "active"
    probability: float = 0.5


# ── Inquiry Bridge payloads ────────────────────────────────────────────────


class BridgeStrategyOutcomePayload(BaseModel):
    """Payload for bridge.strategy_outcome_recorded topic."""

    strategy_name: str
    goal_id: str = ""
    success: bool = False
    insights_count: int = 0


# ── Schema Registry ────────────────────────────────────────────────────────

# Maps topic -> payload model for validation
TOPIC_SCHEMAS: dict[str, type[BaseModel]] = {
    "claims.proposed": ClaimProposedPayload,
    "claims.committed": ClaimCommittedPayload,
    "goals.submitted": GoalSubmittedPayload,
    "goals.completed": GoalCompletedPayload,
    "goals.failed": GoalFailedPayload,
    "tasks.dispatched": TaskDispatchedPayload,
    "system.error": SystemErrorPayload,
    "system.circuit_break": SystemCircuitBreakPayload,
    "system.heartbeat": HeartbeatPayload,
    "system.dlq": DLQPayload,
    "system.gate_denied": GateDeniedPayload,
    "agents.registered": AgentRegisteredPayload,
    "coordination.vote_request": VoteRequestPayload,
    "coordination.vote_response": VoteResponsePayload,
    "tasks.delegated": TaskDelegatedPayload,
    "strategy.selected": StrategySelectedPayload,
    "strategy.switch_requested": StrategySwitchRequestedPayload,
    "strategy.evaluated": StrategyEvaluatedPayload,
    "pool.scale_recommended": PoolScaleRecommendedPayload,
    "pool.scale_executed": PoolScaleExecutedPayload,
    "pool.health_check": PoolHealthCheckPayload,
    "inquiry.started": InquiryStartedPayload,
    "inquiry.phase_completed": InquiryPhaseCompletedPayload,
    "inquiry.question_generated": InquiryQuestionGeneratedPayload,
    "inquiry.investigation_completed": InquiryInvestigationCompletedPayload,
    "inquiry.hypothesis_generated": InquiryHypothesisGeneratedPayload,
    "inquiry.hypothesis_updated": InquiryHypothesisUpdatedPayload,
    "inquiry.insight_generated": InquiryInsightGeneratedPayload,
    "inquiry.completed": InquiryCompletedPayload,
    "inquiry.failed": InquiryFailedPayload,
    "inquiry.budget_warning": InquiryBudgetWarningPayload,
    "prompt.variant_selected": PromptVariantSelectedPayload,
    "prompt.outcome_recorded": PromptOutcomeRecordedPayload,
    "prompt.variant_created": PromptVariantCreatedPayload,
    "prompt.variant_deactivated": PromptVariantDeactivatedPayload,
    "prompt.mutation_cycle_completed": PromptMutationCyclePayload,
    "prompt.variant_promoted": PromptVariantPromotedPayload,
    "knowledge.consolidation_completed": KnowledgeConsolidationCompletedPayload,
    "knowledge.belief_promoted": KnowledgeBeliefPromotedPayload,
    "knowledge.hypothesis_updated": KnowledgeHypothesisUpdatedPayload,
    "bridge.strategy_outcome_recorded": BridgeStrategyOutcomePayload,
}


def validate_payload(topic: str, payload: dict[str, Any]) -> BaseModel | None:
    """Validate a payload against its topic schema.

    Returns the validated model if schema exists, None if no schema
    registered for this topic. Raises ValidationError on bad data.
    """
    schema = TOPIC_SCHEMAS.get(topic)
    if schema is None:
        return None
    return schema.model_validate(payload)


def get_schema(topic: str) -> type[BaseModel] | None:
    """Look up the schema model for a topic."""
    return TOPIC_SCHEMAS.get(topic)

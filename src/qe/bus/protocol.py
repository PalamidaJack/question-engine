from qe.models.envelope import Envelope

TOPICS = {
    # Ingestion
    "observations.raw",
    "observations.structured",
    # Belief Ledger
    "claims.proposed",
    "claims.committed",
    "claims.contradiction_detected",
    "claims.challenged",
    "claims.verification_requested",
    "predictions.proposed",
    "predictions.committed",
    "null_results.committed",
    # Goal orchestration
    "goals.submitted",
    "goals.enriched",
    "goals.completed",
    "goals.failed",
    "goals.drift_detected",
    # Task decomposition
    "tasks.planned",
    "tasks.dispatched",
    "tasks.completed",
    "tasks.verified",
    "tasks.verification_failed",
    "tasks.recovered",
    "tasks.failed",
    "tasks.progress",
    "tasks.checkpoint",
    # Orchestration (legacy)
    "investigations.requested",
    "investigations.completed",
    # Query
    "queries.asked",
    "queries.answered",
    # Entity
    "entities.resolved",
    # Ingestion (lifecycle)
    "ingestion.item_received",
    # HIL
    "hil.approval_required",
    "hil.approved",
    "hil.rejected",
    # Chat
    "chat.message_received",
    "chat.response_sent",
    # Memory
    "memory.updated",
    "memory.preference_set",
    # Analysis
    "analysis.requested",
    "analysis.completed",
    # Synthesis
    "synthesis.requested",
    "synthesis.completed",
    # Monitoring
    "monitor.scheduled",
    "monitor.triggered",
    "monitor.alert",
    # Voice & Multimodal
    "voice.ingested",
    "voice.transcribed",
    "document.ingested",
    "document.parsed",
    # Channels & Notifications
    "channel.message_received",
    "channel.message_sent",
    "notification.queued",
    "notification.delivered",
    # Inference
    "inference.claim_inferred",
    "inference.inconsistency_detected",
    # Predictions
    "predictions.resolved",
    # Security
    "system.gate_denied",
    "system.integrity_violation",
    # System
    "system.heartbeat",
    "system.error",
    "system.circuit_break",
    "system.service_stalled",
    "system.service_restarted",
    "system.budget_alert",
    "system.resource_alert",
    "system.security_alert",
    "system.digest",
    # Doctor / Health
    "system.health.check",
    "system.health.report",
    # Dead Letter Queue
    "system.dlq",
    # Multi-agent coordination
    "agents.registered",
    "agents.deregistered",
    "agents.heartbeat",
    "coordination.vote_request",
    "coordination.vote_response",
    "coordination.consensus",
    "tasks.delegated",
    "tasks.delegation_result",
    # Cognitive Layer (v2)
    "cognitive.approach_selected",
    "cognitive.approach_exhausted",
    "cognitive.absence_detected",
    "cognitive.surprise_detected",
    "cognitive.uncertainty_assessed",
    "cognitive.dialectic_completed",
    "cognitive.assumption_surfaced",
    "cognitive.root_cause_analyzed",
    "cognitive.reframe_suggested",
    "cognitive.lesson_learned",
    "cognitive.insight_crystallized",
    "cognitive.capability_gap",
    # Strategy Loop (Phase 4)
    "strategy.selected",
    "strategy.switch_requested",
    "strategy.evaluated",
    "pool.scale_recommended",
    "pool.scale_executed",
    "pool.health_check",
    # Inquiry Loop (Phase 3)
    "inquiry.started",
    "inquiry.phase_completed",
    "inquiry.question_generated",
    "inquiry.investigation_completed",
    "inquiry.hypothesis_generated",
    "inquiry.hypothesis_updated",
    "inquiry.insight_generated",
    "inquiry.completed",
    "inquiry.failed",
    "inquiry.budget_warning",
    # Prompt Evolution
    "prompt.variant_selected",
    "prompt.outcome_recorded",
    "prompt.variant_created",
    "prompt.variant_deactivated",
    "prompt.mutation_cycle_completed",
    "prompt.variant_promoted",
    # Knowledge Loop
    "knowledge.consolidation_completed",
    "knowledge.belief_promoted",
    "knowledge.hypothesis_updated",
    # Inquiry Bridge
    "bridge.strategy_outcome_recorded",
    # Competitive Arena
    "arena.tournament_started",
    "arena.tournament_completed",
    "arena.match_completed",
    "arena.divergence_checked",
    "arena.sycophancy_fallback",
    "arena.elo_updated",
    # Goal Orchestration Pipeline
    "tasks.contract_violated",
    "goals.synthesized",
    "goals.synthesis_failed",
}


def validate_envelope(envelope: Envelope) -> None:
    """Validate that the envelope's topic is in TOPICS. Raises ValueError if not."""
    if envelope.topic not in TOPICS:
        raise ValueError(f"Unknown topic: {envelope.topic}. Must be one of {TOPICS}")

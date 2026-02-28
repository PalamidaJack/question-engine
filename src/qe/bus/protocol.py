from qe.models.envelope import Envelope

TOPICS = {
    # Ingestion
    "observations.raw",
    "observations.structured",
    # Belief Ledger
    "claims.proposed",
    "claims.committed",
    "claims.contradiction_detected",
    "predictions.proposed",
    "predictions.committed",
    "null_results.committed",
    # Orchestration
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
    # System
    "system.heartbeat",
    "system.error",
    "system.circuit_break",
    "system.service_stalled",
    "system.budget_alert",
}


def validate_envelope(envelope: Envelope) -> None:
    """Validate that the envelope's topic is in TOPICS. Raises ValueError if not."""
    if envelope.topic not in TOPICS:
        raise ValueError(f"Unknown topic: {envelope.topic}. Must be one of {TOPICS}")

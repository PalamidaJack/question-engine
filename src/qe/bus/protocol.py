from qe.models.envelope import Envelope


TOPICS = {
    # Ingestion
    "observations.raw",
    "observations.structured",
    # Belief Ledger
    "claims.proposed",
    "claims.committed",
    "predictions.proposed",
    "predictions.committed",
    "null_results.committed",
    # Orchestration
    "investigations.requested",
    "investigations.completed",
    # HIL
    "hil.approval_required",
    "hil.approved",
    "hil.rejected",
    # System
    "system.heartbeat",
    "system.error",
}


def validate_envelope(envelope: Envelope) -> None:
    """Validate that the envelope's topic is in TOPICS. Raises ValueError if not."""
    if envelope.topic not in TOPICS:
        raise ValueError(f"Unknown topic: {envelope.topic}. Must be one of {TOPICS}")

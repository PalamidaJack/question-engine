"""Ingestion service: publishes text as observations to the bus."""

from __future__ import annotations

import logging

from qe.bus.memory_bus import MemoryBus
from qe.models.envelope import Envelope

log = logging.getLogger(__name__)


class IngestionService:
    """Stateless helper that publishes text observations to the bus."""

    def __init__(self, bus: MemoryBus) -> None:
        self._bus = bus

    def ingest_text(self, text: str, source: str = "ingestor") -> Envelope:
        """Publish a single text observation and return the envelope."""
        envelope = Envelope(
            topic="observations.structured",
            source_service_id=source,
            payload={"text": text},
        )
        self._bus.publish(envelope)

        # Also publish lifecycle event
        self._bus.publish(Envelope(
            topic="ingestion.item_received",
            source_service_id=source,
            correlation_id=envelope.envelope_id,
            payload={"text_length": len(text)},
        ))

        log.info("Ingested observation: %s (%d chars)", envelope.envelope_id, len(text))
        return envelope

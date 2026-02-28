"""Claim validation service: duplicate detection, contradiction detection, then commit."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from qe.models.claim import Claim
from qe.models.envelope import Envelope
from qe.runtime.service import BaseService

log = logging.getLogger(__name__)

_DUPLICATE_WINDOW_HOURS = 24


class ClaimValidatorService(BaseService):
    """Subscribes to claims.proposed, validates, and commits to the ledger."""

    async def _handle_envelope(self, envelope: Envelope) -> None:
        if envelope.topic != "claims.proposed":
            return

        payload = envelope.payload
        if "claim_id" not in payload:
            return

        claim = Claim.model_validate(payload)

        # ── Duplicate detection ──────────────────────────────────────
        existing = await self.substrate.get_claims(
            subject_entity_id=claim.subject_entity_id,
            predicate=claim.predicate,
        )

        cutoff = datetime.now(UTC) - timedelta(hours=_DUPLICATE_WINDOW_HOURS)
        for existing_claim in existing:
            if (
                existing_claim.object_value == claim.object_value
                and existing_claim.created_at > cutoff
            ):
                log.info(
                    "Duplicate claim dropped: %s (matches %s)",
                    claim.claim_id,
                    existing_claim.claim_id,
                )
                return

        # ── Contradiction detection ──────────────────────────────────
        for existing_claim in existing:
            if (
                existing_claim.object_value != claim.object_value
                and existing_claim.superseded_by is None
            ):
                log.warning(
                    "Contradiction detected: %s vs %s for (%s, %s)",
                    claim.object_value,
                    existing_claim.object_value,
                    claim.subject_entity_id,
                    claim.predicate,
                )
                self.bus.publish(
                    Envelope(
                        topic="claims.contradiction_detected",
                        source_service_id=self.blueprint.service_id,
                        correlation_id=envelope.correlation_id or envelope.envelope_id,
                        payload={
                            "new_claim": claim.model_dump(mode="json"),
                            "existing_claim": existing_claim.model_dump(mode="json"),
                            "subject": claim.subject_entity_id,
                            "predicate": claim.predicate,
                        },
                    )
                )

        # ── Commit ───────────────────────────────────────────────────
        committed = await self.substrate.commit_claim(claim)
        log.info("Claim committed: %s", committed.claim_id)

        self.bus.publish(
            Envelope(
                topic="claims.committed",
                source_service_id=self.blueprint.service_id,
                correlation_id=envelope.correlation_id or envelope.envelope_id,
                causation_id=envelope.envelope_id,
                payload=committed.model_dump(mode="json"),
            )
        )

    async def handle_response(self, envelope: Envelope, response) -> None:
        pass  # No LLM calls — purely event-driven

    def get_response_schema(self, topic: str):
        raise NotImplementedError("Validator does not use LLM")

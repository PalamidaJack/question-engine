
from pydantic import BaseModel

from qe.models.claim import Claim
from qe.models.envelope import Envelope
from qe.runtime.service import BaseService
from qe.services.researcher.schemas import ClaimExtractionResponse, ClaimProposal


class ResearcherService(BaseService):
    """Extracts claims from observations and publishes claims.proposed.

    Claim commitment is handled by ClaimValidatorService.
    """

    def get_response_schema(self, topic: str) -> type[BaseModel]:
        if topic == "observations.structured":
            return ClaimExtractionResponse
        raise ValueError(f"Unsupported topic for LLM extraction: {topic}")

    async def _handle_envelope(self, envelope: Envelope) -> None:
        if envelope.topic != "observations.structured":
            return

        cold_storage = getattr(self.substrate, "cold_storage", None)
        if cold_storage is not None:
            cold_storage.append(envelope)

        messages = self.context_manager.build_messages(envelope, self._turn_count)
        model = self.router.select(envelope)
        response = await self._call_llm(
            model, messages, self.get_response_schema(envelope.topic)
        )

        self._turn_count += 1
        if self._turn_count % self.blueprint.reinforcement_interval_turns == 0:
            self.context_manager.reinforce()

        await self.handle_response(envelope, response)

    async def handle_response(
        self, envelope: Envelope, response: ClaimExtractionResponse
    ) -> None:
        for proposal in response.claims:
            claim = self._proposal_to_claim(envelope, proposal)
            self.bus.publish(
                Envelope(
                    topic="claims.proposed",
                    source_service_id=self.blueprint.service_id,
                    correlation_id=envelope.envelope_id,
                    causation_id=envelope.envelope_id,
                    payload=claim.model_dump(mode="json"),
                )
            )

    def _proposal_to_claim(self, envelope: Envelope, proposal: ClaimProposal) -> Claim:
        return Claim(
            subject_entity_id=proposal.subject_entity_id,
            predicate=proposal.predicate,
            object_value=proposal.object_value,
            confidence=proposal.confidence,
            source_service_id=self.blueprint.service_id,
            source_envelope_ids=[envelope.envelope_id],
            metadata={"reasoning": proposal.reasoning},
        )

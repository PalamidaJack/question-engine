from pydantic import BaseModel


class ClaimProposal(BaseModel):
    subject_entity_id: str
    predicate: str
    object_value: str
    confidence: float
    reasoning: str


class ClaimExtractionResponse(BaseModel):
    claims: list[ClaimProposal]

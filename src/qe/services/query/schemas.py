"""Schemas for the query/answer service."""

from pydantic import BaseModel, Field


class AnswerResponse(BaseModel):
    """LLM-generated answer with confidence and reasoning."""

    answer: str = Field(description="The synthesized answer to the question")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the answer based on available evidence",
    )
    reasoning: str = Field(description="Explanation of how the answer was derived")

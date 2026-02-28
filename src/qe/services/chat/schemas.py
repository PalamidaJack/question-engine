"""Pydantic models for the chat interface."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ChatIntent(StrEnum):
    QUESTION = "question"
    OBSERVATION = "observation"
    COMMAND = "command"
    CONVERSATION = "conversation"


class IntentClassification(BaseModel):
    """LLM-generated intent classification for a user message."""

    intent: ChatIntent = Field(description="The detected intent of the user message")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the classification"
    )
    extracted_query: str = Field(
        description="The core query/observation/command extracted from the message"
    )
    reasoning: str = Field(description="Brief reasoning for the classification")


class CommandAction(StrEnum):
    RETRACT_CLAIM = "retract_claim"
    LIST_CLAIMS = "list_claims"
    LIST_ENTITIES = "list_entities"
    SHOW_ENTITY = "show_entity"
    SHOW_BUDGET = "show_budget"
    HELP = "help"
    UNKNOWN = "unknown"


class CommandParse(BaseModel):
    """Parsed command details."""

    action: CommandAction
    target: str | None = Field(
        default=None, description="Target entity/claim ID if applicable"
    )


class ChatResponsePayload(BaseModel):
    """Structured response sent back to the chat client."""

    message_id: str
    reply_text: str
    intent: ChatIntent
    claims: list[dict] = Field(default_factory=list)
    entities: list[dict] = Field(default_factory=list)
    confidence: float | None = None
    reasoning: str | None = None
    pipeline_complete: bool = True
    tracking_envelope_id: str | None = None
    error: str | None = None


class ConversationalResponse(BaseModel):
    """LLM-generated conversational reply."""

    reply: str = Field(description="The conversational response to the user")

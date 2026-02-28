"""Chat service: intent detection, routing, and response generation."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import instructor
import litellm
from dotenv import load_dotenv

from qe.models.envelope import Envelope
from qe.runtime.budget import BudgetTracker
from qe.services.chat.schemas import (
    ChatIntent,
    ChatResponsePayload,
    CommandAction,
    CommandParse,
    ConversationalResponse,
    IntentClassification,
)
from qe.services.query.service import answer_question
from qe.substrate import Substrate

load_dotenv()

log = logging.getLogger(__name__)

_MAX_HISTORY = 50
_HISTORY_TRIM_TO = 30


class ChatSession:
    """Per-session conversation state."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.history: list[dict[str, str]] = []
        self.created_at = datetime.now(UTC)
        self.last_active = datetime.now(UTC)

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})
        self.last_active = datetime.now(UTC)
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self) -> None:
        if len(self.history) > _MAX_HISTORY:
            self.history = self.history[-_HISTORY_TRIM_TO:]


class ChatService:
    """Orchestrates chat: intent detection, routing, response generation."""

    def __init__(
        self,
        substrate: Substrate,
        bus: Any,
        budget_tracker: BudgetTracker | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.substrate = substrate
        self.bus = bus
        self.budget_tracker = budget_tracker
        self.model = model
        self._sessions: dict[str, ChatSession] = {}

    def get_or_create_session(self, session_id: str | None = None) -> ChatSession:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        sid = session_id or str(uuid.uuid4())
        session = ChatSession(sid)
        self._sessions[sid] = session
        return session

    async def handle_message(
        self, session_id: str, user_message: str
    ) -> ChatResponsePayload:
        """Main entry point: receive user message, return structured response."""
        session = self.get_or_create_session(session_id)
        session.add_user_message(user_message)
        message_id = str(uuid.uuid4())

        if self.budget_tracker and self.budget_tracker.remaining_pct() <= 0:
            return ChatResponsePayload(
                message_id=message_id,
                reply_text="Budget exhausted. Unable to process your request.",
                intent=ChatIntent.CONVERSATION,
                error="budget_exhausted",
            )

        try:
            intent_result = await self._classify_intent(user_message, session)

            if intent_result.intent == ChatIntent.QUESTION:
                response = await self._handle_question(
                    intent_result.extracted_query, message_id
                )
            elif intent_result.intent == ChatIntent.OBSERVATION:
                response = await self._handle_observation(
                    intent_result.extracted_query, user_message, message_id
                )
            elif intent_result.intent == ChatIntent.COMMAND:
                response = await self._handle_command(
                    intent_result.extracted_query, message_id
                )
            else:
                response = await self._handle_conversation(
                    user_message, message_id, session
                )

            session.add_assistant_message(response.reply_text)
            return response

        except Exception as e:
            log.exception("Chat handler error for session %s", session_id)
            error_response = ChatResponsePayload(
                message_id=message_id,
                reply_text=f"I encountered an error processing your message: {e}",
                intent=ChatIntent.CONVERSATION,
                error=str(e),
            )
            session.add_assistant_message(error_response.reply_text)
            return error_response

    # ── Intent Classification ───────────────────────────────────────────

    async def _classify_intent(
        self, message: str, session: ChatSession
    ) -> IntentClassification:
        """Use LLM to classify user intent."""
        recent = session.history[-6:]
        history_context = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent[:-1]
        )

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You classify user messages into one of four intents:\n"
                    "- question: User wants to query existing knowledge "
                    "(e.g. 'What do we know about X?', 'Tell me about Y')\n"
                    "- observation: User is providing new information to be ingested "
                    "(e.g. 'I read that X happened', 'NASA announced Y')\n"
                    "- command: User wants to perform an action "
                    "(e.g. 'retract claim X', 'show entities', 'list claims', "
                    "'show budget', 'help')\n"
                    "- conversation: General chat, greetings, clarifications, "
                    "or anything that doesn't fit above\n\n"
                    "Extract the core query/observation/command from the message."
                ),
            },
        ]
        if history_context:
            messages.append({
                "role": "user",
                "content": f"Conversation context:\n{history_context}",
            })
        messages.append({
            "role": "user",
            "content": f"Classify this message: {message}",
        })

        client = instructor.from_litellm(litellm.acompletion)
        result = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=IntentClassification,
        )
        self._record_cost(messages)
        return result

    # ── Question Handler ────────────────────────────────────────────────

    async def _handle_question(
        self, query: str, message_id: str
    ) -> ChatResponsePayload:
        """Route to QueryService and format the response."""
        result = await answer_question(query, self.substrate, model=self.model)

        reply_parts = [result["answer"]]
        if result.get("reasoning"):
            reply_parts.append(f"\n\n**Reasoning:** {result['reasoning']}")

        suggestions: list[str] = []
        for claim in result.get("supporting_claims", [])[:2]:
            entity = claim.get("subject_entity_id")
            if entity:
                suggestions.append(f"Tell me more about {entity}")
        if not suggestions:
            suggestions.append("What else do we know?")
        suggestions.append("Submit a new observation")

        return ChatResponsePayload(
            message_id=message_id,
            reply_text="\n".join(reply_parts),
            intent=ChatIntent.QUESTION,
            claims=result.get("supporting_claims", []),
            confidence=result.get("confidence"),
            reasoning=result.get("reasoning"),
            suggestions=suggestions[:3],
        )

    # ── Observation Handler ─────────────────────────────────────────────

    async def _handle_observation(
        self, extracted: str, original: str, message_id: str
    ) -> ChatResponsePayload:
        """Ingest an observation via the bus and return tracking info."""
        text = extracted or original
        envelope = Envelope(
            topic="observations.structured",
            source_service_id="chat",
            payload={"text": text},
        )
        self.bus.publish(envelope)

        return ChatResponsePayload(
            message_id=message_id,
            reply_text=(
                f"I've submitted your observation for processing. "
                f"The research pipeline will extract claims from it.\n\n"
                f"**Tracking ID:** `{envelope.envelope_id}`\n"
                f"**Pipeline:** observation \u2192 researcher \u2192 validator \u2192 committed"
            ),
            intent=ChatIntent.OBSERVATION,
            pipeline_complete=False,
            tracking_envelope_id=envelope.envelope_id,
            suggestions=[
                "What do we know so far?",
                "Submit another observation",
                "Show all claims",
            ],
        )

    # ── Command Handler ─────────────────────────────────────────────────

    async def _handle_command(
        self, extracted: str, message_id: str
    ) -> ChatResponsePayload:
        """Handle direct commands against the system."""
        cmd = self._parse_command(extracted)

        if cmd.action == CommandAction.LIST_CLAIMS:
            claims = await self.substrate.get_claims()
            if cmd.target:
                claims = [
                    c
                    for c in claims
                    if cmd.target.lower() in c.subject_entity_id.lower()
                ]
            claim_dicts = [c.model_dump(mode="json") for c in claims[:20]]
            count = len(claims)
            about = f' about "{cmd.target}"' if cmd.target else ""
            return ChatResponsePayload(
                message_id=message_id,
                reply_text=f"Found {count} claim{'s' if count != 1 else ''}{about}.",
                intent=ChatIntent.COMMAND,
                claims=claim_dicts,
                suggestions=["Show entities", "Ask a question about these claims"],
            )

        if cmd.action == CommandAction.LIST_ENTITIES:
            entities = await self.substrate.entity_resolver.list_entities()
            first_entity = (
                entities[0].get("canonical_name", entities[0])
                if entities
                else None
            )
            entity_suggestions = ["Show claims"]
            if first_entity:
                entity_suggestions.append(f"Tell me about {first_entity}")
            return ChatResponsePayload(
                message_id=message_id,
                reply_text=f"Found {len(entities)} entities in the knowledge base.",
                intent=ChatIntent.COMMAND,
                entities=entities[:30],
                suggestions=entity_suggestions,
            )

        if cmd.action == CommandAction.SHOW_ENTITY:
            if not cmd.target:
                return ChatResponsePayload(
                    message_id=message_id,
                    reply_text=(
                        "Please specify which entity to show. "
                        "For example: 'show entity SpaceX'"
                    ),
                    intent=ChatIntent.COMMAND,
                )
            canonical = await self.substrate.entity_resolver.resolve(cmd.target)
            claims = await self.substrate.get_claims(subject_entity_id=canonical)
            return ChatResponsePayload(
                message_id=message_id,
                reply_text=f"Entity **{canonical}** has {len(claims)} claims.",
                intent=ChatIntent.COMMAND,
                claims=[c.model_dump(mode="json") for c in claims[:20]],
                suggestions=[
                    "Show all entities",
                    f"What else do we know about {canonical}?",
                ],
            )

        if cmd.action == CommandAction.RETRACT_CLAIM:
            if not cmd.target:
                return ChatResponsePayload(
                    message_id=message_id,
                    reply_text=(
                        "Please specify the claim ID to retract. "
                        "For example: 'retract claim clm_abc123'"
                    ),
                    intent=ChatIntent.COMMAND,
                )
            retracted = await self.substrate.retract_claim(cmd.target)
            if retracted:
                return ChatResponsePayload(
                    message_id=message_id,
                    reply_text=f"Claim `{cmd.target}` has been retracted.",
                    intent=ChatIntent.COMMAND,
                    suggestions=["Show all claims", "List entities"],
                )
            return ChatResponsePayload(
                message_id=message_id,
                reply_text=f"Claim `{cmd.target}` not found.",
                intent=ChatIntent.COMMAND,
                error="claim_not_found",
                suggestions=["Show all claims", "List entities"],
            )

        if cmd.action == CommandAction.SHOW_BUDGET:
            if self.budget_tracker:
                return ChatResponsePayload(
                    message_id=message_id,
                    reply_text=(
                        f"**Budget Status:**\n"
                        f"- Spent: ${self.budget_tracker.total_spend():.4f}\n"
                        f"- Limit: ${self.budget_tracker.monthly_limit_usd:.2f}\n"
                        f"- Remaining: "
                        f"{self.budget_tracker.remaining_pct() * 100:.1f}%"
                    ),
                    intent=ChatIntent.COMMAND,
                    suggestions=["Show all claims", "List entities"],
                )
            return ChatResponsePayload(
                message_id=message_id,
                reply_text="Budget tracking is not active.",
                intent=ChatIntent.COMMAND,
                suggestions=["Show all claims", "List entities"],
            )

        if cmd.action == CommandAction.HELP:
            return ChatResponsePayload(
                message_id=message_id,
                reply_text=(
                    "**What I can do:**\n\n"
                    "**Ask questions** \u2014 "
                    "'What do we know about SpaceX?'\n"
                    "**Submit observations** \u2014 "
                    "'NASA announced water on Mars'\n"
                    "**List claims** \u2014 "
                    "'Show all claims' or 'List claims about JWST'\n"
                    "**List entities** \u2014 'Show entities'\n"
                    "**Show entity** \u2014 'Show entity SpaceX'\n"
                    "**Retract claims** \u2014 "
                    "'Retract claim clm_abc123'\n"
                    "**Check budget** \u2014 'Show budget'\n"
                ),
                intent=ChatIntent.COMMAND,
                suggestions=[
                    "What do we know about exoplanets?",
                    "Show all claims",
                    "Show budget",
                ],
            )

        return ChatResponsePayload(
            message_id=message_id,
            reply_text="I didn't understand that command. Type 'help' to see what I can do.",
            intent=ChatIntent.COMMAND,
        )

    def _parse_command(self, extracted: str) -> CommandParse:
        """Rule-based command parsing (no LLM needed)."""
        lower = extracted.lower().strip()

        if "help" in lower:
            return CommandParse(action=CommandAction.HELP)
        if any(kw in lower for kw in ("budget", "spend", "cost")):
            return CommandParse(action=CommandAction.SHOW_BUDGET)
        if lower.startswith("retract") or lower.startswith("delete claim"):
            target = (
                lower.replace("retract claim", "")
                .replace("delete claim", "")
                .strip()
            )
            return CommandParse(
                action=CommandAction.RETRACT_CLAIM, target=target or None
            )
        if "list entities" in lower or "show entities" in lower:
            return CommandParse(action=CommandAction.LIST_ENTITIES)
        if "show entity" in lower or "entity " in lower:
            target = lower.replace("show entity", "").replace("entity", "").strip()
            return CommandParse(
                action=CommandAction.SHOW_ENTITY, target=target or None
            )
        if "list claims" in lower or "show claims" in lower or "show all claims" in lower:
            target = None
            for kw in ("about", "for", "on", "regarding"):
                if kw in lower:
                    target = lower.split(kw, 1)[1].strip()
                    break
            return CommandParse(action=CommandAction.LIST_CLAIMS, target=target)

        return CommandParse(action=CommandAction.UNKNOWN)

    # ── Conversation Handler ────────────────────────────────────────────

    async def _handle_conversation(
        self, message: str, message_id: str, session: ChatSession
    ) -> ChatResponsePayload:
        """Handle general conversation with system-aware context."""
        claims = await self.substrate.get_claims()
        entities = await self.substrate.entity_resolver.list_entities()

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are the Question Engine assistant. You help users interact "
                    "with a knowledge system that extracts claims from observations, "
                    "validates them, and builds a belief ledger.\n\n"
                    f"Current state: {len(claims)} claims, {len(entities)} entities.\n\n"
                    "You can:\n"
                    "- Answer questions about what the system knows\n"
                    "- Accept observations for claim extraction\n"
                    "- Execute commands (list claims, retract, show entities, etc.)\n"
                    "- Explain how the system works\n\n"
                    "Be concise and helpful. If the user seems to want to ask a "
                    "question or submit an observation, guide them to do so.\n\n"
                    "Also suggest 2-3 short follow-up prompts the user might want to try next."
                ),
            },
        ]
        for m in session.history[-8:]:
            messages.append(m)

        client = instructor.from_litellm(litellm.acompletion)
        result = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=ConversationalResponse,
        )
        self._record_cost(messages)

        return ChatResponsePayload(
            message_id=message_id,
            reply_text=result.reply,
            intent=ChatIntent.CONVERSATION,
            suggestions=result.suggestions[:3],
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _record_cost(self, messages: list[dict]) -> None:
        """Record LLM cost to budget tracker."""
        if self.budget_tracker is None:
            return
        try:
            cost = litellm.completion_cost(
                model=self.model, messages=messages, completion=""
            )
            self.budget_tracker.record_cost(self.model, cost)
        except Exception:
            pass

    def cleanup_stale_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours. Returns count removed."""
        now = datetime.now(UTC)
        stale = [
            sid
            for sid, s in self._sessions.items()
            if (now - s.last_active).total_seconds() > max_age_hours * 3600
        ]
        for sid in stale:
            del self._sessions[sid]
        return len(stale)

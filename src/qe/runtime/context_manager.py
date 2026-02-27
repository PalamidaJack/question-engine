import logging

from litellm import token_counter

from qe.models.envelope import Envelope
from qe.models.genome import Blueprint

log = logging.getLogger(__name__)


class ContextManager:
    def __init__(self, blueprint: Blueprint) -> None:
        self.blueprint = blueprint
        self.history: list[dict] = []
        self.token_limit = blueprint.max_context_tokens

    @property
    def _immutable_count(self) -> int:
        """Number of leading messages that must never be truncated."""
        return 2 if self.blueprint.constitution else 1

    def build_messages(
        self,
        envelope: Envelope,
        turn_count: int = 0,
    ) -> list[dict]:
        """
        Returns the full messages array for this call.
        Structure:
        [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": constitution},  # immutable safety zone
            ... compressed history ...,
            {"role": "user", "content": format_envelope_as_user_message(envelope)}
        ]
        The constitution (if present) is NEVER truncated — it survives all compression.
        """
        messages = [{"role": "system", "content": self.blueprint.system_prompt}]

        # Immutable safety zone: constitution is never compressed away
        if self.blueprint.constitution:
            messages.append({
                "role": "system",
                "content": (
                    "[CONSTITUTION — IMMUTABLE SAFETY CONSTRAINTS]\n"
                    + self.blueprint.constitution
                ),
            })

        # Reinforcement: re-inject system prompt every N turns
        if turn_count > 0 and turn_count % self.blueprint.reinforcement_interval_turns == 0:
            messages.append({
                "role": "user",
                "content": f"[SYSTEM REMINDER] {self.blueprint.system_prompt}",
            })

        messages.extend(self.history)

        # Add current envelope as user message
        messages.append({
            "role": "user",
            "content": self._format_envelope(envelope),
        })

        # Compression: truncate if over token limit
        # Never drop immutable messages (system prompt + constitution)
        threshold = int(self.token_limit * self.blueprint.context_compression_threshold)
        model = "gpt-4o-mini"  # Default model for counting

        token_count = token_counter(model=model, messages=messages)
        immutable = self._immutable_count
        while token_count > threshold and len(messages) > immutable:
            removed = messages.pop(immutable)
            log.warning(
                "Context truncated: removed message role=%s, tokens now=%d",
                removed["role"],
                token_counter(model=model, messages=messages),
            )
            token_count = token_counter(model=model, messages=messages)

        return messages

    def compress(self) -> None:
        """
        Compression strategy (two-stage):
        Stage 1 — Structural summarization: use LLM to summarize history
        Stage 2 — Drop oldest messages until under threshold

        Phase 0: not implemented. Uses truncation in build_messages instead.
        TODO Phase 1: implement LLM summarization.
        """
        # TODO Phase 1: replace truncation with LLM summarization
        pass

    def reinforce(self) -> None:
        """
        Re-inject the system prompt to prevent drift.
        Add a user message with reminder followed by assistant message with system prompt.
        """
        self.history.append({
            "role": "user",
            "content": "Reminder of your operating instructions:"
        })
        self.history.append({
            "role": "assistant",
            "content": self.blueprint.system_prompt
        })

    def _format_envelope(self, envelope: Envelope) -> str:
        """Format an envelope as a user message for the LLM."""
        return (
            f"[{envelope.topic}] from {envelope.source_service_id}:\n"
            f"{envelope.payload}"
        )

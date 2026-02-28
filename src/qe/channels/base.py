"""Base channel adapter for multi-platform communication."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ChannelMessage:
    """Normalized message from any channel."""

    text: str
    user_id: str
    channel_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw: Any = None
    attachments: list[Any] = field(default_factory=list)


class ChannelAdapter(ABC):
    """Abstract base for all channel integrations.

    Each adapter normalizes platform-specific messages into a common
    format, applies input sanitization, and provides a unified
    send/receive interface.
    """

    def __init__(
        self,
        channel_name: str,
        sanitizer: Any | None = None,
        message_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._channel_name = channel_name
        self._sanitizer = sanitizer
        self._message_callback = message_callback
        self._running = False

    @property
    def channel_name(self) -> str:
        """Return the canonical name of this channel."""
        return self._channel_name

    @property
    def is_running(self) -> bool:
        """Return whether the adapter is actively processing messages."""
        return self._running

    def _forward_message(self, result: dict[str, Any]) -> None:
        """Forward a received message through the message callback, if set."""
        if self._message_callback is not None:
            self._message_callback(result)

    @abstractmethod
    async def start(self) -> None:
        """Start listening for incoming messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully stop the adapter."""

    @abstractmethod
    async def send(
        self,
        user_id: str,
        message: str,
        attachments: list[Any] | None = None,
    ) -> None:
        """Send a message to a user on this channel."""

    async def receive(self, raw_message: Any) -> dict[str, Any] | None:
        """Extract, sanitize, and normalize an incoming message.

        Returns a dict with text, channel, user_id, sanitized_text, and
        risk_score.  Returns ``None`` when the risk score exceeds 0.8
        (message is rejected).
        """
        text = self._extract_text(raw_message)
        user_id = self._get_user_id(raw_message)

        sanitized_text = text
        risk_score = 0.0

        if self._sanitizer is not None:
            result = self._sanitizer.sanitize(text)
            sanitized_text = result.text
            risk_score = result.risk_score

            if risk_score > 0.8:
                log.warning(
                    "channel.message_rejected channel=%s user=%s risk=%.2f",
                    self._channel_name,
                    user_id,
                    risk_score,
                )
                return None

        log.debug(
            "channel.message_received channel=%s user=%s risk=%.2f",
            self._channel_name,
            user_id,
            risk_score,
        )

        return {
            "text": text,
            "channel": self._channel_name,
            "user_id": user_id,
            "sanitized_text": sanitized_text,
            "risk_score": risk_score,
        }

    @abstractmethod
    def _extract_text(self, raw_message: Any) -> str:
        """Extract the plain-text body from a platform-specific message."""

    @abstractmethod
    def _get_user_id(self, raw_message: Any) -> str:
        """Extract the user identifier from a platform-specific message."""

    def _is_goal(self, text: str) -> bool:
        """Detect whether the message represents a goal submission.

        A message is considered a goal if it starts with one of the
        recognised prefixes (case-insensitive).
        """
        lower = text.strip().lower()
        prefixes = ("goal:", "research", "find", "analyze")
        return any(lower.startswith(p) for p in prefixes)

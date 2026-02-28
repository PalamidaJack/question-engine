"""Slack channel adapter using slack_bolt."""

from __future__ import annotations

import logging
import os
from typing import Any

from qe.channels.base import ChannelAdapter

log = logging.getLogger(__name__)


class SlackAdapter(ChannelAdapter):
    """Adapter for Slack workspaces via the Bolt SDK.

    Requires ``slack_bolt`` to be installed.  Import is deferred so the
    rest of the system works without it.
    """

    def __init__(
        self,
        bot_token: str = "",
        app_token: str = "",
        default_channel: str = "",
        sanitizer: Any | None = None,
    ) -> None:
        super().__init__(channel_name="slack", sanitizer=sanitizer)
        self._bot_token = bot_token or os.environ.get("SLACK_BOT_TOKEN", "")
        self._app_token = app_token or os.environ.get("SLACK_APP_TOKEN", "")
        self._default_channel = default_channel or os.environ.get(
            "SLACK_DEFAULT_CHANNEL", ""
        )
        self._app: Any = None
        self._handler: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the Slack Bolt app and register event handlers."""
        try:
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
            from slack_bolt.async_app import AsyncApp
        except ImportError:
            log.error(
                "slack_bolt is not installed. "
                "Install with: pip install slack-bolt"
            )
            return

        self._app = AsyncApp(token=self._bot_token)

        # Register event listeners
        self._app.event("app_mention")(self.handle_mention)
        self._app.event("message")(self.handle_dm)

        self._handler = AsyncSocketModeHandler(self._app, self._app_token)
        await self._handler.start_async()
        self._running = True
        log.info("slack.adapter_started")

    async def stop(self) -> None:
        """Shut down the Slack socket-mode handler."""
        if self._handler is not None:
            await self._handler.close_async()
        self._running = False
        log.info("slack.adapter_stopped")

    # ------------------------------------------------------------------
    # Send / Receive helpers
    # ------------------------------------------------------------------

    async def send(
        self,
        user_id: str,
        message: str,
        attachments: list[Any] | None = None,
    ) -> None:
        """Post a message to a Slack channel or DM."""
        if self._app is None:
            log.warning("slack.send_skipped reason=app_not_initialised")
            return

        channel = user_id or self._default_channel
        kwargs: dict[str, Any] = {
            "channel": channel,
            "text": message,
        }
        if attachments:
            kwargs["attachments"] = attachments

        try:
            await self._app.client.chat_postMessage(**kwargs)
            log.debug("slack.message_sent channel=%s", channel)
        except Exception:
            log.exception("slack.send_failed channel=%s", channel)

    def _extract_text(self, raw_message: Any) -> str:
        """Extract text from a Slack event payload."""
        if isinstance(raw_message, dict):
            return raw_message.get("text", "")
        return str(raw_message)

    def _get_user_id(self, raw_message: Any) -> str:
        """Extract the Slack user ID from an event payload."""
        if isinstance(raw_message, dict):
            return raw_message.get("user", "")
        return ""

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def handle_mention(self, event: dict[str, Any], say: Any = None) -> None:
        """Handle @-mention events in channels."""
        result = await self.receive(event)
        if result is None:
            return

        text = result["sanitized_text"]
        if self._is_goal(text):
            log.info(
                "slack.goal_detected user=%s text=%s",
                result["user_id"],
                text[:80],
            )

        if say is not None:
            await say(f"Received: {text[:200]}")

    async def handle_dm(self, event: dict[str, Any], say: Any = None) -> None:
        """Handle direct messages sent to the bot."""
        # Ignore bot messages to avoid loops
        if event.get("subtype") == "bot_message":
            return
        if event.get("bot_id"):
            return

        result = await self.receive(event)
        if result is None:
            return

        text = result["sanitized_text"]
        if self._is_goal(text):
            log.info(
                "slack.dm_goal user=%s text=%s",
                result["user_id"],
                text[:80],
            )

        if say is not None:
            await say(f"Got it: {text[:200]}")

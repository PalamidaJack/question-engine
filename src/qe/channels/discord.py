"""Discord channel adapter using discord.py.

Provides bidirectional chat and /ask slash command.
Gated behind ``discord_integration`` feature flag.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from qe.channels.base import ChannelAdapter

log = logging.getLogger(__name__)


class DiscordAdapter(ChannelAdapter):
    """Adapter for Discord servers via the discord.py library.

    Requires ``discord.py`` to be installed.  Import is deferred so the
    rest of the system works without it.
    """

    def __init__(
        self,
        bot_token: str = "",
        default_channel_id: int = 0,
        sanitizer: Any | None = None,
        message_callback: Any | None = None,
    ) -> None:
        super().__init__(
            channel_name="discord",
            sanitizer=sanitizer,
            message_callback=message_callback,
        )
        self._bot_token = bot_token or os.environ.get(
            "DISCORD_BOT_TOKEN", "",
        )
        self._default_channel_id = default_channel_id or int(
            os.environ.get("DISCORD_DEFAULT_CHANNEL", "0"),
        )
        self._client: Any = None
        self._tree: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the Discord bot and register event handlers."""
        try:
            import discord
            from discord import app_commands
        except ImportError:
            log.error(
                "discord.py is not installed. "
                "Install with: pip install discord.py"
            )
            return

        intents = discord.Intents.default()
        intents.message_content = True

        self._client = discord.Client(intents=intents)
        self._tree = app_commands.CommandTree(self._client)

        adapter = self

        @self._client.event
        async def on_ready() -> None:
            log.info(
                "discord.adapter_started user=%s",
                self._client.user,
            )
            await self._tree.sync()
            adapter._running = True

        @self._client.event
        async def on_message(message: Any) -> None:
            if message.author == self._client.user:
                return
            await adapter._handle_message(message)

        @self._tree.command(
            name="ask",
            description="Ask the question engine",
        )
        async def ask_command(
            interaction: Any, question: str,
        ) -> None:
            await adapter._handle_ask(interaction, question)

        # Start in background (non-blocking)
        log.info("discord.starting")
        await self._client.start(self._bot_token)

    async def stop(self) -> None:
        """Shut down the Discord client."""
        if self._client is not None:
            await self._client.close()
        self._running = False
        log.info("discord.adapter_stopped")

    # ------------------------------------------------------------------
    # Send / Receive helpers
    # ------------------------------------------------------------------

    async def send(
        self,
        user_id: str,
        message: str,
        attachments: list[Any] | None = None,
    ) -> None:
        """Send a message to a Discord channel or user."""
        if self._client is None:
            log.warning(
                "discord.send_skipped reason=client_not_initialised",
            )
            return

        channel = self._client.get_channel(
            int(user_id) if user_id else self._default_channel_id,
        )
        if channel is None:
            log.warning(
                "discord.send_skipped reason=channel_not_found id=%s",
                user_id,
            )
            return

        try:
            # Discord has a 2000 char limit per message
            for i in range(0, len(message), 2000):
                await channel.send(message[i:i + 2000])
            log.debug(
                "discord.message_sent channel=%s", user_id,
            )
        except Exception:
            log.exception(
                "discord.send_failed channel=%s", user_id,
            )

    def _extract_text(self, raw_message: Any) -> str:
        """Extract text from a Discord message object."""
        if isinstance(raw_message, dict):
            return raw_message.get("content", "")
        if hasattr(raw_message, "content"):
            return raw_message.content
        return str(raw_message)

    def _get_user_id(self, raw_message: Any) -> str:
        """Extract the Discord user ID from a message."""
        if isinstance(raw_message, dict):
            return str(raw_message.get("author", {}).get("id", ""))
        if hasattr(raw_message, "author"):
            return str(raw_message.author.id)
        return ""

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_message(self, message: Any) -> None:
        """Handle incoming Discord messages."""
        raw = {
            "content": message.content,
            "author": {"id": str(message.author.id)},
            "channel_id": str(message.channel.id),
        }
        result = await self.receive(raw)
        if result is None:
            return

        text = result["sanitized_text"]
        if self._is_goal(text):
            log.info(
                "discord.goal_detected user=%s text=%s",
                result["user_id"],
                text[:80],
            )
            result["command"] = "goal"

        self._forward_message(result)

    async def _handle_ask(
        self, interaction: Any, question: str,
    ) -> None:
        """Handle the /ask slash command."""
        raw = {
            "content": question,
            "author": {"id": str(interaction.user.id)},
            "channel_id": str(interaction.channel_id),
        }
        result = await self.receive(raw)
        if result is None:
            await interaction.response.send_message(
                "Message rejected by safety filter.",
                ephemeral=True,
            )
            return

        result["command"] = "ask"
        self._forward_message(result)

        await interaction.response.send_message(
            f"Processing: {question[:200]}",
        )

    def server_info(self) -> dict[str, Any]:
        """Return adapter status information."""
        return {
            "channel": "discord",
            "running": self._running,
            "bot_user": (
                str(self._client.user) if self._client else None
            ),
            "default_channel": self._default_channel_id,
        }

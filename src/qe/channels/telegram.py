"""Telegram channel adapter using python-telegram-bot."""

from __future__ import annotations

import logging
import os
from typing import Any

from qe.channels.base import ChannelAdapter

log = logging.getLogger(__name__)


class TelegramAdapter(ChannelAdapter):
    """Adapter for Telegram bots via python-telegram-bot.

    Requires ``python-telegram-bot`` to be installed.  Import is deferred
    so the rest of the system can operate without it.
    """

    def __init__(
        self,
        bot_token: str = "",
        sanitizer: Any | None = None,
    ) -> None:
        super().__init__(channel_name="telegram", sanitizer=sanitizer)
        self._bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._application: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Build and start the Telegram bot application."""
        try:
            from telegram.ext import (
                ApplicationBuilder,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            log.error(
                "python-telegram-bot is not installed. "
                "Install with: pip install python-telegram-bot"
            )
            return

        self._application = (
            ApplicationBuilder().token(self._bot_token).build()
        )

        # Register handlers
        self._application.add_handler(
            CommandHandler("goal", self.handle_command)
        )
        self._application.add_handler(
            CommandHandler("ask", self.handle_command)
        )
        self._application.add_handler(
            CommandHandler("status", self.handle_command)
        )
        self._application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handle_message,
            )
        )

        await self._application.initialize()
        await self._application.start()
        await self._application.updater.start_polling()
        self._running = True
        log.info("telegram.adapter_started")

    async def stop(self) -> None:
        """Gracefully stop the Telegram polling loop."""
        if self._application is not None:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()
        self._running = False
        log.info("telegram.adapter_stopped")

    # ------------------------------------------------------------------
    # Send / Receive helpers
    # ------------------------------------------------------------------

    async def send(
        self,
        user_id: str,
        message: str,
        attachments: list[Any] | None = None,
    ) -> None:
        """Send a text message (and optional documents) to a Telegram chat."""
        if self._application is None:
            log.warning("telegram.send_skipped reason=app_not_initialised")
            return

        bot = self._application.bot
        try:
            await bot.send_message(chat_id=user_id, text=message)

            if attachments:
                for attachment in attachments:
                    if isinstance(attachment, (str, bytes)):
                        await bot.send_document(
                            chat_id=user_id, document=attachment
                        )
            log.debug("telegram.message_sent chat_id=%s", user_id)
        except Exception:
            log.exception("telegram.send_failed chat_id=%s", user_id)

    def _extract_text(self, raw_message: Any) -> str:
        """Extract text from a Telegram update or message object."""
        if isinstance(raw_message, dict):
            # Raw dict representation
            msg = raw_message.get("message", raw_message)
            return msg.get("text", "")

        # python-telegram-bot Update / Message objects
        if hasattr(raw_message, "message") and raw_message.message:
            return raw_message.message.text or ""
        if hasattr(raw_message, "text"):
            return raw_message.text or ""
        return str(raw_message)

    def _get_user_id(self, raw_message: Any) -> str:
        """Extract the chat ID from a Telegram update or message."""
        if isinstance(raw_message, dict):
            msg = raw_message.get("message", raw_message)
            chat = msg.get("chat", {})
            return str(chat.get("id", ""))

        # python-telegram-bot objects
        if hasattr(raw_message, "effective_chat") and raw_message.effective_chat:
            return str(raw_message.effective_chat.id)
        if hasattr(raw_message, "chat") and raw_message.chat:
            return str(raw_message.chat.id)
        return ""

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def handle_message(self, update: Any, context: Any) -> None:
        """Handle a plain-text message (not a command)."""
        result = await self.receive(update)
        if result is None:
            return

        text = result["sanitized_text"]
        if self._is_goal(text):
            log.info(
                "telegram.goal_detected user=%s text=%s",
                result["user_id"],
                text[:80],
            )

        if hasattr(update, "message") and update.message:
            await update.message.reply_text(f"Received: {text[:200]}")

    async def handle_command(self, update: Any, context: Any) -> None:
        """Handle /goal, /ask, and /status commands."""
        if not hasattr(update, "message") or not update.message:
            return

        message_text = update.message.text or ""
        parts = message_text.split(maxsplit=1)
        command = parts[0].lstrip("/").lower() if parts else ""
        args_text = parts[1] if len(parts) > 1 else ""

        user_id = self._get_user_id(update)

        if command == "goal":
            log.info(
                "telegram.goal_command user=%s text=%s",
                user_id,
                args_text[:80],
            )
            await update.message.reply_text(
                f"Goal registered: {args_text[:200]}"
            )

        elif command == "ask":
            log.info(
                "telegram.ask_command user=%s text=%s",
                user_id,
                args_text[:80],
            )
            await update.message.reply_text(
                f"Question received: {args_text[:200]}"
            )

        elif command == "status":
            log.info("telegram.status_command user=%s", user_id)
            await update.message.reply_text("System is running.")

        else:
            await update.message.reply_text(
                f"Unknown command: /{command}"
            )

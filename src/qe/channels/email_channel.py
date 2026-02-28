"""Email channel adapter using IMAP/SMTP."""

from __future__ import annotations

import asyncio
import email
import email.mime.multipart
import email.mime.text
import imaplib
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from qe.channels.base import ChannelAdapter

log = logging.getLogger(__name__)


class EmailAdapter(ChannelAdapter):
    """Adapter for email-based communication via IMAP + SMTP.

    Monitors an IMAP inbox for new messages and sends replies via SMTP.
    All network I/O is offloaded to a thread executor so the async
    event loop is never blocked.
    """

    def __init__(
        self,
        imap_host: str = "",
        smtp_host: str = "",
        username: str = "",
        password: str = "",
        inbox_folder: str = "INBOX",
        sanitizer: Any | None = None,
        message_callback: Any | None = None,
    ) -> None:
        super().__init__(
            channel_name="email",
            sanitizer=sanitizer,
            message_callback=message_callback,
        )
        self._imap_host = imap_host or os.environ.get("EMAIL_IMAP_HOST", "")
        self._smtp_host = smtp_host or os.environ.get("EMAIL_SMTP_HOST", "")
        self._username = username or os.environ.get("EMAIL_USERNAME", "")
        self._password = password or os.environ.get("EMAIL_PASSWORD", "")
        self._inbox_folder = inbox_folder
        self._poll_task: asyncio.Task[None] | None = None
        self._poll_interval: float = 30.0  # seconds

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin monitoring the IMAP inbox for new messages."""
        if not self._imap_host or not self._username:
            log.error(
                "email.adapter_start_failed reason=missing_credentials"
            )
            return

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        log.info("email.adapter_started inbox=%s", self._inbox_folder)

    async def stop(self) -> None:
        """Stop the inbox polling loop."""
        self._running = False
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        log.info("email.adapter_stopped")

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------

    async def send(
        self,
        user_id: str,
        message: str,
        attachments: list[Any] | None = None,
    ) -> None:
        """Send an email to *user_id* (an email address) via SMTP."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, self._send_sync, user_id, message, attachments
            )
            log.debug("email.message_sent to=%s", user_id)
        except Exception:
            log.exception("email.send_failed to=%s", user_id)

    def _send_sync(
        self,
        to_addr: str,
        body: str,
        attachments: list[Any] | None,
    ) -> None:
        """Blocking SMTP send (run inside an executor)."""
        msg = MIMEMultipart()
        msg["From"] = self._username
        msg["To"] = to_addr
        msg["Subject"] = "Question Engine Notification"
        msg.attach(MIMEText(body, "plain"))

        if attachments:
            for att in attachments:
                if isinstance(att, str):
                    part = MIMEText(att, "plain")
                    part.add_header(
                        "Content-Disposition", "attachment", filename="attachment.txt"
                    )
                    msg.attach(part)

        with smtplib.SMTP_SSL(self._smtp_host, 465) as server:
            server.login(self._username, self._password)
            server.send_message(msg)

    # ------------------------------------------------------------------
    # Extract helpers
    # ------------------------------------------------------------------

    def _extract_text(self, raw_message: Any) -> str:
        """Extract the plain-text body from an email message object."""
        if isinstance(raw_message, email.message.Message):
            if raw_message.is_multipart():
                for part in raw_message.walk():
                    ctype = part.get_content_type()
                    if ctype == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            return payload.decode("utf-8", errors="replace")
            else:
                payload = raw_message.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="replace")

        if isinstance(raw_message, dict):
            return raw_message.get("body", raw_message.get("text", ""))

        return str(raw_message)

    def _get_user_id(self, raw_message: Any) -> str:
        """Extract the sender email address."""
        if isinstance(raw_message, email.message.Message):
            sender = raw_message.get("From", "")
            # Extract bare address from "Name <addr>" format
            if "<" in sender and ">" in sender:
                return sender[sender.index("<") + 1 : sender.index(">")]
            return sender

        if isinstance(raw_message, dict):
            return raw_message.get("from", raw_message.get("sender", ""))

        return ""

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _classify_command(self, text: str) -> str:
        """Classify an email message as ask, status, or goal."""
        lower = text.strip().lower()
        if lower.startswith("ask:") or lower.startswith("/ask"):
            return "ask"
        if lower.startswith("status:") or lower.startswith("/status"):
            return "status"
        return "goal"

    async def _poll_loop(self) -> None:
        """Periodically check for new IMAP messages."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                messages = await loop.run_in_executor(None, self._fetch_new_messages)
                for msg in messages:
                    result = await self.receive(msg)
                    if result is not None:
                        result["command"] = self._classify_command(
                            result.get("text", "")
                        )
                        self._forward_message(result)
            except Exception:
                log.exception("email.poll_error")
            await asyncio.sleep(self._poll_interval)

    def _fetch_new_messages(self) -> list[email.message.Message]:
        """Blocking IMAP fetch of unseen messages."""
        results: list[email.message.Message] = []
        try:
            conn = imaplib.IMAP4_SSL(self._imap_host)
            conn.login(self._username, self._password)
            conn.select(self._inbox_folder)

            _status, data = conn.search(None, "UNSEEN")
            if data and data[0]:
                for num in data[0].split():
                    _status, msg_data = conn.fetch(num, "(RFC822)")
                    if msg_data and msg_data[0] and isinstance(msg_data[0], tuple):
                        raw = msg_data[0][1]
                        msg = email.message_from_bytes(raw)
                        results.append(msg)

            conn.logout()
        except Exception:
            log.exception("email.imap_fetch_error")
        return results

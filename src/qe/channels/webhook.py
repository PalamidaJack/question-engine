"""Webhook channel adapter for generic inbound HTTP payloads."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Any

from qe.channels.base import ChannelAdapter

log = logging.getLogger(__name__)


class WebhookAdapter(ChannelAdapter):
    """Adapter for inbound webhook payloads.

    Webhooks are push-based: the external system sends an HTTP request to
    our endpoint.  This adapter validates the request signature, extracts
    the message text, and sanitizes it.

    ``start`` / ``stop`` / ``send`` are no-ops because the adapter does
    not manage its own transport -- it is driven by an HTTP server
    (e.g. FastAPI route) that calls :meth:`process_webhook`.
    """

    def __init__(
        self,
        secret: str = "",
        payload_extractors: dict[str, str] | None = None,
        sanitizer: Any | None = None,
    ) -> None:
        super().__init__(channel_name="webhook", sanitizer=sanitizer)
        self._secret = secret or os.environ.get("WEBHOOK_SECRET", "")
        # Optional mapping of custom payload field names
        self._payload_extractors = payload_extractors or {}

    # ------------------------------------------------------------------
    # Lifecycle (no-ops for webhooks)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """No-op -- webhooks are driven by incoming HTTP requests."""
        self._running = True
        log.info("webhook.adapter_ready")

    async def stop(self) -> None:
        """No-op -- nothing to tear down."""
        self._running = False
        log.info("webhook.adapter_stopped")

    async def send(
        self,
        user_id: str,
        message: str,
        attachments: list[Any] | None = None,
    ) -> None:
        """No-op -- webhooks are inbound only."""
        log.debug("webhook.send_noop (webhooks are inbound-only)")

    # ------------------------------------------------------------------
    # Signature verification
    # ------------------------------------------------------------------

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify an HMAC-SHA256 signature against the shared secret.

        The expected signature format is the hex digest, optionally
        prefixed with ``sha256=``.
        """
        if not self._secret:
            log.warning("webhook.verify_skipped reason=no_secret_configured")
            return True

        expected = hmac.new(
            self._secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        # Strip common prefix
        bare_sig = signature.removeprefix("sha256=")

        return hmac.compare_digest(expected, bare_sig)

    # ------------------------------------------------------------------
    # Process incoming webhook
    # ------------------------------------------------------------------

    async def process_webhook(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any] | None:
        """Validate the signature, extract text, and sanitize.

        Returns the normalised message dict (see
        :meth:`ChannelAdapter.receive`) or ``None`` if the signature is
        invalid or the message is rejected.
        """
        # Signature check (if a signature header is present)
        import json

        sig = headers.get("X-Signature-256") or headers.get(
            "x-hub-signature-256", ""
        )
        if sig:
            raw_bytes = json.dumps(payload, separators=(",", ":")).encode()
            if not self.verify_signature(raw_bytes, sig):
                log.warning("webhook.invalid_signature")
                return None

        return await self.receive(payload)

    # ------------------------------------------------------------------
    # Extract helpers
    # ------------------------------------------------------------------

    def _extract_text(self, raw_message: Any) -> str:
        """Extract text from a webhook payload dict.

        Checks custom extractors first, then falls back to common field
        names: text, message, body, content, data.
        """
        if not isinstance(raw_message, dict):
            return str(raw_message)

        # Custom extractor overrides
        for field_name in self._payload_extractors.values():
            if field_name in raw_message:
                value = raw_message[field_name]
                return str(value) if value is not None else ""

        # Common field names
        for key in ("text", "message", "body", "content", "data"):
            if key in raw_message:
                value = raw_message[key]
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    # Nested object -- try to find text inside
                    for sub in ("text", "body", "content"):
                        if sub in value:
                            return str(value[sub])
                return str(value)

        return ""

    def _get_user_id(self, raw_message: Any) -> str:
        """Extract the user identifier from a webhook payload."""
        if not isinstance(raw_message, dict):
            return ""

        for key in ("user_id", "user", "sender", "from", "author"):
            if key in raw_message:
                value = raw_message[key]
                if isinstance(value, dict):
                    return str(value.get("id", value.get("name", "")))
                return str(value)

        return ""

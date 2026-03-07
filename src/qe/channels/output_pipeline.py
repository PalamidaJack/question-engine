"""Multi-Channel Output Pipeline — format results per channel.

Formats results for web (markdown), chat platforms (plain text),
API (JSON), and email (HTML).
Gated behind ``output_pipeline`` feature flag.
"""

from __future__ import annotations

import html
import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)


class OutputPipeline:
    """Format output for different channels."""

    def format(
        self,
        content: str,
        channel: str = "web",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Format content for the specified channel.

        Channels: web (markdown), chat (plain text), api (JSON), email (HTML).
        """
        formatters = {
            "web": self._format_web,
            "chat": self._format_chat,
            "api": self._format_api,
            "email": self._format_email,
        }
        formatter = formatters.get(channel, self._format_web)
        return formatter(content, metadata or {})

    def _format_web(
        self, content: str, metadata: dict[str, Any],
    ) -> str:
        """Web format: pass through markdown as-is."""
        return content

    def _format_chat(
        self, content: str, metadata: dict[str, Any],
    ) -> str:
        """Chat format: strip markdown to plain text."""
        text = content
        # Remove markdown headers
        text = re.sub(r"#{1,6}\s+", "", text)
        # Remove bold/italic markers
        text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
        # Remove links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove code blocks markers
        text = re.sub(r"```\w*\n?", "", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        return text.strip()

    def _format_api(
        self, content: str, metadata: dict[str, Any],
    ) -> str:
        """API format: structured JSON."""
        return json.dumps({
            "content": content,
            "format": "markdown",
            "metadata": metadata,
        })

    def _format_email(
        self, content: str, metadata: dict[str, Any],
    ) -> str:
        """Email format: convert markdown to basic HTML."""
        text = html.escape(content)
        # Headers
        text = re.sub(
            r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.M,
        )
        text = re.sub(
            r"^## (.+)$", r"<h2>\1</h2>", text, flags=re.M,
        )
        text = re.sub(
            r"^# (.+)$", r"<h1>\1</h1>", text, flags=re.M,
        )
        # Bold
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        # Italic
        text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
        # Line breaks
        text = text.replace("\n\n", "</p><p>")
        text = text.replace("\n", "<br>")
        return f"<div>{text}</div>"

    def supported_channels(self) -> list[str]:
        return ["web", "chat", "api", "email"]

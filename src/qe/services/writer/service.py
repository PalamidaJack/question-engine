"""Writer service for document generation from claims."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_VALID_FORMATS = {"summary", "report", "brief", "memo"}
_VALID_STYLES = {"neutral", "formal", "conversational"}


class WriterService:
    """Generates documents from collections of claims in
    various formats and styles."""

    def __init__(
        self,
        substrate: Any = None,
        bus: Any = None,
    ) -> None:
        self.substrate = substrate
        self.bus = bus

    async def generate_document(
        self,
        claims: list[dict],
        format: str = "summary",
        style: str = "neutral",
    ) -> dict:
        """Generate a document from the given claims.

        Args:
            claims: List of claim dicts to synthesize.
            format: Output format (summary, report, brief,
                memo).
            style: Writing style (neutral, formal,
                conversational).

        Returns a dict with content, format, word count,
        and sources cited.
        """
        if format not in _VALID_FORMATS:
            log.warning(
                "Unknown format: %s, defaulting to summary",
                format,
            )
            format = "summary"

        if style not in _VALID_STYLES:
            log.warning(
                "Unknown style: %s, defaulting to neutral",
                style,
            )
            style = "neutral"

        content = self._build_content(claims, format, style)
        word_count = len(content.split())
        sources_cited = self._count_sources(claims)

        log.info(
            "Document generated: format=%s style=%s "
            "words=%d sources=%d",
            format,
            style,
            word_count,
            sources_cited,
        )

        return {
            "content": content,
            "format": format,
            "word_count": word_count,
            "sources_cited": sources_cited,
        }

    def _build_content(
        self,
        claims: list[dict],
        format: str,
        style: str,
    ) -> str:
        """Build document content from claims."""
        if not claims:
            return "No claims available for document."

        header = self._format_header(format, len(claims))
        body_parts = []
        for claim in claims:
            text = claim.get(
                "text",
                claim.get("object_value", "No text"),
            )
            body_parts.append(f"- {text}")

        body = "\n".join(body_parts)
        return f"{header}\n\n{body}"

    def _format_header(
        self, format: str, claim_count: int
    ) -> str:
        """Generate a header line for the document."""
        titles = {
            "summary": "Summary",
            "report": "Report",
            "brief": "Brief",
            "memo": "Memo",
        }
        title = titles.get(format, "Document")
        return (
            f"{title}: Synthesized from "
            f"{claim_count} claim(s)"
        )

    def _count_sources(self, claims: list[dict]) -> int:
        """Count unique sources referenced in claims."""
        sources: set[str] = set()
        for claim in claims:
            source = claim.get("source", claim.get("source_id"))
            if source:
                sources.add(str(source))
        return len(sources)

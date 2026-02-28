"""Web fetch tool — HTTP GET with content extraction."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from qe.runtime.tools import ToolSpec

log = logging.getLogger(__name__)

web_fetch_spec = ToolSpec(
    name="web_fetch",
    description=(
        "Fetch a web page and extract its main text content. "
        "Returns title, text, and metadata."
    ),
    requires_capability="web_search",
    input_schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "max_length": {
                "type": "integer",
                "description": "Maximum text length to return",
                "default": 5000,
            },
        },
        "required": ["url"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "text": {"type": "string"},
            "url": {"type": "string"},
            "content_length": {"type": "integer"},
        },
    },
    timeout_seconds=30,
    category="web",
)


async def web_fetch(
    url: str, max_length: int = 5000
) -> dict[str, Any]:
    """Fetch a URL and extract main content."""
    async with httpx.AsyncClient(
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0 (compatible; QE/1.0)"},
        follow_redirects=True,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text

    # Try trafilatura first (best quality extraction)
    text = _extract_with_trafilatura(html)
    if not text:
        text = _extract_simple(html)

    title = _extract_title(html)

    # Truncate to max_length
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return {
        "title": title,
        "text": text,
        "url": str(resp.url),
        "content_length": len(text),
    }


def _extract_with_trafilatura(html: str) -> str:
    """Extract main content using trafilatura if available."""
    try:
        import trafilatura

        result = trafilatura.extract(html, include_comments=False, include_tables=True)
        return result or ""
    except ImportError:
        return ""


def _extract_simple(html: str) -> str:
    """Simple HTML-to-text extraction without external dependencies."""
    # Remove script and style elements
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Decode HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return text


def _extract_title(html: str) -> str:
    """Extract the page title."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.DOTALL | re.IGNORECASE)
    if match:
        title = re.sub(r"<[^>]+>", "", match.group(1)).strip()
        return title
    return ""

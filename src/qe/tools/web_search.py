"""Web search tool — DuckDuckGo HTML fallback, optional SearXNG/Brave."""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from qe.runtime.tools import ToolSpec

log = logging.getLogger(__name__)

web_search_spec = ToolSpec(
    name="web_search",
    description=(
        "Search the web for information. "
        "Returns a list of results with title, URL, and snippet."
    ),
    requires_capability="web_search",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    output_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "url": {"type": "string"},
                "snippet": {"type": "string"},
            },
        },
    },
    timeout_seconds=30,
    category="web",
)


async def web_search(
    query: str,
    max_results: int = 5,
    *,
    searxng_url: str | None = None,
    brave_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Search the web using available backends.

    Priority: SearXNG (self-hosted) → Brave Search API → DuckDuckGo HTML.
    """
    if searxng_url:
        try:
            return await _searxng_search(searxng_url, query, max_results)
        except Exception:
            log.warning("searxng_search failed, falling back", exc_info=True)

    if brave_api_key:
        try:
            return await _brave_search(brave_api_key, query, max_results)
        except Exception:
            log.warning("brave_search failed, falling back", exc_info=True)

    return await _ddg_search(query, max_results)


async def _searxng_search(
    base_url: str, query: str, max_results: int
) -> list[dict[str, Any]]:
    """Search via a SearXNG instance."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{base_url}/search",
            params={"q": query, "format": "json", "categories": "general"},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
        })
    return results


async def _brave_search(
    api_key: str, query: str, max_results: int
) -> list[dict[str, Any]]:
    """Search via Brave Search API."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
        })
    return results


async def _ddg_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Search via DuckDuckGo HTML scraping (no API key needed)."""
    async with httpx.AsyncClient(
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (compatible; QE/1.0)"},
        follow_redirects=True,
    ) as client:
        resp = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
        )
        resp.raise_for_status()
        html = resp.text

    results = []
    # Extract result blocks from DDG HTML
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )
    for url, title, snippet in blocks[:max_results]:
        # Clean HTML tags from title and snippet
        clean_title = re.sub(r"<[^>]+>", "", title).strip()
        clean_snippet = re.sub(r"<[^>]+>", "", snippet).strip()
        # DDG wraps URLs in a redirect; extract the actual URL
        actual_url = url
        uddg_match = re.search(r"uddg=([^&]+)", url)
        if uddg_match:
            from urllib.parse import unquote

            actual_url = unquote(uddg_match.group(1))
        results.append({
            "title": clean_title,
            "url": actual_url,
            "snippet": clean_snippet,
        })

    return results

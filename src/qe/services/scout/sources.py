"""Source discovery: search query generation and web search for scout findings."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

import instructor
import litellm
from pydantic import BaseModel, Field

from qe.models.scout import ScoutFinding
from qe.runtime.feature_flags import get_flag_store
from qe.tools.web_search import web_search

log = logging.getLogger(__name__)


class _SearchQueries(BaseModel):
    """LLM-generated search queries for scouting."""

    queries: list[str] = Field(
        default_factory=list,
        description="Targeted web search queries to find improvements for the QE codebase",
    )


class SourceManager:
    """Generate search queries and discover web resources."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        search_topics: list[str] | None = None,
    ) -> None:
        self._model = model
        self._search_topics = search_topics or [
            "Python asyncio patterns",
            "FastAPI best practices",
            "Pydantic v2 patterns",
            "LLM agent architecture",
            "cognitive architecture AI",
        ]
        self._rejected_categories: list[str] = []
        self._rejected_sources: list[str] = []

    def set_rejected_patterns(
        self,
        categories: list[str],
        sources: list[str],
    ) -> None:
        """Update patterns to avoid based on human feedback."""
        self._rejected_categories = categories
        self._rejected_sources = sources

    async def generate_queries(
        self,
        codebase_summary: str = "",
        max_queries: int = 10,
    ) -> list[str]:
        """Use LLM to generate targeted search queries."""
        if not get_flag_store().is_enabled("innovation_scout"):
            return []

        avoid_text = ""
        if self._rejected_categories:
            avoid_text += f"\nAvoid topics related to: {', '.join(self._rejected_categories)}"
        if self._rejected_sources:
            avoid_text += f"\nAvoid sources like: {', '.join(self._rejected_sources)}"

        try:
            client = instructor.from_litellm(litellm.acompletion)
            result = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a technical scout for a Python multi-agent AI system "
                            "(Question Engine). Generate specific, targeted web search queries "
                            "to find improvements, new patterns, libraries, or techniques. "
                            "Focus on practical, actionable results."
                            f"{avoid_text}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Generate {max_queries} search queries based on "
                            f"these topics: {', '.join(self._search_topics)}. "
                            "Context about the codebase: "
                            f"{codebase_summary or 'Python asyncio + FastAPI + Pydantic + litellm'}"
                        ),
                    },
                ],
                response_model=_SearchQueries,
            )
            return result.queries[:max_queries]
        except Exception:
            log.exception("scout.query_generation_failed")
            return [f"{topic} 2026" for topic in self._search_topics[:max_queries]]

    async def search(
        self,
        queries: list[str],
        max_results_per_query: int = 5,
        seen_urls: set[str] | None = None,
    ) -> list[ScoutFinding]:
        """Execute web searches and return deduplicated findings."""
        seen = seen_urls or set()
        findings: list[ScoutFinding] = []

        for query in queries:
            try:
                results = await web_search(query, max_results=max_results_per_query)
            except Exception:
                log.warning("scout.search_failed query=%s", query[:80])
                continue

            for result in results:
                url = result.get("url", "")
                if not url or url in seen:
                    continue
                seen.add(url)

                source_type = _classify_source(url)

                findings.append(
                    ScoutFinding(
                        finding_id=f"fnd_{uuid.uuid4().hex[:12]}",
                        url=url,
                        title=result.get("title", ""),
                        snippet=result.get("snippet", ""),
                        source_type=source_type,
                        discovered_at=datetime.now(UTC),
                        tags=_extract_tags(query),
                    )
                )

        return findings

    async def fetch_content(
        self,
        findings: list[ScoutFinding],
        max_length: int = 5000,
    ) -> list[ScoutFinding]:
        """Fetch full content for each finding."""
        from qe.tools.web_fetch import web_fetch

        enriched: list[ScoutFinding] = []
        for finding in findings:
            try:
                result = await web_fetch(finding.url, max_length=max_length)
                finding.full_content = result.get("text", "")
                enriched.append(finding)
            except Exception:
                log.debug("scout.fetch_failed url=%s", finding.url[:80])
                enriched.append(finding)
        return enriched


def _classify_source(url: str) -> str:
    """Classify a URL into a source type."""
    url_lower = url.lower()
    if "github.com" in url_lower:
        return "github"
    if "news.ycombinator.com" in url_lower:
        return "hackernews"
    if "reddit.com" in url_lower:
        return "reddit"
    if "arxiv.org" in url_lower:
        return "arxiv"
    if any(
        f in url_lower
        for f in ["stackoverflow.com", "discourse", "forum", "discuss"]
    ):
        return "forum"
    return "blog"


def _extract_tags(query: str) -> list[str]:
    """Extract simple tags from a search query."""
    words = query.lower().split()
    stop = {"the", "a", "an", "for", "in", "on", "of", "to", "and", "or", "is", "with"}
    return [w for w in words if w not in stop and len(w) > 2][:5]

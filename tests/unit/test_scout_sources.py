"""Tests for scout SourceManager — query generation, web search, content fetch."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.scout import ScoutFinding
from qe.services.scout.sources import (
    SourceManager,
    _classify_source,
    _extract_tags,
    _SearchQueries,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_search_result(
    title: str = "Example",
    url: str = "https://example.com",
    snippet: str = "A snippet",
) -> dict[str, str]:
    return {"title": title, "url": url, "snippet": snippet}


def _make_finding(
    url: str = "https://example.com",
    title: str = "Example",
    snippet: str = "A snippet",
    source_type: str = "blog",
) -> ScoutFinding:
    return ScoutFinding(
        url=url,
        title=title,
        snippet=snippet,
        source_type=source_type,
    )


def _enabled_flag():
    mock_flag = MagicMock()
    mock_flag.is_enabled.return_value = True
    return mock_flag


def _disabled_flag():
    mock_flag = MagicMock()
    mock_flag.is_enabled.return_value = False
    return mock_flag


# ── _classify_source ────────────────────────────────────────────────────


class TestClassifySource:
    """URL classification into source types."""

    def test_github(self):
        assert _classify_source("https://github.com/user/repo") == "github"

    def test_hackernews(self):
        assert _classify_source("https://news.ycombinator.com/item?id=123") == "hackernews"

    def test_reddit(self):
        assert _classify_source("https://www.reddit.com/r/python/comments/abc") == "reddit"

    def test_arxiv(self):
        assert _classify_source("https://arxiv.org/abs/2301.12345") == "arxiv"

    def test_forum_stackoverflow(self):
        assert _classify_source("https://stackoverflow.com/questions/123") == "forum"

    def test_forum_discourse(self):
        assert _classify_source("https://community.example.com/discourse/topic/1") == "forum"

    def test_blog_fallback(self):
        assert _classify_source("https://blog.example.com/post") == "blog"

    def test_case_insensitive(self):
        assert _classify_source("https://GITHUB.COM/user/repo") == "github"


# ── _extract_tags ───────────────────────────────────────────────────────


class TestExtractTags:
    """Tag extraction from search queries."""

    def test_filters_stopwords(self):
        tags = _extract_tags("the best patterns for asyncio in Python")
        assert "the" not in tags
        assert "for" not in tags
        assert "in" not in tags

    def test_keeps_meaningful_words(self):
        tags = _extract_tags("Python asyncio patterns")
        assert "python" in tags
        assert "asyncio" in tags
        assert "patterns" in tags

    def test_filters_short_words(self):
        tags = _extract_tags("a go is ok to do it")
        # words with len <= 2 are excluded
        assert "go" not in tags
        assert "ok" not in tags

    def test_max_five_tags(self):
        tags = _extract_tags("alpha bravo charlie delta echo foxtrot golf hotel")
        assert len(tags) <= 5

    def test_lowercases(self):
        tags = _extract_tags("FastAPI Pydantic LLM")
        for tag in tags:
            assert tag == tag.lower()


# ── generate_queries ────────────────────────────────────────────────────


class TestGenerateQueries:
    """LLM-powered query generation."""

    @pytest.mark.asyncio
    async def test_generate_queries_with_llm(self):
        """When flag is on, uses instructor to generate queries."""
        sm = SourceManager(model="test-model", search_topics=["asyncio"])

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_SearchQueries(
                queries=["asyncio best practices 2026", "Python concurrency patterns"],
            )
        )

        with (
            patch("qe.services.scout.sources.get_flag_store", return_value=_enabled_flag()),
            patch("qe.services.scout.sources.instructor") as mock_inst,
        ):
            mock_inst.from_litellm.return_value = mock_client
            queries = await sm.generate_queries(codebase_summary="test project", max_queries=5)

        assert queries == ["asyncio best practices 2026", "Python concurrency patterns"]
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["response_model"] is _SearchQueries

    @pytest.mark.asyncio
    async def test_generate_queries_disabled_flag(self):
        """When feature flag is off, returns empty list without calling LLM."""
        sm = SourceManager()

        with patch("qe.services.scout.sources.get_flag_store", return_value=_disabled_flag()):
            queries = await sm.generate_queries()

        assert queries == []

    @pytest.mark.asyncio
    async def test_generate_queries_respects_max(self):
        """Result is truncated to max_queries."""
        sm = SourceManager()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_SearchQueries(queries=["q1", "q2", "q3", "q4", "q5"])
        )

        with (
            patch("qe.services.scout.sources.get_flag_store", return_value=_enabled_flag()),
            patch("qe.services.scout.sources.instructor") as mock_inst,
        ):
            mock_inst.from_litellm.return_value = mock_client
            queries = await sm.generate_queries(max_queries=3)

        assert len(queries) == 3

    @pytest.mark.asyncio
    async def test_generate_queries_fallback_on_llm_error(self):
        """When LLM call raises, falls back to topic-based queries."""
        sm = SourceManager(search_topics=["FastAPI", "Pydantic"])
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("LLM down"))

        with (
            patch("qe.services.scout.sources.get_flag_store", return_value=_enabled_flag()),
            patch("qe.services.scout.sources.instructor") as mock_inst,
        ):
            mock_inst.from_litellm.return_value = mock_client
            queries = await sm.generate_queries(max_queries=5)

        assert len(queries) == 2
        assert "FastAPI 2026" in queries
        assert "Pydantic 2026" in queries

    @pytest.mark.asyncio
    async def test_generate_queries_includes_avoid_text(self):
        """Rejected patterns are included in system prompt."""
        sm = SourceManager()
        sm.set_rejected_patterns(
            categories=["crypto", "blockchain"],
            sources=["medium.com"],
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_SearchQueries(queries=["test query"])
        )

        with (
            patch("qe.services.scout.sources.get_flag_store", return_value=_enabled_flag()),
            patch("qe.services.scout.sources.instructor") as mock_inst,
        ):
            mock_inst.from_litellm.return_value = mock_client
            await sm.generate_queries()

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        system_msg = call_kwargs["messages"][0]["content"]
        assert "crypto" in system_msg
        assert "blockchain" in system_msg
        assert "medium.com" in system_msg


# ── search ──────────────────────────────────────────────────────────────


class TestSearch:
    """Web search execution and result processing."""

    @pytest.mark.asyncio
    async def test_search_deduplicates_urls(self):
        """Duplicate URLs across queries are filtered out."""
        sm = SourceManager()
        results_batch = [
            _make_search_result(title="Result A", url="https://example.com/page1"),
            _make_search_result(title="Result B", url="https://example.com/page1"),  # duplicate
            _make_search_result(title="Result C", url="https://example.com/page2"),
        ]

        with patch("qe.services.scout.sources.web_search", new_callable=AsyncMock) as mock_ws:
            mock_ws.return_value = results_batch
            findings = await sm.search(["query one"])

        assert len(findings) == 2
        urls = [f.url for f in findings]
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls

    @pytest.mark.asyncio
    async def test_search_deduplicates_across_queries(self):
        """Same URL from different queries is only kept once."""
        sm = SourceManager()

        with patch("qe.services.scout.sources.web_search", new_callable=AsyncMock) as mock_ws:
            mock_ws.side_effect = [
                [_make_search_result(url="https://example.com/shared")],
                [_make_search_result(url="https://example.com/shared")],
            ]
            findings = await sm.search(["query1", "query2"])

        assert len(findings) == 1

    @pytest.mark.asyncio
    async def test_search_respects_seen_urls(self):
        """Pre-seen URLs are excluded from results."""
        sm = SourceManager()
        seen = {"https://example.com/already-seen"}

        with patch("qe.services.scout.sources.web_search", new_callable=AsyncMock) as mock_ws:
            mock_ws.return_value = [
                _make_search_result(url="https://example.com/already-seen"),
                _make_search_result(url="https://example.com/new"),
            ]
            findings = await sm.search(["query"], seen_urls=seen)

        assert len(findings) == 1
        assert findings[0].url == "https://example.com/new"

    @pytest.mark.asyncio
    async def test_search_classifies_sources(self):
        """Each finding gets the correct source_type from URL classification."""
        sm = SourceManager()
        results = [
            _make_search_result(url="https://github.com/user/repo"),
            _make_search_result(url="https://news.ycombinator.com/item?id=1"),
            _make_search_result(url="https://reddit.com/r/python"),
            _make_search_result(url="https://arxiv.org/abs/2301.1"),
            _make_search_result(url="https://myblog.dev/post"),
        ]

        with patch("qe.services.scout.sources.web_search", new_callable=AsyncMock) as mock_ws:
            mock_ws.return_value = results
            findings = await sm.search(["test query"])

        source_map = {f.url: f.source_type for f in findings}
        assert source_map["https://github.com/user/repo"] == "github"
        assert source_map["https://news.ycombinator.com/item?id=1"] == "hackernews"
        assert source_map["https://reddit.com/r/python"] == "reddit"
        assert source_map["https://arxiv.org/abs/2301.1"] == "arxiv"
        assert source_map["https://myblog.dev/post"] == "blog"

    @pytest.mark.asyncio
    async def test_search_extracts_tags_from_query(self):
        """Finding tags come from the query that produced them."""
        sm = SourceManager()

        with patch("qe.services.scout.sources.web_search", new_callable=AsyncMock) as mock_ws:
            mock_ws.return_value = [
                _make_search_result(url="https://example.com/page"),
            ]
            findings = await sm.search(["Python asyncio patterns"])

        assert len(findings) == 1
        assert "python" in findings[0].tags
        assert "asyncio" in findings[0].tags

    @pytest.mark.asyncio
    async def test_search_skips_empty_urls(self):
        """Results with empty URL are skipped."""
        sm = SourceManager()

        with patch("qe.services.scout.sources.web_search", new_callable=AsyncMock) as mock_ws:
            mock_ws.return_value = [
                {"title": "No URL", "url": "", "snippet": "nothing"},
                _make_search_result(url="https://example.com/real"),
            ]
            findings = await sm.search(["query"])

        assert len(findings) == 1
        assert findings[0].url == "https://example.com/real"

    @pytest.mark.asyncio
    async def test_search_continues_on_query_failure(self):
        """If one query fails, the rest still run."""
        sm = SourceManager()

        with patch("qe.services.scout.sources.web_search", new_callable=AsyncMock) as mock_ws:
            mock_ws.side_effect = [
                RuntimeError("network error"),
                [_make_search_result(url="https://example.com/ok")],
            ]
            findings = await sm.search(["bad_query", "good_query"])

        assert len(findings) == 1
        assert findings[0].url == "https://example.com/ok"


# ── fetch_content ───────────────────────────────────────────────────────


class TestFetchContent:
    """Content fetching for discovered findings."""

    @pytest.mark.asyncio
    async def test_fetch_content_enriches_findings(self):
        """Fetched text is stored in full_content."""
        sm = SourceManager()
        findings = [
            _make_finding(url="https://example.com/page1"),
            _make_finding(url="https://example.com/page2"),
        ]

        with patch("qe.tools.web_fetch.web_fetch", new_callable=AsyncMock) as mock_wf:
            mock_wf.side_effect = [
                {"text": "Full content of page 1", "title": "Page 1"},
                {"text": "Full content of page 2", "title": "Page 2"},
            ]
            enriched = await sm.fetch_content(findings)

        assert len(enriched) == 2
        assert enriched[0].full_content == "Full content of page 1"
        assert enriched[1].full_content == "Full content of page 2"

    @pytest.mark.asyncio
    async def test_fetch_content_passes_max_length(self):
        """max_length parameter is forwarded to web_fetch."""
        sm = SourceManager()
        findings = [_make_finding(url="https://example.com/page")]

        with patch("qe.tools.web_fetch.web_fetch", new_callable=AsyncMock) as mock_wf:
            mock_wf.return_value = {"text": "content"}
            await sm.fetch_content(findings, max_length=2000)

        mock_wf.assert_called_once_with("https://example.com/page", max_length=2000)

    @pytest.mark.asyncio
    async def test_fetch_content_survives_failure(self):
        """Findings are returned even if fetch fails (without full_content)."""
        sm = SourceManager()
        findings = [
            _make_finding(url="https://example.com/ok"),
            _make_finding(url="https://example.com/broken"),
        ]

        with patch("qe.tools.web_fetch.web_fetch", new_callable=AsyncMock) as mock_wf:
            mock_wf.side_effect = [
                {"text": "Good content"},
                RuntimeError("fetch failed"),
            ]
            enriched = await sm.fetch_content(findings)

        assert len(enriched) == 2
        assert enriched[0].full_content == "Good content"
        # Failed fetch leaves full_content unchanged (empty default)
        assert enriched[1].full_content == ""


# ── set_rejected_patterns ──────────────────────────────────────────────


class TestSetRejectedPatterns:
    """Rejection pattern management."""

    def test_set_rejected_patterns(self):
        """Categories and sources are stored on the instance."""
        sm = SourceManager()
        sm.set_rejected_patterns(
            categories=["crypto", "blockchain"],
            sources=["medium.com", "dev.to"],
        )

        assert sm._rejected_categories == ["crypto", "blockchain"]
        assert sm._rejected_sources == ["medium.com", "dev.to"]

    def test_set_rejected_patterns_replaces(self):
        """Calling set_rejected_patterns again replaces previous values."""
        sm = SourceManager()
        sm.set_rejected_patterns(categories=["old"], sources=["old.com"])
        sm.set_rejected_patterns(categories=["new"], sources=["new.com"])

        assert sm._rejected_categories == ["new"]
        assert sm._rejected_sources == ["new.com"]

    def test_defaults_empty(self):
        """New SourceManager has no rejected patterns."""
        sm = SourceManager()
        assert sm._rejected_categories == []
        assert sm._rejected_sources == []

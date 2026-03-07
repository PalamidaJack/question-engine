"""Tests for Phase 6 enhancements.

Covers:
- #60 MCP server
- #76 Multi-channel output pipeline
- #54 Analyze-then-validate retrieval
- #88 Discord adapter
"""

from __future__ import annotations

import asyncio
import json

import pytest

from qe.runtime.feature_flags import reset_flag_store


@pytest.fixture(autouse=True)
def _reset_flags():
    reset_flag_store()
    yield
    reset_flag_store()


# ── #60 MCP Server ────────────────────────────────────────────────────


class TestMCPServer:
    def test_register_tool(self):
        from qe.runtime.mcp_server import MCPServer, MCPToolDefinition

        server = MCPServer()
        tool = MCPToolDefinition(
            name="query_beliefs",
            description="Search the belief ledger",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        )
        server.register_tool(tool)
        tools = server.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "query_beliefs"

    def test_list_tools_empty(self):
        from qe.runtime.mcp_server import MCPServer

        server = MCPServer()
        assert server.list_tools() == []

    def test_call_unknown_tool(self):
        from qe.runtime.mcp_server import MCPServer

        server = MCPServer()
        result = asyncio.run(
            server.call_tool("nonexistent", {}),
        )
        assert not result.success
        assert "Unknown tool" in result.error

    def test_call_tool_no_registry(self):
        from qe.runtime.mcp_server import (
            MCPServer,
            MCPToolDefinition,
        )

        server = MCPServer(tool_registry=None)
        server.register_tool(
            MCPToolDefinition("test", "Test tool"),
        )
        result = asyncio.run(
            server.call_tool("test", {}),
        )
        assert not result.success
        assert "No tool registry" in result.error

    def test_call_tool_success(self):
        from qe.runtime.mcp_server import (
            MCPServer,
            MCPToolDefinition,
        )

        class MockRegistry:
            async def execute(self, name, args):
                return {"result": "ok"}

        server = MCPServer(tool_registry=MockRegistry())
        server.register_tool(
            MCPToolDefinition("test", "Test tool"),
        )
        result = asyncio.run(
            server.call_tool("test", {"q": "hello"}),
        )
        assert result.success
        assert result.result == {"result": "ok"}

    def test_call_tool_exception(self):
        from qe.runtime.mcp_server import (
            MCPServer,
            MCPToolDefinition,
        )

        class FailRegistry:
            async def execute(self, name, args):
                raise RuntimeError("boom")

        server = MCPServer(tool_registry=FailRegistry())
        server.register_tool(
            MCPToolDefinition("test", "Test tool"),
        )
        result = asyncio.run(
            server.call_tool("test", {}),
        )
        assert not result.success
        assert "boom" in result.error

    def test_server_info(self):
        from qe.runtime.mcp_server import (
            MCPServer,
            MCPToolDefinition,
        )

        server = MCPServer(
            name="qe-test", version="2.0.0",
        )
        server.register_tool(
            MCPToolDefinition("t1", "Tool 1"),
        )
        info = server.server_info()
        assert info["name"] == "qe-test"
        assert info["version"] == "2.0.0"
        assert info["tools"] == 1
        assert info["protocol"] == "mcp"

    def test_multiple_tools(self):
        from qe.runtime.mcp_server import (
            MCPServer,
            MCPToolDefinition,
        )

        server = MCPServer()
        server.register_tool(
            MCPToolDefinition("t1", "Tool 1"),
        )
        server.register_tool(
            MCPToolDefinition("t2", "Tool 2"),
        )
        server.register_tool(
            MCPToolDefinition("t3", "Tool 3"),
        )
        tools = server.list_tools()
        assert len(tools) == 3
        names = {t["name"] for t in tools}
        assert names == {"t1", "t2", "t3"}

    def test_tool_schema_in_list(self):
        from qe.runtime.mcp_server import (
            MCPServer,
            MCPToolDefinition,
        )

        schema = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
        }
        server = MCPServer()
        server.register_tool(
            MCPToolDefinition("t1", "Tool 1", schema),
        )
        tools = server.list_tools()
        assert tools[0]["inputSchema"] == schema


# ── #76 Output Pipeline ───────────────────────────────────────────────


class TestOutputPipeline:
    def test_web_format_passthrough(self):
        from qe.channels.output_pipeline import OutputPipeline

        pipe = OutputPipeline()
        md = "# Hello\n\nThis is **bold**."
        assert pipe.format(md, "web") == md

    def test_chat_strips_markdown(self):
        from qe.channels.output_pipeline import OutputPipeline

        pipe = OutputPipeline()
        md = "## Title\n\n**bold** and *italic*"
        result = pipe.format(md, "chat")
        assert "#" not in result
        assert "**" not in result
        assert "*" not in result
        assert "bold" in result

    def test_chat_strips_links(self):
        from qe.channels.output_pipeline import OutputPipeline

        pipe = OutputPipeline()
        md = "Check [this link](https://example.com) out"
        result = pipe.format(md, "chat")
        assert "this link" in result
        assert "https://example.com" not in result
        assert "[" not in result

    def test_chat_strips_code_blocks(self):
        from qe.channels.output_pipeline import OutputPipeline

        pipe = OutputPipeline()
        md = "Use `code` here\n```python\nprint('hi')\n```"
        result = pipe.format(md, "chat")
        assert "`" not in result
        assert "code" in result

    def test_api_format_json(self):
        from qe.channels.output_pipeline import OutputPipeline

        pipe = OutputPipeline()
        result = pipe.format(
            "Hello", "api",
            metadata={"source": "test"},
        )
        data = json.loads(result)
        assert data["content"] == "Hello"
        assert data["format"] == "markdown"
        assert data["metadata"]["source"] == "test"

    def test_api_format_no_metadata(self):
        from qe.channels.output_pipeline import OutputPipeline

        pipe = OutputPipeline()
        result = pipe.format("Hello", "api")
        data = json.loads(result)
        assert data["metadata"] == {}

    def test_email_format_html(self):
        from qe.channels.output_pipeline import OutputPipeline

        pipe = OutputPipeline()
        md = "# Title\n\n**bold** text"
        result = pipe.format(md, "email")
        assert "<div>" in result
        assert "<h1>" in result
        assert "<strong>" in result

    def test_email_escapes_html(self):
        from qe.channels.output_pipeline import OutputPipeline

        pipe = OutputPipeline()
        md = "Use <script>alert('xss')</script>"
        result = pipe.format(md, "email")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_email_paragraphs(self):
        from qe.channels.output_pipeline import OutputPipeline

        pipe = OutputPipeline()
        md = "Para 1\n\nPara 2"
        result = pipe.format(md, "email")
        assert "</p><p>" in result

    def test_unknown_channel_defaults_to_web(self):
        from qe.channels.output_pipeline import OutputPipeline

        pipe = OutputPipeline()
        md = "# Hello"
        assert pipe.format(md, "unknown") == md

    def test_supported_channels(self):
        from qe.channels.output_pipeline import OutputPipeline

        pipe = OutputPipeline()
        channels = pipe.supported_channels()
        assert set(channels) == {"web", "chat", "api", "email"}


# ── #54 Analyze-then-Validate Retrieval ───────────────────────────────


class TestRetrievalAnalyzer:
    def test_factual_intent(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "What is Python programming?",
        )
        assert analysis.intent == "factual"
        assert len(analysis.validation_criteria) > 0

    def test_comparative_intent(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "Compare Python versus JavaScript",
        )
        assert analysis.intent == "comparative"

    def test_analytical_intent(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "Why does memory management cause issues?",
        )
        assert analysis.intent == "analytical"

    def test_key_entity_extraction(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "What is machine learning used for?",
        )
        assert "machine" in analysis.key_entities
        assert "learning" in analysis.key_entities

    def test_entity_limit(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        long_query = " ".join(
            f"word{i}" for i in range(20)
        )
        analysis = analyzer.analyze_query(long_query)
        assert len(analysis.key_entities) <= 10

    def test_validate_empty_results(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query("test query")
        validation = analyzer.validate_results(
            analysis, [],
        )
        assert validation.total_retrieved == 0
        assert "No results" in validation.issues[0]

    def test_validate_relevant_results(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "Python programming language",
        )
        results = [
            "Python is a popular programming language",
            "Python supports multiple paradigms",
        ]
        validation = analyzer.validate_results(
            analysis, results,
        )
        assert validation.passed > 0
        assert validation.total_retrieved == 2

    def test_validate_irrelevant_results(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "quantum entanglement superconductors",
        )
        results = [
            "The weather is sunny today",
            "Bananas are a yellow fruit",
        ]
        validation = analyzer.validate_results(
            analysis, results,
        )
        assert validation.failed > 0
        assert validation.pass_rate < 0.5

    def test_validate_comparative_coverage(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "Compare Python versus JavaScript",
        )
        # Only cover Python, not JavaScript
        results = [
            "Python is great for data science",
        ]
        validation = analyzer.validate_results(
            analysis, results,
        )
        assert any(
            "not all" in i.lower() or "covered" in i.lower()
            for i in validation.issues
        )

    def test_pass_rate_property(self):
        from qe.runtime.retrieval_analyzer import (
            ValidationResult,
        )

        vr = ValidationResult(
            total_retrieved=10, passed=7, failed=3,
        )
        assert vr.pass_rate == 0.7

    def test_pass_rate_zero_results(self):
        from qe.runtime.retrieval_analyzer import (
            ValidationResult,
        )

        vr = ValidationResult()
        assert vr.pass_rate == 0.0

    def test_analyze_and_validate_pipeline(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        result = analyzer.analyze_and_validate(
            "What is Python?",
            ["Python is a programming language"],
        )
        assert result["query"] == "What is Python?"
        assert result["intent"] == "factual"
        assert "validation" in result
        assert "pass_rate" in result["validation"]

    def test_confidence_threshold_factual(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "What is the capital of France?",
        )
        assert analysis.confidence_threshold == 0.6

    def test_confidence_threshold_analytical(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "Why do computers need memory?",
        )
        assert analysis.confidence_threshold == 0.4

    def test_factual_criteria_content(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "What is quantum computing?",
        )
        assert any(
            "factual" in c.lower()
            for c in analysis.validation_criteria
        )

    def test_comparative_criteria_content(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "Compare React vs Angular",
        )
        assert any(
            "compared" in c.lower()
            for c in analysis.validation_criteria
        )

    def test_analytical_criteria_content(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        analyzer = RetrievalAnalyzer()
        analysis = analyzer.analyze_query(
            "How does garbage collection work?",
        )
        assert any(
            "reasoning" in c.lower()
            for c in analysis.validation_criteria
        )


# ── #88 Discord Adapter ───────────────────────────────────────────────


class TestDiscordAdapter:
    def test_instantiation(self):
        from qe.channels.discord import DiscordAdapter

        adapter = DiscordAdapter(
            bot_token="test-token",
            default_channel_id=12345,
        )
        assert adapter.channel_name == "discord"
        assert not adapter.is_running

    def test_extract_text_dict(self):
        from qe.channels.discord import DiscordAdapter

        adapter = DiscordAdapter()
        text = adapter._extract_text(
            {"content": "Hello world"},
        )
        assert text == "Hello world"

    def test_extract_text_object(self):
        from qe.channels.discord import DiscordAdapter

        class FakeMsg:
            content = "Test message"

        adapter = DiscordAdapter()
        text = adapter._extract_text(FakeMsg())
        assert text == "Test message"

    def test_extract_text_string(self):
        from qe.channels.discord import DiscordAdapter

        adapter = DiscordAdapter()
        text = adapter._extract_text("plain string")
        assert text == "plain string"

    def test_get_user_id_dict(self):
        from qe.channels.discord import DiscordAdapter

        adapter = DiscordAdapter()
        uid = adapter._get_user_id(
            {"author": {"id": "123456"}},
        )
        assert uid == "123456"

    def test_get_user_id_object(self):
        from qe.channels.discord import DiscordAdapter

        class FakeAuthor:
            id = 789

        class FakeMsg:
            author = FakeAuthor()

        adapter = DiscordAdapter()
        uid = adapter._get_user_id(FakeMsg())
        assert uid == "789"

    def test_get_user_id_fallback(self):
        from qe.channels.discord import DiscordAdapter

        adapter = DiscordAdapter()
        uid = adapter._get_user_id("raw string")
        assert uid == ""

    def test_server_info(self):
        from qe.channels.discord import DiscordAdapter

        adapter = DiscordAdapter(
            default_channel_id=999,
        )
        info = adapter.server_info()
        assert info["channel"] == "discord"
        assert info["running"] is False
        assert info["bot_user"] is None
        assert info["default_channel"] == 999

    def test_send_without_client(self):
        from qe.channels.discord import DiscordAdapter

        adapter = DiscordAdapter()
        # Should not raise, just log warning
        asyncio.run(adapter.send("123", "Hello"))

    def test_receive_clean_message(self):
        from qe.channels.discord import DiscordAdapter

        adapter = DiscordAdapter()
        result = asyncio.run(
            adapter.receive(
                {"content": "Hi", "author": {"id": "42"}},
            ),
        )
        assert result is not None
        assert result["text"] == "Hi"
        assert result["user_id"] == "42"

    def test_is_goal_detection(self):
        from qe.channels.discord import DiscordAdapter

        adapter = DiscordAdapter()
        assert adapter._is_goal("goal: research AI")
        assert adapter._is_goal("Research quantum computing")
        assert not adapter._is_goal("Hello there")


# ── Phase 6 Imports ───────────────────────────────────────────────────


class TestPhase6Imports:
    def test_import_mcp_server(self):
        from qe.runtime.mcp_server import MCPServer

        assert MCPServer

    def test_import_output_pipeline(self):
        from qe.channels.output_pipeline import OutputPipeline

        assert OutputPipeline

    def test_import_retrieval_analyzer(self):
        from qe.runtime.retrieval_analyzer import (
            RetrievalAnalyzer,
        )

        assert RetrievalAnalyzer

    def test_import_discord_adapter(self):
        from qe.channels.discord import DiscordAdapter

        assert DiscordAdapter

    def test_import_channel_base(self):
        from qe.channels.base import (
            ChannelAdapter,
            ChannelMessage,
        )

        assert ChannelAdapter and ChannelMessage

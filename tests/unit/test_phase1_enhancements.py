"""Tests for Phase 1 enhancements: #44, #46, #50, #56, #78, #87."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# #78 — Maturity lifecycle
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaturityLifecycle:
    """Tests for FeatureFlagDef.maturity and related store methods."""

    def test_default_maturity_is_stable(self):
        from qe.runtime.feature_flags import FeatureFlagDef

        flag = FeatureFlagDef(name="test")
        assert flag.maturity == "stable"

    def test_custom_maturity(self):
        from qe.runtime.feature_flags import FeatureFlagDef

        flag = FeatureFlagDef(name="test", maturity="experimental")
        assert flag.maturity == "experimental"

    def test_maturity_in_to_dict(self):
        from qe.runtime.feature_flags import FeatureFlagDef

        flag = FeatureFlagDef(name="test", maturity="preview")
        d = flag.to_dict()
        assert d["maturity"] == "preview"

    def test_define_with_maturity(self):
        from qe.runtime.feature_flags import FeatureFlagStore

        store = FeatureFlagStore()
        flag = store.define("x", maturity="experimental")
        assert flag.maturity == "experimental"

    def test_define_updates_maturity(self):
        from qe.runtime.feature_flags import FeatureFlagStore

        store = FeatureFlagStore()
        store.define("x", maturity="experimental")
        store.define("x", maturity="preview")
        assert store.get("x").maturity == "preview"

    def test_list_by_maturity(self):
        from qe.runtime.feature_flags import FeatureFlagStore

        store = FeatureFlagStore()
        store.define("a", maturity="experimental")
        store.define("b", maturity="stable")
        store.define("c", maturity="experimental")
        result = store.list_by_maturity("experimental")
        names = {f.name for f in result}
        assert names == {"a", "c"}

    def test_stats_include_maturity_counts(self):
        from qe.runtime.feature_flags import FeatureFlagStore

        store = FeatureFlagStore()
        store.define("a", maturity="experimental")
        store.define("b", maturity="stable")
        store.define("c", maturity="deprecated")
        stats = store.stats()
        assert stats["maturity"]["experimental"] == 1
        assert stats["maturity"]["stable"] == 1
        assert stats["maturity"]["deprecated"] == 1

    def test_maturity_levels_constant(self):
        from qe.runtime.feature_flags import MATURITY_LEVELS

        assert "experimental" in MATURITY_LEVELS
        assert "preview" in MATURITY_LEVELS
        assert "stable" in MATURITY_LEVELS
        assert "deprecated" in MATURITY_LEVELS

    def test_load_preserves_maturity(self, tmp_path):
        import json

        from qe.runtime.feature_flags import FeatureFlagStore

        store = FeatureFlagStore()
        store.define("x", maturity="preview")
        path = tmp_path / "flags.json"
        store.save(path)

        store2 = FeatureFlagStore()
        store2.load(path)
        assert store2.get("x").maturity == "preview"

    def test_load_defaults_maturity_to_stable(self, tmp_path):
        import json

        path = tmp_path / "flags.json"
        path.write_text(json.dumps({"x": {"enabled": True}}))

        from qe.runtime.feature_flags import FeatureFlagStore

        store = FeatureFlagStore()
        store.load(path)
        assert store.get("x").maturity == "stable"


# ═══════════════════════════════════════════════════════════════════════════════
# #44 — Vector embeddings enhancements
# ═══════════════════════════════════════════════════════════════════════════════


class TestEmbeddingStoreEnhancements:
    """Tests for coverage_stats(), re_embed_all(), list_embedded_ids()."""

    @pytest.fixture
    async def store(self, tmp_path):
        import aiosqlite

        from qe.substrate.embeddings import EmbeddingStore

        db_path = str(tmp_path / "test.db")
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimensions INTEGER NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)
            await db.commit()

        s = EmbeddingStore(db_path, model="text-embedding-3-small")
        return s

    @pytest.mark.asyncio
    async def test_coverage_stats_empty(self, store):
        stats = await store.coverage_stats(total_claims=100)
        assert stats["embedded_count"] == 0
        assert stats["total_claims"] == 100
        assert stats["coverage_pct"] == 0.0

    @pytest.mark.asyncio
    async def test_coverage_stats_with_embeddings(self, store):
        embedding = [0.1] * 10
        await store.store("claim:1", "test claim", embedding=embedding)
        stats = await store.coverage_stats(total_claims=5)
        assert stats["embedded_count"] == 1
        assert stats["coverage_pct"] == 20.0

    @pytest.mark.asyncio
    async def test_coverage_stats_zero_claims(self, store):
        stats = await store.coverage_stats(total_claims=0)
        assert stats["coverage_pct"] == 0.0

    @pytest.mark.asyncio
    async def test_list_embedded_ids_empty(self, store):
        ids = await store.list_embedded_ids()
        assert ids == set()

    @pytest.mark.asyncio
    async def test_list_embedded_ids(self, store):
        embedding = [0.1] * 10
        await store.store("claim:1", "test 1", embedding=embedding)
        await store.store("claim:2", "test 2", embedding=embedding)
        ids = await store.list_embedded_ids()
        assert ids == {"claim:1", "claim:2"}

    @pytest.mark.asyncio
    async def test_re_embed_all(self, store):
        claims = [
            {
                "claim_id": "c1",
                "subject_entity_id": "apple",
                "predicate": "is",
                "object_value": "a company",
            },
            {
                "claim_id": "c2",
                "subject_entity_id": "google",
                "predicate": "is",
                "object_value": "a company",
            },
        ]
        embedding = [0.1] * 10
        with patch.object(store, "embed_text", new_callable=AsyncMock, return_value=embedding):
            result = await store.re_embed_all(claims, batch_size=10)
        assert result["succeeded"] == 2
        assert result["failed"] == 0
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_re_embed_all_with_failure(self, store):
        claims = [
            {"claim_id": "c1", "subject_entity_id": "x", "predicate": "p", "object_value": "v"},
        ]

        call_count = 0

        async def _failing_store(*args, **kwargs):
            raise RuntimeError("embed error")

        with patch.object(store, "store", side_effect=_failing_store):
            result = await store.re_embed_all(claims)
        assert result["failed"] == 1

    @pytest.mark.asyncio
    async def test_re_embed_all_progress_callback(self, store):
        claims = [
            {"claim_id": f"c{i}", "subject_entity_id": "x", "predicate": "p", "object_value": "v"}
            for i in range(5)
        ]
        embedding = [0.1] * 10
        progress_calls = []

        async def on_progress(done, total):
            progress_calls.append((done, total))

        with patch.object(store, "embed_text", new_callable=AsyncMock, return_value=embedding):
            await store.re_embed_all(claims, batch_size=2, on_progress=on_progress)

        assert len(progress_calls) == 3  # batches of 2: [0:2], [2:4], [4:5]


# ═══════════════════════════════════════════════════════════════════════════════
# #46 — Graph-based knowledge retrieval
# ═══════════════════════════════════════════════════════════════════════════════


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph."""

    def _make_claim(self, cid, subj, pred, obj, conf=0.8):
        """Create a mock claim object."""
        c = MagicMock()
        c.claim_id = cid
        c.subject_entity_id = subj
        c.predicate = pred
        c.object_value = obj
        c.confidence = conf
        return c

    def test_empty_graph(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        assert g.node_count == 0
        assert g.edge_count == 0

    def test_build_from_claims(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        claims = [
            self._make_claim("c1", "apple", "is", "technology company"),
            self._make_claim("c2", "apple", "founded_by", "steve jobs"),
            self._make_claim("c3", "steve jobs", "cofounded", "pixar"),
        ]
        g.build(claims)
        assert g.node_count > 0
        assert g.edge_count == 3

    def test_one_hop_neighbors(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        claims = [
            self._make_claim("c1", "apple", "is", "company"),
            self._make_claim("c2", "apple", "founded_by", "steve jobs"),
        ]
        g.build(claims)
        edges = g.neighbors("apple", hops=1)
        assert len(edges) == 2

    def test_two_hop_neighbors(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        claims = [
            self._make_claim("c1", "apple", "founded_by", "steve jobs"),
            self._make_claim("c2", "steve jobs", "cofounded", "pixar"),
        ]
        g.build(claims)
        edges = g.neighbors("apple", hops=2)
        # Should reach both apple→steve jobs and steve jobs→pixar
        cids = {e.claim_id for e in edges}
        assert "c1" in cids
        assert "c2" in cids

    def test_neighbors_with_min_confidence(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        claims = [
            self._make_claim("c1", "apple", "is", "company", conf=0.9),
            self._make_claim("c2", "apple", "might_be", "fruit", conf=0.1),
        ]
        g.build(claims)
        edges = g.neighbors("apple", hops=1, min_confidence=0.5)
        assert len(edges) == 1
        assert edges[0].claim_id == "c1"

    def test_neighbors_unknown_entity(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        g.build([self._make_claim("c1", "apple", "is", "company")])
        edges = g.neighbors("unknown_entity", hops=1)
        assert edges == []

    def test_graph_context(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        claims = [
            self._make_claim("c1", "apple", "is", "company"),
        ]
        g.build(claims)
        ctx = g.graph_context("apple")
        assert "Knowledge Graph" in ctx
        assert "apple" in ctx
        assert "company" in ctx

    def test_graph_context_empty(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        g.build([])
        assert g.graph_context("nothing") == ""

    def test_subgraph(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        claims = [
            self._make_claim("c1", "apple", "is", "company"),
            self._make_claim("c2", "google", "is", "company"),
        ]
        g.build(claims)
        sub = g.subgraph(["apple", "google"])
        assert sub["edge_count"] == 2
        assert "apple" in sub["nodes"]
        assert "google" in sub["nodes"]

    def test_stats(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        assert g.stats()["built"] is False
        g.build([self._make_claim("c1", "a", "p", "o")])
        assert g.stats()["built"] is True

    def test_build_skips_empty_subject(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        c = MagicMock()
        c.claim_id = "c1"
        c.subject_entity_id = ""
        c.predicate = "is"
        c.object_value = "val"
        c.confidence = 0.5
        g.build([c])
        assert g.edge_count == 0

    def test_deduplication_in_neighbors(self):
        from qe.substrate.knowledge_graph import KnowledgeGraph

        g = KnowledgeGraph()
        # Bidirectional reference creates potential duplicates
        claims = [
            self._make_claim("c1", "apple", "related_to", "steve jobs"),
        ]
        g.build(claims)
        edges = g.neighbors("apple", hops=2)
        cids = [e.claim_id for e in edges]
        # Should not have duplicates
        assert len(cids) == len(set(cids))


# ═══════════════════════════════════════════════════════════════════════════════
# #50 — Per-tool quality metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolMetrics:
    """Tests for ToolMetrics."""

    def test_empty_summary(self):
        from qe.runtime.tools import ToolMetrics

        tm = ToolMetrics()
        s = tm.summary()
        assert s["total_tool_calls"] == 0
        assert s["tools_tracked"] == 0

    def test_record_success(self):
        from qe.runtime.tools import ToolMetrics

        tm = ToolMetrics()
        tm.record_success("web_search", 150.0)
        stats = tm.get_stats("web_search")
        assert stats["total_calls"] == 1
        assert stats["successes"] == 1
        assert stats["failures"] == 0
        assert stats["avg_latency_ms"] == 150.0
        assert stats["success_rate"] == 1.0

    def test_record_failure(self):
        from qe.runtime.tools import ToolMetrics

        tm = ToolMetrics()
        tm.record_failure("web_search", 200.0, "timeout")
        stats = tm.get_stats("web_search")
        assert stats["failures"] == 1
        assert stats["success_rate"] == 0.0
        assert stats["last_error"] == "timeout"

    def test_mixed_success_and_failure(self):
        from qe.runtime.tools import ToolMetrics

        tm = ToolMetrics()
        tm.record_success("tool_a", 100.0)
        tm.record_success("tool_a", 200.0)
        tm.record_failure("tool_a", 300.0, "err")
        stats = tm.get_stats("tool_a")
        assert stats["total_calls"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert stats["success_rate"] == pytest.approx(0.6667, abs=0.001)
        assert stats["avg_latency_ms"] == 200.0

    def test_all_stats(self):
        from qe.runtime.tools import ToolMetrics

        tm = ToolMetrics()
        tm.record_success("b_tool", 10.0)
        tm.record_success("a_tool", 20.0)
        all_s = tm.all_stats()
        assert len(all_s) == 2
        assert all_s[0]["name"] == "a_tool"  # sorted

    def test_summary_overall_success_rate(self):
        from qe.runtime.tools import ToolMetrics

        tm = ToolMetrics()
        tm.record_success("t1", 10.0)
        tm.record_success("t2", 10.0)
        tm.record_failure("t2", 10.0, "err")
        s = tm.summary()
        assert s["total_tool_calls"] == 3
        assert s["total_errors"] == 1
        assert s["overall_success_rate"] == pytest.approx(0.6667, abs=0.001)

    def test_get_stats_unknown_tool(self):
        from qe.runtime.tools import ToolMetrics

        tm = ToolMetrics()
        assert tm.get_stats("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════════════════
# #56 — 3-tier routing enhancements
# ═══════════════════════════════════════════════════════════════════════════════


class TestThreeTierRouting:
    """Tests for preferred_tier on ToolSpec and select_for_tool."""

    def test_tool_spec_preferred_tier(self):
        from qe.runtime.tools import ToolSpec

        spec = ToolSpec(name="test", description="d", preferred_tier="fast")
        assert spec.preferred_tier == "fast"

    def test_tool_spec_default_tier_empty(self):
        from qe.runtime.tools import ToolSpec

        spec = ToolSpec(name="test", description="d")
        assert spec.preferred_tier == ""

    def test_registry_get_tier_for_tool(self):
        from qe.runtime.tools import ToolRegistry, ToolSpec

        registry = ToolRegistry()
        spec = ToolSpec(name="web_search", description="d", preferred_tier="fast")
        registry.register(spec, AsyncMock())
        assert registry.get_tier_for_tool("web_search") == "fast"

    def test_registry_get_tier_for_tool_none(self):
        from qe.runtime.tools import ToolRegistry, ToolSpec

        registry = ToolRegistry()
        spec = ToolSpec(name="web_search", description="d")
        registry.register(spec, AsyncMock())
        assert registry.get_tier_for_tool("web_search") is None

    def test_registry_get_tier_unknown_tool(self):
        from qe.runtime.tools import ToolRegistry

        registry = ToolRegistry()
        assert registry.get_tier_for_tool("unknown") is None

    def test_router_tier_config(self):
        from qe.models.genome import ModelPreference
        from qe.runtime.router import AutoRouter

        router = AutoRouter(ModelPreference(tier="balanced"))
        config = router.tier_config()
        assert "tiers" in config
        assert "active_tier" in config
        assert config["active_tier"] == "balanced"
        assert "task_tier_map" in config

    def test_router_select_for_tool_with_preferred_tier(self):
        from qe.models.genome import ModelPreference
        from qe.models.envelope import Envelope
        from qe.runtime.router import AutoRouter
        from qe.runtime.tools import ToolRegistry, ToolSpec

        registry = ToolRegistry()
        spec = ToolSpec(name="web_search", description="d", preferred_tier="fast")
        registry.register(spec, AsyncMock())

        router = AutoRouter(
            ModelPreference(tier="powerful"),
            tool_registry=registry,
        )
        env = Envelope(topic="test", payload={}, source_service_id="test")
        model = router.select_for_tool(env, tool_name="web_search")
        # Should select from fast tier, not powerful
        from qe.runtime.router import TIER_MODELS
        assert model in TIER_MODELS["fast"]

    def test_router_select_for_tool_no_preferred(self):
        from qe.models.genome import ModelPreference
        from qe.models.envelope import Envelope
        from qe.runtime.router import AutoRouter

        router = AutoRouter(ModelPreference(tier="balanced"))
        env = Envelope(topic="test", payload={}, source_service_id="test")
        model = router.select_for_tool(env, tool_name=None)
        # Falls through to normal select
        from qe.runtime.router import TIER_MODELS
        assert model in TIER_MODELS["balanced"]


# ═══════════════════════════════════════════════════════════════════════════════
# #50 — ToolRegistry execute with metrics
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolRegistryWithMetrics:
    """Test ToolRegistry.execute records metrics."""

    @pytest.mark.asyncio
    async def test_execute_records_success(self):
        from qe.runtime.tools import ToolMetrics, ToolRegistry, ToolSpec

        tm = ToolMetrics()
        registry = ToolRegistry(tool_metrics=tm)

        async def handler(**kwargs):
            return "ok"

        spec = ToolSpec(name="mytool", description="d")
        registry.register(spec, handler)

        result = await registry.execute("mytool", {})
        assert result == "ok"
        stats = tm.get_stats("mytool")
        assert stats["successes"] == 1

    @pytest.mark.asyncio
    async def test_execute_records_failure(self):
        from qe.runtime.tools import ToolMetrics, ToolRegistry, ToolSpec

        tm = ToolMetrics()
        registry = ToolRegistry(tool_metrics=tm)

        async def handler(**kwargs):
            raise ValueError("boom")

        spec = ToolSpec(name="mytool", description="d")
        registry.register(spec, handler)

        with pytest.raises(ValueError, match="boom"):
            await registry.execute("mytool", {})

        stats = tm.get_stats("mytool")
        assert stats["failures"] == 1
        assert "boom" in stats["last_error"]

    @pytest.mark.asyncio
    async def test_execute_without_metrics(self):
        from qe.runtime.tools import ToolRegistry, ToolSpec

        registry = ToolRegistry()  # no metrics

        async def handler(**kwargs):
            return 42

        spec = ToolSpec(name="t", description="d")
        registry.register(spec, handler)

        result = await registry.execute("t", {})
        assert result == 42


# ═══════════════════════════════════════════════════════════════════════════════
# #87 — Unified LLM abstraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnifiedLLM:
    """Tests for UnifiedLLM."""

    def test_default_stats(self):
        from qe.runtime.llm import UnifiedLLM

        llm = UnifiedLLM()
        s = llm.stats()
        assert s["total_calls"] == 0
        assert s["total_cost_usd"] == 0.0

    def test_recent_calls_empty(self):
        from qe.runtime.llm import UnifiedLLM

        llm = UnifiedLLM()
        assert llm.recent_calls() == []

    @pytest.mark.asyncio
    async def test_complete_success(self):
        from qe.runtime.llm import UnifiedLLM

        llm = UnifiedLLM(max_retries=0)
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            with patch("litellm.completion_cost", return_value=0.001):
                result = await llm.complete(
                    [{"role": "user", "content": "hello"}],
                    model="test-model",
                )
        assert result == mock_response
        assert llm.stats()["total_calls"] == 1
        assert llm.stats()["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_complete_retry_on_failure(self):
        from qe.runtime.llm import UnifiedLLM

        llm = UnifiedLLM(max_retries=1, base_delay=0.01)
        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10

        call_count = 0

        async def _flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("temporary failure")
            return mock_response

        with patch("litellm.acompletion", side_effect=_flaky):
            with patch("litellm.completion_cost", return_value=0.0):
                result = await llm.complete(
                    [{"role": "user", "content": "hello"}],
                    model="test-model",
                )
        assert result == mock_response
        assert llm.stats()["total_calls"] == 2
        assert llm.stats()["total_errors"] == 1

    @pytest.mark.asyncio
    async def test_complete_exhausts_retries(self):
        from qe.runtime.llm import UnifiedLLM

        llm = UnifiedLLM(max_retries=1, base_delay=0.01)

        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError, match="fail"):
                await llm.complete(
                    [{"role": "user", "content": "hello"}],
                    model="test-model",
                )
        assert llm.stats()["total_calls"] == 2  # initial + 1 retry
        assert llm.stats()["total_errors"] == 2

    @pytest.mark.asyncio
    async def test_complete_with_metrics(self):
        from qe.runtime.llm import UnifiedLLM
        from qe.runtime.metrics import MetricsCollector

        mc = MetricsCollector()
        llm = UnifiedLLM(max_retries=0, metrics=mc)

        mock_response = MagicMock()
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            with patch("litellm.completion_cost", return_value=0.0):
                await llm.complete([{"role": "user", "content": "hi"}])

        assert mc.counter("llm_calls_total").value >= 1

    @pytest.mark.asyncio
    async def test_provider_config_applied(self):
        from qe.runtime.llm import UnifiedLLM

        llm = UnifiedLLM(
            max_retries=0,
            provider_config={"openai/": {"temperature": 0.5, "max_tokens": 100}},
        )
        mock_response = MagicMock()
        mock_response.usage = None

        captured_kwargs = {}

        async def _capture(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return mock_response

        with patch("litellm.acompletion", side_effect=_capture):
            with patch("litellm.completion_cost", return_value=0.0):
                await llm.complete(
                    [{"role": "user", "content": "hi"}],
                    model="openai/gpt-4o",
                )

        assert captured_kwargs.get("temperature") == 0.5
        assert captured_kwargs.get("max_tokens") == 100

    def test_recent_calls_limit(self):
        from qe.runtime.llm import LLMCallRecord, UnifiedLLM

        llm = UnifiedLLM()
        for i in range(10):
            llm._record(
                LLMCallRecord(
                    model="m",
                    latency_ms=i,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    success=True,
                )
            )
        calls = llm.recent_calls(limit=3)
        assert len(calls) == 3

    def test_llm_call_record_fields(self):
        from qe.runtime.llm import LLMCallRecord

        r = LLMCallRecord(
            model="test",
            latency_ms=100.5,
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            success=True,
        )
        assert r.model == "test"
        assert r.error is None

    @pytest.mark.asyncio
    async def test_complete_no_usage(self):
        """Handle response with no usage attribute."""
        from qe.runtime.llm import UnifiedLLM

        llm = UnifiedLLM(max_retries=0)
        mock_response = MagicMock()
        mock_response.usage = None

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            with patch("litellm.completion_cost", return_value=0.0):
                result = await llm.complete([{"role": "user", "content": "hi"}])
        assert result == mock_response


# ═══════════════════════════════════════════════════════════════════════════════
# #50 — tools API endpoint
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolsEndpoint:
    """Tests for the /api/tools/* endpoints."""

    def test_router_exists(self):
        from qe.api.endpoints.tools import router

        routes = [r.path for r in router.routes]
        assert "/metrics" in routes or "/api/tools/metrics" in routes

    def test_get_app_globals(self):
        from qe.api.endpoints.tools import _get_app_globals

        mod = _get_app_globals()
        assert hasattr(mod, "_tool_metrics")

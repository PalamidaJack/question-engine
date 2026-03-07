"""Tests for Phase 2 enhancements: #47, #55, #59, #63, #67, #68, #71."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# #71 — Token budget management
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenBudgetManagement:
    def test_default_section_budget_sums_to_1(self):
        from qe.runtime.context_curator import DEFAULT_SECTION_BUDGET

        assert abs(sum(DEFAULT_SECTION_BUDGET.values()) - 1.0) < 0.001

    def test_assemble_sections_system_only(self):
        from qe.runtime.context_curator import ContextCurator

        cc = ContextCurator()
        messages = cc.assemble_sections(
            system_content="You are an assistant.",
            max_tokens=1000,
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_assemble_sections_all_sections(self):
        from qe.runtime.context_curator import ContextCurator

        cc = ContextCurator()
        messages = cc.assemble_sections(
            max_tokens=2000,
            system_content="System prompt",
            knowledge_items=["Claim A", "Claim B"],
            memory_items=["Memory 1"],
            history_messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )
        # System + knowledge + memory + 2 history messages
        assert len(messages) == 5
        assert messages[0]["content"] == "System prompt"
        assert "[KNOWLEDGE]" in messages[1]["content"]
        assert "[MEMORY]" in messages[2]["content"]

    def test_assemble_sections_history_trimming(self):
        from qe.runtime.context_curator import ContextCurator

        cc = ContextCurator()
        # Create many history messages that exceed the budget
        history = [
            {"role": "user", "content": "x" * 500}
            for _ in range(100)
        ]
        messages = cc.assemble_sections(
            max_tokens=500,
            history_messages=history,
        )
        # Should trim to fit budget (25% of 500 = 125 tokens ≈ 500 chars)
        hist_count = len(messages)
        assert hist_count < 100

    def test_assemble_sections_custom_budget(self):
        from qe.runtime.context_curator import ContextCurator

        cc = ContextCurator()
        custom = {"system": 0.50, "knowledge": 0.20, "memory": 0.10, "history": 0.15, "margin": 0.05}
        messages = cc.assemble_sections(
            max_tokens=1000,
            system_content="Prompt",
            section_budget=custom,
        )
        assert len(messages) == 1

    def test_trim_to_tokens(self):
        from qe.runtime.context_curator import ContextCurator

        cc = ContextCurator()
        long_text = "x" * 1000
        trimmed = cc._trim_to_tokens(long_text, 50)
        assert len(trimmed) <= 203  # 50*4 + "..."

    def test_pack_items(self):
        from qe.runtime.context_curator import ContextCurator

        cc = ContextCurator()
        items = ["short", "a" * 200, "b" * 200]
        packed = cc._pack_items(items, budget_tokens=20)
        assert len(packed) <= 2

    def test_trim_history_recency(self):
        from qe.runtime.context_curator import ContextCurator

        cc = ContextCurator()
        msgs = [
            {"role": "user", "content": f"msg {i}"}
            for i in range(10)
        ]
        trimmed = cc._trim_history(msgs, budget_tokens=10)
        # Should keep most recent messages
        assert trimmed[-1]["content"] == "msg 9"


# ═══════════════════════════════════════════════════════════════════════════════
# #47 — Adaptive memory tier weighting
# ═══════════════════════════════════════════════════════════════════════════════


class TestQueryClassifier:
    def test_classify_factual(self):
        from qe.runtime.query_classifier import classify_query

        assert classify_query("What is the capital of France?") == "factual"

    def test_classify_procedural(self):
        from qe.runtime.query_classifier import classify_query

        assert classify_query("How do I install Python?") == "procedural"

    def test_classify_analytical(self):
        from qe.runtime.query_classifier import classify_query

        assert classify_query("Why did the project fail?") == "analytical"

    def test_classify_creative(self):
        from qe.runtime.query_classifier import classify_query

        assert classify_query("Create a poem about nature") == "creative"

    def test_classify_meta(self):
        from qe.runtime.query_classifier import classify_query

        assert classify_query("What do you remember from our previous conversation?") == "meta"

    def test_classify_conversational(self):
        from qe.runtime.query_classifier import classify_query

        assert classify_query("hello") == "conversational"

    def test_classify_default_factual(self):
        from qe.runtime.query_classifier import classify_query

        assert classify_query("xyzzy") == "factual"

    def test_get_memory_weights(self):
        from qe.runtime.query_classifier import get_memory_weights

        weights = get_memory_weights("How do I configure this?")
        assert abs(sum(weights.values()) - 1.0) < 0.001
        assert weights["procedural"] > weights["semantic"]

    def test_get_memory_weights_for_type(self):
        from qe.runtime.query_classifier import get_memory_weights_for_type

        weights = get_memory_weights_for_type("creative")
        assert weights["working"] > weights["procedural"]

    def test_weights_sum_to_one(self):
        from qe.runtime.query_classifier import QUERY_TYPE_WEIGHTS

        for qtype, weights in QUERY_TYPE_WEIGHTS.items():
            assert abs(sum(weights.values()) - 1.0) < 0.001, f"{qtype} weights don't sum to 1"


# ═══════════════════════════════════════════════════════════════════════════════
# #55 — Checkpoint/resume for workflows
# ═══════════════════════════════════════════════════════════════════════════════


class TestWorkflowCheckpoints:
    @pytest.fixture
    def executor(self, tmp_path):
        from qe.runtime.workflow_executor import WorkflowExecutor

        return WorkflowExecutor(workflows_dir=str(tmp_path / "workflows"))

    def _make_workflow(self):
        from qe.models.workflow import WorkflowDefinition, WorkflowNode, WorkflowEdge

        return WorkflowDefinition(
            id="wf1",
            name="Test Workflow",
            nodes=[
                WorkflowNode(id="n1", type="standard", config={}),
                WorkflowNode(id="n2", type="standard", config={}),
            ],
            edges=[
                WorkflowEdge(from_node="n1", to_node="n2"),
            ],
        )

    def test_checkpoint_serialization(self, executor):
        from qe.runtime.workflow_executor import WorkflowExecution

        wf = self._make_workflow()
        ex = WorkflowExecution(wf)
        ex.status = "paused"
        ex.current_node = "n1"
        ex.node_outputs = {"n1": {"result": "ok"}}

        cp = executor.checkpoint(ex)
        assert cp["status"] == "paused"
        assert cp["current_node"] == "n1"
        assert cp["workflow_id"] == "wf1"

    def test_save_and_load_checkpoint(self, executor):
        from qe.runtime.workflow_executor import WorkflowExecution

        wf = self._make_workflow()
        ex = WorkflowExecution(wf)
        ex.status = "paused"

        executor.save_checkpoint(ex)
        loaded = executor.load_checkpoint(ex.id)
        assert loaded is not None
        assert loaded["status"] == "paused"

    def test_load_nonexistent_checkpoint(self, executor):
        assert executor.load_checkpoint("nonexistent") is None

    def test_list_checkpoints(self, executor):
        from qe.runtime.workflow_executor import WorkflowExecution

        wf = self._make_workflow()
        ex1 = WorkflowExecution(wf)
        ex2 = WorkflowExecution(wf)

        executor.save_checkpoint(ex1)
        executor.save_checkpoint(ex2)
        cps = executor.list_checkpoints()
        assert len(cps) == 2

    def test_delete_checkpoint(self, executor):
        from qe.runtime.workflow_executor import WorkflowExecution

        wf = self._make_workflow()
        ex = WorkflowExecution(wf)
        executor.save_checkpoint(ex)
        assert executor.delete_checkpoint(ex.id)
        assert executor.load_checkpoint(ex.id) is None

    def test_delete_nonexistent(self, executor):
        assert not executor.delete_checkpoint("nope")


# ═══════════════════════════════════════════════════════════════════════════════
# #68 — Multi-provider failover
# ═══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        from qe.runtime.llm import CircuitBreaker

        cb = CircuitBreaker()
        assert cb.state("model-a") == "closed"
        assert cb.is_available("model-a")

    def test_opens_after_threshold(self):
        from qe.runtime.llm import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("model-a")
        assert cb.state("model-a") == "closed"
        cb.record_failure("model-a")
        assert cb.state("model-a") == "open"
        assert not cb.is_available("model-a")

    def test_half_open_after_cooldown(self):
        from qe.runtime.llm import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure("model-a")
        assert cb.state("model-a") == "open"
        time.sleep(0.02)
        assert cb.state("model-a") == "half_open"
        assert cb.is_available("model-a")

    def test_success_resets(self):
        from qe.runtime.llm import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("model-a")
        cb.record_failure("model-a")
        assert cb.state("model-a") == "open"
        cb.record_success("model-a")
        assert cb.state("model-a") == "closed"

    def test_all_states(self):
        from qe.runtime.llm import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure("m1")
        cb.record_success("m2")
        states = cb.all_states()
        assert states["m1"] == "open"
        assert states["m2"] == "closed"


class TestFailoverChain:
    def test_available_models(self):
        from qe.runtime.llm import CircuitBreaker, FailoverChain

        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure("m2")
        chain = FailoverChain(["m1", "m2", "m3"], circuit_breaker=cb)
        assert chain.available_models() == ["m1", "m3"]

    @pytest.mark.asyncio
    async def test_failover_success(self):
        from qe.runtime.llm import FailoverChain, UnifiedLLM

        llm = UnifiedLLM(max_retries=0)
        chain = FailoverChain(["m1", "m2"], unified_llm=llm)

        mock_resp = MagicMock()
        mock_resp.usage = None

        call_models = []

        async def _fake_complete(messages, *, model=None, **kwargs):
            call_models.append(model)
            if model == "m1":
                raise RuntimeError("m1 down")
            return mock_resp

        with patch.object(llm, "complete", side_effect=_fake_complete):
            result = await chain.complete([{"role": "user", "content": "hi"}])
        assert result == mock_resp
        assert "m1" in call_models
        assert "m2" in call_models

    def test_status(self):
        from qe.runtime.llm import FailoverChain

        chain = FailoverChain(["m1", "m2"])
        s = chain.status()
        assert s["models"] == ["m1", "m2"]

    @pytest.mark.asyncio
    async def test_failover_no_llm_raises(self):
        from qe.runtime.llm import FailoverChain

        chain = FailoverChain(["m1"])
        with pytest.raises(RuntimeError, match="requires a UnifiedLLM"):
            await chain.complete([{"role": "user", "content": "hi"}])


# ═══════════════════════════════════════════════════════════════════════════════
# #63 — Filesystem artifact store
# ═══════════════════════════════════════════════════════════════════════════════


class TestFilesystemArtifactStore:
    @pytest.fixture
    def store(self, tmp_path):
        from qe.runtime.artifact_store import FilesystemArtifactStore

        return FilesystemArtifactStore(base_dir=tmp_path / "artifacts")

    def test_store_and_retrieve(self, store):
        handle = store.store("sess1", "art1", "big content here")
        assert handle == "[artifact:art1]"
        content = store.retrieve("sess1", "art1")
        assert content == "big content here"

    def test_retrieve_nonexistent(self, store):
        assert store.retrieve("sess1", "nope") is None

    def test_list_artifacts(self, store):
        store.store("sess1", "a1", "content 1")
        store.store("sess1", "a2", "content 2")
        arts = store.list_artifacts("sess1")
        assert len(arts) == 2

    def test_delete(self, store):
        store.store("sess1", "a1", "content")
        assert store.delete("sess1", "a1")
        assert store.retrieve("sess1", "a1") is None

    def test_delete_nonexistent(self, store):
        assert not store.delete("sess1", "nope")

    def test_ttl_expiry(self, store):
        store.store("sess1", "a1", "content", ttl_seconds=0)
        # TTL=0 means no expiry
        assert store.retrieve("sess1", "a1") is not None

    def test_cleanup_expired(self, store):
        # Store with very short TTL
        store.store("sess1", "a1", "old", ttl_seconds=1)
        store.store("sess1", "a2", "new", ttl_seconds=3600)
        time.sleep(1.1)
        removed = store.cleanup_expired()
        assert removed == 1
        assert store.retrieve("sess1", "a2") is not None

    def test_stats(self, store):
        store.store("s1", "a1", "content")
        s = store.stats()
        assert s["total_artifacts"] == 1
        assert s["sessions"] == 1

    def test_empty_stats(self, store):
        s = store.stats()
        assert s["total_artifacts"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# #59 — BM25 hybrid search
# ═══════════════════════════════════════════════════════════════════════════════


class TestBM25Reranking:
    def _make_claim(self, text):
        c = MagicMock()
        parts = text.split()
        c.subject_entity_id = parts[0] if parts else ""
        c.predicate = parts[1] if len(parts) > 1 else ""
        c.object_value = " ".join(parts[2:]) if len(parts) > 2 else ""
        return c

    def test_bm25_rerank_basic(self):
        from qe.substrate import _bm25_rerank

        claims = [
            self._make_claim("python is great programming language"),
            self._make_claim("java is also programming language"),
            self._make_claim("cat is furry animal"),
        ]
        result = _bm25_rerank(claims, "python programming")
        # Python claim should rank first
        assert result[0].subject_entity_id == "python"

    def test_bm25_rerank_empty_query(self):
        from qe.substrate import _bm25_rerank

        claims = [self._make_claim("test is claim")]
        result = _bm25_rerank(claims, "")
        assert len(result) == 1

    def test_bm25_rerank_empty_claims(self):
        from qe.substrate import _bm25_rerank

        result = _bm25_rerank([], "test query")
        assert result == []

    def test_bm25_configurable_params(self):
        from qe.substrate import _bm25_rerank

        claims = [
            self._make_claim("apple is fruit"),
            self._make_claim("banana is fruit"),
        ]
        result1 = _bm25_rerank(claims, "apple", k1=1.0, b=0.5)
        result2 = _bm25_rerank(claims, "apple", k1=2.0, b=1.0)
        assert result1[0].subject_entity_id == "apple"
        assert result2[0].subject_entity_id == "apple"


# ═══════════════════════════════════════════════════════════════════════════════
# #67 — Fast path bypass
# ═══════════════════════════════════════════════════════════════════════════════


class TestFastPathBypass:
    @pytest.fixture(autouse=True)
    def _reset_flags(self):
        from qe.runtime.feature_flags import reset_flag_store
        reset_flag_store()
        yield
        reset_flag_store()

    def _make_chat_service(self):
        """Create a minimal ChatService mock for fast path testing."""
        from qe.services.chat.service import ChatService

        svc = MagicMock(spec=ChatService)
        svc._fast_path_check = ChatService._fast_path_check.__get__(svc)
        svc._FAST_PATH_GREETINGS = ChatService._FAST_PATH_GREETINGS
        svc._FAST_PATH_ACKS = ChatService._FAST_PATH_ACKS
        return svc

    def test_fast_path_disabled_by_default(self):
        svc = self._make_chat_service()
        result = svc._fast_path_check("hello")
        assert result is None

    def test_fast_path_greeting(self):
        svc = self._make_chat_service()
        from qe.runtime.feature_flags import get_flag_store

        get_flag_store().define("fast_path_bypass", enabled=True)
        result = svc._fast_path_check("hello")
        assert result is not None
        assert "Hello" in result

    def test_fast_path_thanks(self):
        svc = self._make_chat_service()
        from qe.runtime.feature_flags import get_flag_store

        get_flag_store().define("fast_path_bypass", enabled=True)
        result = svc._fast_path_check("thanks")
        assert result is not None
        assert "welcome" in result.lower()

    def test_fast_path_ack(self):
        svc = self._make_chat_service()
        from qe.runtime.feature_flags import get_flag_store

        get_flag_store().define("fast_path_bypass", enabled=True)
        result = svc._fast_path_check("got it")
        assert result is not None

    def test_fast_path_long_message_skipped(self):
        svc = self._make_chat_service()
        from qe.runtime.feature_flags import get_flag_store

        get_flag_store().define("fast_path_bypass", enabled=True)
        result = svc._fast_path_check("x" * 200)
        assert result is None

    def test_fast_path_complex_query_skipped(self):
        svc = self._make_chat_service()
        from qe.runtime.feature_flags import get_flag_store

        get_flag_store().define("fast_path_bypass", enabled=True)
        result = svc._fast_path_check("What is the meaning of life?")
        assert result is None

    def test_fast_path_bye(self):
        svc = self._make_chat_service()
        from qe.runtime.feature_flags import get_flag_store

        get_flag_store().define("fast_path_bypass", enabled=True)
        result = svc._fast_path_check("bye")
        assert result is not None
        assert "Goodbye" in result

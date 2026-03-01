"""Tests for ProceduralMemory — CRUD, success_rate, ordering, in-memory mode."""

from __future__ import annotations

import pytest

from qe.runtime.procedural_memory import (
    ProceduralMemory,
    QuestionTemplate,
    ToolSequence,
)

# ── Model tests ─────────────────────────────────────────────────────────────


class TestQuestionTemplate:
    def test_success_rate_zero(self):
        t = QuestionTemplate(pattern="What is {X}?")
        assert t.success_rate == 0.0

    def test_success_rate(self):
        t = QuestionTemplate(pattern="Why {X}?", success_count=3, failure_count=1)
        assert t.success_rate == pytest.approx(0.75)


class TestToolSequence:
    def test_success_rate_zero(self):
        s = ToolSequence(tool_names=["web_search"])
        assert s.success_rate == 0.0

    def test_success_rate(self):
        s = ToolSequence(
            tool_names=["web_search", "code_exec"],
            success_count=7, failure_count=3,
        )
        assert s.success_rate == pytest.approx(0.7)


# ── In-memory mode tests ───────────────────────────────────────────────────


class TestProceduralMemoryInMemory:
    @pytest.mark.asyncio
    async def test_record_template_success(self):
        mem = ProceduralMemory()
        await mem.initialize()

        t = await mem.record_template_outcome(
            template_id="qt_1", pattern="What is {X}?",
            question_type="factual", success=True, info_gain=0.8,
        )
        assert t.success_count == 1
        assert t.failure_count == 0
        assert t.avg_info_gain == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_record_template_failure(self):
        mem = ProceduralMemory()
        await mem.initialize()

        t = await mem.record_template_outcome(
            template_id="qt_2", pattern="Why {X}?",
            question_type="causal", success=False, info_gain=0.1,
        )
        assert t.success_count == 0
        assert t.failure_count == 1

    @pytest.mark.asyncio
    async def test_record_sequence_success(self):
        mem = ProceduralMemory()
        await mem.initialize()

        s = await mem.record_sequence_outcome(
            sequence_id="ts_1", tool_names=["web_search"],
            success=True, cost_usd=0.01,
        )
        assert s.success_count == 1
        assert s.avg_cost_usd == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_get_best_templates(self):
        mem = ProceduralMemory()
        await mem.initialize()

        await mem.record_template_outcome(
            "qt_a", "Good pattern", "factual", True, 0.9, "general"
        )
        await mem.record_template_outcome(
            "qt_b", "Bad pattern", "factual", False, 0.1, "general"
        )

        best = await mem.get_best_templates("general", top_k=5)
        assert len(best) == 2
        assert best[0].template_id == "qt_a"  # Higher success rate

    @pytest.mark.asyncio
    async def test_get_best_sequences(self):
        mem = ProceduralMemory()
        await mem.initialize()

        await mem.record_sequence_outcome(
            "ts_a", ["web_search"], True, 0.01, "general"
        )
        await mem.record_sequence_outcome(
            "ts_b", ["code_exec"], False, 0.05, "general"
        )

        best = await mem.get_best_sequences("general", top_k=5)
        assert len(best) == 2
        assert best[0].sequence_id == "ts_a"


# ── SQLite mode tests ─────────────────────────────────────────────────────


class TestProceduralMemorySQLite:
    @pytest.mark.asyncio
    async def test_record_and_get_templates(self, tmp_path):
        db_path = str(tmp_path / "proc.db")
        mem = ProceduralMemory(db_path)
        await mem.initialize()

        await mem.record_template_outcome(
            "qt_x", "Pattern X", "causal", True, 0.7, "general"
        )
        await mem.record_template_outcome(
            "qt_x", "Pattern X", "causal", True, 0.9, "general"
        )

        best = await mem.get_best_templates("general")
        assert len(best) == 1
        assert best[0].success_count == 2
        assert best[0].avg_info_gain == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_record_and_get_sequences(self, tmp_path):
        db_path = str(tmp_path / "proc.db")
        mem = ProceduralMemory(db_path)
        await mem.initialize()

        await mem.record_sequence_outcome(
            "ts_x", ["web_search", "code_exec"], True, 0.02, "general"
        )

        best = await mem.get_best_sequences("general")
        assert len(best) == 1
        assert best[0].tool_names == ["web_search", "code_exec"]

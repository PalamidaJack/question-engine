"""Tests for Phases 6-12."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qe.sdk.service import ServiceBase, handles
from qe.sdk.testing import MockBus, ServiceTestHarness
from qe.sdk.tool import _SDK_TOOLS, tool
from qe.sdk.validate import validate_genome
from qe.services.analyst.service import AnalystService
from qe.services.coder.service import CoderService
from qe.services.digest.service import DigestService
from qe.services.fact_checker.service import FactCheckerService
from qe.services.memory.service import MemoryService
from qe.services.monitor.service import MonitorService
from qe.services.security.service import SecurityMonitor
from qe.services.writer.service import WriterService
from qe.substrate.memory_store import MemoryStore

# ── Phase 6: Memory Store ──────────────────────────────────────────


class TestMemoryStore:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmp) / "mem.db")
        self.store = MemoryStore(self.db_path)

    @pytest.mark.asyncio
    async def test_set_and_get_preference(self):
        entry = await self.store.set_preference("style", "brief")
        assert entry.memory_id.startswith("mem_")
        assert entry.key == "style"
        assert entry.value == "brief"
        prefs = await self.store.get_preferences()
        assert len(prefs) == 1
        assert prefs[0].value == "brief"

    @pytest.mark.asyncio
    async def test_supersede_preference(self):
        await self.store.set_preference("style", "brief")
        await self.store.set_preference("style", "detailed")
        prefs = await self.store.get_preferences()
        assert len(prefs) == 1
        assert prefs[0].value == "detailed"

    @pytest.mark.asyncio
    async def test_project_context(self):
        await self.store.set_project_context(
            "p1", "topic", "fusion energy"
        )
        ctx = await self.store.get_project_context("p1")
        assert len(ctx) == 1
        assert "fusion energy" in ctx[0].value

    @pytest.mark.asyncio
    async def test_entity_memory(self):
        await self.store.set_entity_memory(
            "entity_nasa", "type", "space agency"
        )
        memories = await self.store.get_entity_memories("entity_nasa")
        assert len(memories) == 1
        assert memories[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_delete(self):
        entry = await self.store.set_preference("key", "val")
        deleted = await self.store.delete(entry.memory_id)
        assert deleted is True
        assert await self.store.count() == 0

    @pytest.mark.asyncio
    async def test_count(self):
        await self.store.set_preference("a", "1")
        await self.store.set_preference("b", "2")
        assert await self.store.count() == 2

    @pytest.mark.asyncio
    async def test_create_project(self):
        proj = await self.store.create_project(
            "Fusion Research", "Studying fusion"
        )
        assert proj["project_id"].startswith("proj_")
        projects = await self.store.list_projects()
        assert len(projects) == 1
        assert projects[0]["name"] == "Fusion Research"

    @pytest.mark.asyncio
    async def test_get_all_active(self):
        await self.store.set_preference("a", "1")
        await self.store.set_preference("b", "2")
        entries = await self.store.get_all_active()
        assert len(entries) == 2


# ── Phase 6: Memory Service ───────────────────────────────────────


class TestMemoryService:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmp) / "mem.db")
        self.store = MemoryStore(self.db_path)
        self.svc = MemoryService(memory_store=self.store)

    @pytest.mark.asyncio
    async def test_enrichment_with_preferences(self):
        await self.store.set_preference("output_format", "detailed")
        ctx = await self.svc.get_enrichment_context()
        assert "preferences" in ctx
        assert ctx["preferences"][0]["key"] == "output_format"

    @pytest.mark.asyncio
    async def test_enrichment_empty(self):
        ctx = await self.svc.get_enrichment_context()
        assert ctx == {}

    @pytest.mark.asyncio
    async def test_format_context_for_prompt(self):
        await self.store.set_preference("style", "brief")
        ctx = await self.svc.get_enrichment_context()
        text = self.svc.format_context_for_prompt(ctx)
        assert "[USER PREFERENCES]" in text
        assert "style: brief" in text

    @pytest.mark.asyncio
    async def test_infer_entity_memory(self):
        # First mention — not stored yet
        await self.svc.infer_entity_memory(
            "nasa", "type", "space agency"
        )
        memories = await self.store.get_entity_memories("nasa")
        assert len(memories) == 0

        # Second mention — now stored
        await self.svc.infer_entity_memory(
            "nasa", "type", "space agency"
        )
        memories = await self.store.get_entity_memories("nasa")
        assert len(memories) == 1


# ── Phase 7: Service Types ────────────────────────────────────────


class TestFactCheckerService:
    def setup_method(self):
        self.svc = FactCheckerService()

    @pytest.mark.asyncio
    async def test_check_claim_basic(self):
        result = await self.svc.check_claim("The sky is blue")
        assert "verdict" in result
        assert "confidence" in result
        assert result["verdict"] in (
            "supported",
            "challenged",
            "insufficient_evidence",
        )

    @pytest.mark.asyncio
    async def test_find_contradictions(self):
        claims = [
            {"text": "The sky is blue", "confidence": 0.9},
            {"text": "The sky is red", "confidence": 0.5},
        ]
        result = await self.svc.find_contradictions(
            "The sky is green", claims
        )
        assert isinstance(result, list)


class TestAnalystService:
    @pytest.mark.asyncio
    async def test_analyze_claims(self):
        svc = AnalystService()
        claims = [{"text": "Claim A"}, {"text": "Claim B"}]
        result = await svc.analyze_claims(claims, "trend")
        assert "analysis_type" in result
        assert "findings" in result


class TestWriterService:
    @pytest.mark.asyncio
    async def test_generate_document(self):
        svc = WriterService()
        claims = [{"text": "Important fact"}]
        result = await svc.generate_document(claims, format="summary")
        assert "content" in result
        assert "word_count" in result


class TestCoderService:
    @pytest.mark.asyncio
    async def test_execute_task(self):
        svc = CoderService()
        result = await svc.execute_task("Calculate 2+2")
        assert "code" in result
        assert "success" in result


class TestMonitorService:
    def setup_method(self):
        self.svc = MonitorService()

    @pytest.mark.asyncio
    async def test_add_schedule(self):
        result = await self.svc.add_schedule(
            "Check news", interval="daily"
        )
        assert "monitor_id" in result
        assert result["active"] is True

    @pytest.mark.asyncio
    async def test_list_schedules(self):
        await self.svc.add_schedule("Check A")
        await self.svc.add_schedule("Check B")
        schedules = await self.svc.list_schedules()
        assert len(schedules) == 2

    @pytest.mark.asyncio
    async def test_remove_schedule(self):
        result = await self.svc.add_schedule("Check C")
        removed = await self.svc.remove_schedule(result["monitor_id"])
        assert removed is True
        schedules = await self.svc.list_schedules()
        assert len(schedules) == 0


class TestDigestService:
    @pytest.mark.asyncio
    async def test_generate_digest(self):
        svc = DigestService()
        digest = await svc.generate_daily_digest()
        assert "system_health" in digest
        assert "generated_at" in digest


class TestSecurityMonitor:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.svc = SecurityMonitor(genome_dir=self.tmp)

    @pytest.mark.asyncio
    async def test_record_baseline(self):
        # Create a test genome file
        genome_path = Path(self.tmp) / "test.toml"
        genome_path.write_text('[service]\nservice_id = "test"')
        await self.svc.record_baseline()
        assert len(self.svc._known_hashes) == 1
        key = next(iter(self.svc._known_hashes))
        assert key.endswith("test.toml")

    @pytest.mark.asyncio
    async def test_integrity_check_clean(self):
        genome_path = Path(self.tmp) / "test.toml"
        genome_path.write_text('[service]\nservice_id = "test"')
        await self.svc.record_baseline()
        alerts = await self.svc.integrity_check()
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_integrity_check_modified(self):
        genome_path = Path(self.tmp) / "test.toml"
        genome_path.write_text('[service]\nservice_id = "test"')
        await self.svc.record_baseline()
        # Modify the file
        genome_path.write_text('[service]\nservice_id = "hacked"')
        alerts = await self.svc.integrity_check()
        assert len(alerts) == 1

    @pytest.mark.asyncio
    async def test_behavioral_audit(self):
        result = await self.svc.behavioral_audit()
        assert isinstance(result, list)


# ── Phase 12: SDK ──────────────────────────────────────────────────


class TestSDKService:
    def test_handler_discovery(self):
        class MyService(ServiceBase):
            @handles("test.topic")
            async def handle_test(self, envelope):
                return "handled"

        svc = MyService()
        assert "test.topic" in svc.get_subscribed_topics()

    @pytest.mark.asyncio
    async def test_start_stop(self):
        svc = ServiceBase()
        await svc.start()
        assert svc.is_running is True
        await svc.stop()
        assert svc.is_running is False


class TestSDKTool:
    def test_tool_decorator(self):
        initial_count = len(_SDK_TOOLS)

        @tool(name="test_sdk_tool", description="A test tool")
        async def my_tool(query: str):
            return {"result": query}

        assert len(_SDK_TOOLS) == initial_count + 1
        assert hasattr(my_tool, "_tool_spec")
        assert my_tool._tool_spec.name == "test_sdk_tool"


class TestSDKTestHarness:
    @pytest.mark.asyncio
    async def test_mock_bus(self):
        bus = MockBus()
        bus.publish(MagicMock(topic="test.topic"))
        assert len(bus.published) == 1

    @pytest.mark.asyncio
    async def test_harness_start_stop(self):
        harness = ServiceTestHarness()
        svc = await harness.start_service(ServiceBase)
        assert svc.is_running is True
        await harness.stop()
        assert svc.is_running is False


class TestGenomeValidation:
    def test_missing_file(self):
        errors = validate_genome("/nonexistent/file.toml")
        assert any(e.field == "path" for e in errors)

    def test_valid_genome(self):
        tmp = tempfile.mkdtemp()
        genome = Path(tmp) / "test.toml"
        genome.write_text(
            '[service]\nservice_id = "test"\n\n'
            '[prompts]\nbalanced = "You are a test service."'
        )
        errors = validate_genome(str(genome))
        assert len(errors) == 0

    def test_missing_service_section(self):
        tmp = tempfile.mkdtemp()
        genome = Path(tmp) / "test.toml"
        genome.write_text(
            '[prompts]\nbalanced = "You are a test service."'
        )
        errors = validate_genome(str(genome))
        assert any(e.field == "service" for e in errors)

    def test_missing_prompts(self):
        tmp = tempfile.mkdtemp()
        genome = Path(tmp) / "test.toml"
        genome.write_text(
            '[service]\nservice_id = "test"'
        )
        errors = validate_genome(str(genome))
        assert any(e.field == "prompts" for e in errors)

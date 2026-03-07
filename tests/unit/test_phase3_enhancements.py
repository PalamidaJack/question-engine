"""Tests for Phase 3 enhancements.

Covers:
- #48 Cognitive tool personas
- #49 Orchestrator handoff rules
- #73 Verification protocol
- #74 Two-stage review
- #72 Structured design-plan-execute workflow
- #80 RPI methodology
- #79 Constraint-based guardrails
- #51 Specialist profile playbooks
- #81 /common-ground command
- #86 Auto session memory extraction
"""

from __future__ import annotations

import asyncio

import pytest

from qe.runtime.feature_flags import get_flag_store, reset_flag_store


@pytest.fixture(autouse=True)
def _reset_flags():
    """Reset flag store before and after each test."""
    reset_flag_store()
    yield
    reset_flag_store()


# ── #48 Cognitive Tool Personas ──────────────────────────────────────────


class TestCognitivePersonas:
    def test_builtin_personas_exist(self):
        from qe.runtime.persona import BUILTIN_PERSONAS

        assert "researcher" in BUILTIN_PERSONAS
        assert "analyst" in BUILTIN_PERSONAS
        assert "coder" in BUILTIN_PERSONAS
        assert "synthesizer" in BUILTIN_PERSONAS

    def test_persona_manager_get(self):
        from qe.runtime.persona import PersonaManager

        mgr = PersonaManager()
        p = mgr.get_persona("researcher")
        assert p is not None
        assert p.name == "researcher"
        assert "web_search" in p.tool_categories

    def test_persona_for_tool_category(self):
        from qe.runtime.persona import PersonaManager

        mgr = PersonaManager()
        p = mgr.get_persona_for_tool("web_search")
        assert p is not None
        assert p.name == "researcher"

    def test_persona_for_unknown_tool(self):
        from qe.runtime.persona import PersonaManager

        mgr = PersonaManager()
        assert mgr.get_persona_for_tool("nonexistent_tool") is None

    def test_register_custom_persona(self):
        from qe.runtime.persona import CognitivePersona, PersonaManager

        mgr = PersonaManager()
        custom = CognitivePersona(
            name="custom",
            description="Custom persona",
            tool_categories=["custom_tool"],
            system_addendum="Be custom.",
        )
        mgr.register(custom)
        assert mgr.get_persona("custom") is not None
        assert mgr.get_persona_for_tool("custom_tool").name == "custom"

    def test_list_personas(self):
        from qe.runtime.persona import PersonaManager

        mgr = PersonaManager()
        listing = mgr.list_personas()
        assert len(listing) >= 4
        names = [p["name"] for p in listing]
        assert "researcher" in names

    def test_persona_style_attribute(self):
        from qe.runtime.persona import BUILTIN_PERSONAS

        assert BUILTIN_PERSONAS["coder"].style == "concise"
        assert BUILTIN_PERSONAS["researcher"].style == "thorough"


# ── #49 Orchestrator Handoff Rules ──────────────────────────────────────


class TestOrchestratorHandoff:
    def test_default_rules_exist(self):
        from qe.runtime.orchestrator import DEFAULT_RULES

        assert len(DEFAULT_RULES) >= 4
        names = [r.name for r in DEFAULT_RULES]
        assert "compare_to_swarm" in names

    def test_evaluate_comparison_query(self):
        from qe.runtime.orchestrator import ToolOrchestrator

        orch = ToolOrchestrator()
        rule = orch.evaluate("Compare Python vs JavaScript")
        assert rule is not None
        assert rule.name == "compare_to_swarm"
        assert rule.action == "swarm"

    def test_evaluate_no_match(self):
        from qe.runtime.orchestrator import ToolOrchestrator

        orch = ToolOrchestrator()
        rule = orch.evaluate("Hello world")
        assert rule is None

    def test_evaluate_fallback(self):
        from qe.runtime.orchestrator import ToolOrchestrator

        orch = ToolOrchestrator()
        rule = orch.evaluate("web_search failed with error")
        assert rule is not None
        assert rule.action == "fallback"

    def test_evaluate_deep_research(self):
        from qe.runtime.orchestrator import ToolOrchestrator

        orch = ToolOrchestrator()
        rule = orch.evaluate("Do a thorough analysis of this topic")
        assert rule is not None
        assert rule.name == "deep_research"

    def test_evaluate_all(self):
        from qe.runtime.orchestrator import ToolOrchestrator

        orch = ToolOrchestrator()
        rules = orch.evaluate_all(
            "Do a thorough comparison vs alternatives"
        )
        names = [r.name for r in rules]
        assert "compare_to_swarm" in names
        assert "deep_research" in names

    def test_add_custom_rule(self):
        from qe.runtime.orchestrator import HandoffRule, ToolOrchestrator

        orch = ToolOrchestrator()
        custom = HandoffRule(
            name="custom",
            trigger=r"\bcustom_trigger\b",
            action="route_to",
            target="custom_tool",
            priority=100,
        )
        orch.add_rule(custom)
        rule = orch.evaluate("custom_trigger here")
        assert rule is not None
        assert rule.name == "custom"

    def test_list_rules(self):
        from qe.runtime.orchestrator import ToolOrchestrator

        orch = ToolOrchestrator()
        listing = orch.list_rules()
        assert len(listing) >= 4
        assert all("name" in r for r in listing)

    def test_handoff_rule_matches(self):
        from qe.runtime.orchestrator import HandoffRule

        rule = HandoffRule(
            name="test", trigger=r"\btest\b",
            action="route_to", target="x",
        )
        assert rule.matches("this is a test")
        assert not rule.matches("nothing here")

    def test_priority_ordering(self):
        from qe.runtime.orchestrator import ToolOrchestrator

        orch = ToolOrchestrator()
        listing = orch.list_rules()
        priorities = [r["priority"] for r in listing]
        assert priorities == sorted(priorities, reverse=True)


# ── #73 Verification Protocol ───────────────────────────────────────────


class TestVerificationProtocol:
    def test_empty_output_fails(self):
        from qe.runtime.verification import ToolVerifier

        verifier = ToolVerifier()
        result = asyncio.run(
            verifier.verify("What is Python?", "query", "")
        )
        assert not result.passed
        assert any("empty" in i.lower() for i in result.issues)

    def test_short_output_fails(self):
        from qe.runtime.verification import ToolVerifier

        verifier = ToolVerifier()
        result = asyncio.run(
            verifier.verify("What is Python?", "query", "short")
        )
        assert not result.passed

    def test_good_output_passes(self):
        from qe.runtime.verification import ToolVerifier

        verifier = ToolVerifier()
        result = asyncio.run(
            verifier.verify(
                "What is Python?",
                "query",
                "Python is a high-level programming language "
                "known for its simplicity and readability.",
            )
        )
        assert result.passed
        assert result.recommendation == "accept"

    def test_hedging_detected(self):
        from qe.runtime.verification import ToolVerifier

        verifier = ToolVerifier()
        result = asyncio.run(
            verifier.verify(
                "What is Python?",
                "query",
                "As an AI, I think Python might be a language.",
            )
        )
        assert not result.passed
        assert any("hedging" in i.lower() for i in result.issues)

    def test_low_relevance_detected(self):
        from qe.runtime.verification import ToolVerifier

        verifier = ToolVerifier()
        result = asyncio.run(
            verifier.verify(
                "quantum entanglement superconductors",
                "query",
                "Bananas mangoes papayas pineapples guavas "
                "dragonfruit lychee starfruit kumquat persimmon.",
            )
        )
        assert not result.passed
        assert any("overlap" in i.lower() for i in result.issues)

    def test_confidence_decreases_with_issues(self):
        from qe.runtime.verification import ToolVerifier

        verifier = ToolVerifier()
        result = asyncio.run(
            verifier.verify("test", "query", "")
        )
        assert result.confidence < 1.0

    def test_reinvoke_recommendation(self):
        from qe.runtime.verification import ToolVerifier

        verifier = ToolVerifier()
        result = asyncio.run(
            verifier.verify(
                "complex question about many things?",
                "query",
                "As an AI, I don't have access. In theory it is.",
            )
        )
        assert result.recommendation in ("re_invoke", "flag")


# ── #74 Two-Stage Review ────────────────────────────────────────────────


class TestTwoStageReview:
    def test_empty_results(self):
        from qe.runtime.verification import TwoStageReviewer

        reviewer = TwoStageReviewer()
        result = asyncio.run(
            reviewer.review_merged("test", [])
        )
        assert result["consistency_score"] == 0.0
        assert result["recommendation"] == "re_invoke"

    def test_consistent_results(self):
        from qe.runtime.verification import TwoStageReviewer

        reviewer = TwoStageReviewer()
        results = [
            {"summary": "Python is a programming language used widely"},
            {"summary": "Python is a popular programming language"},
        ]
        result = asyncio.run(
            reviewer.review_merged("What is Python?", results)
        )
        assert result["consistency_score"] > 0.5
        assert result["results_reviewed"] == 2

    def test_contradictory_results(self):
        from qe.runtime.verification import TwoStageReviewer

        reviewer = TwoStageReviewer()
        results = [
            {"summary": "alpha beta gamma delta epsilon zeta"},
            {"summary": "uno dos tres cuatro cinco seis"},
        ]
        result = asyncio.run(
            reviewer.review_merged("test", results)
        )
        assert len(result["issues"]) > 0

    def test_single_result(self):
        from qe.runtime.verification import TwoStageReviewer

        reviewer = TwoStageReviewer()
        results = [{"summary": "Single result here"}]
        result = asyncio.run(
            reviewer.review_merged("test", results)
        )
        assert result["consistency_score"] == 1.0
        assert result["recommendation"] == "accept"


# ── #72 Structured Workflow ─────────────────────────────────────────────


class TestStructuredWorkflow:
    def test_initial_phase(self):
        from qe.services.chat.structured_workflow import (
            StructuredWorkflow,
            WorkflowPhase,
        )

        wf = StructuredWorkflow(goal="Test")
        assert wf.current_phase == WorkflowPhase.DESIGN
        assert not wf.completed

    def test_advance_through_phases(self):
        from qe.services.chat.structured_workflow import (
            PhaseResult,
            StructuredWorkflow,
            WorkflowPhase,
        )

        wf = StructuredWorkflow(goal="Test")
        next_phase = wf.advance(PhaseResult(phase="design"))
        assert next_phase == WorkflowPhase.PLAN

        next_phase = wf.advance(PhaseResult(phase="plan"))
        assert next_phase == WorkflowPhase.EXECUTE

        next_phase = wf.advance(PhaseResult(phase="execute"))
        assert next_phase is None
        assert wf.completed

    def test_exit_criteria_not_met(self):
        from qe.services.chat.structured_workflow import (
            PhaseResult,
            StructuredWorkflow,
            WorkflowPhase,
        )

        wf = StructuredWorkflow(goal="Test")
        next_phase = wf.advance(
            PhaseResult(
                phase="design",
                exit_criteria_met=False,
                issues=["Requirements unclear"],
            )
        )
        assert next_phase == WorkflowPhase.DESIGN
        assert not wf.completed

    def test_phase_prompts(self):
        from qe.services.chat.structured_workflow import (
            StructuredWorkflow,
        )

        wf = StructuredWorkflow(goal="Test")
        prompt = wf.get_phase_prompt()
        assert "DESIGN" in prompt

    def test_status(self):
        from qe.services.chat.structured_workflow import (
            StructuredWorkflow,
        )

        wf = StructuredWorkflow(goal="Test goal")
        status = wf.status()
        assert status["goal"] == "Test goal"
        assert status["current_phase"] == "design"
        assert not status["completed"]

    def test_phases_recorded(self):
        from qe.services.chat.structured_workflow import (
            PhaseResult,
            StructuredWorkflow,
        )

        wf = StructuredWorkflow(goal="Test")
        wf.advance(PhaseResult(phase="design"))
        assert "design" in wf.phases


# ── #80 RPI Methodology ─────────────────────────────────────────────────


class TestRPIWorkflow:
    def test_initial_phase(self):
        from qe.services.chat.structured_workflow import (
            RPIWorkflow,
            WorkflowPhase,
        )

        wf = RPIWorkflow(goal="Test")
        assert wf.current_phase == WorkflowPhase.RESEARCH

    def test_advance_through_rpi(self):
        from qe.services.chat.structured_workflow import (
            PhaseResult,
            RPIWorkflow,
            WorkflowPhase,
        )

        wf = RPIWorkflow(goal="Test")
        next_phase = wf.advance(PhaseResult(phase="research"))
        assert next_phase == WorkflowPhase.PLAN

        next_phase = wf.advance(PhaseResult(phase="plan"))
        assert next_phase == WorkflowPhase.IMPLEMENT

        next_phase = wf.advance(PhaseResult(phase="implement"))
        assert next_phase is None
        assert wf.completed

    def test_rpi_phase_prompts(self):
        from qe.services.chat.structured_workflow import RPIWorkflow

        wf = RPIWorkflow(goal="Test")
        prompt = wf.get_phase_prompt()
        assert "RESEARCH" in prompt
        assert "confidence" in prompt.lower()

    def test_rpi_exit_criteria_not_met(self):
        from qe.services.chat.structured_workflow import (
            PhaseResult,
            RPIWorkflow,
            WorkflowPhase,
        )

        wf = RPIWorkflow(goal="Test")
        next_phase = wf.advance(
            PhaseResult(phase="research", exit_criteria_met=False)
        )
        assert next_phase == WorkflowPhase.RESEARCH

    def test_confidence_in_phase_result(self):
        from qe.services.chat.structured_workflow import PhaseResult

        result = PhaseResult(phase="research", confidence=0.95)
        assert result.confidence == 0.95


# ── #79 Constraint-Based Guardrails ─────────────────────────────────────


class TestConstraintGuardrails:
    def test_initial_session_state(self):
        from qe.runtime.guardrails import ConstraintGuardrails

        cg = ConstraintGuardrails()
        status = cg.session_status("s1")
        assert status["tool_calls"] == 0
        assert status["cost_usd"] == 0.0
        assert status["tokens_used"] == 0

    def test_record_tool_calls(self):
        from qe.runtime.guardrails import ConstraintGuardrails

        cg = ConstraintGuardrails(max_tool_calls=3)
        cg.record_tool_call("s1")
        cg.record_tool_call("s1")
        assert cg.check_tool_calls("s1").passed
        cg.record_tool_call("s1")
        assert not cg.check_tool_calls("s1").passed

    def test_record_cost(self):
        from qe.runtime.guardrails import ConstraintGuardrails

        cg = ConstraintGuardrails(max_cost_usd=1.0)
        cg.record_cost("s1", 0.5)
        assert cg.check_cost("s1").passed
        cg.record_cost("s1", 0.6)
        assert not cg.check_cost("s1").passed

    def test_record_tokens(self):
        from qe.runtime.guardrails import ConstraintGuardrails

        cg = ConstraintGuardrails(max_tokens=1000)
        cg.record_tokens("s1", 500)
        assert cg.check_tokens("s1").passed
        cg.record_tokens("s1", 600)
        assert not cg.check_tokens("s1").passed

    def test_domain_blocking(self):
        from qe.runtime.guardrails import ConstraintGuardrails

        cg = ConstraintGuardrails(
            blocked_domains=["evil.com"]
        )
        assert not cg.check_domain("https://evil.com/page").passed
        assert cg.check_domain("https://good.com/page").passed

    def test_domain_allowlist(self):
        from qe.runtime.guardrails import ConstraintGuardrails

        cg = ConstraintGuardrails(
            allowed_domains=["trusted.com"]
        )
        assert cg.check_domain("https://trusted.com/api").passed
        assert not cg.check_domain("https://other.com/api").passed

    def test_check_all(self):
        from qe.runtime.guardrails import ConstraintGuardrails

        cg = ConstraintGuardrails(max_tool_calls=1)
        cg.record_tool_call("s1")
        results = cg.check_all("s1")
        assert len(results) == 3
        blocked = [r for r in results if not r.passed]
        assert len(blocked) == 1

    def test_reset_session(self):
        from qe.runtime.guardrails import ConstraintGuardrails

        cg = ConstraintGuardrails()
        cg.record_tool_call("s1")
        cg.reset_session("s1")
        status = cg.session_status("s1")
        assert status["tool_calls"] == 0

    def test_isolated_sessions(self):
        from qe.runtime.guardrails import ConstraintGuardrails

        cg = ConstraintGuardrails()
        cg.record_tool_call("s1")
        cg.record_tool_call("s1")
        assert cg.session_status("s1")["tool_calls"] == 2
        assert cg.session_status("s2")["tool_calls"] == 0

    def test_extract_domain(self):
        from qe.runtime.guardrails import _extract_domain

        assert _extract_domain("https://example.com/path") == "example.com"
        assert _extract_domain("http://sub.example.com:8080/p") == "sub.example.com"
        assert _extract_domain("example.com/path") == "example.com"


# ── #51 Specialist Profile Playbooks ────────────────────────────────────


class TestSpecialistProfiles:
    def test_builtin_specialists(self):
        from qe.runtime.profiles import BUILTIN_SPECIALISTS

        assert len(BUILTIN_SPECIALISTS) >= 8
        assert "analyst" in BUILTIN_SPECIALISTS
        assert "researcher" in BUILTIN_SPECIALISTS
        assert "coder" in BUILTIN_SPECIALISTS
        assert "strategist" in BUILTIN_SPECIALISTS
        assert "critic" in BUILTIN_SPECIALISTS
        assert "synthesizer" in BUILTIN_SPECIALISTS
        assert "factchecker" in BUILTIN_SPECIALISTS
        assert "creative" in BUILTIN_SPECIALISTS

    def test_specialist_manager_get(self):
        from qe.runtime.profiles import SpecialistManager

        mgr = SpecialistManager()
        s = mgr.get("analyst")
        assert s is not None
        assert s.name == "analyst"
        assert "query_beliefs" in s.tool_whitelist

    def test_specialist_manager_list(self):
        from qe.runtime.profiles import SpecialistManager

        mgr = SpecialistManager()
        listing = mgr.list_specialists()
        assert len(listing) >= 8
        assert all("name" in s for s in listing)

    def test_get_by_tag(self):
        from qe.runtime.profiles import SpecialistManager

        mgr = SpecialistManager()
        results = mgr.get_by_tag("research")
        assert any(s.name == "researcher" for s in results)

    def test_register_custom_specialist(self):
        from qe.runtime.profiles import SpecialistManager, SpecialistProfile

        mgr = SpecialistManager()
        custom = SpecialistProfile(
            name="custom_specialist",
            description="Custom",
            system_prompt="Be custom.",
            tool_whitelist=["query_beliefs"],
        )
        mgr.register(custom)
        assert mgr.get("custom_specialist") is not None

    def test_specialist_to_dict(self):
        from qe.runtime.profiles import BUILTIN_SPECIALISTS

        d = BUILTIN_SPECIALISTS["analyst"].to_dict()
        assert d["name"] == "analyst"
        assert "tool_whitelist" in d
        assert "style" in d

    def test_specialist_names(self):
        from qe.runtime.profiles import SpecialistManager

        mgr = SpecialistManager()
        names = mgr.names()
        assert "analyst" in names
        assert "researcher" in names

    def test_specialist_temperature(self):
        from qe.runtime.profiles import BUILTIN_SPECIALISTS

        assert BUILTIN_SPECIALISTS["coder"].temperature == 0.3
        assert BUILTIN_SPECIALISTS["creative"].temperature == 0.9


# ── #81 /common-ground Command ──────────────────────────────────────────


class TestCommonGround:
    def _make_chat_service(self):
        """Create a minimal ChatService for testing."""
        from unittest.mock import MagicMock

        from qe.services.chat import ChatService

        svc = ChatService.__new__(ChatService)
        svc._sessions = {}
        svc._session_metadata = {}
        svc._lock = asyncio.Lock()
        svc.model = "test-model"
        svc.substrate = MagicMock()
        svc.bus = None
        svc.budget_tracker = None
        svc._procedural_memory = None
        svc._episodic_memory = None
        svc._guardrails = None
        svc._sanitizer = None
        svc._chat_store = None
        svc._current_session = None
        return svc

    def test_common_ground_disabled(self):
        svc = self._make_chat_service()
        result = asyncio.run(
            svc.common_ground("s1")
        )
        assert "error" in result

    def test_common_ground_no_session(self):
        store = get_flag_store()
        store.define("structured_workflow", enabled=True)
        svc = self._make_chat_service()
        result = asyncio.run(
            svc.common_ground("nonexistent")
        )
        assert result["assumptions"] == []

    def test_common_ground_with_conversation(self):
        from unittest.mock import MagicMock

        store = get_flag_store()
        store.define("structured_workflow", enabled=True)

        svc = self._make_chat_service()
        session = MagicMock()
        session.history = [
            {"role": "user", "content": "Python is a great language."},
            {"role": "user", "content": "I think we should use FastAPI."},
            {"role": "user", "content": "What about performance?"},
        ]
        svc._sessions["s1"] = session

        result = asyncio.run(
            svc.common_ground("s1")
        )
        assert len(result["assumptions"]) > 0 or len(result["premises"]) > 0
        assert len(result["gaps"]) > 0  # question detected
        assert result["summary"]


# ── #86 Auto Session Memory Extraction ──────────────────────────────────


class TestAutoSessionMemory:
    def _make_chat_service(self):
        from unittest.mock import MagicMock

        from qe.services.chat import ChatService

        svc = ChatService.__new__(ChatService)
        svc._sessions = {}
        svc._session_metadata = {}
        svc._lock = asyncio.Lock()
        svc.model = "test-model"
        svc.substrate = MagicMock()
        svc.bus = None
        svc.budget_tracker = None
        svc._procedural_memory = None
        svc._episodic_memory = None
        svc._guardrails = None
        svc._sanitizer = None
        svc._chat_store = None
        svc._current_session = None
        return svc

    def test_disabled_returns_empty(self):
        svc = self._make_chat_service()
        result = asyncio.run(
            svc.extract_session_memory("s1")
        )
        assert result["extracted"] == 0

    def test_no_session_returns_empty(self):
        store = get_flag_store()
        store.define("auto_session_memory", enabled=True)

        svc = self._make_chat_service()
        result = asyncio.run(
            svc.extract_session_memory("nonexistent")
        )
        assert result["extracted"] == 0

    def test_extract_decisions(self):
        from unittest.mock import MagicMock

        store = get_flag_store()
        store.define("auto_session_memory", enabled=True)

        svc = self._make_chat_service()
        session = MagicMock()
        session.history = [
            {"role": "user", "content": "We decided to use PostgreSQL."},
            {"role": "assistant", "content": "Great choice."},
        ]
        svc._sessions["s1"] = session

        result = asyncio.run(
            svc.extract_session_memory("s1")
        )
        assert result["extracted"] >= 1
        types = [i["type"] for i in result["items"]]
        assert "decision" in types

    def test_extract_facts(self):
        from unittest.mock import MagicMock

        store = get_flag_store()
        store.define("auto_session_memory", enabled=True)

        svc = self._make_chat_service()
        session = MagicMock()
        session.history = [
            {
                "role": "assistant",
                "content": "I found that the API returns JSON.",
            },
        ]
        svc._sessions["s1"] = session

        result = asyncio.run(
            svc.extract_session_memory("s1")
        )
        assert result["extracted"] >= 1
        types = [i["type"] for i in result["items"]]
        assert "fact" in types

    def test_extract_preferences(self):
        from unittest.mock import MagicMock

        store = get_flag_store()
        store.define("auto_session_memory", enabled=True)

        svc = self._make_chat_service()
        session = MagicMock()
        session.history = [
            {"role": "user", "content": "I prefer dark mode always."},
        ]
        svc._sessions["s1"] = session

        result = asyncio.run(
            svc.extract_session_memory("s1")
        )
        assert result["extracted"] >= 1
        types = [i["type"] for i in result["items"]]
        assert "preference" in types

    def test_stores_to_episodic_memory(self):
        from unittest.mock import AsyncMock, MagicMock

        store = get_flag_store()
        store.define("auto_session_memory", enabled=True)

        svc = self._make_chat_service()
        svc._episodic_memory = MagicMock()
        svc._episodic_memory.store = AsyncMock()

        session = MagicMock()
        session.history = [
            {"role": "user", "content": "We decided to use Redis."},
        ]
        svc._sessions["s1"] = session

        asyncio.run(
            svc.extract_session_memory("s1")
        )
        assert svc._episodic_memory.store.called


# ── Phase 3 Feature Flags ──────────────────────────────────────────────


class TestPhase3FeatureFlags:
    def test_phase3_flags_defined(self):
        """Verify Phase 3 flags are defined when app creates them."""
        store = get_flag_store()
        phase3_flags = [
            "cognitive_personas",
            "orchestrator_handoff",
            "specialist_profiles",
            "verification_protocol",
            "two_stage_review",
            "structured_workflow",
            "rpi_methodology",
            "auto_session_memory",
            "constraint_guardrails",
        ]
        for flag_name in phase3_flags:
            store.define(flag_name, enabled=False)
            assert not store.is_enabled(flag_name)

    def test_phase3_flags_toggleable(self):
        store = get_flag_store()
        store.define("cognitive_personas", enabled=False)
        store.enable("cognitive_personas")
        assert store.is_enabled("cognitive_personas")


# ── Integration: module imports ─────────────────────────────────────────


class TestPhase3Imports:
    def test_import_persona(self):
        from qe.runtime.persona import (
            BUILTIN_PERSONAS,
            CognitivePersona,
            PersonaManager,
        )

        assert CognitivePersona
        assert PersonaManager
        assert BUILTIN_PERSONAS

    def test_import_orchestrator(self):
        from qe.runtime.orchestrator import (
            DEFAULT_RULES,
            HandoffRule,
            ToolOrchestrator,
        )

        assert HandoffRule
        assert ToolOrchestrator
        assert DEFAULT_RULES

    def test_import_verification(self):
        from qe.runtime.verification import (
            ToolVerifier,
            TwoStageReviewer,
            VerificationResult,
        )

        assert VerificationResult
        assert ToolVerifier
        assert TwoStageReviewer

    def test_import_structured_workflow(self):
        from qe.services.chat.structured_workflow import (
            PhaseResult,
            RPIWorkflow,
            StructuredWorkflow,
            WorkflowPhase,
        )

        assert WorkflowPhase
        assert PhaseResult
        assert StructuredWorkflow
        assert RPIWorkflow

    def test_import_constraint_guardrails(self):
        from qe.runtime.guardrails import ConstraintGuardrails

        assert ConstraintGuardrails

    def test_import_specialist_profiles(self):
        from qe.runtime.profiles import (
            BUILTIN_SPECIALISTS,
            SpecialistManager,
            SpecialistProfile,
        )

        assert SpecialistProfile
        assert SpecialistManager
        assert BUILTIN_SPECIALISTS

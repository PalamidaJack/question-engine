"""Tests for Phase 5 enhancements.

Covers:
- #77 4-tier artifact system
- #64 Composable workflow skills
- #82 Multi-skill chaining
- #89 Skill catalog
- #62 Progressive tool loading
- #66 Learning loop
- #70 BDI mental state tracking
"""

from __future__ import annotations

import asyncio

import pytest

from qe.runtime.feature_flags import reset_flag_store


@pytest.fixture(autouse=True)
def _reset_flags():
    reset_flag_store()
    yield
    reset_flag_store()


# ── #77 Artifact System ─────────────────────────────────────────────────


class TestArtifactSystem:
    def test_register_and_get(self):
        from qe.runtime.artifacts import Artifact, ArtifactRegistry

        reg = ArtifactRegistry()
        a = Artifact("a1", "Test Prompt", "prompt", description="A test")
        reg.register(a)
        assert reg.get("a1") is not None
        assert reg.get("a1").name == "Test Prompt"

    def test_invalid_type(self):
        from qe.runtime.artifacts import Artifact, ArtifactRegistry

        reg = ArtifactRegistry()
        a = Artifact("a1", "Bad", "invalid_type")
        with pytest.raises(ValueError):
            reg.register(a)

    def test_list_by_type(self):
        from qe.runtime.artifacts import Artifact, ArtifactRegistry

        reg = ArtifactRegistry()
        reg.register(Artifact("a1", "P1", "prompt"))
        reg.register(Artifact("a2", "P2", "prompt"))
        reg.register(Artifact("a3", "S1", "skill"))
        prompts = reg.list_by_type("prompt")
        assert len(prompts) == 2

    def test_list_by_tag(self):
        from qe.runtime.artifacts import Artifact, ArtifactRegistry

        reg = ArtifactRegistry()
        reg.register(
            Artifact("a1", "P1", "prompt", tags=["research"])
        )
        reg.register(Artifact("a2", "P2", "prompt", tags=["code"]))
        results = reg.list_by_tag("research")
        assert len(results) == 1

    def test_search(self):
        from qe.runtime.artifacts import Artifact, ArtifactRegistry

        reg = ArtifactRegistry()
        reg.register(
            Artifact(
                "a1", "Research Helper", "prompt",
                description="Helps with research tasks",
            )
        )
        results = reg.search("research")
        assert len(results) == 1

    def test_dependency_resolution(self):
        from qe.runtime.artifacts import Artifact, ArtifactRegistry

        reg = ArtifactRegistry()
        reg.register(Artifact("a1", "Base", "skill"))
        reg.register(
            Artifact("a2", "Mid", "skill", dependencies=["a1"])
        )
        reg.register(
            Artifact("a3", "Top", "skill", dependencies=["a2"])
        )
        deps = reg.resolve_dependencies("a3")
        assert deps == ["a1", "a2", "a3"]

    def test_stats(self):
        from qe.runtime.artifacts import Artifact, ArtifactRegistry

        reg = ArtifactRegistry()
        reg.register(Artifact("a1", "P1", "prompt"))
        reg.register(Artifact("a2", "S1", "skill"))
        stats = reg.stats()
        assert stats["total"] == 2
        assert stats["by_type"]["prompt"] == 1


# ── #64 + #82 Composable Skills + Chaining ──────────────────────────────


class TestSkillEngine:
    def test_register_and_execute(self):
        from qe.runtime.skill_engine import (
            Skill,
            SkillEngine,
            SkillStep,
        )

        engine = SkillEngine()
        engine.register_handler("greet", lambda: "hello")
        skill = Skill(
            "s1", "Greet",
            steps=[
                SkillStep("step1", "greet", output_key="greeting")
            ],
        )
        engine.register_skill(skill)
        result = asyncio.run(engine.execute("s1"))
        assert result.success
        assert result.outputs["greeting"] == "hello"

    def test_skill_not_found(self):
        from qe.runtime.skill_engine import SkillEngine

        engine = SkillEngine()
        result = asyncio.run(engine.execute("nonexistent"))
        assert not result.success

    def test_step_with_params(self):
        from qe.runtime.skill_engine import (
            Skill,
            SkillEngine,
            SkillStep,
        )

        engine = SkillEngine()
        engine.register_handler("add", lambda a, b: a + b)
        skill = Skill(
            "s1", "Add",
            steps=[
                SkillStep(
                    "step1", "add",
                    params={"a": 1, "b": 2},
                    output_key="sum",
                )
            ],
        )
        engine.register_skill(skill)
        result = asyncio.run(engine.execute("s1"))
        assert result.success
        assert result.outputs["sum"] == 3

    def test_chain_skills(self):
        from qe.runtime.skill_engine import (
            Skill,
            SkillEngine,
            SkillStep,
        )

        engine = SkillEngine()
        engine.register_handler("double", lambda x: x * 2)
        engine.register_handler(
            "stringify", lambda x: str(x)
        )

        s1 = Skill(
            "s1", "Double",
            steps=[
                SkillStep(
                    "step1", "double",
                    params={"x": "$input"},
                    output_key="x",
                )
            ],
        )
        s2 = Skill(
            "s2", "Stringify",
            steps=[
                SkillStep(
                    "step1", "stringify",
                    params={"x": "$x"},
                    output_key="result",
                )
            ],
        )
        engine.register_skill(s1)
        engine.register_skill(s2)

        results = asyncio.run(
            engine.chain(["s1", "s2"], {"input": 5})
        )
        assert len(results) == 2
        assert results[0].success
        assert results[1].success
        assert results[1].outputs["result"] == "10"

    def test_step_dependencies(self):
        from qe.runtime.skill_engine import (
            Skill,
            SkillEngine,
            SkillStep,
        )

        engine = SkillEngine()
        engine.register_handler("inc", lambda v: v + 1)
        skill = Skill(
            "s1", "Inc Twice",
            steps=[
                SkillStep(
                    "a", "inc", params={"v": 1},
                    output_key="v1",
                ),
                SkillStep(
                    "b", "inc", params={"v": "$v1"},
                    output_key="v2", depends_on=["a"],
                ),
            ],
        )
        engine.register_skill(skill)
        result = asyncio.run(engine.execute("s1"))
        assert result.success
        assert result.outputs["v2"] == 3

    def test_list_skills(self):
        from qe.runtime.skill_engine import Skill, SkillEngine

        engine = SkillEngine()
        engine.register_skill(Skill("s1", "Skill 1"))
        engine.register_skill(Skill("s2", "Skill 2"))
        listing = engine.list_skills()
        assert len(listing) == 2


# ── #89 Skill Catalog ───────────────────────────────────────────────────


class TestSkillCatalog:
    def test_add_and_get(self):
        from qe.runtime.skill_catalog import (
            CatalogEntry,
            SkillCatalog,
        )

        cat = SkillCatalog()
        cat.add_entry(CatalogEntry("e1", "Test Skill"))
        assert cat.get("e1") is not None

    def test_search(self):
        from qe.runtime.skill_catalog import (
            CatalogEntry,
            SkillCatalog,
        )

        cat = SkillCatalog()
        cat.add_entry(
            CatalogEntry(
                "e1", "Research Helper",
                description="Helps research",
            )
        )
        results = cat.search("research")
        assert len(results) == 1

    def test_install_uninstall(self):
        from qe.runtime.skill_catalog import (
            CatalogEntry,
            SkillCatalog,
        )

        cat = SkillCatalog()
        cat.add_entry(CatalogEntry("e1", "Test"))
        assert cat.install("e1")
        assert cat.get("e1").installed
        assert cat.uninstall("e1")
        assert not cat.get("e1").installed

    def test_list_installed(self):
        from qe.runtime.skill_catalog import (
            CatalogEntry,
            SkillCatalog,
        )

        cat = SkillCatalog()
        cat.add_entry(CatalogEntry("e1", "A"))
        cat.add_entry(CatalogEntry("e2", "B"))
        cat.install("e1")
        installed = cat.list_installed()
        assert len(installed) == 1
        assert installed[0]["entry_id"] == "e1"

    def test_by_tag(self):
        from qe.runtime.skill_catalog import (
            CatalogEntry,
            SkillCatalog,
        )

        cat = SkillCatalog()
        cat.add_entry(
            CatalogEntry("e1", "A", tags=["research"])
        )
        results = cat.by_tag("research")
        assert len(results) == 1

    def test_stats(self):
        from qe.runtime.skill_catalog import (
            CatalogEntry,
            SkillCatalog,
        )

        cat = SkillCatalog()
        cat.add_entry(CatalogEntry("e1", "A"))
        cat.add_entry(CatalogEntry("e2", "B"))
        cat.install("e1")
        stats = cat.stats()
        assert stats["total_entries"] == 2
        assert stats["installed"] == 1


# ── #62 Progressive Tool Loading ────────────────────────────────────────


class TestProgressiveToolLoading:
    def test_core_tools_returned(self):
        from qe.runtime.tools import ToolRegistry, ToolSpec

        reg = ToolRegistry()
        reg.register(
            ToolSpec(name="query_beliefs", description="Search"),
            handler=lambda: None,
        )
        schemas = reg.get_schemas_for_intent("conversation")
        assert len(schemas) >= 1

    def test_research_intent_adds_tools(self):
        from qe.runtime.tools import ToolRegistry, ToolSpec

        reg = ToolRegistry()
        for name in [
            "query_beliefs", "deep_research",
            "swarm_research", "crystallize_insights",
        ]:
            reg.register(
                ToolSpec(name=name, description=name),
                handler=lambda: None,
            )
        schemas = reg.get_schemas_for_intent("research")
        names = [s["function"]["name"] for s in schemas]
        assert "deep_research" in names
        assert "query_beliefs" in names

    def test_tool_names(self):
        from qe.runtime.tools import ToolRegistry, ToolSpec

        reg = ToolRegistry()
        reg.register(
            ToolSpec(name="t1", description="x"),
            handler=lambda: None,
        )
        assert "t1" in reg.tool_names()


# ── #66 Learning Loop ───────────────────────────────────────────────────


class TestLearningLoop:
    def test_distill_pattern(self):
        from qe.runtime.learning_router import LearningLoop

        loop = LearningLoop()
        pattern = loop.distill(
            "Use deep_research for complex queries",
            domain="routing",
        )
        assert pattern.pattern_id.startswith("pat_")

    def test_retrieve_patterns(self):
        from qe.runtime.learning_router import LearningLoop

        loop = LearningLoop()
        loop.distill("Pattern A", domain="routing")
        loop.distill("Pattern B", domain="routing")
        results = loop.retrieve("routing")
        assert len(results) == 2

    def test_judge_updates_quality(self):
        from qe.runtime.learning_router import LearningLoop

        loop = LearningLoop()
        p = loop.distill("Test", quality=0.5)
        loop.judge(p, True)
        assert p.quality_score > 0.5

    def test_consolidate_prunes(self):
        from qe.runtime.learning_router import LearningLoop

        loop = LearningLoop()
        p = loop.distill("Bad pattern", quality=0.05)
        p.frequency = 10  # Enough observations
        pruned = loop.consolidate()
        assert pruned == 1

    def test_suggest(self):
        from qe.runtime.learning_router import LearningLoop

        loop = LearningLoop()
        loop.distill("Use research for complex", quality=0.8)
        suggestions = loop.suggest("I need complex research")
        assert len(suggestions) >= 1

    def test_stats(self):
        from qe.runtime.learning_router import LearningLoop

        loop = LearningLoop()
        loop.distill("Test")
        stats = loop.stats()
        assert stats["total_patterns"] == 1


# ── #70 BDI Mental State Tracking ────────────────────────────────────────


class TestBDITracking:
    def test_add_belief(self):
        from qe.runtime.bdi import BDITracker, Belief

        tracker = BDITracker()
        tracker.add_belief(
            Belief("Python", "is_type", "language", 0.9)
        )
        state = tracker.get_state()
        assert len(state.beliefs) == 1

    def test_add_desire(self):
        from qe.runtime.bdi import BDITracker, Desire

        tracker = BDITracker()
        tracker.add_desire(
            Desire("g1", "Research Python frameworks", 0.8)
        )
        state = tracker.get_state()
        assert len(state.desires) == 1

    def test_add_intention(self):
        from qe.runtime.bdi import BDITracker, Intention

        tracker = BDITracker()
        tracker.add_intention(
            Intention("w1", "Execute research plan", "step_1", 0.3)
        )
        state = tracker.get_state()
        assert len(state.intentions) == 1

    def test_update_belief(self):
        from qe.runtime.bdi import BDITracker, Belief

        tracker = BDITracker()
        tracker.add_belief(
            Belief("Python", "version", "3.12")
        )
        tracker.update_belief("Python", "version", "3.14", 0.95)
        state = tracker.get_state()
        assert state.beliefs[0].value == "3.14"

    def test_achieve_goal(self):
        from qe.runtime.bdi import BDITracker, Desire

        tracker = BDITracker()
        tracker.add_desire(Desire("g1", "Learn Python"))
        tracker.achieve_goal("g1")
        state = tracker.get_state()
        assert state.desires[0].status == "achieved"

    def test_context_prompt(self):
        from qe.runtime.bdi import (
            BDITracker,
            Belief,
            Desire,
            Intention,
        )

        tracker = BDITracker()
        tracker.add_belief(Belief("X", "is", "Y", 0.8))
        tracker.add_desire(Desire("g1", "Do something"))
        tracker.add_intention(
            Intention("w1", "Plan A", "step_2", 0.5)
        )
        state = tracker.get_state()
        prompt = state.context_prompt()
        assert "Beliefs" in prompt
        assert "Goals" in prompt
        assert "Plans" in prompt

    def test_summary(self):
        from qe.runtime.bdi import BDITracker, Belief, Desire

        tracker = BDITracker()
        tracker.add_belief(Belief("X", "is", "Y"))
        tracker.add_desire(Desire("g1", "Goal 1"))
        state = tracker.get_state()
        summary = state.summary()
        assert summary["beliefs"] == 1
        assert summary["desires"] == 1
        assert summary["active_goals"] == 1

    def test_clear(self):
        from qe.runtime.bdi import BDITracker, Belief

        tracker = BDITracker()
        tracker.add_belief(Belief("X", "is", "Y"))
        tracker.clear()
        state = tracker.get_state()
        assert len(state.beliefs) == 0

    def test_update_intention_progress(self):
        from qe.runtime.bdi import BDITracker, Intention

        tracker = BDITracker()
        tracker.add_intention(
            Intention("w1", "Plan", "step1", 0.0)
        )
        tracker.update_intention_progress("w1", 0.75, "step3")
        state = tracker.get_state()
        assert state.intentions[0].progress == 0.75
        assert state.intentions[0].current_step == "step3"


# ── Phase 5 imports ─────────────────────────────────────────────────────


class TestPhase5Imports:
    def test_import_artifacts(self):
        from qe.runtime.artifacts import (
            Artifact,
            ArtifactRegistry,
        )

        assert Artifact and ArtifactRegistry

    def test_import_skill_engine(self):
        from qe.runtime.skill_engine import Skill, SkillEngine

        assert Skill and SkillEngine

    def test_import_skill_catalog(self):
        from qe.runtime.skill_catalog import (
            CatalogEntry,
            SkillCatalog,
        )

        assert CatalogEntry and SkillCatalog

    def test_import_bdi(self):
        from qe.runtime.bdi import BDITracker, MentalState

        assert BDITracker and MentalState

    def test_import_learning_loop(self):
        from qe.runtime.learning_router import LearningLoop

        assert LearningLoop

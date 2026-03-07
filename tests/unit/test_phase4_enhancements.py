"""Tests for Phase 4 enhancements.

Covers:
- #57 Belief clusters + causal chains
- #58 Contradiction cascade
- #45 Self-learning routing (EMA extensions)
- #53 Parallel domain-specific swarms
- #65 Swarm consensus
- #69 Context degradation detection
- #83 L0/L1/L2 tiered context loading
- #85 Filesystem knowledge organization
- #75 CRON-based scheduling
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from qe.runtime.feature_flags import reset_flag_store


@pytest.fixture(autouse=True)
def _reset_flags():
    reset_flag_store()
    yield
    reset_flag_store()


# ── Fake claim for testing ───────────────────────────────────────────────


@dataclass
class FakeClaim:
    claim_id: str = ""
    subject_entity_id: str = ""
    predicate: str = ""
    object_value: str = ""
    confidence: float = 0.8


# ── #57 Belief Clusters ─────────────────────────────────────────────────


class TestBeliefClusters:
    def test_cluster_by_entity(self):
        from qe.substrate.belief_clusters import BeliefClusterEngine

        engine = BeliefClusterEngine()
        claims = [
            FakeClaim("c1", "Python", "is_type", "language"),
            FakeClaim("c2", "Python", "created_by", "Guido"),
            FakeClaim("c3", "Java", "is_type", "language"),
            FakeClaim("c4", "Java", "created_by", "Gosling"),
        ]
        clusters = engine.cluster_claims(claims)
        assert len(clusters) == 2
        labels = {c.label for c in clusters}
        assert "python" in labels
        assert "java" in labels

    def test_cluster_for_claim(self):
        from qe.substrate.belief_clusters import BeliefClusterEngine

        engine = BeliefClusterEngine()
        claims = [
            FakeClaim("c1", "Python", "is_type", "language"),
            FakeClaim("c2", "Python", "version", "3.14"),
        ]
        engine.cluster_claims(claims)
        cluster = engine.get_cluster_for_claim("c1")
        assert cluster is not None
        assert "c2" in cluster.claim_ids

    def test_no_clusters_for_singletons(self):
        from qe.substrate.belief_clusters import BeliefClusterEngine

        engine = BeliefClusterEngine()
        claims = [
            FakeClaim("c1", "Python", "is_type", "language"),
            FakeClaim("c2", "Java", "is_type", "language"),
        ]
        clusters = engine.cluster_claims(claims)
        assert len(clusters) == 0  # Each entity has only 1 claim

    def test_causal_chain_detection(self):
        from qe.substrate.belief_clusters import BeliefClusterEngine

        engine = BeliefClusterEngine()
        claims = [
            FakeClaim("c1", "Rain", "causes", "Flooding"),
            FakeClaim("c2", "Flooding", "causes", "Damage"),
        ]
        chains = engine.detect_causal_chains(claims)
        assert len(chains) >= 1
        # Should detect Rain→Flooding→Damage
        long_chains = [c for c in chains if c.length >= 2]
        assert len(long_chains) >= 1

    def test_stats(self):
        from qe.substrate.belief_clusters import BeliefClusterEngine

        engine = BeliefClusterEngine()
        claims = [
            FakeClaim("c1", "X", "p", "v"),
            FakeClaim("c2", "X", "q", "w"),
        ]
        engine.cluster_claims(claims)
        stats = engine.stats()
        assert stats["total_clusters"] >= 1

    def test_list_clusters(self):
        from qe.substrate.belief_clusters import BeliefClusterEngine

        engine = BeliefClusterEngine()
        claims = [
            FakeClaim("c1", "X", "p", "v"),
            FakeClaim("c2", "X", "q", "w"),
        ]
        engine.cluster_claims(claims)
        listing = engine.list_clusters()
        assert len(listing) >= 1
        assert "cluster_id" in listing[0]


# ── #58 Contradiction Cascade ────────────────────────────────────────────


class TestContradictionCascade:
    def test_no_dependents(self):
        from qe.substrate.contradiction_cascade import (
            ContradictionCascade,
        )

        cc = ContradictionCascade()
        claims = [FakeClaim("c1", "X", "is", "Y")]
        result = cc.analyze("c1", claims)
        assert result.total_affected == 0

    def test_direct_dependents(self):
        from qe.substrate.contradiction_cascade import (
            ContradictionCascade,
        )

        cc = ContradictionCascade()
        claims = [
            FakeClaim("c1", "Python", "is", "language"),
            FakeClaim("c2", "Flask", "uses", "Python"),
        ]
        result = cc.analyze("c1", claims)
        # c2 references Python in its object_value
        assert result.total_affected >= 1

    def test_cascade_depth(self):
        from qe.substrate.contradiction_cascade import (
            ContradictionCascade,
        )

        cc = ContradictionCascade()
        claims = [
            FakeClaim("c1", "A", "is", "base"),
            FakeClaim("c2", "B", "depends_on", "A"),
            FakeClaim("c3", "C", "depends_on", "B"),
        ]
        result = cc.analyze("c1", claims)
        assert result.total_affected >= 1

    def test_preview_retraction(self):
        from qe.substrate.contradiction_cascade import (
            ContradictionCascade,
        )

        cc = ContradictionCascade()
        claims = [
            FakeClaim("c1", "X", "is", "Y"),
            FakeClaim("c2", "Z", "uses", "X"),
        ]
        preview = cc.preview_retraction("c1", claims)
        assert "summary" in preview
        assert "affected_count" in preview

    def test_empty_claims(self):
        from qe.substrate.contradiction_cascade import (
            ContradictionCascade,
        )

        cc = ContradictionCascade()
        result = cc.analyze("c1", [])
        assert result.total_affected == 0


# ── #45 Self-Learning Routing (EMA) ─────────────────────────────────────


class TestSelfLearningRouting:
    def test_ema_record(self):
        from qe.runtime.routing_optimizer import RoutingOptimizer

        opt = RoutingOptimizer()
        opt.ema_record("gpt-4o", "analysis", True, 500.0, 0.9)
        opt.ema_record("gpt-4o", "analysis", True, 300.0, 0.8)
        stats = opt._stats[("gpt-4o", "analysis")]
        assert stats["total"] == 2
        assert stats["successes"] == 2
        assert 0 < stats["ema_latency"] < 500
        assert 0 < stats["ema_quality"] <= 0.9

    def test_dynamic_rankings(self):
        from qe.runtime.routing_optimizer import RoutingOptimizer

        opt = RoutingOptimizer()
        opt.ema_record("model_a", "task", True, 100.0, 0.9)
        opt.ema_record("model_b", "task", False, 500.0, 0.3)
        rankings = opt.dynamic_rankings(
            "task", ["model_a", "model_b"]
        )
        assert rankings[0]["model"] == "model_a"
        assert rankings[0]["score"] > rankings[1]["score"]

    def test_thompson_select(self):
        from qe.runtime.routing_optimizer import RoutingOptimizer

        opt = RoutingOptimizer()
        # Record many successes for model_a
        for _ in range(20):
            opt.ema_record("model_a", "t", True, 100.0, 0.9)
        for _ in range(20):
            opt.ema_record("model_b", "t", False, 500.0, 0.1)
        # Thompson should prefer model_a most of the time
        selections = [
            opt.thompson_select_model("t", ["model_a", "model_b"])
            for _ in range(50)
        ]
        a_count = selections.count("model_a")
        assert a_count > 30  # Should be picked more often

    def test_uninformed_rankings(self):
        from qe.runtime.routing_optimizer import RoutingOptimizer

        opt = RoutingOptimizer()
        rankings = opt.dynamic_rankings(
            "new_task", ["m1", "m2"]
        )
        # Both should get default prior score
        assert rankings[0]["score"] == rankings[1]["score"]


# ── #53 Swarm Coordinator ───────────────────────────────────────────────


class TestSwarmCoordinator:
    def test_assign_experts(self):
        from qe.runtime.swarm_coordinator import SwarmCoordinator

        coord = SwarmCoordinator()
        experts = coord.assign_experts(
            "Compare Python software frameworks"
        )
        assert len(experts) >= 1
        names = [e.name for e in experts]
        assert "tech_expert" in names

    def test_assign_business_expert(self):
        from qe.runtime.swarm_coordinator import SwarmCoordinator

        coord = SwarmCoordinator()
        experts = coord.assign_experts(
            "Analyze market growth and revenue strategy"
        )
        names = [e.name for e in experts]
        assert "business_expert" in names

    def test_weighted_merge(self):
        from qe.runtime.swarm_coordinator import (
            SwarmCoordinator,
            SwarmResult,
        )

        coord = SwarmCoordinator()
        results = [
            SwarmResult("tech", "technology", "Python is fast", 0.9),
            SwarmResult("biz", "business", "Good ROI", 0.7),
        ]
        merged = coord.weighted_merge(results)
        assert merged["agent_count"] == 2
        assert merged["total_confidence"] > 0
        assert "Python" in merged["merged_text"]

    def test_empty_merge(self):
        from qe.runtime.swarm_coordinator import SwarmCoordinator

        coord = SwarmCoordinator()
        merged = coord.weighted_merge([])
        assert merged["total_confidence"] == 0.0

    def test_list_experts(self):
        from qe.runtime.swarm_coordinator import SwarmCoordinator

        coord = SwarmCoordinator()
        listing = coord.list_experts()
        assert len(listing) >= 3


# ── #65 Swarm Consensus ─────────────────────────────────────────────────


class TestSwarmConsensus:
    def test_majority_vote(self):
        from qe.runtime.consensus import ConsensusEngine, ConsensusVote

        engine = ConsensusEngine()
        votes = [
            ConsensusVote("a1", "yes"),
            ConsensusVote("a2", "yes"),
            ConsensusVote("a3", "no"),
        ]
        result = engine.majority_vote(votes)
        assert result.winner == "yes"
        assert result.agreement_score > 0.5

    def test_weighted_vote(self):
        from qe.runtime.consensus import ConsensusEngine, ConsensusVote

        engine = ConsensusEngine()
        votes = [
            ConsensusVote("a1", "yes", confidence=0.9, weight=2.0),
            ConsensusVote("a2", "no", confidence=0.3, weight=1.0),
        ]
        result = engine.weighted_vote(votes)
        assert result.winner == "yes"

    def test_unanimous(self):
        from qe.runtime.consensus import ConsensusEngine, ConsensusVote

        engine = ConsensusEngine()
        votes = [
            ConsensusVote("a1", "yes"),
            ConsensusVote("a2", "yes"),
        ]
        result = engine.unanimous(votes)
        assert result.winner == "yes"
        assert result.agreement_score == 1.0
        assert result.details["unanimous"] is True

    def test_no_unanimity(self):
        from qe.runtime.consensus import ConsensusEngine, ConsensusVote

        engine = ConsensusEngine()
        votes = [
            ConsensusVote("a1", "yes"),
            ConsensusVote("a2", "no"),
        ]
        result = engine.unanimous(votes)
        assert result.details["unanimous"] is False

    def test_auto_select(self):
        from qe.runtime.consensus import ConsensusEngine, ConsensusVote

        engine = ConsensusEngine()
        # All agree → unanimous
        votes = [
            ConsensusVote("a1", "yes"),
            ConsensusVote("a2", "yes"),
        ]
        result = engine.auto_select(votes)
        assert result.agreement_score == 1.0

    def test_empty_votes(self):
        from qe.runtime.consensus import ConsensusEngine

        engine = ConsensusEngine()
        result = engine.majority_vote([])
        assert result.winner == ""
        assert result.agreement_score == 0.0


# ── #69 Context Degradation Detection ────────────────────────────────────


class TestContextHealthCheck:
    def test_healthy_context(self):
        from qe.runtime.context_curator import ContextCurator

        curator = ContextCurator.__new__(ContextCurator)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = curator.check_context_health(messages)
        assert result["health_score"] > 0.5
        assert result["total_tokens"] > 0

    def test_large_context_warning(self):
        from qe.runtime.context_curator import ContextCurator

        curator = ContextCurator.__new__(ContextCurator)
        # Create messages that exceed 70% of a small token limit
        messages = [
            {"role": "user", "content": "x " * 500}
            for _ in range(10)
        ]
        result = curator.check_context_health(
            messages, max_tokens=200
        )
        assert any("full" in i.lower() or "usage" in i.lower() for i in result["issues"])

    def test_duplicate_detection(self):
        from qe.runtime.context_curator import ContextCurator

        curator = ContextCurator.__new__(ContextCurator)
        messages = [
            {"role": "user", "content": "Same message here"}
        ] * 5
        result = curator.check_context_health(messages)
        assert any(
            "redundancy" in i.lower() or "duplicate" in i.lower()
            for i in result["issues"]
        )

    def test_remediation_suggestions(self):
        from qe.runtime.context_curator import ContextCurator

        curator = ContextCurator.__new__(ContextCurator)
        messages = [
            {"role": "user", "content": "Same here"}
        ] * 5
        result = curator.check_context_health(messages)
        assert len(result["remediation"]) > 0


# ── #83 Tiered Context Loading ───────────────────────────────────────────


class TestTieredContextLoading:
    def test_l0_system_only(self):
        from qe.runtime.context_curator import ContextCurator

        curator = ContextCurator.__new__(ContextCurator)
        result = curator.tiered_context(
            "hi", "You are a helpful assistant", tier="L0"
        )
        assert result["tier"] == "L0"
        assert len(result["messages"]) == 1
        assert result["claims_included"] == 0

    def test_l1_top_claims(self):
        from qe.runtime.context_curator import ContextCurator

        curator = ContextCurator.__new__(ContextCurator)
        claims = [FakeClaim(f"c{i}", "E", f"p{i}", f"v{i}")
                  for i in range(10)]
        result = curator.tiered_context(
            "Tell me about E",
            "System prompt",
            claims=claims,
            history=[{"role": "user", "content": "prev"}],
            tier="L1",
        )
        assert result["tier"] == "L1"
        assert result["claims_included"] == 5
        assert result["history_included"] == 1

    def test_l2_full_assembly(self):
        from qe.runtime.context_curator import ContextCurator

        curator = ContextCurator.__new__(ContextCurator)
        claims = [FakeClaim(f"c{i}", "E", f"p{i}", f"v{i}")
                  for i in range(10)]
        history = [
            {"role": "user", "content": f"msg{i}"}
            for i in range(5)
        ]
        result = curator.tiered_context(
            "complex query about many things",
            "System prompt",
            claims=claims,
            history=history,
            tier="L2",
        )
        assert result["tier"] == "L2"
        assert result["claims_included"] == 10
        assert result["history_included"] == 5

    def test_auto_tier_short_query(self):
        from qe.runtime.context_curator import ContextCurator

        curator = ContextCurator.__new__(ContextCurator)
        result = curator.tiered_context("hi", "System")
        assert result["tier"] == "L0"

    def test_auto_tier_medium_query(self):
        from qe.runtime.context_curator import ContextCurator

        curator = ContextCurator.__new__(ContextCurator)
        result = curator.tiered_context(
            "What do you know about Python?", "System"
        )
        assert result["tier"] == "L1"

    def test_auto_tier_complex_query(self):
        from qe.runtime.context_curator import ContextCurator

        curator = ContextCurator.__new__(ContextCurator)
        result = curator.tiered_context(
            "Compare and contrast the performance characteristics "
            "of Python and Java in large-scale distributed systems "
            "considering memory usage and concurrency",
            "System",
        )
        assert result["tier"] == "L2"


# ── #85 Knowledge Tree ───────────────────────────────────────────────────


class TestKnowledgeTree:
    def test_build_and_ls(self):
        from qe.substrate.knowledge_tree import KnowledgeTree

        tree = KnowledgeTree()
        claims = [
            FakeClaim("c1", "Python", "is_type", "language"),
            FakeClaim("c2", "Python", "version", "3.14"),
        ]
        tree.build(claims)
        listing = tree.ls("/")
        assert len(listing) >= 1
        assert listing[0]["name"] == "python"

    def test_ls_entity(self):
        from qe.substrate.knowledge_tree import KnowledgeTree

        tree = KnowledgeTree()
        claims = [
            FakeClaim("c1", "Python", "is_type", "language"),
            FakeClaim("c2", "Python", "version", "3.14"),
        ]
        tree.build(claims)
        listing = tree.ls("/python")
        assert len(listing) == 2  # is_type and version dirs

    def test_cat_claim(self):
        from qe.substrate.knowledge_tree import KnowledgeTree

        tree = KnowledgeTree()
        claims = [
            FakeClaim("c1", "Python", "is_type", "language"),
        ]
        tree.build(claims)
        data = tree.cat("/python/is_type/c1")
        assert data is not None
        assert data["claim_id"] == "c1"

    def test_tree_output(self):
        from qe.substrate.knowledge_tree import KnowledgeTree

        tree = KnowledgeTree()
        claims = [
            FakeClaim("c1", "Python", "is_type", "language"),
        ]
        tree.build(claims)
        lines = tree.tree("/")
        assert len(lines) >= 1

    def test_find(self):
        from qe.substrate.knowledge_tree import KnowledgeTree

        tree = KnowledgeTree()
        claims = [
            FakeClaim("c1", "Python", "is_type", "language"),
            FakeClaim("c2", "Java", "is_type", "language"),
        ]
        tree.build(claims)
        matches = tree.find("python")
        assert any("python" in m for m in matches)

    def test_stats(self):
        from qe.substrate.knowledge_tree import KnowledgeTree

        tree = KnowledgeTree()
        claims = [
            FakeClaim("c1", "Python", "is_type", "language"),
        ]
        tree.build(claims)
        stats = tree.stats()
        assert stats["total_claims"] >= 1
        assert stats["top_level_entities"] >= 1

    def test_cat_nonexistent(self):
        from qe.substrate.knowledge_tree import KnowledgeTree

        tree = KnowledgeTree()
        tree.build([])
        assert tree.cat("/nonexistent") is None

    def test_ls_nonexistent(self):
        from qe.substrate.knowledge_tree import KnowledgeTree

        tree = KnowledgeTree()
        tree.build([])
        assert tree.ls("/nonexistent") == []


# ── #75 Task Scheduler ──────────────────────────────────────────────────


class TestTaskScheduler:
    def test_schedule_task(self):
        from qe.runtime.scheduler import TaskScheduler

        sched = TaskScheduler()
        tid = sched.schedule(
            "test", "*/5 * * * *", "consolidate"
        )
        assert tid.startswith("sched_")

    def test_due_tasks(self):
        import time

        from qe.runtime.scheduler import TaskScheduler

        sched = TaskScheduler()
        sched.schedule("test", "every:1", "action")
        # Immediately due since last_run=0
        due = sched.get_due_tasks(now=time.time())
        assert len(due) == 1

    def test_not_due_yet(self):
        import time

        from qe.runtime.scheduler import TaskScheduler

        sched = TaskScheduler()
        tid = sched.schedule("test", "every:3600", "action")
        # Mark as just run
        sched._tasks[tid].last_run = time.time()
        due = sched.get_due_tasks()
        assert len(due) == 0

    def test_tick_with_handler(self):
        from qe.runtime.scheduler import TaskScheduler

        sched = TaskScheduler()
        results = []
        sched.register_handler(
            "test_action", lambda: results.append("ran")
        )
        sched.schedule("test", "every:1", "test_action")
        asyncio.run(sched.tick())
        assert results == ["ran"]

    def test_disable_enable(self):
        import time

        from qe.runtime.scheduler import TaskScheduler

        sched = TaskScheduler()
        tid = sched.schedule("test", "every:1", "action")
        sched.disable(tid)
        due = sched.get_due_tasks(now=time.time() + 100)
        assert len(due) == 0
        sched.enable(tid)
        due = sched.get_due_tasks(now=time.time() + 100)
        assert len(due) == 1

    def test_unschedule(self):
        from qe.runtime.scheduler import TaskScheduler

        sched = TaskScheduler()
        tid = sched.schedule("test", "every:1", "action")
        assert sched.unschedule(tid)
        assert not sched.unschedule(tid)  # Already removed

    def test_max_runs(self):
        from qe.runtime.scheduler import TaskScheduler

        sched = TaskScheduler()
        tid = sched.schedule(
            "test", "every:1", "action", max_runs=2
        )
        task = sched._tasks[tid]
        task.mark_run()
        task.mark_run()
        assert not task.is_due()

    def test_list_tasks(self):
        from qe.runtime.scheduler import TaskScheduler

        sched = TaskScheduler()
        sched.schedule("t1", "every:60", "a1")
        sched.schedule("t2", "*/5 * * * *", "a2")
        listing = sched.list_tasks()
        assert len(listing) == 2

    def test_stats(self):
        from qe.runtime.scheduler import TaskScheduler

        sched = TaskScheduler()
        sched.schedule("t1", "every:60", "action")
        sched.register_handler("action", lambda: None)
        stats = sched.stats()
        assert stats["total_tasks"] == 1
        assert stats["enabled"] == 1
        assert "action" in stats["handlers"]

    def test_interval_parsing(self):
        from qe.runtime.scheduler import ScheduledTask

        # */5 format (minutes)
        t = ScheduledTask("id", "test", "*/5 * * * *", "a")
        assert t.interval_seconds == 300

        # every:N format (seconds)
        t2 = ScheduledTask("id", "test", "every:120", "a")
        assert t2.interval_seconds == 120


# ── Phase 4 imports ─────────────────────────────────────────────────────


class TestPhase4Imports:
    def test_import_belief_clusters(self):
        from qe.substrate.belief_clusters import (
            BeliefClusterEngine,
            CausalChain,
            ClaimCluster,
        )

        assert BeliefClusterEngine
        assert ClaimCluster
        assert CausalChain

    def test_import_contradiction_cascade(self):
        from qe.substrate.contradiction_cascade import (
            CascadeResult,
            ContradictionCascade,
        )

        assert ContradictionCascade
        assert CascadeResult

    def test_import_swarm_coordinator(self):
        from qe.runtime.swarm_coordinator import (
            SwarmCoordinator,
            SwarmResult,
        )

        assert SwarmCoordinator
        assert SwarmResult

    def test_import_consensus(self):
        from qe.runtime.consensus import (
            ConsensusEngine,
            ConsensusResult,
            ConsensusVote,
        )

        assert ConsensusEngine
        assert ConsensusVote
        assert ConsensusResult

    def test_import_knowledge_tree(self):
        from qe.substrate.knowledge_tree import KnowledgeTree

        assert KnowledgeTree

    def test_import_scheduler(self):
        from qe.runtime.scheduler import (
            ScheduledTask,
            TaskScheduler,
        )

        assert TaskScheduler
        assert ScheduledTask

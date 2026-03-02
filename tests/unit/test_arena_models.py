"""Tests for arena Pydantic models."""

from __future__ import annotations

from qe.models.arena import (
    AgentEloRating,
    ArenaConfig,
    ArenaMatch,
    ArenaResult,
    CrossExamination,
    DefenseResponse,
    DivergenceCheck,
    MatchJudgment,
)


class TestAgentEloRating:
    def test_defaults(self):
        r = AgentEloRating(agent_id="a1")
        assert r.elo == 1200.0
        assert r.wins == 0
        assert r.losses == 0
        assert r.draws == 0
        assert r.total_matches == 0

    def test_total_matches(self):
        r = AgentEloRating(agent_id="a1", wins=3, losses=2, draws=1)
        assert r.total_matches == 6

    def test_serialization_roundtrip(self):
        r = AgentEloRating(agent_id="a1", elo=1500.0, wins=5)
        data = r.model_dump()
        r2 = AgentEloRating.model_validate(data)
        assert r2.agent_id == "a1"
        assert r2.elo == 1500.0
        assert r2.wins == 5


class TestCrossExamination:
    def test_id_prefix(self):
        ex = CrossExamination(challenger_id="a1", defender_id="a2")
        assert ex.examination_id.startswith("xex_")

    def test_fields(self):
        ex = CrossExamination(
            challenger_id="a1",
            defender_id="a2",
            challenges=["weak evidence"],
            weaknesses_identified=["no source"],
            questions_raised=["what about X?"],
        )
        assert len(ex.challenges) == 1
        assert len(ex.weaknesses_identified) == 1
        assert len(ex.questions_raised) == 1


class TestDefenseResponse:
    def test_fields(self):
        d = DefenseResponse(
            defender_id="a2",
            examination_id="xex_abc123",
            rebuttals=["source provided"],
            concessions=["point about Y is valid"],
        )
        assert d.defender_id == "a2"
        assert len(d.rebuttals) == 1
        assert len(d.concessions) == 1


class TestMatchJudgment:
    def test_id_prefix(self):
        j = MatchJudgment(agent_a_id="a1", agent_b_id="a2")
        assert j.judgment_id.startswith("jdg_")

    def test_defaults(self):
        j = MatchJudgment(agent_a_id="a1", agent_b_id="a2")
        assert j.winner == "draw"
        assert j.agent_a_score == 0.0
        assert j.agent_b_score == 0.0

    def test_score_constraints(self):
        j = MatchJudgment(
            agent_a_id="a1",
            agent_b_id="a2",
            agent_a_score=8.5,
            agent_b_score=6.0,
            winner="agent_a",
            reasoning="A had better evidence",
        )
        assert j.agent_a_score == 8.5
        assert j.winner == "agent_a"


class TestDivergenceCheck:
    def test_id_prefix(self):
        d = DivergenceCheck()
        assert d.divergence_id.startswith("div_")

    def test_sycophancy_flag(self):
        d = DivergenceCheck(
            similarity_score=0.9,
            shared_claims=["claim1", "claim2"],
            sycophancy_risk=True,
        )
        assert d.sycophancy_risk is True
        assert d.similarity_score == 0.9

    def test_low_similarity(self):
        d = DivergenceCheck(
            similarity_score=0.1,
            divergent_claims=["unique_a", "unique_b"],
        )
        assert d.sycophancy_risk is False


class TestArenaMatch:
    def test_id_prefix(self):
        m = ArenaMatch(agent_a_id="a1", agent_b_id="a2")
        assert m.match_id.startswith("mtch_")

    def test_empty_match(self):
        m = ArenaMatch(agent_a_id="a1", agent_b_id="a2")
        assert m.winner is None
        assert m.judgment is None
        assert len(m.examinations) == 0
        assert len(m.defenses) == 0


class TestArenaConfig:
    def test_defaults(self):
        c = ArenaConfig()
        assert c.enabled is False
        assert c.max_rounds == 2
        assert c.divergence_threshold == 0.3
        assert c.budget_limit_usd == 0.50
        assert c.tournament_style == "round_robin"

    def test_custom(self):
        c = ArenaConfig(
            enabled=True,
            max_rounds=3,
            divergence_threshold=0.5,
            budget_limit_usd=1.0,
            tournament_style="single_elimination",
        )
        assert c.enabled is True
        assert c.tournament_style == "single_elimination"


class TestArenaResult:
    def test_id_prefix(self):
        r = ArenaResult()
        assert r.arena_id.startswith("arena_")

    def test_fields(self):
        r = ArenaResult(
            goal_id="g1",
            winner_id="a1",
            sycophancy_detected=False,
            total_cost_usd=0.05,
        )
        assert r.goal_id == "g1"
        assert r.winner_id == "a1"
        assert r.sycophancy_detected is False

    def test_serialization(self):
        r = ArenaResult(goal_id="g1", winner_id="a1")
        data = r.model_dump()
        r2 = ArenaResult.model_validate(data)
        assert r2.goal_id == "g1"
        assert r2.winner_id == "a1"

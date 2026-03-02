"""Tests for CompetitiveArena — tournament-style agent competition."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qe.models.arena import (
    ArenaConfig,
    ArenaResult,
    CrossExamination,
    DefenseResponse,
)
from qe.runtime.competitive_arena import (
    BetaArm,
    CompetitiveArena,
    _CrossExamResult,
    _DefenseResult,
    _DivergenceResult,
    _JudgeResult,
)
from qe.services.inquiry.schemas import InquiryResult

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_bus():
    bus = MagicMock()
    bus.publish = MagicMock()
    return bus


def _make_result(
    goal_id: str = "g1",
    findings: str = "Some findings",
    cost: float = 0.01,
) -> InquiryResult:
    return InquiryResult(
        inquiry_id="inq_test",
        goal_id=goal_id,
        status="completed",
        findings_summary=findings,
        total_cost_usd=cost,
    )


def _mock_instructor_client(
    divergence: _DivergenceResult | None = None,
    cross_exam: _CrossExamResult | None = None,
    defense: _DefenseResult | None = None,
    judge: _JudgeResult | None = None,
):
    """Build a mock instructor client that returns different results based on response_model."""
    mock_client = MagicMock()

    async def _create(**kwargs):
        model_cls = kwargs.get("response_model")
        if model_cls is _DivergenceResult:
            return divergence or _DivergenceResult()
        if model_cls is _CrossExamResult:
            return cross_exam or _CrossExamResult(
                challenges=["weak evidence"],
                weaknesses_identified=["no citations"],
                questions_raised=["what about X?"],
            )
        if model_cls is _DefenseResult:
            return defense or _DefenseResult(
                rebuttals=["I have sources"],
                concessions=["fair point about Y"],
            )
        if model_cls is _JudgeResult:
            return judge or _JudgeResult(
                agent_a_score=7.0,
                agent_b_score=5.0,
                winner="agent_a",
                reasoning="A was better.",
            )
        return MagicMock()

    mock_client.chat.completions.create = _create
    return mock_client


# ── Elo Math Tests ───────────────────────────────────────────────────────


class TestEloRating:
    def test_initial_elo(self):
        arena = CompetitiveArena()
        rating = arena.get_elo_rating("agent_1")
        assert rating.elo == 1200.0
        assert rating.wins == 0
        assert rating.losses == 0

    def test_win_increases_elo(self):
        arena = CompetitiveArena()
        arena.get_elo_rating("winner")
        arena.get_elo_rating("loser")
        arena._update_elo("winner", "loser", draw=False)

        assert arena._elo_ratings["winner"].elo > 1200.0
        assert arena._elo_ratings["loser"].elo < 1200.0
        assert arena._elo_ratings["winner"].wins == 1
        assert arena._elo_ratings["loser"].losses == 1

    def test_draw_elo(self):
        arena = CompetitiveArena()
        arena.get_elo_rating("a")
        arena.get_elo_rating("b")
        arena._update_elo("a", "b", draw=True)

        # With equal starting Elo, draw should keep them approximately equal
        assert abs(arena._elo_ratings["a"].elo - 1200.0) < 0.01
        assert abs(arena._elo_ratings["b"].elo - 1200.0) < 0.01
        assert arena._elo_ratings["a"].draws == 1
        assert arena._elo_ratings["b"].draws == 1

    def test_elo_sum_preserved(self):
        """Total Elo in the system should be preserved."""
        arena = CompetitiveArena()
        arena.get_elo_rating("a")
        arena.get_elo_rating("b")
        initial_sum = arena._elo_ratings["a"].elo + arena._elo_ratings["b"].elo

        arena._update_elo("a", "b", draw=False)
        final_sum = arena._elo_ratings["a"].elo + arena._elo_ratings["b"].elo
        assert abs(initial_sum - final_sum) < 0.01

    def test_upset_win_larger_elo_change(self):
        """Low-Elo beating high-Elo should yield a bigger swing."""
        arena = CompetitiveArena()
        arena.get_elo_rating("underdog")
        arena.get_elo_rating("favorite")
        arena._elo_ratings["favorite"].elo = 1600.0
        arena._elo_ratings["underdog"].elo = 800.0

        arena._update_elo("underdog", "favorite", draw=False)
        # Underdog should gain a lot (close to K_FACTOR)
        assert arena._elo_ratings["underdog"].elo > 828  # expected ~831
        assert arena._elo_ratings["favorite"].elo < 1572

    def test_expected_win_small_elo_change(self):
        """High-Elo beating low-Elo should yield a small swing."""
        arena = CompetitiveArena()
        arena.get_elo_rating("strong")
        arena.get_elo_rating("weak")
        arena._elo_ratings["strong"].elo = 1600.0
        arena._elo_ratings["weak"].elo = 800.0

        arena._update_elo("strong", "weak", draw=False)
        # Strong gains very little
        assert arena._elo_ratings["strong"].elo < 1605

    def test_rankings_sorted(self):
        arena = CompetitiveArena()
        arena.get_elo_rating("low")
        arena.get_elo_rating("high")
        arena._elo_ratings["low"].elo = 1000.0
        arena._elo_ratings["high"].elo = 1500.0

        rankings = arena.get_rankings()
        assert rankings[0].agent_id == "high"
        assert rankings[1].agent_id == "low"


# ── Thompson Sampling Tests ──────────────────────────────────────────────


class TestBetaArm:
    def test_initial_sample_range(self):
        arm = BetaArm()
        samples = [arm.sample() for _ in range(100)]
        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_update_success(self):
        arm = BetaArm()
        arm.update(success=True)
        assert arm.alpha == 2.0
        assert arm.beta == 1.0

    def test_update_failure(self):
        arm = BetaArm()
        arm.update(success=False)
        assert arm.alpha == 1.0
        assert arm.beta == 2.0

    def test_strong_prior_biases_samples(self):
        arm = BetaArm(alpha=100.0, beta=1.0)
        samples = [arm.sample() for _ in range(50)]
        assert sum(s > 0.8 for s in samples) > 40  # mostly high values


class TestAgentSelection:
    def test_select_all_when_n_exceeds_available(self):
        arena = CompetitiveArena()
        selected = arena.select_agents_for_arena(["a", "b"], n=5)
        assert set(selected) == {"a", "b"}

    def test_select_n_agents(self):
        arena = CompetitiveArena()
        selected = arena.select_agents_for_arena(
            ["a", "b", "c", "d"], n=2,
        )
        assert len(selected) == 2

    def test_strong_prior_influences_selection(self):
        arena = CompetitiveArena()
        # Give "good" agent a very strong prior
        arena._agent_arms["good"] = BetaArm(alpha=100.0, beta=1.0)
        arena._agent_arms["bad"] = BetaArm(alpha=1.0, beta=100.0)
        arena._agent_arms["neutral"] = BetaArm(alpha=1.0, beta=1.0)

        # Over many selections, "good" should be picked most often
        picks = {"good": 0, "bad": 0, "neutral": 0}
        for _ in range(100):
            selected = arena.select_agents_for_arena(
                ["good", "bad", "neutral"], n=1,
            )
            picks[selected[0]] += 1

        assert picks["good"] > picks["bad"]


# ── Divergence Check Tests ───────────────────────────────────────────────


class TestDivergenceCheck:
    @pytest.mark.asyncio
    async def test_high_similarity_flags_sycophancy(self):
        arena = CompetitiveArena(
            config=ArenaConfig(divergence_threshold=0.3),
        )

        mock_client = _mock_instructor_client(
            divergence=_DivergenceResult(
                similarity_score=0.9,
                shared_claims=["everything is great"],
                divergent_claims=[],
                sycophancy_risk=True,
            ),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await arena._check_divergence(
                "test goal",
                [_make_result(findings="great"), _make_result(findings="also great")],
                ["a1", "a2"],
            )

        assert result.sycophancy_risk is True
        assert result.similarity_score == 0.9

    @pytest.mark.asyncio
    async def test_low_similarity_no_sycophancy(self):
        arena = CompetitiveArena(
            config=ArenaConfig(divergence_threshold=0.3),
        )

        mock_client = _mock_instructor_client(
            divergence=_DivergenceResult(
                similarity_score=0.1,
                shared_claims=[],
                divergent_claims=["A says X", "B says Y"],
                sycophancy_risk=False,
            ),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await arena._check_divergence(
                "test goal",
                [_make_result(findings="X"), _make_result(findings="Y")],
                ["a1", "a2"],
            )

        assert result.sycophancy_risk is False
        assert result.similarity_score == 0.1

    @pytest.mark.asyncio
    async def test_divergence_fallback_on_error(self):
        arena = CompetitiveArena()

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("LLM down")
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await arena._check_divergence(
                "goal", [_make_result()], ["a1"],
            )

        assert result.similarity_score == 0.0
        assert result.sycophancy_risk is False


# ── Cross-Examination Tests ──────────────────────────────────────────────


class TestCrossExamination:
    @pytest.mark.asyncio
    async def test_generates_challenges(self):
        arena = CompetitiveArena()
        mock_client = _mock_instructor_client(
            cross_exam=_CrossExamResult(
                challenges=["No sources cited", "Circular reasoning"],
                weaknesses_identified=["Weak methodology"],
                questions_raised=["What about edge cases?"],
            ),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            exam = await arena._cross_examine(
                "test goal", "challenger_1", "defender_1", "Some findings",
            )

        assert exam.challenger_id == "challenger_1"
        assert exam.defender_id == "defender_1"
        assert len(exam.challenges) == 2
        assert len(exam.weaknesses_identified) == 1
        assert exam.examination_id.startswith("xex_")

    @pytest.mark.asyncio
    async def test_cross_examine_fallback_on_error(self):
        arena = CompetitiveArena()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("LLM error")
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            exam = await arena._cross_examine(
                "goal", "c1", "d1", "findings",
            )

        assert exam.challenger_id == "c1"
        assert exam.challenges == []


class TestDefense:
    @pytest.mark.asyncio
    async def test_generates_defense(self):
        arena = CompetitiveArena()
        mock_client = _mock_instructor_client(
            defense=_DefenseResult(
                rebuttals=["Evidence is in appendix"],
                concessions=["Fair point about methodology"],
                additional_evidence=["New data supports claim"],
            ),
        )
        exam = CrossExamination(
            challenger_id="c1",
            defender_id="d1",
            challenges=["No evidence"],
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            defense = await arena._defend("goal", "d1", exam, "findings")

        assert defense.defender_id == "d1"
        assert len(defense.rebuttals) == 1
        assert len(defense.concessions) == 1


# ── Judgment Tests ───────────────────────────────────────────────────────


class TestJudgment:
    @pytest.mark.asyncio
    async def test_judge_picks_winner(self):
        arena = CompetitiveArena()
        mock_client = _mock_instructor_client(
            judge=_JudgeResult(
                agent_a_score=8.0,
                agent_b_score=5.0,
                factual_accuracy_a=9.0,
                factual_accuracy_b=4.0,
                winner="agent_a",
                reasoning="Agent A provided better evidence.",
            ),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            judgment = await arena._judge_match(
                goal_description="test goal",
                agent_a_id="a1",
                agent_b_id="a2",
                result_a=_make_result(findings="A findings"),
                result_b=_make_result(findings="B findings"),
                exam_a_to_b=CrossExamination(challenger_id="a1", defender_id="a2"),
                exam_b_to_a=CrossExamination(challenger_id="a2", defender_id="a1"),
                defense_a=DefenseResponse(defender_id="a1", examination_id="x1"),
                defense_b=DefenseResponse(defender_id="a2", examination_id="x2"),
            )

        assert judgment.winner == "agent_a"
        assert judgment.agent_a_score == 8.0
        assert judgment.agent_b_score == 5.0

    @pytest.mark.asyncio
    async def test_judge_declares_draw(self):
        arena = CompetitiveArena()
        mock_client = _mock_instructor_client(
            judge=_JudgeResult(
                agent_a_score=6.0,
                agent_b_score=6.0,
                winner="draw",
                reasoning="Both equally good.",
            ),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            judgment = await arena._judge_match(
                goal_description="goal",
                agent_a_id="a1", agent_b_id="a2",
                result_a=_make_result(), result_b=_make_result(),
                exam_a_to_b=CrossExamination(challenger_id="a1", defender_id="a2"),
                exam_b_to_a=CrossExamination(challenger_id="a2", defender_id="a1"),
                defense_a=DefenseResponse(defender_id="a1", examination_id="x1"),
                defense_b=DefenseResponse(defender_id="a2", examination_id="x2"),
            )

        assert judgment.winner == "draw"

    @pytest.mark.asyncio
    async def test_judge_fallback_on_error(self):
        arena = CompetitiveArena()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("LLM error")
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            judgment = await arena._judge_match(
                goal_description="goal",
                agent_a_id="a1", agent_b_id="a2",
                result_a=_make_result(), result_b=_make_result(),
                exam_a_to_b=CrossExamination(challenger_id="a1", defender_id="a2"),
                exam_b_to_a=CrossExamination(challenger_id="a2", defender_id="a1"),
                defense_a=DefenseResponse(defender_id="a1", examination_id="x1"),
                defense_b=DefenseResponse(defender_id="a2", examination_id="x2"),
            )

        assert judgment.winner == "draw"
        assert "failed" in judgment.reasoning.lower()


# ── Full Tournament Tests ────────────────────────────────────────────────


class TestTournament:
    @pytest.mark.asyncio
    async def test_two_agent_tournament(self):
        bus = _make_bus()
        arena = CompetitiveArena(bus=bus)
        mock_client = _mock_instructor_client(
            divergence=_DivergenceResult(similarity_score=0.1),
            judge=_JudgeResult(winner="agent_a", agent_a_score=8, agent_b_score=5),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await arena.run_tournament(
                goal_id="g1",
                goal_description="test goal",
                results=[
                    _make_result(findings="A's work"),
                    _make_result(findings="B's work"),
                ],
                agent_ids=["agent_a", "agent_b"],
            )

        assert isinstance(result, ArenaResult)
        assert result.goal_id == "g1"
        assert result.winner_id == "agent_a"
        assert result.sycophancy_detected is False
        assert len(result.matches) == 1  # 2 agents = 1 match

    @pytest.mark.asyncio
    async def test_three_agent_round_robin(self):
        arena = CompetitiveArena(
            config=ArenaConfig(tournament_style="round_robin"),
        )
        mock_client = _mock_instructor_client(
            divergence=_DivergenceResult(similarity_score=0.1),
            judge=_JudgeResult(winner="agent_a", agent_a_score=8, agent_b_score=5),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await arena.run_tournament(
                goal_id="g1",
                goal_description="test",
                results=[_make_result(), _make_result(), _make_result()],
                agent_ids=["a", "b", "c"],
            )

        # 3 agents round-robin = 3 matches (C(3,2))
        assert len(result.matches) == 3
        assert result.winner_id is not None

    @pytest.mark.asyncio
    async def test_single_elimination_pairing(self):
        arena = CompetitiveArena(
            config=ArenaConfig(tournament_style="single_elimination"),
        )
        pairs = arena._get_pairs(["a", "b", "c", "d"])
        assert len(pairs) == 2  # (0,1) and (2,3)

    @pytest.mark.asyncio
    async def test_budget_exhaustion_stops_early(self):
        arena = CompetitiveArena(
            config=ArenaConfig(budget_limit_usd=0.0),  # already over budget
        )
        arena._total_cost_usd = 0.01  # over the 0.0 limit

        mock_client = _mock_instructor_client(
            divergence=_DivergenceResult(similarity_score=0.1),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await arena.run_tournament(
                goal_id="g1",
                goal_description="test",
                results=[_make_result(), _make_result()],
                agent_ids=["a", "b"],
            )

        # No matches should have run due to budget
        assert len(result.matches) == 0


# ── Sycophancy Fallback Tests ────────────────────────────────────────────


class TestSycophancyFallback:
    @pytest.mark.asyncio
    async def test_sycophancy_skips_cross_examination(self):
        bus = _make_bus()
        arena = CompetitiveArena(
            bus=bus,
            config=ArenaConfig(divergence_threshold=0.3),
        )

        mock_client = _mock_instructor_client(
            divergence=_DivergenceResult(
                similarity_score=0.9,
                sycophancy_risk=True,
            ),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            result = await arena.run_tournament(
                goal_id="g1",
                goal_description="test",
                results=[
                    _make_result(findings="short"),
                    _make_result(findings="much longer findings win majority vote"),
                ],
                agent_ids=["a", "b"],
            )

        assert result.sycophancy_detected is True
        assert len(result.matches) == 0  # no cross-exam happened
        assert result.winner_id == "b"  # longer summary wins

    @pytest.mark.asyncio
    async def test_sycophancy_publishes_fallback_event(self):
        bus = _make_bus()
        arena = CompetitiveArena(
            bus=bus,
            config=ArenaConfig(divergence_threshold=0.3),
        )

        mock_client = _mock_instructor_client(
            divergence=_DivergenceResult(similarity_score=0.95, sycophancy_risk=True),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            await arena.run_tournament(
                goal_id="g1",
                goal_description="test",
                results=[_make_result(), _make_result()],
                agent_ids=["a", "b"],
            )

        topics = [c.args[0].topic for c in bus.publish.call_args_list]
        assert "arena.sycophancy_fallback" in topics


# ── Bus Event Tests ──────────────────────────────────────────────────────


class TestBusEvents:
    @pytest.mark.asyncio
    async def test_tournament_publishes_correct_topics(self):
        bus = _make_bus()
        arena = CompetitiveArena(bus=bus)

        mock_client = _mock_instructor_client(
            divergence=_DivergenceResult(similarity_score=0.1),
            judge=_JudgeResult(winner="agent_a"),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            await arena.run_tournament(
                goal_id="g1",
                goal_description="test",
                results=[_make_result(), _make_result()],
                agent_ids=["a", "b"],
            )

        topics = [c.args[0].topic for c in bus.publish.call_args_list]
        assert "arena.tournament_started" in topics
        assert "arena.divergence_checked" in topics
        assert "arena.match_completed" in topics
        assert "arena.elo_updated" in topics
        assert "arena.tournament_completed" in topics

    @pytest.mark.asyncio
    async def test_no_events_without_bus(self):
        arena = CompetitiveArena(bus=None)

        mock_client = _mock_instructor_client(
            divergence=_DivergenceResult(similarity_score=0.1),
            judge=_JudgeResult(winner="agent_a"),
        )

        with patch("qe.runtime.competitive_arena.instructor") as mock_inst:
            mock_inst.from_litellm.return_value = mock_client
            # Should not raise
            result = await arena.run_tournament(
                goal_id="g1",
                goal_description="test",
                results=[_make_result(), _make_result()],
                agent_ids=["a", "b"],
            )

        assert result.winner_id is not None


# ── Status Tests ─────────────────────────────────────────────────────────


class TestArenaStatus:
    def test_status_empty(self):
        arena = CompetitiveArena()
        s = arena.status()
        assert s["total_cost_usd"] == 0.0
        assert s["agents_rated"] == 0

    def test_status_with_agents(self):
        arena = CompetitiveArena()
        arena.get_elo_rating("a")
        arena.get_elo_rating("b")
        s = arena.status()
        assert s["agents_rated"] == 2
        assert len(s["rankings"]) == 2


# ── Match Pairing Tests ─────────────────────────────────────────────────


class TestMatchPairing:
    def test_round_robin_two(self):
        arena = CompetitiveArena(
            config=ArenaConfig(tournament_style="round_robin"),
        )
        pairs = arena._get_pairs(["a", "b"])
        assert pairs == [(0, 1)]

    def test_round_robin_four(self):
        arena = CompetitiveArena(
            config=ArenaConfig(tournament_style="round_robin"),
        )
        pairs = arena._get_pairs(["a", "b", "c", "d"])
        assert len(pairs) == 6  # C(4,2)

    def test_single_elimination_four(self):
        arena = CompetitiveArena(
            config=ArenaConfig(tournament_style="single_elimination"),
        )
        pairs = arena._get_pairs(["a", "b", "c", "d"])
        assert len(pairs) == 2
        assert pairs == [(0, 1), (2, 3)]

    def test_single_elimination_three(self):
        arena = CompetitiveArena(
            config=ArenaConfig(tournament_style="single_elimination"),
        )
        pairs = arena._get_pairs(["a", "b", "c"])
        assert len(pairs) == 1  # only (0,1), c gets a bye

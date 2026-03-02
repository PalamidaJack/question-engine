"""CompetitiveArena — tournament-style verification competition between agents.

Implements agent-vs-agent competition with Elo ranking, cross-examination,
divergence checking (anti-sycophancy), and Thompson sampling agent selection.

Key principle: "Cooperate on generation, compete on verification."
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import uuid
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

import instructor
import litellm
from pydantic import BaseModel, Field

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
from qe.models.envelope import Envelope
from qe.services.inquiry.schemas import InquiryResult

log = logging.getLogger(__name__)

# Elo constants
_K_FACTOR = 32
_DEFAULT_ELO = 1200.0


# ---------------------------------------------------------------------------
# LLM Prompt Templates
# ---------------------------------------------------------------------------

_CROSS_EXAMINE_PROMPT = """\
You are a rigorous research critic. You are examining the findings of another \
research agent on the following goal: {goal_description}

The defender's findings:
{defender_findings}

Your task: identify weaknesses, unsupported claims, logical gaps, and \
questionable reasoning in the above findings. You CANNOT simply agree — \
you MUST find at least one genuine weakness or question.

Provide structured challenges, weaknesses, and questions."""

_DEFENSE_PROMPT = """\
You are a research agent defending your findings. Another agent has challenged \
your work on: {goal_description}

Your original findings:
{your_findings}

The challenges raised:
{challenges}

Respond with honest rebuttals where you can defend your work, \
concessions where the criticism is valid, and any additional evidence \
that supports your position. Do NOT blindly defend — concede where appropriate."""

_JUDGE_PROMPT = """\
You are an impartial research judge evaluating a debate between two agents \
investigating: {goal_description}

Agent A ({agent_a_id}) findings: {agent_a_findings}
Agent B ({agent_b_id}) findings: {agent_b_findings}

Agent A's challenges of B: {a_challenges_b}
Agent B's defense: {b_defense}
Agent B's challenges of A: {b_challenges_a}
Agent A's defense: {a_defense}

Score each agent (0-10) on three axes:
1. Factual accuracy — claims supported by evidence
2. Evidence quality — strength and relevance of evidence
3. Novelty — unique insights not found by the other agent

Determine a winner or declare a draw. Provide reasoning."""

_DIVERGENCE_PROMPT = """\
Compare the following research findings from multiple agents investigating: \
{goal_description}

{agent_findings}

Identify:
1. Claims that ALL agents share (shared_claims)
2. Claims that only SOME agents make (divergent_claims)
3. A similarity_score (0.0 = completely different, 1.0 = identical)

If similarity_score > {threshold}, flag sycophancy_risk as true — \
this means agents may be echoing each other rather than independently verifying."""


# ---------------------------------------------------------------------------
# Internal structured output models for LLM calls
# ---------------------------------------------------------------------------


class _CrossExamResult(BaseModel):
    challenges: list[str] = Field(default_factory=list)
    weaknesses_identified: list[str] = Field(default_factory=list)
    questions_raised: list[str] = Field(default_factory=list)


class _DefenseResult(BaseModel):
    rebuttals: list[str] = Field(default_factory=list)
    concessions: list[str] = Field(default_factory=list)
    additional_evidence: list[str] = Field(default_factory=list)


class _JudgeResult(BaseModel):
    agent_a_score: float = Field(default=5.0, ge=0.0, le=10.0)
    agent_b_score: float = Field(default=5.0, ge=0.0, le=10.0)
    factual_accuracy_a: float = Field(default=5.0, ge=0.0, le=10.0)
    factual_accuracy_b: float = Field(default=5.0, ge=0.0, le=10.0)
    evidence_quality_a: float = Field(default=5.0, ge=0.0, le=10.0)
    evidence_quality_b: float = Field(default=5.0, ge=0.0, le=10.0)
    novelty_a: float = Field(default=5.0, ge=0.0, le=10.0)
    novelty_b: float = Field(default=5.0, ge=0.0, le=10.0)
    winner: str = "draw"  # "agent_a", "agent_b", "draw"
    reasoning: str = ""


class _DivergenceResult(BaseModel):
    similarity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    shared_claims: list[str] = Field(default_factory=list)
    divergent_claims: list[str] = Field(default_factory=list)
    sycophancy_risk: bool = False


# ---------------------------------------------------------------------------
# Thompson Sampling Arm (local, lightweight)
# ---------------------------------------------------------------------------


class BetaArm:
    """Beta-Binomial conjugate prior for Thompson sampling."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0


# ---------------------------------------------------------------------------
# CompetitiveArena
# ---------------------------------------------------------------------------


class CompetitiveArena:
    """Tournament-style verification competition between cognitive agents."""

    def __init__(
        self,
        bus: Any = None,
        config: ArenaConfig | None = None,
        prompt_registry: Any = None,
    ) -> None:
        self._bus = bus
        self._config = config or ArenaConfig()
        self._prompt_registry = prompt_registry
        self._elo_ratings: dict[str, AgentEloRating] = {}
        self._agent_arms: dict[str, BetaArm] = {}
        self._total_cost_usd: float = 0.0

    # ── Public API ────────────────────────────────────────────────────

    async def run_tournament(
        self,
        goal_id: str,
        goal_description: str,
        results: list[InquiryResult],
        agent_ids: list[str],
    ) -> ArenaResult:
        """Run a full tournament between agents.

        Flow:
        1. Divergence check (anti-sycophancy)
        2. Cross-examination (per pair)
        3. Judging (per pair)
        4. Elo update
        """
        arena_id = f"arena_{uuid.uuid4().hex[:12]}"

        self._publish("arena.tournament_started", {
            "arena_id": arena_id,
            "goal_id": goal_id,
            "agent_ids": agent_ids,
        })

        # Ensure we have Elo ratings for all agents
        for aid in agent_ids:
            self.get_elo_rating(aid)

        # Phase 1: Divergence check
        divergence = await self._check_divergence(
            goal_description, results, agent_ids,
        )

        self._publish("arena.divergence_checked", {
            "arena_id": arena_id,
            "similarity_score": divergence.similarity_score,
            "sycophancy_risk": divergence.sycophancy_risk,
        })

        # If sycophancy detected, skip expensive cross-examination
        if divergence.sycophancy_risk:
            self._publish("arena.sycophancy_fallback", {
                "arena_id": arena_id,
                "similarity_score": divergence.similarity_score,
            })
            # Fallback: pick the agent with the best (longest) summary
            winner_idx = max(
                range(len(results)),
                key=lambda i: len(results[i].findings_summary),
            )
            winner_id = agent_ids[winner_idx]

            result = ArenaResult(
                arena_id=arena_id,
                goal_id=goal_id,
                divergence=divergence,
                winner_id=winner_id,
                rankings=self.get_rankings(),
                sycophancy_detected=True,
                total_cost_usd=self._total_cost_usd,
            )

            self._publish("arena.tournament_completed", {
                "arena_id": arena_id,
                "goal_id": goal_id,
                "winner_id": winner_id,
                "sycophancy_detected": True,
            })
            return result

        # Phase 2 & 3: Cross-examination + judging per pair
        pairs = self._get_pairs(agent_ids)
        matches: list[ArenaMatch] = []

        for a_idx, b_idx in pairs:
            # Budget check
            if self._total_cost_usd >= self._config.budget_limit_usd:
                log.warning(
                    "arena.budget_exceeded total=%.4f limit=%.4f",
                    self._total_cost_usd,
                    self._config.budget_limit_usd,
                )
                break

            match = await self._run_match(
                goal_description=goal_description,
                agent_a_id=agent_ids[a_idx],
                agent_b_id=agent_ids[b_idx],
                result_a=results[a_idx],
                result_b=results[b_idx],
            )
            matches.append(match)

            self._publish("arena.match_completed", {
                "arena_id": arena_id,
                "match_id": match.match_id,
                "agent_a_id": match.agent_a_id,
                "agent_b_id": match.agent_b_id,
                "winner": match.winner,
            })

        # Phase 4: Update Elo from match results
        for match in matches:
            if match.winner:
                loser = (
                    match.agent_b_id
                    if match.winner == match.agent_a_id
                    else match.agent_a_id
                )
                if match.winner == "draw":
                    self._update_elo(match.agent_a_id, match.agent_b_id, draw=True)
                else:
                    self._update_elo(match.winner, loser, draw=False)

        self._publish("arena.elo_updated", {
            "arena_id": arena_id,
            "ratings": {
                aid: self._elo_ratings[aid].elo
                for aid in agent_ids
                if aid in self._elo_ratings
            },
        })

        # Determine overall winner: most match wins, Elo as tiebreaker
        win_counts: dict[str, int] = dict.fromkeys(agent_ids, 0)
        for match in matches:
            if match.winner and match.winner != "draw":
                win_counts[match.winner] = win_counts.get(match.winner, 0) + 1

        def _agent_score(aid: str) -> tuple[int, float]:
            elo = self._elo_ratings.get(
                aid, AgentEloRating(agent_id=aid),
            ).elo
            return (win_counts.get(aid, 0), elo)

        winner_id = max(agent_ids, key=_agent_score)

        result = ArenaResult(
            arena_id=arena_id,
            goal_id=goal_id,
            matches=matches,
            divergence=divergence,
            winner_id=winner_id,
            rankings=self.get_rankings(),
            sycophancy_detected=False,
            total_cost_usd=self._total_cost_usd,
        )

        self._publish("arena.tournament_completed", {
            "arena_id": arena_id,
            "goal_id": goal_id,
            "winner_id": winner_id,
            "sycophancy_detected": False,
            "match_count": len(matches),
        })

        return result

    def get_elo_rating(self, agent_id: str) -> AgentEloRating:
        """Get or create Elo rating for an agent."""
        if agent_id not in self._elo_ratings:
            self._elo_ratings[agent_id] = AgentEloRating(agent_id=agent_id)
            self._agent_arms[agent_id] = BetaArm()
        return self._elo_ratings[agent_id]

    def get_rankings(self) -> list[AgentEloRating]:
        """All agents sorted by Elo descending."""
        return sorted(
            self._elo_ratings.values(),
            key=lambda r: r.elo,
            reverse=True,
        )

    def select_agents_for_arena(
        self, available_ids: list[str], n: int = 2,
    ) -> list[str]:
        """Select agents via Thompson sampling for arena participation."""
        if len(available_ids) <= n:
            return list(available_ids)

        # Ensure all have arms
        for aid in available_ids:
            if aid not in self._agent_arms:
                self._agent_arms[aid] = BetaArm()

        scored = [(aid, self._agent_arms[aid].sample()) for aid in available_ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scored[:n]]

    def status(self) -> dict[str, Any]:
        """Monitoring snapshot."""
        return {
            "total_cost_usd": self._total_cost_usd,
            "agents_rated": len(self._elo_ratings),
            "config": self._config.model_dump(),
            "rankings": [
                {"agent_id": r.agent_id, "elo": r.elo, "matches": r.total_matches}
                for r in self.get_rankings()
            ],
        }

    # ── Private: Tournament Phases ─────────────────────────────────────

    async def _check_divergence(
        self,
        goal_description: str,
        results: list[InquiryResult],
        agent_ids: list[str],
    ) -> DivergenceCheck:
        """Phase 1: Check similarity between agent outputs."""
        agent_findings = "\n\n".join(
            f"Agent {agent_ids[i]}: {r.findings_summary}"
            for i, r in enumerate(results)
        )

        prompt = _DIVERGENCE_PROMPT.format(
            goal_description=goal_description,
            agent_findings=agent_findings,
            threshold=self._config.divergence_threshold,
        )

        try:
            client = instructor.from_litellm(litellm.acompletion)
            result: _DivergenceResult = await client.chat.completions.create(
                model=self._config.judge_model,
                messages=[
                    {"role": "system", "content": "You are a research divergence analyst."},
                    {"role": "user", "content": prompt},
                ],
                response_model=_DivergenceResult,
            )

            return DivergenceCheck(
                similarity_score=result.similarity_score,
                shared_claims=result.shared_claims,
                divergent_claims=result.divergent_claims,
                sycophancy_risk=(
                    result.similarity_score > self._config.divergence_threshold
                ),
            )
        except Exception:
            log.warning("arena.divergence_check_failed", exc_info=True)
            return DivergenceCheck(similarity_score=0.0, sycophancy_risk=False)

    async def _run_match(
        self,
        goal_description: str,
        agent_a_id: str,
        agent_b_id: str,
        result_a: InquiryResult,
        result_b: InquiryResult,
    ) -> ArenaMatch:
        """Run a full head-to-head match between two agents."""
        match = ArenaMatch(agent_a_id=agent_a_id, agent_b_id=agent_b_id)

        # Phase 2: Cross-examination (both directions in parallel)
        exam_a_to_b, exam_b_to_a = await asyncio.gather(
            self._cross_examine(
                goal_description, agent_a_id, agent_b_id,
                result_b.findings_summary,
            ),
            self._cross_examine(
                goal_description, agent_b_id, agent_a_id,
                result_a.findings_summary,
            ),
        )
        match.examinations = [exam_a_to_b, exam_b_to_a]

        # Defenses (both directions in parallel)
        defense_b, defense_a = await asyncio.gather(
            self._defend(
                goal_description, agent_b_id, exam_a_to_b,
                result_b.findings_summary,
            ),
            self._defend(
                goal_description, agent_a_id, exam_b_to_a,
                result_a.findings_summary,
            ),
        )
        match.defenses = [defense_b, defense_a]

        # Phase 3: Judge
        judgment = await self._judge_match(
            goal_description=goal_description,
            agent_a_id=agent_a_id,
            agent_b_id=agent_b_id,
            result_a=result_a,
            result_b=result_b,
            exam_a_to_b=exam_a_to_b,
            exam_b_to_a=exam_b_to_a,
            defense_a=defense_a,
            defense_b=defense_b,
        )
        match.judgment = judgment

        # Determine winner
        if judgment.winner == "agent_a":
            match.winner = agent_a_id
        elif judgment.winner == "agent_b":
            match.winner = agent_b_id
        else:
            match.winner = "draw"

        return match

    async def _cross_examine(
        self,
        goal_description: str,
        challenger_id: str,
        defender_id: str,
        defender_findings: str,
    ) -> CrossExamination:
        """Generate a cross-examination challenge."""
        prompt = _CROSS_EXAMINE_PROMPT.format(
            goal_description=goal_description,
            defender_findings=defender_findings,
        )

        try:
            client = instructor.from_litellm(litellm.acompletion)
            result: _CrossExamResult = await client.chat.completions.create(
                model=self._config.examination_model,
                messages=[
                    {"role": "system", "content": "You are a rigorous research critic."},
                    {"role": "user", "content": prompt},
                ],
                response_model=_CrossExamResult,
            )
            return CrossExamination(
                challenger_id=challenger_id,
                defender_id=defender_id,
                challenges=result.challenges,
                weaknesses_identified=result.weaknesses_identified,
                questions_raised=result.questions_raised,
            )
        except Exception:
            log.warning("arena.cross_examine_failed", exc_info=True)
            return CrossExamination(
                challenger_id=challenger_id,
                defender_id=defender_id,
            )

    async def _defend(
        self,
        goal_description: str,
        defender_id: str,
        examination: CrossExamination,
        findings: str,
    ) -> DefenseResponse:
        """Generate a defense response."""
        challenges_text = "\n".join(
            f"- {c}" for c in examination.challenges
        )
        prompt = _DEFENSE_PROMPT.format(
            goal_description=goal_description,
            your_findings=findings,
            challenges=challenges_text,
        )

        try:
            client = instructor.from_litellm(litellm.acompletion)
            result: _DefenseResult = await client.chat.completions.create(
                model=self._config.examination_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research agent defending your findings.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_model=_DefenseResult,
            )
            return DefenseResponse(
                defender_id=defender_id,
                examination_id=examination.examination_id,
                rebuttals=result.rebuttals,
                concessions=result.concessions,
                additional_evidence=result.additional_evidence,
            )
        except Exception:
            log.warning("arena.defense_failed", exc_info=True)
            return DefenseResponse(
                defender_id=defender_id,
                examination_id=examination.examination_id,
            )

    async def _judge_match(
        self,
        goal_description: str,
        agent_a_id: str,
        agent_b_id: str,
        result_a: InquiryResult,
        result_b: InquiryResult,
        exam_a_to_b: CrossExamination,
        exam_b_to_a: CrossExamination,
        defense_a: DefenseResponse,
        defense_b: DefenseResponse,
    ) -> MatchJudgment:
        """Independent judge scoring."""
        prompt = _JUDGE_PROMPT.format(
            goal_description=goal_description,
            agent_a_id=agent_a_id,
            agent_b_id=agent_b_id,
            agent_a_findings=result_a.findings_summary,
            agent_b_findings=result_b.findings_summary,
            a_challenges_b="; ".join(exam_a_to_b.challenges),
            b_defense="; ".join(defense_b.rebuttals),
            b_challenges_a="; ".join(exam_b_to_a.challenges),
            a_defense="; ".join(defense_a.rebuttals),
        )

        try:
            client = instructor.from_litellm(litellm.acompletion)
            result: _JudgeResult = await client.chat.completions.create(
                model=self._config.judge_model,
                messages=[
                    {"role": "system", "content": "You are an impartial research judge."},
                    {"role": "user", "content": prompt},
                ],
                response_model=_JudgeResult,
            )

            winner = result.winner
            if winner not in ("agent_a", "agent_b", "draw"):
                winner = "draw"

            return MatchJudgment(
                agent_a_id=agent_a_id,
                agent_b_id=agent_b_id,
                agent_a_score=result.agent_a_score,
                agent_b_score=result.agent_b_score,
                factual_accuracy_a=result.factual_accuracy_a,
                factual_accuracy_b=result.factual_accuracy_b,
                evidence_quality_a=result.evidence_quality_a,
                evidence_quality_b=result.evidence_quality_b,
                novelty_a=result.novelty_a,
                novelty_b=result.novelty_b,
                winner=winner,
                reasoning=result.reasoning,
            )
        except Exception:
            log.warning("arena.judge_failed", exc_info=True)
            return MatchJudgment(
                agent_a_id=agent_a_id,
                agent_b_id=agent_b_id,
                winner="draw",
                reasoning="Judgment failed — defaulting to draw.",
            )

    # ── Private: Elo Math ─────────────────────────────────────────────

    def _update_elo(
        self, winner_id: str, loser_id: str, *, draw: bool = False,
    ) -> None:
        """Standard Elo update with K=32."""
        winner = self.get_elo_rating(winner_id)
        loser = self.get_elo_rating(loser_id)

        expected_w = 1.0 / (1.0 + math.pow(10, (loser.elo - winner.elo) / 400.0))
        expected_l = 1.0 - expected_w

        if draw:
            score_w = 0.5
            score_l = 0.5
            winner.draws += 1
            loser.draws += 1
        else:
            score_w = 1.0
            score_l = 0.0
            winner.wins += 1
            loser.losses += 1

        winner.elo += _K_FACTOR * (score_w - expected_w)
        loser.elo += _K_FACTOR * (score_l - expected_l)
        winner.last_updated = datetime.now(UTC)
        loser.last_updated = datetime.now(UTC)

        # Update Thompson arms
        if winner_id in self._agent_arms:
            self._agent_arms[winner_id].update(not draw)
        if loser_id in self._agent_arms:
            self._agent_arms[loser_id].update(draw)

    # ── Private: Helpers ──────────────────────────────────────────────

    def _get_pairs(self, agent_ids: list[str]) -> list[tuple[int, int]]:
        """Get match pairs based on tournament style."""
        if self._config.tournament_style == "round_robin":
            return list(combinations(range(len(agent_ids)), 2))
        # single_elimination: sequential pairs
        pairs = []
        for i in range(0, len(agent_ids) - 1, 2):
            pairs.append((i, i + 1))
        return pairs

    def _publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish bus event if bus is available."""
        if self._bus is None:
            return
        try:
            self._bus.publish(
                Envelope(
                    topic=topic,
                    source_service_id="competitive_arena",
                    payload=payload,
                )
            )
        except Exception:
            log.debug("arena.publish_failed topic=%s", topic)

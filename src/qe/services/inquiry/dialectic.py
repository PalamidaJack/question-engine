"""Dialectic Engine — adversarial self-critique and perspective rotation.

For every significant conclusion, generates counterarguments, rotates
perspectives, surfaces assumptions, and attempts falsification. All prompts
are engineered so the LLM CANNOT just agree — it MUST produce genuine
counterarguments.
"""

from __future__ import annotations

import logging
from typing import Any

import instructor
import litellm
from pydantic import BaseModel

from qe.models.cognition import (
    AssumptionChallenge,
    Counterargument,
    DialecticReport,
    PerspectiveAnalysis,
)
from qe.runtime.episodic_memory import Episode, EpisodicMemory

log = logging.getLogger(__name__)

# Domain-aware perspective sets
PERSPECTIVE_SETS: dict[str, list[str]] = {
    "financial": [
        "bullish analyst",
        "bearish analyst",
        "risk manager",
        "retail investor",
        "regulator",
    ],
    "technology": [
        "early adopter",
        "skeptic",
        "enterprise buyer",
        "security researcher",
        "end user",
    ],
    "general": [
        "optimist",
        "pessimist",
        "contrarian",
        "insider",
        "outsider",
    ],
    "scientific": [
        "proponent",
        "critic",
        "methodologist",
        "replication advocate",
        "practitioner",
    ],
}

_DEVILS_ADVOCATE_PROMPT = """\
You are a ruthlessly honest devil's advocate. Your ONLY job is to find \
flaws in the following conclusion. You MUST argue against it, even if \
you personally agree with it.

CONCLUSION: {conclusion}
EVIDENCE PRESENTED: {evidence}

Rules:
1. You MUST find at least one strong counterargument. Saying "the \
conclusion seems solid" is NOT acceptable.
2. For each counterargument, explain what evidence would disprove \
the original conclusion.
3. Rate the strength of each counterargument honestly.
4. State what the counterargument CONCEDES (what's right about \
the original).
"""

_PERSPECTIVE_PROMPT = """\
You are analyzing the following situation from the perspective of a \
{perspective_name}.

SITUATION: {situation}
EVIDENCE: {evidence}

As a {perspective_name}, analyze:
1. What key observations stand out to you?
2. What risks do you see that others might miss?
3. What opportunities do you see?
4. What is your overall assessment?

Stay firmly in character. Do not hedge or give a balanced view — \
give the view this specific perspective would hold.
"""

_ASSUMPTION_SURFACING_PROMPT = """\
You are an assumption-detection module. Every analysis rests on hidden \
assumptions. Your job is to FIND and CHALLENGE them.

ANALYSIS: {analysis}
EXPLICIT ASSUMPTIONS: {explicit_assumptions}

Instructions:
1. List HIDDEN assumptions that are NOT in the explicit list \
(these are more important).
2. For EACH assumption (hidden and explicit), describe:
   - What happens if this assumption is WRONG
   - Whether it can be TESTED
   - HOW to test it
3. Focus on assumptions that, if wrong, would INVALIDATE the analysis.
"""

_RED_TEAM_PROMPT = """\
You are a red team analyst. Your job is to construct the STRONGEST \
possible case that the following finding is WRONG.

FINDING: {finding}
SUPPORTING EVIDENCE: {evidence}

Construct your attack:
1. What alternative explanations exist for the same evidence?
2. What biases could have led to this conclusion?
3. What evidence would DISPROVE this finding?
4. If you had to bet AGAINST this finding, what would your argument be?

You MUST produce a substantive attack. "The finding seems correct" \
is NOT acceptable output.
"""


class DialecticEngine:
    """Adversarial self-critique: challenges conclusions before they
    become insights.

    All methods return structured outputs that feed into the
    InsightCrystallizer. Nothing passes into the final output without
    surviving dialectic scrutiny.
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory | None = None,
        model: str = "openai/anthropic/claude-sonnet-4",
        prompt_registry: Any | None = None,
    ) -> None:
        self._episodic = episodic_memory
        self._model = model
        self._registry = prompt_registry
        self._fallbacks: dict[str, str] = {
            "dialectic.challenge.system": (
                "You are a devil's advocate. "
                "You MUST argue against the conclusion."
            ),
            "dialectic.challenge.user": _DEVILS_ADVOCATE_PROMPT,
            "dialectic.perspectives.system": (
                "You are a {perspective_name}. Stay in character."
            ),
            "dialectic.perspectives.user": _PERSPECTIVE_PROMPT,
            "dialectic.assumptions.system": (
                "You are an assumption-detection module."
            ),
            "dialectic.assumptions.user": _ASSUMPTION_SURFACING_PROMPT,
            "dialectic.red_team.system": (
                "You are a red team analyst. "
                "You MUST attack the finding."
            ),
            "dialectic.red_team.user": _RED_TEAM_PROMPT,
        }

    def _get_prompt(self, slot_key: str, **fmt: Any) -> tuple[str, str]:
        """Get prompt content, preferring registry variants when available."""
        if self._registry is not None:
            content, vid = self._registry.get_prompt(slot_key)
            try:
                return content.format(**fmt) if fmt else content, vid
            except KeyError:
                pass  # Variant has wrong format keys — fall back
        fallback = self._fallbacks.get(slot_key, "")
        return (fallback.format(**fmt) if fmt else fallback), "baseline"

    async def challenge(
        self,
        goal_id: str,
        conclusion: str,
        evidence: str = "",
    ) -> list[Counterargument]:
        """Generate devil's advocate counterarguments."""
        sys_content, _ = self._get_prompt("dialectic.challenge.system")
        user_content, user_vid = self._get_prompt(
            "dialectic.challenge.user",
            conclusion=conclusion,
            evidence=evidence or "Not explicitly provided.",
        )

        class CounterargumentList(BaseModel):
            counterarguments: list[Counterargument]

        client = instructor.from_litellm(litellm.acompletion)
        try:
            result = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=CounterargumentList,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    user_vid, "dialectic.challenge.user", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                user_vid, "dialectic.challenge.user", success=True
            )

        if self._episodic:
            await self._episodic.store(
                Episode(
                    goal_id=goal_id,
                    episode_type="observation",
                    content={
                        "phase": "dialectic",
                        "type": "challenge",
                        "count": len(result.counterarguments),
                    },
                    summary=(
                        f"Devil's advocate produced "
                        f"{len(result.counterarguments)} counterarguments"
                    ),
                    relevance_to_goal=0.85,
                )
            )

        return result.counterarguments

    async def rotate_perspectives(
        self,
        goal_id: str,
        situation: str,
        evidence: str = "",
        domain: str = "general",
        custom_perspectives: list[str] | None = None,
    ) -> list[PerspectiveAnalysis]:
        """Analyze from multiple forced perspectives."""
        perspectives = custom_perspectives or PERSPECTIVE_SETS.get(
            domain, PERSPECTIVE_SETS["general"]
        )
        analyses: list[PerspectiveAnalysis] = []

        for perspective in perspectives:
            sys_content, _ = self._get_prompt(
                "dialectic.perspectives.system",
                perspective_name=perspective,
            )
            user_content, user_vid = self._get_prompt(
                "dialectic.perspectives.user",
                perspective_name=perspective,
                situation=situation,
                evidence=evidence or "General knowledge.",
            )

            client = instructor.from_litellm(litellm.acompletion)
            try:
                analysis = await client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": sys_content},
                        {"role": "user", "content": user_content},
                    ],
                    response_model=PerspectiveAnalysis,
                )
            except Exception:
                if self._registry:
                    self._registry.record_outcome(
                        user_vid, "dialectic.perspectives.user", success=False
                    )
                raise
            if self._registry:
                self._registry.record_outcome(
                    user_vid, "dialectic.perspectives.user", success=True
                )
            analysis.perspective_name = perspective
            analyses.append(analysis)

        return analyses

    async def surface_assumptions(
        self,
        goal_id: str,
        analysis: str,
        explicit_assumptions: list[str] | None = None,
    ) -> list[AssumptionChallenge]:
        """Find and challenge both explicit and hidden assumptions."""
        assumptions_str = (
            "\n".join(f"- {a}" for a in (explicit_assumptions or []))
            or "None stated."
        )
        sys_content, _ = self._get_prompt("dialectic.assumptions.system")
        user_content, user_vid = self._get_prompt(
            "dialectic.assumptions.user",
            analysis=analysis,
            explicit_assumptions=assumptions_str,
        )

        class AssumptionList(BaseModel):
            assumptions: list[AssumptionChallenge]

        client = instructor.from_litellm(litellm.acompletion)
        try:
            result = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=AssumptionList,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    user_vid, "dialectic.assumptions.user", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                user_vid, "dialectic.assumptions.user", success=True
            )
        return result.assumptions

    async def red_team(
        self,
        goal_id: str,
        finding: str,
        evidence: str = "",
    ) -> Counterargument:
        """Construct the strongest possible case AGAINST a finding."""
        sys_content, _ = self._get_prompt("dialectic.red_team.system")
        user_content, user_vid = self._get_prompt(
            "dialectic.red_team.user",
            finding=finding,
            evidence=evidence or "Not explicitly provided.",
        )

        client = instructor.from_litellm(litellm.acompletion)
        try:
            result = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=Counterargument,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    user_vid, "dialectic.red_team.user", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                user_vid, "dialectic.red_team.user", success=True
            )
        return result

    async def full_dialectic(
        self,
        goal_id: str,
        conclusion: str,
        evidence: str = "",
        explicit_assumptions: list[str] | None = None,
        domain: str = "general",
    ) -> DialecticReport:
        """Run the complete dialectic pipeline on a conclusion.

        Runs: challenge + perspective rotation + assumption surfacing.
        Computes revised confidence and flags investigation questions.
        """
        counterarguments = await self.challenge(
            goal_id, conclusion, evidence
        )
        perspectives = await self.rotate_perspectives(
            goal_id, conclusion, evidence, domain
        )
        assumptions = await self.surface_assumptions(
            goal_id, conclusion, explicit_assumptions
        )

        # Revised confidence based on dialectic results
        strong_counters = sum(
            1
            for c in counterarguments
            if c.strength in ("strong", "decisive")
        )
        total_counters = len(counterarguments)
        hidden_assumptions = sum(
            1 for a in assumptions if not a.is_explicit
        )

        revised_confidence = 0.7
        if total_counters > 0:
            revised_confidence -= 0.1 * (
                strong_counters / total_counters
            )
        if hidden_assumptions > 2:
            revised_confidence -= 0.1
        revised_confidence = max(0.1, min(0.95, revised_confidence))

        should_investigate = (
            strong_counters > 0 or hidden_assumptions > 2
        )

        investigation_questions: list[str] = []
        for c in counterarguments:
            if (
                c.strength in ("strong", "decisive")
                and c.evidence_needed_to_resolve
            ):
                investigation_questions.append(
                    c.evidence_needed_to_resolve
                )
        for a in assumptions:
            if not a.is_explicit and a.testable and a.test_method:
                investigation_questions.append(
                    f"Test assumption: {a.test_method}"
                )

        synthesis = (
            f"Original conclusion "
            f"({revised_confidence:.0%} confidence after dialectic). "
        )
        if strong_counters == 0:
            synthesis += "No decisive counterarguments found. "
        else:
            synthesis += (
                f"{strong_counters} strong counterarguments "
                f"require attention. "
            )
        if hidden_assumptions > 0:
            synthesis += (
                f"{hidden_assumptions} hidden assumptions surfaced. "
            )

        report = DialecticReport(
            original_conclusion=conclusion,
            counterarguments=counterarguments,
            perspectives=perspectives,
            assumptions_challenged=assumptions,
            revised_confidence=revised_confidence,
            should_investigate_further=should_investigate,
            investigation_questions=investigation_questions,
            synthesis=synthesis,
        )

        if self._episodic:
            await self._episodic.store(
                Episode(
                    goal_id=goal_id,
                    episode_type="synthesis",
                    content={
                        "phase": "dialectic",
                        "revised_confidence": revised_confidence,
                        "strong_counters": strong_counters,
                        "hidden_assumptions": hidden_assumptions,
                    },
                    summary=(
                        f"Full dialectic: {len(counterarguments)} "
                        f"counters, confidence {revised_confidence:.0%}"
                    ),
                    relevance_to_goal=0.95,
                )
            )

        return report

    def status(self) -> dict[str, Any]:
        """Monitoring snapshot."""
        return {"model": self._model}

"""Persistence Engine — determination, reframing, and root cause analysis.

Key difference from RecoveryOrchestrator:
- Recovery answers "what to do next mechanically" (retry, escalate, simplify)
- Persistence answers "WHY did this fail and HOW to think about it differently"

Implements Why-Why-Why root cause chains (min 3 levels), 7 named reframing
strategies, and lesson accumulation across goals.
"""

from __future__ import annotations

import logging
from typing import Any

import instructor
import litellm

from qe.models.cognition import (
    ReframingResult,
    RootCauseAnalysis,
)
from qe.runtime.episodic_memory import Episode, EpisodicMemory

log = logging.getLogger(__name__)

_ROOT_CAUSE_PROMPT = """\
You are a root cause analysis module. Apply the "5 Whys" technique to \
understand why a failure occurred.

FAILURE: {failure_summary}
CONTEXT: {context}

For each "Why?" level:
1. Ask "Why did [X] happen?"
2. Provide the answer
3. Assess confidence in that answer
4. Note if the answer is actionable (can we fix it?)

Go at least 3 levels deep. Stop when you reach a root cause that is \
actionable or when you hit an external constraint we cannot change.
"""

_REFRAME_PROMPT = """\
You are a creative problem reframing module. The system is STUCK on a \
problem and standard approaches have failed.

ORIGINAL PROBLEM: {original}
WHAT WAS TRIED: {tried}
WHY IT FAILED: {failure}

Reframe this problem using the specified strategy: {strategy}

Strategy descriptions:
- inversion: "Instead of finding X, what would IMPLY X?"
- implication: "What would be true IF X were true/false?"
- proxy: "What measurable proxy could substitute for X?"
- decompose_differently: "Break the problem into different pieces"
- change_domain: "Look for X in a completely different domain"
- stakeholder_shift: "Who else would know about X?"
- temporal_shift: "When was X different? What changed?"

Provide a SPECIFIC reframed question/approach, not a vague suggestion.
"""

REFRAMING_STRATEGIES: list[str] = [
    "inversion",
    "proxy",
    "stakeholder_shift",
    "decompose_differently",
    "implication",
    "change_domain",
    "temporal_shift",
]


class PersistenceEngine:
    """Makes the system determined: root cause analysis, reframing,
    and lesson accumulation.
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory | None = None,
        model: str = "openai/google/gemini-2.0-flash",
    ) -> None:
        self._episodic = episodic_memory
        self._model = model
        self._reframe_history: dict[str, list[str]] = {}
        self._lessons: list[dict[str, str]] = []

    # -------------------------------------------------------------------
    # Root Cause Analysis
    # -------------------------------------------------------------------

    async def analyze_root_cause(
        self,
        goal_id: str,
        failure_summary: str,
        context: str = "",
    ) -> RootCauseAnalysis:
        """Why-Why-Why analysis, minimum 3 levels deep."""
        prompt = _ROOT_CAUSE_PROMPT.format(
            failure_summary=failure_summary,
            context=context or "No additional context.",
        )

        client = instructor.from_litellm(litellm.acompletion)
        analysis = await client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a root cause analysis expert.",
                },
                {"role": "user", "content": prompt},
            ],
            response_model=RootCauseAnalysis,
        )

        if len(analysis.chain) < 3:
            log.warning(
                "Root cause chain only %d levels deep, expected >= 3",
                len(analysis.chain),
            )

        # Store lesson
        if analysis.lesson_learned:
            self._lessons.append({
                "goal_id": goal_id,
                "failure": failure_summary,
                "root_cause": analysis.root_cause,
                "lesson": analysis.lesson_learned,
                "prevention": analysis.prevention_strategy,
            })

        if self._episodic:
            await self._episodic.store(
                Episode(
                    goal_id=goal_id,
                    episode_type="observation",
                    content={
                        "phase": "persistence",
                        "root_cause": analysis.model_dump(),
                    },
                    summary=f"Root cause: {analysis.root_cause}",
                    relevance_to_goal=0.9,
                )
            )

        return analysis

    # -------------------------------------------------------------------
    # Reframing
    # -------------------------------------------------------------------

    async def reframe(
        self,
        goal_id: str,
        original_problem: str,
        tried_approaches: str = "",
        failure_reason: str = "",
        strategy: str | None = None,
    ) -> ReframingResult:
        """Reframe a stuck problem using a specific strategy.

        If no strategy specified, picks the next untried one.
        """
        history = self._reframe_history.setdefault(goal_id, [])

        if strategy is None:
            for s in REFRAMING_STRATEGIES:
                if s not in history:
                    strategy = s
                    break
            if strategy is None:
                strategy = "decompose_differently"

        history.append(strategy)

        prompt = _REFRAME_PROMPT.format(
            original=original_problem,
            tried=tried_approaches or "None specified.",
            failure=failure_reason or "Not specified.",
            strategy=strategy,
        )

        client = instructor.from_litellm(litellm.acompletion)
        result = await client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a creative problem reframing module."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_model=ReframingResult,
        )

        if self._episodic:
            await self._episodic.store(
                Episode(
                    goal_id=goal_id,
                    episode_type="observation",
                    content={
                        "phase": "persistence",
                        "strategy": strategy,
                        "reframed": result.reframed_question[:100],
                    },
                    summary=(
                        f"Reframed via {strategy}: "
                        f"{result.reframed_question[:80]}"
                    ),
                    relevance_to_goal=0.85,
                )
            )

        return result

    async def reframe_cascade(
        self,
        goal_id: str,
        original_problem: str,
        tried_approaches: str = "",
        failure_reason: str = "",
        max_reframes: int = 3,
    ) -> list[ReframingResult]:
        """Try multiple reframing strategies, return all results."""
        results: list[ReframingResult] = []
        for _ in range(max_reframes):
            result = await self.reframe(
                goal_id,
                original_problem,
                tried_approaches,
                failure_reason,
            )
            results.append(result)
            if result.estimated_tractability > 0.7:
                break
        return results

    # -------------------------------------------------------------------
    # Lesson Management
    # -------------------------------------------------------------------

    def get_relevant_lessons(
        self,
        context: str,
        top_k: int = 5,
    ) -> list[dict[str, str]]:
        """Retrieve lessons relevant to a context (keyword match)."""
        if not self._lessons:
            return []
        context_words = set(context.lower().split())
        scored: list[tuple[int, dict[str, str]]] = []
        for lesson in self._lessons:
            lesson_text = (
                f"{lesson.get('failure', '')} "
                f"{lesson.get('root_cause', '')} "
                f"{lesson.get('lesson', '')}"
            ).lower()
            lesson_words = set(lesson_text.split())
            overlap = len(context_words & lesson_words)
            if overlap > 0:
                scored.append((overlap, lesson))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [lesson for _, lesson in scored[:top_k]]

    def reframing_strategies_remaining(self, goal_id: str) -> list[str]:
        """Which reframing strategies haven't been tried yet?"""
        tried = set(self._reframe_history.get(goal_id, []))
        return [s for s in REFRAMING_STRATEGIES if s not in tried]

    # -------------------------------------------------------------------
    # Cleanup & Status
    # -------------------------------------------------------------------

    def clear_goal(self, goal_id: str) -> None:
        """Clean up state for a completed goal."""
        self._reframe_history.pop(goal_id, None)

    def status(self) -> dict[str, Any]:
        """Monitoring snapshot."""
        return {
            "total_lessons": len(self._lessons),
            "active_reframe_histories": len(self._reframe_history),
        }

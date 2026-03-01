"""Epistemic Reasoner — manages what the system knows vs. doesn't know.

After every investigation step, explicitly reasons about absences,
quantifies uncertainty, maintains known unknowns, and detects surprises
by comparing findings against existing beliefs.
"""

from __future__ import annotations

import logging
from typing import Any

import instructor
import litellm
from pydantic import BaseModel

from qe.models.cognition import (
    AbsenceDetection,
    EpistemicState,
    KnownUnknown,
    SurpriseDetection,
    UncertaintyAssessment,
)
from qe.runtime.episodic_memory import Episode, EpisodicMemory
from qe.substrate.bayesian_belief import BayesianBeliefStore

log = logging.getLogger(__name__)


_ABSENCE_DETECTION_PROMPT = """\
You are an epistemic reasoning module. After an investigation step, \
you must identify what EXPECTED data was NOT found.

GOAL: {goal_description}
INVESTIGATION STEP: {investigation_description}
RESULTS OBTAINED: {results_summary}

Think carefully:
1. Given the goal, what data would a knowledgeable analyst EXPECT to find?
2. Which of those expected data points are MISSING from the results?
3. For each missing item, why might it be absent?
4. How significant is each absence?

Focus on INFORMATIVE absences — things whose absence tells us something.
Do not list trivial or irrelevant missing data.
"""

_UNCERTAINTY_PROMPT = """\
You are an epistemic reasoning module assessing uncertainty.

FINDING: {finding}
SOURCE: {source}
EVIDENCE: {evidence}

Assess this finding's uncertainty:
1. How confident should we be? (very_low/low/moderate/high/very_high)
2. What is the evidence quality? (primary/secondary/hearsay/inferred)
3. What biases might affect this finding?
4. What information gaps remain?
5. Under what conditions could this finding be WRONG?
"""

_SURPRISE_PROMPT = """\
You are a surprise detection module. Compare a new finding against \
existing beliefs.

NEW FINDING: {new_finding}
EXISTING BELIEFS about {entity}:
{existing_beliefs}

Is this finding surprising given the existing beliefs?
If so, explain:
1. What was expected instead?
2. How surprising is it (0.0 = mildly unexpected, 1.0 = shocking)?
3. What are the implications?
"""


class EpistemicReasoner:
    """Manages the system's epistemic state per goal.

    Core functions:
    - detect_absences(): find what's missing after investigation
    - assess_uncertainty(): structured confidence per finding
    - detect_surprise(): flag contradictions with beliefs
    - maintain known unknowns + blind spot warnings
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory | None = None,
        belief_store: BayesianBeliefStore | None = None,
        model: str = "openai/google/gemini-2.0-flash",
        prompt_registry: Any | None = None,
    ) -> None:
        self._episodic = episodic_memory
        self._belief_store = belief_store
        self._model = model
        self._states: dict[str, EpistemicState] = {}
        self._registry = prompt_registry
        self._fallbacks: dict[str, str] = {
            "epistemic.absence.system": "You are an epistemic reasoning module.",
            "epistemic.absence.user": _ABSENCE_DETECTION_PROMPT,
            "epistemic.uncertainty.system": "You are an epistemic reasoning module.",
            "epistemic.uncertainty.user": _UNCERTAINTY_PROMPT,
            "epistemic.surprise.system": "You are a surprise detection module.",
            "epistemic.surprise.user": _SURPRISE_PROMPT,
        }

    def _get_prompt(self, slot_key: str, **fmt: Any) -> tuple[str, str]:
        """Get prompt content, preferring registry variants when available."""
        if self._registry is not None:
            content, vid = self._registry.get_prompt(slot_key)
            try:
                return content.format(**fmt) if fmt else content, vid
            except KeyError:
                pass
        fallback = self._fallbacks.get(slot_key, "")
        return (fallback.format(**fmt) if fmt else fallback), "baseline"

    def get_or_create_state(self, goal_id: str) -> EpistemicState:
        """Get or create epistemic state for a goal."""
        if goal_id not in self._states:
            self._states[goal_id] = EpistemicState(goal_id=goal_id)
        return self._states[goal_id]

    # -------------------------------------------------------------------
    # Absence Detection
    # -------------------------------------------------------------------

    async def detect_absences(
        self,
        goal_id: str,
        goal_description: str,
        investigation_description: str,
        results: list[dict[str, Any]],
    ) -> list[AbsenceDetection]:
        """After an investigation step, reason about what was NOT found."""
        state = self.get_or_create_state(goal_id)

        # Heuristic: empty results = significant absence
        if not results:
            empty_absence = AbsenceDetection(
                expected_data=(
                    f"Any results for: {investigation_description}"
                ),
                why_expected=(
                    "Investigation was designed to find this"
                ),
                search_scope=investigation_description,
                significance="high",
                possible_explanations=[
                    "Data does not exist",
                    "Search terms were wrong",
                    "Data is behind access restrictions",
                    "Data exists under different terminology",
                ],
            )
            state.absences.append(empty_absence)

        # LLM reasoning about informative absences
        results_summary = (
            "\n".join(str(r)[:200] for r in results[:10])
            if results
            else "No results obtained."
        )

        sys_content, _ = self._get_prompt("epistemic.absence.system")
        user_content, user_vid = self._get_prompt(
            "epistemic.absence.user",
            goal_description=goal_description,
            investigation_description=investigation_description,
            results_summary=results_summary,
        )

        class AbsenceList(BaseModel):
            absences: list[AbsenceDetection]

        client = instructor.from_litellm(litellm.acompletion)
        try:
            result = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=AbsenceList,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    user_vid, "epistemic.absence.user", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                user_vid, "epistemic.absence.user", success=True
            )

        state.absences.extend(result.absences)

        if self._episodic:
            await self._episodic.store(
                Episode(
                    goal_id=goal_id,
                    episode_type="observation",
                    content={
                        "phase": "epistemic",
                        "absences": [a.model_dump() for a in result.absences],
                    },
                    summary=(
                        f"Detected {len(result.absences)} informative absences"
                    ),
                    relevance_to_goal=0.8,
                )
            )

        return result.absences

    # -------------------------------------------------------------------
    # Uncertainty Assessment
    # -------------------------------------------------------------------

    async def assess_uncertainty(
        self,
        goal_id: str,
        finding: str,
        source: str = "",
        evidence: str = "",
    ) -> UncertaintyAssessment:
        """Produce a structured uncertainty assessment for a finding."""
        sys_content, _ = self._get_prompt("epistemic.uncertainty.system")
        user_content, user_vid = self._get_prompt(
            "epistemic.uncertainty.user",
            finding=finding,
            source=source or "unknown",
            evidence=evidence or "not specified",
        )

        client = instructor.from_litellm(litellm.acompletion)
        try:
            assessment = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=UncertaintyAssessment,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    user_vid, "epistemic.uncertainty.user", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                user_vid, "epistemic.uncertainty.user", success=True
            )

        state = self.get_or_create_state(goal_id)
        state.known_facts.append(assessment)
        return assessment

    # -------------------------------------------------------------------
    # Surprise Detection
    # -------------------------------------------------------------------

    async def detect_surprise(
        self,
        goal_id: str,
        entity_id: str,
        new_finding: str,
    ) -> SurpriseDetection | None:
        """Check if a finding contradicts existing beliefs about an entity."""
        if not self._belief_store:
            return None

        beliefs = await self._belief_store.get_beliefs_for_entity(entity_id)
        if not beliefs:
            return None

        beliefs_summary = "\n".join(
            f"- {b.claim.predicate}: {b.claim.object_value} "
            f"(confidence: {b.posterior:.2f})"
            for b in beliefs
        )

        sys_content, _ = self._get_prompt("epistemic.surprise.system")
        user_content, user_vid = self._get_prompt(
            "epistemic.surprise.user",
            new_finding=new_finding,
            entity=entity_id,
            existing_beliefs=beliefs_summary,
        )

        client = instructor.from_litellm(litellm.acompletion)
        try:
            surprise = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=SurpriseDetection,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    user_vid, "epistemic.surprise.user", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                user_vid, "epistemic.surprise.user", success=True
            )

        if surprise.surprise_magnitude > 0.3:
            state = self.get_or_create_state(goal_id)
            surprise.related_belief_ids = [
                b.claim.claim_id for b in beliefs
            ]
            state.surprises.append(surprise)
            return surprise

        return None

    # -------------------------------------------------------------------
    # Known Unknowns
    # -------------------------------------------------------------------

    def register_unknown(
        self,
        goal_id: str,
        question: str,
        why_unknown: str,
        importance: str = "medium",
    ) -> KnownUnknown:
        """Register a question the system cannot currently answer."""
        unknown = KnownUnknown(
            question=question,
            why_unknown=why_unknown,
            importance=importance,
        )
        state = self.get_or_create_state(goal_id)
        state.known_unknowns.append(unknown)
        return unknown

    def resolve_unknown(self, goal_id: str, unknown_id: str) -> None:
        """Remove a known unknown when it gets answered."""
        state = self._states.get(goal_id)
        if state:
            state.known_unknowns = [
                u for u in state.known_unknowns
                if u.unknown_id != unknown_id
            ]

    # -------------------------------------------------------------------
    # Blind Spot Warnings
    # -------------------------------------------------------------------

    def get_blind_spot_warning(self, goal_id: str) -> str:
        """Generate a warning about potential blind spots."""
        state = self.get_or_create_state(goal_id)
        warnings = []
        if not state.known_unknowns:
            warnings.append(
                "No known unknowns registered. This is suspicious — "
                "are we asking the right questions?"
            )
        if not state.absences:
            warnings.append(
                "No absences detected. Have we checked what SHOULD "
                "be there but isn't?"
            )
        if (
            state.known_facts
            and all(
                f.confidence_level in ("high", "very_high")
                for f in state.known_facts
            )
        ):
            warnings.append(
                "All findings have high confidence. This is unusual "
                "and may indicate confirmation bias."
            )
        return " | ".join(warnings) if warnings else ""

    # -------------------------------------------------------------------
    # State Access & Cleanup
    # -------------------------------------------------------------------

    def get_epistemic_state(self, goal_id: str) -> EpistemicState:
        """Get the full epistemic state for a goal."""
        return self.get_or_create_state(goal_id)

    def clear_goal(self, goal_id: str) -> None:
        """Clean up state for a completed goal."""
        self._states.pop(goal_id, None)

    def status(self) -> dict[str, Any]:
        """Monitoring snapshot."""
        return {
            "active_goals": len(self._states),
            "total_unknowns": sum(
                len(s.known_unknowns) for s in self._states.values()
            ),
            "total_absences": sum(
                len(s.absences) for s in self._states.values()
            ),
            "total_surprises": sum(
                len(s.surprises) for s in self._states.values()
            ),
        }

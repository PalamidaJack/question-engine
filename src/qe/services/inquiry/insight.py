"""Insight Crystallizer — transforms findings into genuine insights.

The final filter: assesses novelty (strict gate), extracts causal
mechanisms (specific, not vague), scores actionability, builds provenance
chains, and finds cross-domain connections. Only genuinely novel findings
that survived dialectic scrutiny become CrystallizedInsights.
"""

from __future__ import annotations

import logging
from typing import Any

import instructor
import litellm
from pydantic import BaseModel, Field

from qe.models.cognition import (
    ActionabilityResult,
    CrystallizedInsight,
    MechanismExplanation,
    NoveltyAssessment,
    ProvenanceChain,
)
from qe.runtime.episodic_memory import Episode, EpisodicMemory
from qe.substrate.bayesian_belief import BayesianBeliefStore

log = logging.getLogger(__name__)

_NOVELTY_PROMPT = """\
You are a novelty assessment module. Determine if a finding is \
genuinely novel or just a restatement of common knowledge.

FINDING: {finding}
DOMAIN: {domain}
EXISTING KNOWLEDGE (from our belief store): {existing_knowledge}

Assess:
1. Is this novel? Or could any informed person have stated this?
2. What TYPE of novelty is it?
   - contradicts_consensus: goes against conventional wisdom
   - new_connection: links two previously unconnected ideas
   - unexpected_magnitude: known relationship but surprising degree
   - temporal_anomaly: timing is unexpected
   - structural_analogy: pattern from another domain applies here
   - absence_significant: something important is missing
   - not_novel: commonly known
3. Who would find this surprising? (experts, general public, etc.)
4. What existing fact would this most likely be confused with?

Be STRICT. Most findings are NOT novel. Only rate as novel if an \
expert in the field would genuinely be surprised.
"""

_MECHANISM_PROMPT = """\
You are a causal reasoning module. For a given finding, explain the \
underlying MECHANISM — not just WHAT, but WHY and HOW.

FINDING: {finding}
EVIDENCE: {evidence}

Explain:
1. WHAT happens (the observable fact)
2. WHY it happens (the causal drivers)
3. HOW it works (the mechanism/process)
4. What are the KEY CAUSAL LINKS in the chain?

Be specific. "Market forces" is not a mechanism. "Sector classification \
systems group grid infrastructure with volatile renewables, causing \
institutional investors with sector-based allocation rules to \
underweight grid stocks" IS a mechanism.
"""

_ACTIONABILITY_PROMPT = """\
You are an actionability assessment module.

INSIGHT: {insight}
MECHANISM: {mechanism}

Assess:
1. Can someone ACT on this insight? (0.0 = purely informational, \
1.0 = immediately actionable)
2. WHO could act on it? (specific roles/stakeholders)
3. WHAT specific action could they take?
4. What is the TIME HORIZON for acting? (days, weeks, months, years)
"""

_CROSS_DOMAIN_PROMPT = """\
You are a cross-domain connection finder. Given a finding in one domain, \
identify structural analogies in OTHER domains.

FINDING: {finding} (domain: {domain})
KNOWLEDGE GRAPH CONTEXT: {graph_context}

Find connections to other domains:
1. What structural pattern does this finding exemplify?
2. Where else does this same pattern appear?
3. What can we learn from those other domains?

Focus on STRUCTURAL analogies (same pattern, different domain), \
not surface similarities.
"""


class InsightCrystallizer:
    """Transforms raw findings into genuine, novel, actionable insights.

    Pipeline:
    1. Novelty filter (is this genuinely new?)
    2. Mechanism extraction (WHY, not just WHAT)
    3. Actionability scoring (can someone act on this?)
    4. Cross-domain connections (structural analogies)
    5. Provenance chain (full audit trail)
    6. Final crystallization
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory | None = None,
        belief_store: BayesianBeliefStore | None = None,
        model: str = "openai/anthropic/claude-sonnet-4",
        prompt_registry: Any | None = None,
    ) -> None:
        self._episodic = episodic_memory
        self._belief_store = belief_store
        self._model = model
        self._registry = prompt_registry
        self._fallbacks: dict[str, str] = {
            "insight.novelty.system": (
                "You are a strict novelty assessor. "
                "Most findings are NOT novel."
            ),
            "insight.novelty.user": _NOVELTY_PROMPT,
            "insight.mechanism.system": (
                "You are a causal reasoning module. Be specific."
            ),
            "insight.mechanism.user": _MECHANISM_PROMPT,
            "insight.actionability.system": "You are an actionability assessor.",
            "insight.actionability.user": _ACTIONABILITY_PROMPT,
            "insight.cross_domain.system": (
                "You find structural analogies across domains."
            ),
            "insight.cross_domain.user": _CROSS_DOMAIN_PROMPT,
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

    async def assess_novelty(
        self,
        finding: str,
        domain: str = "general",
        entity_id: str = "",
    ) -> NoveltyAssessment:
        """Is this finding genuinely novel?"""
        existing_knowledge = ""
        if self._belief_store and entity_id:
            beliefs = await self._belief_store.get_beliefs_for_entity(
                entity_id
            )
            existing_knowledge = "\n".join(
                f"- {b.claim.predicate}: {b.claim.object_value} "
                f"(conf: {b.posterior:.2f})"
                for b in beliefs
            )

        sys_content, _ = self._get_prompt("insight.novelty.system")
        user_content, user_vid = self._get_prompt(
            "insight.novelty.user",
            finding=finding,
            domain=domain,
            existing_knowledge=(
                existing_knowledge or "No existing beliefs recorded."
            ),
        )

        client = instructor.from_litellm(litellm.acompletion)
        try:
            result = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=NoveltyAssessment,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    user_vid, "insight.novelty.user", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                user_vid, "insight.novelty.user", success=True
            )
        return result

    async def extract_mechanism(
        self,
        finding: str,
        evidence: str = "",
    ) -> MechanismExplanation:
        """Extract the causal mechanism behind a finding."""
        sys_content, _ = self._get_prompt("insight.mechanism.system")
        user_content, user_vid = self._get_prompt(
            "insight.mechanism.user",
            finding=finding,
            evidence=evidence or "Not provided.",
        )

        client = instructor.from_litellm(litellm.acompletion)
        try:
            result = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=MechanismExplanation,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    user_vid, "insight.mechanism.user", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                user_vid, "insight.mechanism.user", success=True
            )
        return result

    async def score_actionability(
        self,
        insight: str,
        mechanism: str,
    ) -> ActionabilityResult:
        """Score how actionable an insight is."""
        sys_content, _ = self._get_prompt("insight.actionability.system")
        user_content, user_vid = self._get_prompt(
            "insight.actionability.user",
            insight=insight,
            mechanism=mechanism,
        )

        client = instructor.from_litellm(litellm.acompletion)
        try:
            result = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=ActionabilityResult,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    user_vid, "insight.actionability.user", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                user_vid, "insight.actionability.user", success=True
            )
        return result

    async def find_cross_domain_connections(
        self,
        finding: str,
        domain: str = "general",
        graph_context: str = "",
    ) -> list[str]:
        """Find structural analogies across domains."""
        sys_content, _ = self._get_prompt("insight.cross_domain.system")
        user_content, user_vid = self._get_prompt(
            "insight.cross_domain.user",
            finding=finding,
            domain=domain,
            graph_context=(
                graph_context
                or "No knowledge graph context available."
            ),
        )

        class Connections(BaseModel):
            connections: list[str] = Field(default_factory=list)

        client = instructor.from_litellm(litellm.acompletion)
        try:
            result = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=Connections,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    user_vid, "insight.cross_domain.user", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                user_vid, "insight.cross_domain.user", success=True
            )
        return result.connections

    async def crystallize(
        self,
        goal_id: str,
        finding: str,
        evidence: str = "",
        original_question: str = "",
        reasoning_steps: list[str] | None = None,
        assumptions: list[str] | None = None,
        domain: str = "general",
        entity_id: str = "",
        dialectic_survived: bool = False,
    ) -> CrystallizedInsight | None:
        """Full crystallization pipeline.

        Returns None if the finding is not novel enough.
        """
        # Step 1: Novelty gate
        novelty = await self.assess_novelty(finding, domain, entity_id)
        if not novelty.is_novel:
            log.info(
                "Finding rejected by novelty filter: %s", finding[:80]
            )
            return None

        # Step 2: Mechanism
        mechanism = await self.extract_mechanism(finding, evidence)

        # Step 3: Actionability
        actionability = await self.score_actionability(
            finding, mechanism.why_it_happens
        )

        # Step 4: Cross-domain connections
        connections = await self.find_cross_domain_connections(
            finding, domain
        )

        # Step 5: Provenance chain
        provenance = ProvenanceChain(
            original_question=original_question,
            evidence_items=[evidence] if evidence else [],
            reasoning_steps=reasoning_steps or [],
            assumptions_made=assumptions or [],
            insight=finding,
            confidence=mechanism.confidence_in_mechanism,
        )

        # Step 6: Assemble
        headline = (
            finding if len(finding) < 120 else finding[:117] + "..."
        )
        insight = CrystallizedInsight(
            headline=headline,
            mechanism=mechanism,
            novelty=novelty,
            provenance=provenance,
            actionability_score=actionability.score,
            actionability_description=(
                f"{actionability.who_can_act} can "
                f"{actionability.what_action} "
                f"({actionability.time_horizon})"
            ),
            cross_domain_connections=connections,
            dialectic_survivor=dialectic_survived,
            confidence=mechanism.confidence_in_mechanism,
        )

        if self._episodic:
            await self._episodic.store(
                Episode(
                    goal_id=goal_id,
                    episode_type="synthesis",
                    content={
                        "phase": "crystallization",
                        "headline": insight.headline[:80],
                        "novelty_type": novelty.novelty_type,
                        "actionability": actionability.score,
                    },
                    summary=(
                        f"Crystallized insight: {insight.headline[:80]}"
                    ),
                    relevance_to_goal=1.0,
                )
            )

        log.info(
            "Crystallized insight: novelty=%s actionability=%.2f "
            "dialectic=%s headline=%s",
            novelty.novelty_type,
            actionability.score,
            dialectic_survived,
            insight.headline[:60],
        )

        return insight

    def status(self) -> dict[str, Any]:
        """Monitoring snapshot."""
        return {"model": self._model}

"""HypothesisManager — POPPER-inspired hypothesis generation and testing.

Contradictions → competing hypotheses → falsification criteria.
Integrates with BayesianBeliefStore for evidence accumulation.
"""

from __future__ import annotations

import logging
from typing import Any

import instructor
import litellm
from pydantic import BaseModel, Field

from qe.services.inquiry.schemas import Question
from qe.substrate.bayesian_belief import BayesianBeliefStore, EvidenceRecord, Hypothesis

log = logging.getLogger(__name__)

_HYPOTHESIS_GEN_SYSTEM = (
    "You are a hypothesis generator following Popperian "
    "philosophy. Every hypothesis MUST be falsifiable. "
    "Generate competing hypotheses that cover different "
    "explanations for the same observations."
)


# ---------------------------------------------------------------------------
# LLM response models
# ---------------------------------------------------------------------------


class HypothesisSpec(BaseModel):
    """A single hypothesis from LLM generation."""

    statement: str
    falsification_criteria: list[str] = Field(default_factory=list)
    experiments: list[str] = Field(default_factory=list)
    prior_probability: float = Field(default=0.5, ge=0.0, le=1.0)


class GeneratedHypotheses(BaseModel):
    """LLM response: a batch of competing hypotheses."""

    hypotheses: list[HypothesisSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# HypothesisManager
# ---------------------------------------------------------------------------


class HypothesisManager:
    """Manages hypothesis lifecycle: generation, falsification, evidence updating."""

    def __init__(
        self,
        belief_store: BayesianBeliefStore | None = None,
        model: str = "openai/anthropic/claude-sonnet-4",
        prompt_registry: Any | None = None,
    ) -> None:
        self._belief_store = belief_store
        self._model = model
        self._client = instructor.from_litellm(litellm.acompletion)
        # In-memory store for hypotheses when no belief_store
        self._local_hypotheses: dict[str, Hypothesis] = {}
        self._registry = prompt_registry
        self._fallbacks: dict[str, str] = {
            "hypothesis.generate.system": _HYPOTHESIS_GEN_SYSTEM,
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

    async def generate_hypotheses(
        self,
        goal: str,
        contradictions: list[str] | None = None,
        existing_beliefs: list[str] | None = None,
    ) -> list[Hypothesis]:
        """Generate competing hypotheses from contradictions."""
        contras = contradictions or []
        beliefs = existing_beliefs or []

        prompt = f"Goal: {goal}\n\n"
        if contras:
            prompt += "Contradictions found:\n" + "\n".join(f"- {c}" for c in contras) + "\n\n"
        if beliefs:
            prompt += "Existing beliefs:\n" + "\n".join(f"- {b}" for b in beliefs) + "\n\n"

        prompt += (
            "Generate 2-3 competing hypotheses that could explain these findings "
            "or contradictions. For each, provide: a clear statement, "
            "falsification criteria (what would disprove it), "
            "and experiments (what to investigate). "
            "Set prior_probability between 0.1 and 0.9."
        )

        sys_content, sys_vid = self._get_prompt("hypothesis.generate.system")

        try:
            result = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": prompt},
                ],
                response_model=GeneratedHypotheses,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    sys_vid, "hypothesis.generate.system", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                sys_vid, "hypothesis.generate.system", success=True
            )

        hypotheses: list[Hypothesis] = []
        for spec in result.hypotheses:
            h = Hypothesis(
                statement=spec.statement,
                prior_probability=spec.prior_probability,
                current_probability=spec.prior_probability,
                falsification_criteria=spec.falsification_criteria,
                experiments=spec.experiments,
            )

            # Persist via belief store if available
            if self._belief_store is not None:
                h = await self._belief_store.store_hypothesis(h)
            else:
                self._local_hypotheses[h.hypothesis_id] = h

            hypotheses.append(h)

        log.info(
            "hypothesis_manager.generated goal=%s count=%d",
            goal[:60],
            len(hypotheses),
        )
        return hypotheses

    def create_falsification_questions(self, hypothesis: Hypothesis) -> list[Question]:
        """Create falsification-typed questions from hypothesis criteria."""
        questions: list[Question] = []
        for criterion in hypothesis.falsification_criteria:
            q = Question(
                text=f"Can we find evidence that: {criterion}",
                question_type="falsification",
                expected_info_gain=0.8,
                relevance_to_goal=0.7,
                novelty_score=0.6,
                hypothesis_id=hypothesis.hypothesis_id,
            )
            questions.append(q)
        return questions

    async def update_with_evidence(
        self,
        hypothesis_id: str,
        evidence: EvidenceRecord,
    ) -> Hypothesis:
        """Update a hypothesis with new evidence via belief store."""
        if self._belief_store is not None:
            return await self._belief_store.update_hypothesis(hypothesis_id, evidence)

        # Fallback: local update
        h = self._local_hypotheses.get(hypothesis_id)
        if h is None:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")

        # Simple Bayesian update
        prior = h.current_probability
        if evidence.supports:
            lr = 1.0 + evidence.strength * 4.0
        else:
            lr = 1.0 / (1.0 + evidence.strength * 4.0)
        numerator = prior * lr
        denominator = numerator + (1.0 - prior)
        posterior = max(0.01, min(0.99, numerator / denominator)) if denominator > 0 else prior

        h.current_probability = posterior
        if posterior >= 0.95:
            h.status = "confirmed"
        elif posterior <= 0.05:
            h.status = "falsified"

        return h

    async def get_active_hypotheses(self) -> list[Hypothesis]:
        """Get all active hypotheses."""
        if self._belief_store is not None:
            return await self._belief_store.get_active_hypotheses()

        return [
            h for h in self._local_hypotheses.values()
            if h.status == "active"
        ]

    @staticmethod
    def compute_bayes_factor(
        hypothesis_a: Hypothesis,
        hypothesis_b: Hypothesis,
    ) -> float:
        """Compute Bayes factor: P(H_a)/P(H_b).

        Interpretation:
        - BF > 10 (log10 > 1.0): strong evidence for H_a
        - BF > 3 (log10 > 0.48): substantial evidence for H_a
        - BF ~ 1: inconclusive
        - BF < 1/3: substantial evidence for H_b
        """
        p_a = max(hypothesis_a.current_probability, 1e-10)
        p_b = max(hypothesis_b.current_probability, 1e-10)
        return p_a / p_b

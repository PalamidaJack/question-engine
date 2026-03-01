"""LLM-powered question generation and prioritization for the Inquiry Loop."""

from __future__ import annotations

import logging
from typing import Any

import instructor
import litellm
from pydantic import BaseModel, Field

from qe.services.inquiry.schemas import Question

log = logging.getLogger(__name__)

_QUESTION_GEN_SYSTEM = (
    "You are a research question generator. Generate precise, "
    "non-overlapping questions that maximize information gain. "
    "Later iterations should focus on gaps and surprises. "
    "If hypotheses are present, include falsification questions."
)


# ---------------------------------------------------------------------------
# LLM response models
# ---------------------------------------------------------------------------


class GeneratedQuestion(BaseModel):
    """Single question from LLM generation."""

    text: str
    question_type: str = "factual"
    expected_info_gain: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_to_goal: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    hypothesis_id: str | None = None


class GeneratedQuestions(BaseModel):
    """LLM response: a batch of generated questions."""

    questions: list[GeneratedQuestion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# QuestionGenerator
# ---------------------------------------------------------------------------


class QuestionGenerator:
    """Generates and prioritizes questions using an LLM."""

    def __init__(
        self,
        model: str = "openai/google/gemini-2.0-flash",
        prompt_registry: Any | None = None,
    ) -> None:
        self._model = model
        self._client = instructor.from_litellm(litellm.acompletion)
        self._registry = prompt_registry
        self._fallbacks: dict[str, str] = {
            "question_gen.generate.system": _QUESTION_GEN_SYSTEM,
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

    async def generate(
        self,
        goal: str,
        findings_summary: str = "",
        asked_questions: list[str] | None = None,
        epistemic_state: dict[str, Any] | None = None,
        hypotheses_summary: str = "",
        iteration: int = 0,
        max_iterations: int = 10,
        n_questions: int = 3,
    ) -> list[Question]:
        """Generate new questions to advance the inquiry."""
        asked = asked_questions or []
        epi = epistemic_state or {}

        prompt = (
            f"Goal: {goal}\n\n"
            f"Iteration: {iteration + 1}/{max_iterations}\n\n"
        )

        if findings_summary:
            prompt += f"Findings so far:\n{findings_summary}\n\n"

        if asked:
            prompt += "Questions already asked:\n" + "\n".join(f"- {q}" for q in asked) + "\n\n"

        if epi:
            prompt += f"Epistemic state: {epi}\n\n"

        if hypotheses_summary:
            prompt += f"Active hypotheses:\n{hypotheses_summary}\n\n"

        prompt += (
            f"Generate exactly {n_questions} NEW questions that would most advance "
            f"understanding of the goal. DO NOT repeat questions already asked. "
            f"Include question_type (factual/causal/comparative/hypothetical/"
            f"clarifying/falsification/meta), expected_info_gain (0-1), "
            f"relevance_to_goal (0-1), and novelty_score (0-1)."
        )

        sys_content, sys_vid = self._get_prompt("question_gen.generate.system")

        try:
            result = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": prompt},
                ],
                response_model=GeneratedQuestions,
            )
        except Exception:
            if self._registry:
                self._registry.record_outcome(
                    sys_vid, "question_gen.generate.system", success=False
                )
            raise
        if self._registry:
            self._registry.record_outcome(
                sys_vid, "question_gen.generate.system", success=True
            )

        questions = []
        for gq in result.questions[:n_questions]:
            # Validate question_type
            valid_types = {
                "factual", "causal", "comparative", "hypothetical",
                "clarifying", "falsification", "meta",
            }
            q_type = gq.question_type if gq.question_type in valid_types else "factual"

            q = Question(
                text=gq.text,
                question_type=q_type,
                expected_info_gain=gq.expected_info_gain,
                relevance_to_goal=gq.relevance_to_goal,
                novelty_score=gq.novelty_score,
                hypothesis_id=gq.hypothesis_id,
                iteration_generated=iteration,
            )
            questions.append(q)

        log.info(
            "question_generator.generated goal=%s iteration=%d count=%d",
            goal[:60],
            iteration,
            len(questions),
        )
        return questions

    async def prioritize(
        self,
        goal: str,
        questions: list[Question],
    ) -> list[Question]:
        """Sort questions by computed priority score (no LLM call needed)."""
        scored = sorted(
            questions,
            key=lambda q: self.compute_priority_score(q),
            reverse=True,
        )
        return scored

    @staticmethod
    def compute_priority_score(q: Question) -> float:
        """Weighted priority: info_gain*0.4 + relevance*0.35 + novelty*0.25."""
        return (
            q.expected_info_gain * 0.4
            + q.relevance_to_goal * 0.35
            + q.novelty_score * 0.25
        )

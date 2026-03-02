"""LLM-powered relevance and feasibility analysis for scout findings."""

from __future__ import annotations

import logging
import uuid

import instructor
import litellm
from pydantic import BaseModel, Field

from qe.models.scout import ImprovementIdea, ScoutFinding
from qe.runtime.feature_flags import get_flag_store

log = logging.getLogger(__name__)

# Static codebase summary for the LLM prompt
_CODEBASE_SUMMARY = """
Question Engine (QE) is a Python multi-agent AI system built with:
- Python 3.14, FastAPI, Pydantic v2, aiosqlite
- litellm + instructor for LLM calls with structured output
- MemoryBus event system with topic-based pub/sub
- Three nested loops: Inquiry (seconds-minutes), Knowledge (minutes-hours), Strategy (hours-days)
- Four-tier memory: Working (ContextCurator), Episodic, Semantic (BayesianBeliefStore), Procedural
- Cognitive layer: Metacognitor, Epistemic Reasoner, Dialectic Engine,
  Persistence Engine, Insight Crystallizer
- Tool system: web_search, web_fetch, file_read, file_write, code_execute
- Competitive Arena for agent-vs-agent verification
- Prompt Evolution with Thompson sampling
Key directories: src/qe/services/, src/qe/runtime/, src/qe/substrate/, src/qe/api/, src/qe/tools/
"""


class _RelevanceAssessment(BaseModel):
    """LLM-structured relevance assessment."""

    title: str = Field(description="Concise improvement title")
    description: str = Field(description="What this improvement would do")
    category: str = Field(
        description="One of: performance, feature, refactor, testing, "
        "security, dependency, pattern, model, other",
    )
    relevance_score: float = Field(ge=0.0, le=1.0, description="How relevant to QE (0-1)")
    feasibility_score: float = Field(ge=0.0, le=1.0, description="How easy to implement (0-1)")
    impact_score: float = Field(ge=0.0, le=1.0, description="How much improvement (0-1)")
    rationale: str = Field(description="Why this is relevant and how it would help")
    affected_files: list[str] = Field(
        default_factory=list, description="Files that would need changes",
    )


class ScoutAnalyzer:
    """Analyze findings for relevance, feasibility, and impact."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        min_composite_score: float = 0.5,
    ) -> None:
        self._model = model
        self._min_composite_score = min_composite_score
        self._approved_patterns: list[str] = []

    def set_approved_patterns(self, patterns: list[str]) -> None:
        """Bias towards patterns humans have approved."""
        self._approved_patterns = patterns

    def update_threshold(self, new_threshold: float) -> None:
        """Dynamic threshold adjustment based on approval rate."""
        self._min_composite_score = max(0.1, min(0.9, new_threshold))

    async def analyze(
        self,
        findings: list[ScoutFinding],
    ) -> list[ImprovementIdea]:
        """Score each finding and return filtered improvement ideas."""
        if not get_flag_store().is_enabled("innovation_scout"):
            return []

        ideas: list[ImprovementIdea] = []
        for finding in findings:
            try:
                idea = await self._analyze_one(finding)
                if idea and idea.composite_score >= self._min_composite_score:
                    ideas.append(idea)
            except Exception:
                log.warning(
                    "scout.analysis_failed finding=%s",
                    finding.finding_id,
                )
        return ideas

    async def _analyze_one(self, finding: ScoutFinding) -> ImprovementIdea | None:
        """Analyze a single finding."""
        content = finding.full_content or finding.snippet
        if not content.strip():
            return None

        bias_text = ""
        if self._approved_patterns:
            bias_text = (
                "\nHumans have previously approved improvements related to: "
                + ", ".join(self._approved_patterns[:5])
            )

        try:
            client = instructor.from_litellm(litellm.acompletion)
            assessment = await client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert software architect assessing whether a web "
                            "resource contains ideas that could improve the Question Engine "
                            "codebase. Score relevance, feasibility, and impact on a 0-1 scale."
                            f"\n\nCodebase context:\n{_CODEBASE_SUMMARY}"
                            f"{bias_text}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Title: {finding.title}\n"
                            f"URL: {finding.url}\n"
                            f"Source: {finding.source_type}\n\n"
                            f"Content:\n{content[:3000]}"
                        ),
                    },
                ],
                response_model=_RelevanceAssessment,
            )
        except Exception:
            log.warning("scout.llm_analysis_failed finding=%s", finding.finding_id)
            return None

        # Validate category
        valid_categories = {
            "performance", "feature", "refactor", "testing",
            "security", "dependency", "pattern", "model", "other",
        }
        category = assessment.category if assessment.category in valid_categories else "other"

        composite = (
            assessment.relevance_score * 0.4
            + assessment.feasibility_score * 0.3
            + assessment.impact_score * 0.3
        )

        return ImprovementIdea(
            idea_id=f"idea_{uuid.uuid4().hex[:12]}",
            finding_id=finding.finding_id,
            title=assessment.title,
            description=assessment.description,
            category=category,
            relevance_score=assessment.relevance_score,
            feasibility_score=assessment.feasibility_score,
            impact_score=assessment.impact_score,
            composite_score=round(composite, 3),
            source_url=finding.url,
            rationale=assessment.rationale,
            affected_files=assessment.affected_files,
        )

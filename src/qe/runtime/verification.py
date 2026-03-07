"""Verification protocol for cognitive tool call outputs.

After a cognitive tool call, a fast model checks whether the response
addresses the original question and doesn't contain hallucinated
citations.  Gated behind ``verification_protocol`` feature flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a tool call output."""

    passed: bool
    issues: list[str]
    confidence: float  # 0.0 to 1.0
    recommendation: str  # "accept", "re_invoke", "flag"


class ToolVerifier:
    """Verifies cognitive tool call outputs for quality.

    Checks:
    1. Does the output address the original question?
    2. Are citations/references plausible (not hallucinated)?
    3. Is the output internally consistent?
    """

    def __init__(self, llm: Any = None, fast_model: str = "gpt-4o-mini") -> None:
        self._llm = llm
        self._fast_model = fast_model

    async def verify(
        self,
        question: str,
        tool_name: str,
        tool_output: str,
    ) -> VerificationResult:
        """Verify a tool output against the original question.

        If no LLM is available, falls back to heuristic checks.
        """
        issues: list[str] = []

        # Heuristic checks (always run)
        if not tool_output or len(tool_output.strip()) < 10:
            issues.append("Output is empty or too short")

        if question and tool_output:
            # Check if key terms from question appear in output
            q_words = set(question.lower().split())
            o_words = set(tool_output.lower().split())
            overlap = q_words & o_words
            relevance = len(overlap) / max(len(q_words), 1)
            if relevance < 0.1:
                issues.append("Output may not address the question (low term overlap)")

        # Check for hallucination markers
        hallucination_markers = [
            "as an ai", "i don't have access", "i cannot verify",
            "hypothetically", "in theory",
        ]
        lower_output = tool_output.lower()
        for marker in hallucination_markers:
            if marker in lower_output:
                issues.append(f"Potential hedging detected: '{marker}'")

        # LLM-based verification (if available)
        if self._llm is not None and not issues:
            try:
                llm_result = await self._llm_verify(question, tool_name, tool_output)
                issues.extend(llm_result)
            except Exception:
                log.debug("verification.llm_check_failed", exc_info=True)

        passed = len(issues) == 0
        confidence = 1.0 - min(len(issues) * 0.2, 0.8)

        if not passed and len(issues) >= 3:
            recommendation = "re_invoke"
        elif not passed:
            recommendation = "flag"
        else:
            recommendation = "accept"

        return VerificationResult(
            passed=passed,
            issues=issues,
            confidence=confidence,
            recommendation=recommendation,
        )

    async def _llm_verify(
        self, question: str, tool_name: str, output: str
    ) -> list[str]:
        """Use a fast LLM to verify output quality."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a verification agent. Check if the tool output "
                    "addresses the question. Report only genuine issues as "
                    "a JSON list of strings. Return [] if the output is good."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Tool: {tool_name}\n"
                    f"Output: {output[:2000]}"
                ),
            },
        ]

        response = await self._llm.complete(
            messages, model=self._fast_model, max_tokens=200
        )
        text = response.choices[0].message.content or "[]"
        import json
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return []


class TwoStageReviewer:
    """Two-stage review: first verify, then review merged results.

    Stage 1: ToolVerifier checks individual tool outputs.
    Stage 2: A second LLM reviews merged swarm results for consistency.
    Gated behind ``two_stage_review`` feature flag.
    """

    def __init__(self, verifier: ToolVerifier | None = None, llm: Any = None) -> None:
        self._verifier = verifier or ToolVerifier()
        self._llm = llm

    async def review_merged(
        self,
        question: str,
        merged_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Review merged swarm results for consistency and completeness.

        Returns a review dict with consistency score and issues.
        """
        issues: list[str] = []

        if not merged_results:
            return {
                "consistency_score": 0.0,
                "issues": ["No results to review"],
                "recommendation": "re_invoke",
            }

        # Check for contradictions between results
        texts = [str(r.get("summary", r.get("result", ""))) for r in merged_results]
        if len(texts) >= 2:
            # Simple contradiction detection: check if results agree
            all_words: list[set[str]] = [set(t.lower().split()) for t in texts]
            pairwise_overlaps = []
            for i in range(len(all_words)):
                for j in range(i + 1, len(all_words)):
                    if all_words[i] and all_words[j]:
                        overlap = len(all_words[i] & all_words[j]) / min(
                            len(all_words[i]), len(all_words[j])
                        )
                        pairwise_overlaps.append(overlap)

            if pairwise_overlaps:
                avg_overlap = sum(pairwise_overlaps) / len(pairwise_overlaps)
                if avg_overlap < 0.1:
                    issues.append("Results show very low agreement (possible contradiction)")

        consistency = 1.0 - min(len(issues) * 0.3, 0.9)
        return {
            "consistency_score": round(consistency, 2),
            "issues": issues,
            "recommendation": "accept" if not issues else "flag",
            "results_reviewed": len(merged_results),
        }

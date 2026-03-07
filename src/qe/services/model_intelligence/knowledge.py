"""Model profile narrative generation and storage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import litellm

log = logging.getLogger(__name__)

_PROFILE_PROMPT_TEMPLATE = """\
You are a technical writer specialising in LLM model evaluations.
Given the benchmark scores and metadata below, write a rich narrative
profile in Markdown.  Use this exact structure:

# {model_id}
## Overview
Maker: ... | Context: ... | Cost: ... | Providers: ...

## Strengths
- (bullet points based on high-scoring categories)

## Weaknesses
- (bullet points based on low-scoring categories)

## Best For
(paragraph — which tasks this model excels at)

## Avoid For
(paragraph — which tasks this model struggles with)

## Benchmark Highlights
(table or bullets summarising per-category scores)

## Stability: ...
(one-word rating: Excellent / Good / Fair / Poor, plus brief justification)

---

MODEL ID: {model_id}

METADATA:
{metadata_block}

BENCHMARK SCORES:
{scores_block}

Write the profile now.  Output ONLY Markdown, no preamble.
"""


class ModelKnowledgeAgent:
    """Generates, saves, and retrieves narrative Markdown profiles for models.

    Uses a small, inexpensive LLM (default ``gpt-4o-mini``) to author
    the narrative given structured benchmark data.
    """

    def __init__(
        self,
        profiles_dir: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._profiles_dir = Path(profiles_dir)
        self._model = model

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate_profile(
        self,
        model_id: str,
        scores: dict[str, Any],
        metadata: dict[str, Any],
    ) -> str:
        """Generate a narrative Markdown profile via LLM.

        Parameters
        ----------
        model_id:
            The identifier of the model being profiled.
        scores:
            Per-category score dictionaries as returned by
            ``ModelIntelligenceService.get_model_scores``.
        metadata:
            Supplementary metadata (provider, maker, context length, etc.).

        Returns
        -------
        str
            The generated Markdown text.
        """
        metadata_block = self._format_metadata(model_id, metadata)
        scores_block = self._format_scores(scores)

        prompt = _PROFILE_PROMPT_TEMPLATE.format(
            model_id=model_id,
            metadata_block=metadata_block,
            scores_block=scores_block,
        )

        response = await litellm.acompletion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            timeout=90,
        )
        content: str = response.choices[0].message.content or ""
        return content.strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_profile(self, model_id: str, markdown: str) -> None:
        """Save a Markdown profile to disk under ``profiles_dir``."""
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_id.replace("/", "__").replace(":", "_")
        path = self._profiles_dir / f"{safe_name}.md"
        path.write_text(markdown, encoding="utf-8")
        log.info("knowledge.profile_saved path=%s", path)

    def load_profile(self, model_id: str) -> str | None:
        """Load a previously saved profile, or return ``None``."""
        safe_name = model_id.replace("/", "__").replace(":", "_")
        path = self._profiles_dir / f"{safe_name}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    def list_profiles(self) -> list[str]:
        """Return a list of model IDs that have saved profiles."""
        if not self._profiles_dir.exists():
            return []
        results: list[str] = []
        for p in sorted(self._profiles_dir.glob("*.md")):
            name = p.stem.replace("__", "/").replace("_", ":")
            results.append(name)
        return results

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_metadata(
        model_id: str, metadata: dict[str, Any],
    ) -> str:
        """Format metadata dict into a human-readable block."""
        lines: list[str] = [f"Model ID: {model_id}"]
        if "provider" in metadata:
            lines.append(f"Provider: {metadata['provider']}")
        if "maker" in metadata:
            lines.append(f"Maker: {metadata['maker']}")
        if "context_length" in metadata:
            lines.append(f"Context length: {metadata['context_length']}")
        if "is_free" in metadata:
            cost = "Free" if metadata["is_free"] else "Paid"
            lines.append(f"Cost tier: {cost}")
        caps = metadata.get("capabilities", [])
        if caps:
            lines.append(f"Capabilities: {', '.join(caps)}")
        return "\n".join(lines)

    @staticmethod
    def _format_scores(scores: dict[str, Any]) -> str:
        """Format scores dict into a readable block."""
        if not scores:
            return "(no benchmark scores available)"
        lines: list[str] = []
        for category, data in scores.items():
            if isinstance(data, dict):
                avg = data.get("avg_score", "?")
                p50 = data.get("p50_latency_ms", "?")
                p95 = data.get("p95_latency_ms", "?")
                consistency = data.get("consistency_score", "?")
                n = data.get("sample_count", "?")
                lines.append(
                    f"  {category}: avg={avg}  p50={p50}ms  "
                    f"p95={p95}ms  consistency={consistency}  n={n}"
                )
            else:
                lines.append(f"  {category}: {data}")
        return "\n".join(lines)

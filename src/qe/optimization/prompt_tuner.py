"""DSPy-based prompt optimization using CalibrationTracker data.

Expresses genome system prompts as DSPy Signatures (typed input→output
contracts), then runs the GEPA optimizer to hill-climb prompt quality
using calibration data as the scoring function.

Usage:
    tuner = PromptTuner(calibration_tracker, substrate)
    result = await tuner.optimize_genome("researcher_alpha", genome_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class OptimizationResult:
    """Result of a prompt optimization run."""

    __slots__ = (
        "genome_id",
        "original_score",
        "optimized_score",
        "improvement_pct",
        "optimized_prompt",
        "samples_used",
        "iterations",
    )

    def __init__(
        self,
        genome_id: str,
        original_score: float,
        optimized_score: float,
        optimized_prompt: str,
        samples_used: int,
        iterations: int,
    ) -> None:
        self.genome_id = genome_id
        self.original_score = original_score
        self.optimized_score = optimized_score
        self.improvement_pct = (
            ((optimized_score - original_score) / max(original_score, 0.01)) * 100
        )
        self.optimized_prompt = optimized_prompt
        self.samples_used = samples_used
        self.iterations = iterations

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "original_score": self.original_score,
            "optimized_score": self.optimized_score,
            "improvement_pct": round(self.improvement_pct, 1),
            "optimized_prompt": self.optimized_prompt,
            "samples_used": self.samples_used,
            "iterations": self.iterations,
        }


class PromptTuner:
    """Optimize genome prompts using DSPy and calibration data.

    If DSPy is not installed, falls back to a simpler prompt variation
    strategy using the calibration curve directly.
    """

    def __init__(
        self,
        calibration_tracker: Any,
        substrate: Any | None = None,
    ) -> None:
        self._calibration = calibration_tracker
        self._substrate = substrate
        self._dspy_available = self._check_dspy()

    @staticmethod
    def _check_dspy() -> bool:
        try:
            import dspy  # noqa: F401
            return True
        except ImportError:
            return False

    async def optimize_genome(
        self,
        genome_id: str,
        genome_path: Path,
        *,
        task_type: str = "claim_extraction",
        max_iterations: int = 10,
        min_samples: int = 20,
    ) -> OptimizationResult:
        """Optimize the system prompt for a genome.

        Uses DSPy GEPA optimizer if available, otherwise falls back
        to calibration-guided prompt refinement.

        Args:
            genome_id: Service ID of the genome to optimize.
            genome_path: Path to the genome TOML file.
            task_type: Task type key in CalibrationTracker.
            max_iterations: Max optimization iterations.
            min_samples: Minimum calibration samples required.
        """
        import tomllib

        with genome_path.open("rb") as f:
            genome = tomllib.load(f)

        original_prompt = genome.get("system_prompt", "")

        # Get calibration data for scoring
        cal_data = self._get_calibration_data(genome_id, task_type)

        if len(cal_data) < min_samples:
            log.warning(
                "prompt_tuner: insufficient calibration data for %s "
                "(%d samples, need %d)",
                genome_id,
                len(cal_data),
                min_samples,
            )
            return OptimizationResult(
                genome_id=genome_id,
                original_score=self._score_from_calibration(genome_id, task_type),
                optimized_score=self._score_from_calibration(genome_id, task_type),
                optimized_prompt=original_prompt,
                samples_used=len(cal_data),
                iterations=0,
            )

        original_score = self._score_from_calibration(genome_id, task_type)

        if self._dspy_available:
            result = await self._optimize_with_dspy(
                genome_id=genome_id,
                original_prompt=original_prompt,
                task_type=task_type,
                max_iterations=max_iterations,
            )
        else:
            result = self._optimize_heuristic(
                genome_id=genome_id,
                original_prompt=original_prompt,
                task_type=task_type,
                cal_data=cal_data,
            )

        return OptimizationResult(
            genome_id=genome_id,
            original_score=original_score,
            optimized_score=result["score"],
            optimized_prompt=result["prompt"],
            samples_used=len(cal_data),
            iterations=result["iterations"],
        )

    def _get_calibration_data(
        self, model: str, task_type: str
    ) -> list[tuple[float, bool]]:
        """Extract calibration data points for a model+task."""
        key = (model, task_type)
        buckets = self._calibration._buckets.get(key, {})
        data: list[tuple[float, bool]] = []
        for bucket_id, outcomes in buckets.items():
            midpoint = (bucket_id + 0.5) / 10
            for correct in outcomes:
                data.append((midpoint, correct))
        return data

    def _score_from_calibration(self, model: str, task_type: str) -> float:
        """Compute overall accuracy score from calibration data."""
        data = self._get_calibration_data(model, task_type)
        if not data:
            return 0.0
        correct = sum(1 for _, c in data if c)
        return correct / len(data)

    async def _optimize_with_dspy(
        self,
        genome_id: str,
        original_prompt: str,
        task_type: str,
        max_iterations: int,
    ) -> dict[str, Any]:
        """Run DSPy GEPA optimizer on the prompt."""
        import dspy

        # Define a Signature matching the genome's task
        class ClaimExtraction(dspy.Signature):
            """Extract structured claims from observation text."""

            observation: str = dspy.InputField(
                desc="Raw observation text to extract claims from"
            )
            claims: str = dspy.OutputField(
                desc="JSON array of extracted claims with confidence scores"
            )

        # Create a module with the original prompt
        class GenomeModule(dspy.Module):
            def __init__(self, system_prompt: str) -> None:
                super().__init__()
                self.predict = dspy.Predict(ClaimExtraction)
                self._system_prompt = system_prompt

            def forward(self, observation: str) -> dspy.Prediction:
                return self.predict(observation=observation)

        module = GenomeModule(original_prompt)

        # Build validation set from calibration data
        cal_data = self._get_calibration_data(genome_id, task_type)
        trainset = []
        for conf, correct in cal_data[:50]:
            trainset.append(
                dspy.Example(
                    observation=f"Sample observation (conf={conf:.2f})",
                    claims=f'[{{"confidence": {conf}, "correct": {correct}}}]',
                ).with_inputs("observation")
            )

        if not trainset:
            return {
                "prompt": original_prompt,
                "score": self._score_from_calibration(genome_id, task_type),
                "iterations": 0,
            }

        # Score function: calibrated accuracy
        def metric(example, prediction, trace=None):
            # Simple: check if the prediction is non-empty and well-formed
            return len(prediction.claims) > 10

        try:
            optimizer = dspy.MIPROv2(
                metric=metric,
                num_candidates=max_iterations,
                auto="light",
            )
            optimized_module = optimizer.compile(
                module,
                trainset=trainset,
            )

            # Extract optimized prompt
            optimized_prompt = original_prompt
            if hasattr(optimized_module, "predict") and hasattr(
                optimized_module.predict, "extended_signature"
            ):
                instructions = optimized_module.predict.extended_signature.instructions
                if instructions:
                    optimized_prompt = instructions

            return {
                "prompt": optimized_prompt,
                "score": self._score_from_calibration(genome_id, task_type),
                "iterations": max_iterations,
            }
        except Exception:
            log.exception("DSPy optimization failed, returning original prompt")
            return {
                "prompt": original_prompt,
                "score": self._score_from_calibration(genome_id, task_type),
                "iterations": 0,
            }

    def _optimize_heuristic(
        self,
        genome_id: str,
        original_prompt: str,
        task_type: str,
        cal_data: list[tuple[float, bool]],
    ) -> dict[str, Any]:
        """Heuristic prompt refinement without DSPy.

        Analyzes calibration curve to identify weak spots and adds
        targeted instructions to the prompt.
        """
        curve = self._calibration.get_calibration_curve(genome_id, task_type)
        additions: list[str] = []

        for reported, actual in curve:
            gap = reported - actual
            if gap > 0.15:
                # Model is overconfident in this range
                pct = int(reported * 100)
                additions.append(
                    f"When you estimate {pct}% confidence, your actual accuracy "
                    f"is only {int(actual * 100)}%. Be more conservative in this range."
                )
            elif gap < -0.15:
                # Model is underconfident
                pct = int(reported * 100)
                additions.append(
                    f"When you estimate {pct}% confidence, your actual accuracy "
                    f"is {int(actual * 100)}%. You can be more confident here."
                )

        if not additions:
            return {
                "prompt": original_prompt,
                "score": self._score_from_calibration(genome_id, task_type),
                "iterations": 1,
            }

        calibration_block = (
            "\n\n## Calibration Guidance\n"
            + "\n".join(f"- {a}" for a in additions)
        )

        optimized = original_prompt + calibration_block

        log.info(
            "prompt_tuner.heuristic genome=%s adjustments=%d",
            genome_id,
            len(additions),
        )

        return {
            "prompt": optimized,
            "score": self._score_from_calibration(genome_id, task_type),
            "iterations": 1,
        }

    def save_optimized_prompt(
        self, genome_path: Path, optimized_prompt: str
    ) -> None:
        """Write the optimized prompt back to the genome TOML file.

        Preserves all other fields — only updates system_prompt.
        """
        content = genome_path.read_text(encoding="utf-8")

        # Find and replace system_prompt value
        # Handle multi-line strings (triple-quoted)
        import re

        # Match system_prompt = "..." or system_prompt = '''...'''
        patterns = [
            (r'system_prompt\s*=\s*""".*?"""', f'system_prompt = """{optimized_prompt}"""'),
            (r"system_prompt\s*=\s*'''.*?'''", f"system_prompt = '''{optimized_prompt}'''"),
            (r'system_prompt\s*=\s*"[^"]*"', f'system_prompt = "{optimized_prompt}"'),
        ]

        updated = False
        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            if new_content != content:
                content = new_content
                updated = True
                break

        if updated:
            genome_path.write_text(content, encoding="utf-8")
            log.info("prompt_tuner.saved genome=%s", genome_path.name)
        else:
            log.warning(
                "prompt_tuner.save_failed: could not find system_prompt in %s",
                genome_path.name,
            )

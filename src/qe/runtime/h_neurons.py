"""H-Neuron profiling for local open-weight models.

Identifies hallucination-associated neurons using the CETT method
and sparse logistic regression, inspired by the Tsinghua H-Neurons paper.

Requires optional dependency: torch, scikit-learn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_PROFILE_DIR = "data/h_neuron_profiles"


@dataclass
class HNeuronProfile:
    """Profile of hallucination-associated neurons for a model."""

    model_name: str
    neurons_by_layer: dict[int, list[int]] = field(default_factory=dict)
    total_neurons: int = 0
    creation_timestamp: str = ""
    calibration_size: int = 0

    def __post_init__(self) -> None:
        if not self.creation_timestamp:
            self.creation_timestamp = datetime.now(UTC).isoformat()
        if self.total_neurons == 0 and self.neurons_by_layer:
            self.total_neurons = sum(len(v) for v in self.neurons_by_layer.values())


def _require_torch() -> Any:
    """Import and return torch, raising a clear error if unavailable."""
    try:
        import torch  # type: ignore[import-untyped]

        return torch
    except ImportError:
        raise ImportError(
            "torch is required for H-Neuron profiling. "
            "Install it with: pip install torch"
        ) from None


def _require_sklearn() -> Any:
    """Import and return sklearn.linear_model, raising a clear error if unavailable."""
    try:
        from sklearn import linear_model  # type: ignore[import-untyped]

        return linear_model
    except ImportError:
        raise ImportError(
            "scikit-learn is required for H-Neuron profiling. "
            "Install it with: pip install scikit-learn"
        ) from None


class HNeuronProfiler:
    """Profiles a local open-weight model to identify hallucination-associated neurons.

    Uses the Contrastive Entity Token Tracking (CETT) method:
    1. Run calibration items through the model.
    2. Collect intermediate activations per layer.
    3. Label each item as faithful or hallucinated (comparing model output to expected answer).
    4. Train a sparse logistic regression (L1 penalty) per layer.
    5. Non-zero weight indices identify the H-Neurons.
    """

    def __init__(self, profile_dir: str = _DEFAULT_PROFILE_DIR) -> None:
        self._profile_dir = Path(profile_dir)

    async def profile_model(
        self,
        model_path: str,
        calibration_data: list[dict[str, str]],
    ) -> HNeuronProfile:
        """Profile a model to find H-Neurons.

        Args:
            model_path: Path to the local model weights (HuggingFace-compatible).
            calibration_data: List of {"question": str, "expected_answer": str} dicts.

        Returns:
            HNeuronProfile with the identified neurons.
        """
        torch = _require_torch()
        linear_model = _require_sklearn()

        log.info(
            "h_neuron.profile_start model=%s calibration_items=%d",
            model_path,
            len(calibration_data),
        )

        # Load model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            output_hidden_states=True,
        )
        model.eval()

        # Determine the number of transformer layers
        num_layers = model.config.num_hidden_layers

        # Collect activations and labels for each layer
        # activations_by_layer[layer_idx] is a list of 1-D mean-pooled activation vectors
        activations_by_layer: dict[int, list[Any]] = {i: [] for i in range(num_layers)}
        labels: list[int] = []  # 0 = faithful, 1 = hallucinated

        for item in calibration_data:
            question = item["question"]
            expected = item["expected_answer"]

            # Tokenize and run forward pass
            inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Extract hidden states: tuple of (num_layers + 1) tensors
            hidden_states = outputs.hidden_states  # (embedding + num_layers)

            for layer_idx in range(num_layers):
                # hidden_states[0] is the embedding layer; transformer layers start at index 1
                layer_hidden = hidden_states[layer_idx + 1]  # (batch=1, seq_len, hidden_dim)
                # Mean-pool over the sequence dimension
                pooled = layer_hidden.mean(dim=1).squeeze(0).cpu().float().numpy()
                activations_by_layer[layer_idx].append(pooled)

            # Generate model output and compare to expected answer
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
            generated_text = tokenizer.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            is_hallucinated = self._is_hallucinated(generated_text, expected)
            labels.append(1 if is_hallucinated else 0)

        log.info(
            "h_neuron.activations_collected layers=%d items=%d hallucinated=%d faithful=%d",
            num_layers,
            len(labels),
            sum(labels),
            len(labels) - sum(labels),
        )

        # Need both classes to train a classifier
        if len(set(labels)) < 2:
            log.warning(
                "h_neuron.insufficient_label_diversity "
                "All calibration items produced the same label. "
                "Returning empty profile."
            )
            return HNeuronProfile(
                model_name=model_path,
                neurons_by_layer={},
                total_neurons=0,
                calibration_size=len(calibration_data),
            )

        # Train sparse logistic regression per layer to find H-Neurons
        import numpy as np  # type: ignore[import-untyped]

        y = np.array(labels)
        neurons_by_layer: dict[int, list[int]] = {}

        for layer_idx in range(num_layers):
            features = np.stack(activations_by_layer[layer_idx])  # (n_samples, hidden_dim)

            # L1-regularized logistic regression (sparse)
            clf = linear_model.LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1.0,
                max_iter=1000,
                random_state=42,
            )
            clf.fit(features, y)

            # Non-zero weight indices are the H-Neurons for this layer
            nonzero_indices = np.nonzero(clf.coef_[0])[0].tolist()
            if nonzero_indices:
                neurons_by_layer[layer_idx] = nonzero_indices
                log.debug(
                    "h_neuron.layer_result layer=%d h_neurons=%d",
                    layer_idx,
                    len(nonzero_indices),
                )

        total = sum(len(v) for v in neurons_by_layer.values())
        profile = HNeuronProfile(
            model_name=model_path,
            neurons_by_layer=neurons_by_layer,
            total_neurons=total,
            calibration_size=len(calibration_data),
        )

        log.info(
            "h_neuron.profile_complete model=%s total_h_neurons=%d layers_with_h_neurons=%d",
            model_path,
            total,
            len(neurons_by_layer),
        )
        return profile

    def _is_hallucinated(self, generated: str, expected: str) -> bool:
        """Determine if a generated answer is hallucinated relative to the expected answer.

        Uses a simple heuristic: the expected answer (lowercased) should appear
        somewhere in the generated text. If it does not, we consider the output hallucinated.
        More sophisticated methods (e.g. NLI-based) could be substituted here.
        """
        gen_lower = generated.lower().strip()
        exp_lower = expected.lower().strip()

        if not exp_lower:
            return False

        # Direct substring match
        if exp_lower in gen_lower:
            return False

        # Token overlap: if the majority of expected tokens appear in generated text
        exp_tokens = set(exp_lower.split())
        if not exp_tokens:
            return False

        gen_tokens = set(gen_lower.split())
        overlap = len(exp_tokens & gen_tokens) / len(exp_tokens)
        return overlap < 0.5

    def save_profile(self, profile: HNeuronProfile, path: str | Path | None = None) -> Path:
        """Save a profile to JSON.

        Args:
            profile: The profile to save.
            path: Optional explicit path. If None, saves to profile_dir/<model_name>.json.

        Returns:
            The path where the profile was saved.
        """
        if path is None:
            self._profile_dir.mkdir(parents=True, exist_ok=True)
            safe_name = profile.model_name.replace("/", "__").replace("\\", "__")
            path = self._profile_dir / f"{safe_name}.json"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(profile)
        # JSON requires string keys; convert int layer indices
        data["neurons_by_layer"] = {
            str(k): v for k, v in data["neurons_by_layer"].items()
        }
        path.write_text(json.dumps(data, indent=2))
        log.info("h_neuron.profile_saved path=%s", path)
        return path

    def load_profile(self, model_name: str) -> HNeuronProfile | None:
        """Load a cached profile for a model.

        Args:
            model_name: The model name (as used during profiling).

        Returns:
            The loaded profile, or None if not found.
        """
        safe_name = model_name.replace("/", "__").replace("\\", "__")
        path = self._profile_dir / f"{safe_name}.json"

        if not path.exists():
            log.debug("h_neuron.profile_not_found model=%s path=%s", model_name, path)
            return None

        try:
            data = json.loads(path.read_text())
            # Convert string keys back to int
            data["neurons_by_layer"] = {
                int(k): v for k, v in data["neurons_by_layer"].items()
            }
            profile = HNeuronProfile(**data)
            log.info(
                "h_neuron.profile_loaded model=%s total_neurons=%d",
                model_name,
                profile.total_neurons,
            )
            return profile
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("h_neuron.profile_load_error model=%s error=%s", model_name, exc)
            return None

    def list_profiles(self) -> list[str]:
        """List model names of all cached profiles.

        Returns:
            List of model name strings.
        """
        if not self._profile_dir.exists():
            return []

        profiles: list[str] = []
        for path in sorted(self._profile_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                profiles.append(data.get("model_name", path.stem.replace("__", "/")))
            except (json.JSONDecodeError, KeyError):
                log.warning("h_neuron.invalid_profile path=%s", path)
        return profiles

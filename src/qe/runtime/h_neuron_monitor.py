"""Real-time H-Neuron activation monitoring during inference."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from qe.runtime.h_neurons import HNeuronProfile

log = logging.getLogger(__name__)


def _require_torch() -> Any:
    """Import and return torch, raising a clear error if unavailable."""
    try:
        import torch  # type: ignore[import-untyped]

        return torch
    except ImportError:
        raise ImportError(
            "torch is required for H-Neuron monitoring. "
            "Install it with: pip install torch"
        ) from None


class HNeuronMonitor:
    """Monitors H-Neuron activations during model inference to estimate hallucination risk.

    Installs forward hooks on transformer layers that contain identified H-Neurons.
    After a forward pass, call ``get_hallucination_risk()`` to obtain an aggregated
    risk score between 0.0 (low risk) and 1.0 (high risk).
    """

    def __init__(self, profile: HNeuronProfile) -> None:
        self._profile = profile
        self._hooks: list[Any] = []
        # Per-layer activation magnitudes collected during the current generation
        self._layer_activations: dict[int, list[float]] = defaultdict(list)
        # Running statistics for normalisation (mean/std per layer, built up over calls)
        self._layer_stats: dict[int, dict[str, float]] = {}

    def install_hooks(self, model: Any) -> None:
        """Register forward hooks on layers containing H-Neurons.

        Args:
            model: A HuggingFace-style transformer model whose layers are accessible
                   via ``model.model.layers`` (LLaMA-like) or ``model.transformer.h``
                   (GPT-like).
        """
        _require_torch()
        self.remove_hooks()

        layers = self._get_model_layers(model)
        installed = 0
        for layer_idx, neuron_indices in self._profile.neurons_by_layer.items():
            if layer_idx >= len(layers):
                log.warning(
                    "h_neuron_monitor.layer_out_of_range layer=%d total_layers=%d",
                    layer_idx,
                    len(layers),
                )
                continue

            hook = layers[layer_idx].register_forward_hook(
                self._make_monitor_hook(layer_idx, neuron_indices)
            )
            self._hooks.append(hook)
            installed += 1

        log.info(
            "h_neuron_monitor.hooks_installed count=%d model=%s",
            installed,
            self._profile.model_name,
        )

    def remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _get_model_layers(self, model: Any) -> list[Any]:
        """Resolve the list of transformer layers from the model object.

        Supports common HuggingFace architectures:
        - LLaMA / Mistral: model.model.layers
        - GPT-2 / GPT-Neo: model.transformer.h
        - Falcon: model.transformer.h
        - Phi: model.model.layers
        """
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return list(model.gpt_neox.layers)
        raise ValueError(
            f"Cannot locate transformer layers on model of type {type(model).__name__}. "
            "Supported architectures: LLaMA/Mistral (.model.layers), "
            "GPT-2/Neo (.transformer.h), GPT-NeoX (.gpt_neox.layers)."
        )

    def _make_monitor_hook(
        self, layer_id: int, neuron_indices: list[int]
    ) -> Callable[..., None]:
        """Create a forward-hook closure that records H-Neuron activation magnitudes."""
        torch = _require_torch()
        index_tensor: Any = None  # lazily built on first call

        def hook_fn(module: Any, input: Any, output: Any) -> None:
            nonlocal index_tensor
            # output may be a tuple (hidden_states, ...) or a tensor
            hidden = output[0] if isinstance(output, tuple) else output

            if index_tensor is None or index_tensor.device != hidden.device:
                index_tensor = torch.tensor(
                    neuron_indices, dtype=torch.long, device=hidden.device
                )

            # hidden shape: (batch, seq_len, hidden_dim)
            # Select only the H-Neuron dimensions
            h_neuron_activations = hidden[:, :, index_tensor]  # (batch, seq_len, n_h_neurons)

            # Mean absolute activation across batch and sequence dimensions
            mean_abs = h_neuron_activations.abs().mean().item()
            self._layer_activations[layer_id].append(mean_abs)

        return hook_fn

    def get_hallucination_risk(self) -> float:
        """Compute an aggregated hallucination risk score.

        Returns:
            Float between 0.0 (low risk) and 1.0 (high risk).
            The score is based on the mean activation magnitude of H-Neurons
            across all monitored layers, passed through a sigmoid-like normalisation.
        """
        if not self._layer_activations:
            return 0.0

        layer_means: list[float] = []
        for layer_id, values in self._layer_activations.items():
            if values:
                layer_mean = sum(values) / len(values)
                layer_means.append(layer_mean)
                # Update running stats for the layer
                self._update_layer_stats(layer_id, layer_mean)

        if not layer_means:
            return 0.0

        # Global mean activation across layers
        global_mean = sum(layer_means) / len(layer_means)

        # Normalise using running statistics if available, else use a sigmoid heuristic
        if self._layer_stats:
            # Z-score relative to historical baseline
            all_hist_means = [s["mean"] for s in self._layer_stats.values() if s["count"] > 1]
            all_hist_stds = [s["std"] for s in self._layer_stats.values() if s["count"] > 1]
            if all_hist_means and all_hist_stds:
                baseline_mean = sum(all_hist_means) / len(all_hist_means)
                baseline_std = sum(all_hist_stds) / len(all_hist_stds)
                if baseline_std > 1e-8:
                    z = (global_mean - baseline_mean) / baseline_std
                    return _sigmoid(z)

        # Fallback: use a simple sigmoid on the raw activation
        # Centred at 1.0, scale controls sensitivity
        return _sigmoid((global_mean - 1.0) * 2.0)

    def _update_layer_stats(self, layer_id: int, value: float) -> None:
        """Update running mean/std for a layer using Welford's algorithm."""
        if layer_id not in self._layer_stats:
            self._layer_stats[layer_id] = {"mean": 0.0, "m2": 0.0, "count": 0, "std": 0.0}

        stats = self._layer_stats[layer_id]
        stats["count"] += 1
        delta = value - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = value - stats["mean"]
        stats["m2"] += delta * delta2
        if stats["count"] > 1:
            stats["std"] = math.sqrt(stats["m2"] / (stats["count"] - 1))

    def reset(self) -> None:
        """Clear activation scores for the next generation."""
        self._layer_activations.clear()

    def get_activation_summary(self) -> dict[str, Any]:
        """Return per-layer activation statistics.

        Returns:
            Dictionary with per-layer stats and overall summary.
        """
        summary: dict[str, Any] = {"layers": {}, "overall": {}}
        all_values: list[float] = []

        for layer_id in sorted(self._layer_activations.keys()):
            values = self._layer_activations[layer_id]
            if values:
                layer_mean = sum(values) / len(values)
                layer_max = max(values)
                layer_min = min(values)
                summary["layers"][layer_id] = {
                    "mean_activation": layer_mean,
                    "max_activation": layer_max,
                    "min_activation": layer_min,
                    "num_observations": len(values),
                    "num_h_neurons": len(self._profile.neurons_by_layer.get(layer_id, [])),
                }
                all_values.extend(values)

        if all_values:
            summary["overall"] = {
                "mean_activation": sum(all_values) / len(all_values),
                "max_activation": max(all_values),
                "min_activation": min(all_values),
                "total_observations": len(all_values),
                "monitored_layers": len(self._layer_activations),
                "risk_score": self.get_hallucination_risk(),
            }
        else:
            summary["overall"] = {
                "mean_activation": 0.0,
                "max_activation": 0.0,
                "min_activation": 0.0,
                "total_observations": 0,
                "monitored_layers": 0,
                "risk_score": 0.0,
            }

        return summary


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid mapping to [0, 1]."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class HNeuronSuppressor:
    """Suppresses H-Neuron activations during inference to reduce hallucination.

    Installs pre-forward hooks (``register_forward_pre_hook``) that scale down
    the activations of identified H-Neurons by a configurable factor.
    """

    def __init__(
        self,
        profile: HNeuronProfile,
        suppression_factor: float = 0.5,
    ) -> None:
        """
        Args:
            profile: H-Neuron profile identifying which neurons to suppress.
            suppression_factor: Factor to multiply H-Neuron activations by.
                0.0 = full suppression, 1.0 = no suppression. Default 0.5.
        """
        if not 0.0 <= suppression_factor <= 1.0:
            raise ValueError(f"suppression_factor must be in [0, 1], got {suppression_factor}")
        self._profile = profile
        self._suppression_factor = suppression_factor
        self._hooks: list[Any] = []

    def install_hooks(self, model: Any) -> None:
        """Register pre-forward hooks that suppress H-Neuron activations.

        Args:
            model: A HuggingFace-style transformer model.
        """
        _require_torch()
        self.remove_hooks()

        layers = self._get_model_layers(model)
        installed = 0
        for layer_idx, neuron_indices in self._profile.neurons_by_layer.items():
            if layer_idx >= len(layers):
                log.warning(
                    "h_neuron_suppressor.layer_out_of_range layer=%d total_layers=%d",
                    layer_idx,
                    len(layers),
                )
                continue

            hook = layers[layer_idx].register_forward_pre_hook(
                self._make_suppression_hook(layer_idx, neuron_indices)
            )
            self._hooks.append(hook)
            installed += 1

        log.info(
            "h_neuron_suppressor.hooks_installed count=%d factor=%.2f model=%s",
            installed,
            self._suppression_factor,
            self._profile.model_name,
        )

    def remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _get_model_layers(self, model: Any) -> list[Any]:
        """Resolve transformer layers from the model (same logic as HNeuronMonitor)."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            return list(model.gpt_neox.layers)
        raise ValueError(
            f"Cannot locate transformer layers on model of type {type(model).__name__}. "
            "Supported architectures: LLaMA/Mistral (.model.layers), "
            "GPT-2/Neo (.transformer.h), GPT-NeoX (.gpt_neox.layers)."
        )

    def _make_suppression_hook(
        self, layer_id: int, neuron_indices: list[int]
    ) -> Callable[..., Any]:
        """Create a pre-forward hook that scales down H-Neuron activations in the input."""
        torch = _require_torch()
        index_tensor: Any = None
        factor = self._suppression_factor

        def hook_fn(module: Any, args: Any) -> Any:
            nonlocal index_tensor

            # Pre-forward hook receives the input tuple
            if not args:
                return args

            hidden = args[0]
            if not isinstance(hidden, torch.Tensor):
                return args

            if index_tensor is None or index_tensor.device != hidden.device:
                index_tensor = torch.tensor(
                    neuron_indices, dtype=torch.long, device=hidden.device
                )

            # Validate indices are within hidden dimension
            if index_tensor.max() >= hidden.shape[-1]:
                log.warning(
                    "h_neuron_suppressor.index_out_of_range layer=%d max_idx=%d hidden_dim=%d",
                    layer_id,
                    index_tensor.max().item(),
                    hidden.shape[-1],
                )
                return args

            # Scale H-Neuron activations by the suppression factor
            modified = hidden.clone()
            modified[:, :, index_tensor] *= factor

            # Return modified args tuple
            return (modified,) + args[1:]

        return hook_fn

"""Shared initialization formulas."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config_utils import optional_int

_COMPLETEP_PARAMETERIZATIONS = frozenset({"completeP", "completep", "spectralP", "SpectralP", "spectralp"})


def spectral_std(*, n_in: int, n_out: int, hidden_init_std_factor: float) -> float:
    if n_in <= 0:
        raise ValueError(f"n_in must be positive, got {n_in}.")
    if n_out <= 0:
        raise ValueError(f"n_out must be positive, got {n_out}.")
    if hidden_init_std_factor < 0.0:
        raise ValueError(f"hidden_init_std_factor must be >= 0, got {hidden_init_std_factor}.")
    return (hidden_init_std_factor / math.sqrt(float(n_in))) * min(1.0, math.sqrt(float(n_out) / float(n_in)))


def _linear_fan_in_out(linear: nn.Module) -> tuple[int, int]:
    in_features = getattr(linear, "in_features", None)
    out_features = getattr(linear, "out_features", None)
    if isinstance(in_features, int) and isinstance(out_features, int):
        return in_features, out_features

    weight = getattr(linear, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim < 2:
        raise TypeError(
            f"Expected {type(linear).__name__} to expose in_features/out_features or a rank >= 2 weight tensor."
        )
    return int(weight.shape[1]), int(weight.shape[0])


def linear_spectral_std(linear: nn.Module, *, hidden_init_std_factor: float) -> float:
    n_in, n_out = _linear_fan_in_out(linear)
    return spectral_std(
        n_in=n_in,
        n_out=n_out,
        hidden_init_std_factor=hidden_init_std_factor,
    )


def residual_output_projection_depth_divisor(config: object, *, num_layers: int) -> float:
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")

    parameterization = getattr(config, "parameterization", None)
    if isinstance(parameterization, str) and parameterization in _COMPLETEP_PARAMETERIZATIONS:
        num_layers_base = optional_int(config, "num_hidden_layers_base", num_layers)
        if num_layers_base <= 0:
            raise ValueError(f"num_hidden_layers_base must be positive, got {num_layers_base}.")
        return float(num_layers_base)
    return float(num_layers)


def residual_output_projection_std(
    linear: nn.Module,
    *,
    hidden_init_std_factor: float,
    depth_divisor: float,
) -> float:
    """Match `init_gpt_weights` residual projection scaling.

    Residual branch output projections use spectral fan scaling plus the
    parameterization-specific depth divisor selected by
    `residual_output_projection_depth_divisor`.
    """

    if depth_divisor <= 0.0:
        raise ValueError(f"depth_divisor must be positive, got {depth_divisor}.")
    return linear_spectral_std(linear, hidden_init_std_factor=hidden_init_std_factor) / depth_divisor


__all__ = [
    "spectral_std",
    "linear_spectral_std",
    "residual_output_projection_depth_divisor",
    "residual_output_projection_std",
]

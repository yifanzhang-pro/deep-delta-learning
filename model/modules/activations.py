from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F

ActivationName = Literal["silu", "relu", "elu+1", "identity"]

_ACTIVATION_NAMES: set[str] = {"silu", "relu", "elu+1", "identity"}


def apply_activation(x: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation is None or activation == "identity":
        return x
    if activation == "silu":
        return F.silu(x)
    if activation == "relu":
        return F.relu(x)
    if activation == "elu+1":
        return (F.elu(x, 1.0, False) + 1.0).to(dtype=x.dtype)
    raise ValueError(f"Unsupported activation {activation!r}. Expected one of: {sorted(_ACTIVATION_NAMES)!r}.")


def validate_activation_name(value: object, *, field_name: str) -> ActivationName | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None, got {type(value).__name__}.")
    if value == "elu":
        value = "elu+1"
    if value not in _ACTIVATION_NAMES:
        raise ValueError(f"Unsupported {field_name}={value!r}. Expected one of: {sorted(_ACTIVATION_NAMES)!r}.")
    return cast(ActivationName, value)

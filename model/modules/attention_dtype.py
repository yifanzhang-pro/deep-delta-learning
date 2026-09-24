from __future__ import annotations

from typing import Literal, cast

import torch

AttentionDType = Literal["auto", "bfloat16", "float32"]


def validate_attention_dtype(value: object) -> AttentionDType:
    if value is None:
        return "auto"
    if not isinstance(value, str):
        raise TypeError(f"attention_dtype must be a string, got {type(value).__name__}.")
    if value not in ("auto", "bfloat16", "float32"):
        raise ValueError(f"attention_dtype must be one of auto|bfloat16|float32, got {value!r}.")
    return cast(AttentionDType, value)


def attention_dtype_to_torch_dtype(value: AttentionDType) -> torch.dtype | None:
    if value == "auto":
        return None
    if value == "bfloat16":
        return torch.bfloat16
    return torch.float32

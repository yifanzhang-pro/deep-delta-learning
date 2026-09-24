from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def disable_autocast(device_type: str, *, disable_tf32: bool = False) -> Iterator[None]:
    """Disable autocast, and optionally CUDA TF32, where supported."""
    allow_tf32_matmul_prev: bool | None = None
    allow_tf32_cudnn_prev: bool | None = None
    if disable_tf32 and device_type == "cuda":
        allow_tf32_matmul_prev = bool(torch.backends.cuda.matmul.allow_tf32)
        allow_tf32_cudnn_prev = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    try:
        try:
            with torch.autocast(device_type=device_type, enabled=False):
                yield
        except TypeError, RuntimeError, ValueError:
            yield
    finally:
        if allow_tf32_matmul_prev is not None and allow_tf32_cudnn_prev is not None:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32_matmul_prev
            torch.backends.cudnn.allow_tf32 = allow_tf32_cudnn_prev


def linear_fp32(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Run a linear projection in float32 with autocast disabled."""
    bias: torch.Tensor | None = linear.bias
    with disable_autocast(x.device.type):
        return F.linear(
            x.to(dtype=torch.float32),
            linear.weight.to(dtype=torch.float32),
            None if bias is None else bias.to(dtype=torch.float32),
        )

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict

from .activations import ActivationName, apply_activation


class _Transpose12Contiguous(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous()

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (grad_output.transpose(1, 2).contiguous(),)


class ShortConvSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    use_q: bool = False
    use_k: bool = False
    use_v: bool = False
    kernel_size: int = 4
    shift_right1_k: bool = True

    @classmethod
    def from_pretrained_config(cls, config: Any) -> ShortConvSpec:
        if getattr(config, "shortconv_kernel", None) is not None:
            raise ValueError("Unsupported config key 'shortconv_kernel'; use 'shortconv_kernel_size'.")
        return cls(
            use_q=bool(getattr(config, "shortconvq", False)),
            use_k=bool(getattr(config, "shortconvk", False)),
            use_v=bool(getattr(config, "shortconvv", False)),
            kernel_size=int(getattr(config, "shortconv_kernel_size", 4)),
            shift_right1_k=bool(getattr(config, "shortconv_shift_right1_k", True)),
        )


class DepthwiseShortConv1d(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        kernel_size: int,
        activation: ActivationName | None = None,
        shift_right1: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}.")
        self.activation = activation
        self.shift_right1 = bool(shift_right1)
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.kernel_size,
            padding=0,
            groups=hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()  # (B, C, T)
        pad_left = self.kernel_size - 1 + (1 if self.shift_right1 else 0)
        x = F.pad(x, (pad_left, 0)).contiguous()
        x = self.conv(x)
        if self.shift_right1:
            x = x[:, :, :-1]
        if x.device.type == "mps":
            x = _Transpose12Contiguous.apply(x)
        else:
            x = x.transpose(1, 2).contiguous()  # (B, T, C)
        return apply_activation(x, self.activation)


__all__ = ["DepthwiseShortConv1d", "ShortConvSpec"]

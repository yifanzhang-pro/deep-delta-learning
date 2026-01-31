from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def token_shift_torch(x: torch.Tensor, prev_weight: torch.Tensor, curr_weight: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"expected x of shape (B, T, H, D), got {tuple(x.shape)}")
    if prev_weight.ndim != 3:
        raise ValueError(f"expected prev_weight of shape (B, T, H), got {tuple(prev_weight.shape)}")
    if curr_weight.ndim != 3:
        raise ValueError(f"expected curr_weight of shape (B, T, H), got {tuple(curr_weight.shape)}")

    x_prev = torch.roll(x, shifts=1, dims=1)
    x_prev[:, 0, :, :] = 0.0
    return x_prev * prev_weight.unsqueeze(-1) + x * curr_weight.unsqueeze(-1)


class ShiftLinearTorch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        bias: bool,
        shift_bias: bool = False,
    ):
        super().__init__()

        if output_dim % num_heads != 0:
            raise ValueError("output_dim must be divisible by num_heads")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.shift_proj = nn.Linear(input_dim, num_heads, bias=shift_bias)

    def forward(self, x: torch.Tensor, shift_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x of shape (B, T, D), got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.size()
        out = self.linear(x)

        alpha = torch.sigmoid(self.shift_proj(x).float())  # (B, T, H) in fp32
        out_per_head = out.view(batch_size, seq_len, self.num_heads, -1)

        if seq_len > 1:
            result_per_head = token_shift_torch(out_per_head, alpha, 1.0 - alpha)
        else:
            if shift_state is None:
                result_per_head = out_per_head
            else:
                shift_state_per_head = shift_state.view(batch_size, 1, self.num_heads, -1)
                result_per_head = (
                    alpha.unsqueeze(-1) * shift_state_per_head + (1.0 - alpha).unsqueeze(-1) * out_per_head
                )

        result_per_head = result_per_head.to(out.dtype)

        if shift_state is not None:
            shift_state.copy_(out[:, -1, :])

        return result_per_head.reshape(batch_size, seq_len, self.output_dim)

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class Rotary(nn.Module):
    def __init__(self, dim: int, *, base: float = 10000.0, rope_ratio: float = 1.0) -> None:
        super().__init__()
        if not (0.0 <= float(rope_ratio) <= 1.0):
            raise ValueError(f"rope_ratio must be in [0, 1], got {rope_ratio}.")
        rotary_dim = int(dim * float(rope_ratio))
        rotary_dim -= rotary_dim % 2
        rotary_dim = max(0, min(rotary_dim, int(dim)))

        self.rotary_dim = rotary_dim
        if rotary_dim <= 0:
            inv_freq = torch.empty(0)
        else:
            inv_freq = 1.0 / (float(base) ** (torch.arange(0, rotary_dim, 2).float() / float(rotary_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached: int | None = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rotary:
        super()._apply(fn)
        self.inv_freq = self.inv_freq.float()
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        return self

    def _set_cos_sin_cache(self, seq_len: int, *, device: torch.device) -> None:
        self.seq_len_cached = seq_len
        # Always compute RoPE angles/sin/cos on CPU in float32, then move to the target device.
        t_cpu = torch.arange(seq_len, device=torch.device("cpu"), dtype=torch.float32)
        inv_freq_cpu = self.inv_freq.detach().to(device=torch.device("cpu"), dtype=torch.float32)
        freqs_cpu = torch.outer(t_cpu, inv_freq_cpu)
        cos_cpu = freqs_cpu.cos()
        sin_cpu = freqs_cpu.sin()

        self.cos_cached = cos_cpu.to(device=device, dtype=torch.float32)
        self.sin_cached = sin_cpu.to(device=device, dtype=torch.float32)

    def forward(
        self,
        x: torch.Tensor,
        *,
        seq_len_offset: int = 0,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len_offset < 0:
            raise ValueError(f"seq_len_offset must be >= 0, got {seq_len_offset}")

        if position_ids is not None:
            max_pos = int(position_ids.max().item()) + 1
            if (
                self.cos_cached is None
                or self.sin_cached is None
                or self.seq_len_cached is None
                or max_pos > self.seq_len_cached
                or self.cos_cached.device != x.device
            ):
                self._set_cos_sin_cache(max_pos, device=x.device)
            assert self.cos_cached is not None
            assert self.sin_cached is not None
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
            return cos[:, :, None, :], sin[:, :, None, :]

        seq_len = int(x.shape[1])
        total_len = seq_len + seq_len_offset
        if (
            self.cos_cached is None
            or self.sin_cached is None
            or self.seq_len_cached is None
            or total_len > self.seq_len_cached
            or self.cos_cached.device != x.device
        ):
            self._set_cos_sin_cache(total_len, device=x.device)
        assert self.cos_cached is not None
        assert self.sin_cached is not None
        cos = self.cos_cached[seq_len_offset:total_len]
        sin = self.sin_cached[seq_len_offset:total_len]
        return cos[None, :, None, :], sin[None, :, None, :]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"apply_rotary_emb expects x.ndim==4, got {x.ndim}")
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    d = x_rot.shape[3] // 2
    x1 = x_rot[..., :d]
    x2 = x_rot[..., d:]
    # Standard RoPE rotation by +theta given cos(theta)/sin(theta).
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y_rot = torch.cat([y1, y2], dim=3).type_as(x)
    return torch.cat([y_rot, x_pass], dim=3).type_as(x)


__all__ = ["Rotary", "apply_rotary_emb"]

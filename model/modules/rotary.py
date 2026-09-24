from __future__ import annotations

from collections.abc import Callable
from typing import Final

import torch
import torch.nn as nn


_POSITION_ID_DTYPES: Final[tuple[torch.dtype, ...]] = (torch.int32, torch.int64)


class Rotary(nn.Module):
    def __init__(self, dim: int, *, base: float = 10000.0, rope_ratio: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.base = float(base)
        self.rope_ratio = float(rope_ratio)
        if not (0.0 <= self.rope_ratio <= 1.0):
            raise ValueError(f"rope_ratio must be in [0, 1], got {rope_ratio}.")
        rotary_dim = int(self.dim * self.rope_ratio)
        rotary_dim -= rotary_dim % 2
        rotary_dim = max(0, min(rotary_dim, self.dim))

        self.rotary_dim = rotary_dim
        inv_freq = self._build_inv_freq(device=torch.device("cpu"))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached: int | None = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def _build_inv_freq(self, *, device: torch.device) -> torch.Tensor:
        if self.rotary_dim <= 0:
            return torch.empty(0, device=device)
        return 1.0 / (
            self.base ** (torch.arange(0, self.rotary_dim, 2, device=device).float() / float(self.rotary_dim))
        )

    def refresh_derived_buffers(self) -> None:
        self.inv_freq = self._build_inv_freq(device=self.inv_freq.device)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

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
        validate_position_ids: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len_offset < 0:
            raise ValueError(f"seq_len_offset must be >= 0, got {seq_len_offset}")

        if position_ids is not None:
            if position_ids.dtype not in _POSITION_ID_DTYPES:
                raise ValueError(f"position_ids must be an int32 or int64 tensor, got dtype={position_ids.dtype}.")
            expected_shape = (int(x.shape[0]), int(x.shape[1]))
            if position_ids.shape != expected_shape:
                raise ValueError(f"position_ids must have shape (B, T), got {tuple(position_ids.shape)}.")
            if position_ids.numel() == 0:
                empty_shape = (*position_ids.shape, 1, int(self.inv_freq.numel()))
                empty = torch.empty(empty_shape, device=x.device, dtype=torch.float32)
                return empty, empty.clone()
            if position_ids.device.type == "meta":
                raise ValueError("position_ids on meta device cannot be materialized.")
            if position_ids.device.type != "cpu":
                if validate_position_ids:
                    # Debug-only scalar validation synchronizes accelerator
                    # tensors; leave it disabled on performance-critical paths.
                    min_pos = int(position_ids.min().item())
                    if min_pos < 0:
                        raise ValueError("position_ids must be non-negative.")
                # Exact min/max validation would synchronize accelerator
                # tensors. The default hot path trusts caller-provided IDs and
                # computes only the requested angles directly; validation/debug
                # runs can request scalar checks through validate_position_ids.
                inv_freq = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
                position_values = position_ids.to(dtype=torch.float32)
                # apply_rotary_emb treats cos/sin's last dimension as half of
                # the rotated width and pairs it with the second half of x_rot.
                freqs = position_values.unsqueeze(-1) * inv_freq
                return freqs.cos()[:, :, None, :], freqs.sin()[:, :, None, :]
            min_pos = int(position_ids.min().item())
            if min_pos < 0:
                raise ValueError("position_ids must be non-negative.")
            cache_len = int(position_ids.max().item()) + 1
            if (
                self.cos_cached is None
                or self.sin_cached is None
                or self.seq_len_cached is None
                or cache_len > self.seq_len_cached
                or self.cos_cached.device != x.device
            ):
                self._set_cos_sin_cache(cache_len, device=x.device)
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

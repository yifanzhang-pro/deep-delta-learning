"""Shared DDL input embedding helpers."""

from __future__ import annotations

from typing import Any, Callable, Self, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..DDL_utils import _TemporalShortConvExpandFn
from .shortconv_cache import compact_current_tokens, compact_shortconv_history, gather_compacted_current_slots


def _validate_input_embed_x(x: torch.Tensor, *, hidden_size: int) -> tuple[int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (B, T, d), got {tuple(x.shape)}")
    batch_size, seq_len, actual_hidden_size = (int(dim) for dim in x.shape)
    if actual_hidden_size != hidden_size:
        raise ValueError(f"Expected x feature dim {hidden_size}, got {actual_hidden_size}.")
    return batch_size, seq_len, actual_hidden_size


def _prepare_input_embed_past(
    past: torch.Tensor,
    *,
    batch_size: int,
    hidden_size: int,
    past_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if (
        past.ndim != 3
        or int(past.shape[0]) != batch_size
        or int(past.shape[1]) != hidden_size
        or int(past.shape[2]) != past_len
    ):
        raise ValueError(f"Expected past with shape (B, d, {past_len}), got {tuple(past.shape)}")
    if past.device == device and past.dtype == dtype:
        return past
    if past.device.type == "meta":
        raise ValueError("input embedding past cache on meta device cannot be materialized.")
    return past.to(device=device, dtype=dtype)


class _InputEmbedConvCompatView:
    def __init__(self, owner: InputEmbedShortConvExpander) -> None:
        self._owner = owner

    @property
    def weight(self) -> torch.Tensor:
        return self._owner.weight.reshape(
            self._owner.hidden_size * self._owner.value_channels,
            1,
            self._owner.kernel_size,
        )


class InputEmbedShortConvExpander(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size: int = int(getattr(config, "hidden_size"))
        self.value_channels: int = int(getattr(config, "ddl_value_channels", 4))
        if self.value_channels <= 1:
            raise ValueError("ddl_value_channels must be > 1 for expanded-state DDL.")

        kernel_size = int(getattr(config, "input_embed_shortconv_kernel_size", 4))
        if kernel_size <= 0:
            raise ValueError(f"input_embed_shortconv_kernel_size must be positive, got {kernel_size}.")
        self.kernel_size: int = kernel_size

        self.weight = nn.Parameter(torch.empty(self.hidden_size, self.value_channels, self.kernel_size))
        self.conv = _InputEmbedConvCompatView(self)
        self.reset_parameters_identity()

    def reset_deterministic_parameters(self) -> None:
        self.reset_parameters_identity()

    def reset_parameters_identity(self) -> None:
        with torch.no_grad():
            self.weight.zero_()
            self.weight[:, :, self.kernel_size - 1] = 1.0

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        was_meta = self.weight.device.type == "meta"
        result = super()._apply(fn, recurse=recurse)
        if was_meta and self.weight.device.type != "meta":
            self.reset_parameters_identity()
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = _validate_input_embed_x(x, hidden_size=self.hidden_size)
        if seq_len == 0:
            return x.new_empty((batch_size, seq_len, hidden_size, self.value_channels))

        x_t = x.transpose(1, 2).contiguous()
        pad_left = self.kernel_size - 1
        x_t = F.pad(x_t, (pad_left, 0)).contiguous()
        y = F.conv1d(x_t, self.conv.weight, bias=None, stride=1, padding=0, groups=self.hidden_size)
        y = y.transpose(1, 2).contiguous()
        return y.reshape(batch_size, seq_len, hidden_size, self.value_channels)

    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = _validate_input_embed_x(x, hidden_size=self.hidden_size)

        x_t = x.transpose(1, 2).contiguous()
        past_len = self.kernel_size - 1
        if past_len <= 0:
            empty = x_t[:, :, :0].contiguous()
            if seq_len == 0:
                expanded = x.new_empty((batch_size, seq_len, hidden_size, self.value_channels))
                return expanded, empty
            expanded = self.forward(x)
            return expanded, empty

        if past is None:
            past = torch.zeros((batch_size, hidden_size, past_len), device=x.device, dtype=x.dtype)
        else:
            past = _prepare_input_embed_past(
                past,
                batch_size=batch_size,
                hidden_size=hidden_size,
                past_len=past_len,
                device=x.device,
                dtype=x.dtype,
            )

        if seq_len == 0:
            expanded = x.new_empty((batch_size, seq_len, hidden_size, self.value_channels))
            return expanded, past.contiguous()

        x_for_conv = x
        current_mask: torch.Tensor | None = None
        active_ranks: torch.Tensor | None = None
        if token_mask is not None:
            x_for_conv, current_mask, active_ranks = compact_current_tokens(x, token_mask=token_mask)
            x_t = x_for_conv.transpose(1, 2).contiguous()

        x_cat = torch.cat([past, x_t], dim=-1)
        y = F.conv1d(x_cat, self.conv.weight, bias=None, stride=1, padding=0, groups=self.hidden_size)
        y = y.transpose(1, 2).contiguous()
        expanded = y.reshape(batch_size, seq_len, hidden_size, self.value_channels)
        if current_mask is not None and active_ranks is not None:
            expanded = gather_compacted_current_slots(
                expanded,
                current_mask=current_mask,
                active_ranks=active_ranks,
            )
        past_tokens = past.transpose(1, 2)
        past_out = (
            compact_shortconv_history(
                x,
                history=past_tokens,
                token_mask=token_mask,
                history_size=past_len,
            )
            .transpose(1, 2)
            .contiguous()
        )
        return expanded, past_out


class TritonInputEmbedShortConvExpander(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size: int = int(getattr(config, "hidden_size"))
        self.value_channels: int = int(getattr(config, "ddl_value_channels", 4))
        if self.value_channels <= 1:
            raise ValueError("ddl_value_channels must be > 1 for expanded-state DDL.")

        kernel_size = int(getattr(config, "input_embed_shortconv_kernel_size", 4))
        if kernel_size <= 0:
            raise ValueError(f"input_embed_shortconv_kernel_size must be positive, got {kernel_size}.")
        self.kernel_size: int = kernel_size

        self.weight = nn.Parameter(torch.empty(self.hidden_size, self.value_channels, self.kernel_size))
        self.reset_parameters_identity()

    def reset_deterministic_parameters(self) -> None:
        self.reset_parameters_identity()

    def reset_parameters_identity(self) -> None:
        with torch.no_grad():
            self.weight.zero_()
            self.weight[:, :, self.kernel_size - 1] = 1.0

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        was_meta = self.weight.device.type == "meta"
        result = super()._apply(fn, recurse=recurse)
        if was_meta and self.weight.device.type != "meta":
            self.reset_parameters_identity()
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = _validate_input_embed_x(x, hidden_size=self.hidden_size)
        if seq_len == 0:
            return x.new_empty((batch_size, seq_len, hidden_size, self.value_channels))
        if not x.is_cuda:
            raise RuntimeError("TritonInputEmbedShortConvExpander requires CUDA tensors for Triton input shortconv.")
        # The Triton kernel owns causal left padding through temporal bounds
        # checks, matching the explicit `F.pad` used by the Torch path.
        return cast(torch.Tensor, _TemporalShortConvExpandFn.apply(x, self.weight))

    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = _validate_input_embed_x(x, hidden_size=self.hidden_size)

        past_len = self.kernel_size - 1
        if past_len <= 0:
            empty = x.transpose(1, 2)[:, :, :0].contiguous()
            if seq_len == 0:
                expanded = x.new_empty((batch_size, seq_len, hidden_size, self.value_channels))
                return expanded, empty
            expanded = self.forward(x)
            return expanded, empty

        if past is None:
            past = torch.zeros((batch_size, hidden_size, past_len), device=x.device, dtype=x.dtype)
        else:
            past = _prepare_input_embed_past(
                past,
                batch_size=batch_size,
                hidden_size=hidden_size,
                past_len=past_len,
                device=x.device,
                dtype=x.dtype,
            )

        if seq_len == 0:
            expanded = x.new_empty((batch_size, seq_len, hidden_size, self.value_channels))
            return expanded, past.contiguous()

        if not x.is_cuda:
            raise RuntimeError("TritonInputEmbedShortConvExpander requires CUDA tensors for Triton input shortconv.")

        x_for_conv = x
        current_mask: torch.Tensor | None = None
        active_ranks: torch.Tensor | None = None
        if token_mask is not None:
            x_for_conv, current_mask, active_ranks = compact_current_tokens(x, token_mask=token_mask)

        past_tokens = past.transpose(1, 2)
        x_cat = torch.cat([past_tokens, x_for_conv], dim=1).contiguous()
        expanded_full = cast(torch.Tensor, _TemporalShortConvExpandFn.apply(x_cat, self.weight))
        expanded = expanded_full[:, past_len:, :, :]
        if current_mask is not None and active_ranks is not None:
            expanded = gather_compacted_current_slots(
                expanded,
                current_mask=current_mask,
                active_ranks=active_ranks,
            )
        past_out = (
            compact_shortconv_history(
                x,
                history=past_tokens,
                token_mask=token_mask,
                history_size=past_len,
            )
            .transpose(1, 2)
            .contiguous()
        )
        return expanded, past_out


__all__ = ["InputEmbedShortConvExpander", "TritonInputEmbedShortConvExpander"]

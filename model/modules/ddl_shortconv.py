"""Shared DDL short-convolution compressors.

The DDL entrypoints keep their local ``ResidualShortConvCompressor`` names, but
delegate the common parameter layout, initialization, and fast paths here.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Self, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict

from ..DDL_utils import (
    _ShortConvFn,
    _TemporalShortConvReadFn,
    _TRITON_AVAILABLE,
    build_shortconv_kernel_index_and_mask,
)
from .activations import ActivationName, apply_activation
from .shortconv_cache import compact_current_tokens, compact_shortconv_history, gather_compacted_current_slots

CcShortConvImplementation = Literal["torch", "triton_optional", "triton_required"]
TemporalShortConvImplementation = Literal["torch", "triton_optional", "triton_required"]


def _has_configured_activation(activation: str | None) -> bool:
    return activation is not None and activation != "identity"


class _CcShortConvSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_size: int
    value_channels: int
    kernel_size: int
    causal: bool
    read_init: float

    @classmethod
    def from_config(cls, config: Any) -> _CcShortConvSpec:
        hidden_size = int(getattr(config, "hidden_size"))
        value_channels = int(getattr(config, "ddl_value_channels", 4))
        if value_channels <= 1:
            raise ValueError("ddl_value_channels must be > 1 for expanded-state DDL.")

        kernel_size = int(getattr(config, "ddl_state_shortconv_kernel_size", 4))
        if kernel_size <= 0:
            raise ValueError(f"ddl_state_shortconv_kernel_size must be positive, got {kernel_size}.")
        if kernel_size > value_channels:
            raise ValueError(
                f"ddl_state_shortconv_kernel_size must be <= ddl_value_channels ({value_channels}), got {kernel_size}."
            )

        read_init_raw = getattr(config, "ddl_state_read_init", None)
        read_init = 1.0 / float(value_channels) if read_init_raw is None else float(read_init_raw)
        return cls(
            hidden_size=hidden_size,
            value_channels=value_channels,
            kernel_size=kernel_size,
            causal=bool(getattr(config, "ddl_state_dv_shortconv_causal", False)),
            read_init=read_init,
        )


class _TemporalShortConvSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hidden_size: int
    value_channels: int
    kernel_size: int
    read_init: float

    @classmethod
    def from_config(cls, config: Any) -> _TemporalShortConvSpec:
        hidden_size = int(getattr(config, "hidden_size"))
        value_channels = int(getattr(config, "ddl_value_channels", 4))
        if value_channels <= 1:
            raise ValueError("ddl_value_channels must be > 1 for expanded-state DDL.")

        kernel_size = int(getattr(config, "ddl_state_shortconv_kernel_size", 4))
        if kernel_size <= 0:
            raise ValueError(f"ddl_state_shortconv_kernel_size must be positive, got {kernel_size}.")

        read_init_raw = getattr(config, "ddl_state_read_init", None)
        read_init = 1.0 / float(value_channels) if read_init_raw is None else float(read_init_raw)
        return cls(
            hidden_size=hidden_size,
            value_channels=value_channels,
            kernel_size=kernel_size,
            read_init=read_init,
        )


class _CcShortConvCompatView:
    def __init__(self, owner: CcResidualShortConvCompressor) -> None:
        self._owner = owner
        self.conv = self

    @property
    def weight(self) -> torch.Tensor:
        return self._owner.weight.unsqueeze(1)

    @property
    def kernel_size(self) -> int:
        return self._owner.kernel_size


class _TemporalShortConvCompatView:
    def __init__(self, owner: TemporalResidualShortConvCompressor) -> None:
        self._owner = owner
        self.shift_right1 = False
        self.activation: ActivationName | None = None
        self.conv = self

    @property
    def weight(self) -> torch.Tensor:
        return self._owner.weight.reshape(self._owner.residual_size, 1, self._owner.kernel_size)

    @property
    def kernel_size(self) -> int:
        return self._owner.kernel_size


class CcResidualShortConvCompressor(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        implementation: CcShortConvImplementation = "torch",
        module_name: str = "DDL",
    ) -> None:
        super().__init__()
        spec = _CcShortConvSpec.from_config(config)
        self.hidden_size = spec.hidden_size
        self.value_channels = spec.value_channels
        self.dv_shortconv_causal = spec.causal
        self.kernel_size = spec.kernel_size
        self.read_init = spec.read_init
        self._implementation = implementation
        self._module_name = module_name

        # This is a residual state readout, not a standalone feature extractor:
        # the neutral average starts as the legacy DDL aggregation, while
        # upstream projections and the learned read vector break channel
        # symmetry during training.
        self.weight = nn.Parameter(torch.full((self.hidden_size, self.kernel_size), 1.0 / float(self.kernel_size)))
        self.shortconv = _CcShortConvCompatView(self)
        kernel_index, valid_mask = build_shortconv_kernel_index_and_mask(
            value_channels=self.value_channels,
            kernel_size=self.kernel_size,
            causal=self.dv_shortconv_causal,
            device=torch.device("cpu"),
        )
        self.register_buffer("_kernel_index", kernel_index, persistent=False)
        self.register_buffer("_kernel_valid", valid_mask, persistent=False)
        self.read = nn.Parameter(torch.full((self.value_channels,), spec.read_init))

    def _reset_kernel_buffers(self) -> None:
        kernel_index, valid_mask = build_shortconv_kernel_index_and_mask(
            value_channels=self.value_channels,
            kernel_size=self.kernel_size,
            causal=self.dv_shortconv_causal,
            device=self.weight.device,
        )
        self.register_buffer("_kernel_index", kernel_index, persistent=False)
        self.register_buffer("_kernel_valid", valid_mask, persistent=False)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        was_meta = self.weight.device.type == "meta" or self.read.device.type == "meta"
        result = super()._apply(fn, recurse=recurse)
        if was_meta and self.weight.device.type != "meta":
            self.reset_deterministic_parameters()
        return result

    def reset_deterministic_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0 / float(self.kernel_size))
            self.read.fill_(self.read_init)
        self._reset_kernel_buffers()

    def _get_w_eff(self, *, batch_size: int = 1) -> torch.Tensor:
        kernel_offset_index = cast(torch.Tensor, self._kernel_index)
        kernel_valid = cast(torch.Tensor, self._kernel_valid)
        # `_kernel_index` stores clamped kernel offsets in [0, kernel_size),
        # with axes (readout_dv, input_dv). It is not indexing value channels.
        if batch_size > 1:
            # Preserve the sequence axis until read and kernel gradients have
            # been contracted; summing sequences first magnifies cancellation.
            weight = self.weight.float().unsqueeze(0).expand(batch_size, -1, -1)
            read = self.read.float().unsqueeze(0).expand(batch_size, -1)
            new_weight = weight[:, :, kernel_offset_index] * kernel_valid[None, None, :, :]
            return (new_weight * read[:, None, :, None]).sum(dim=2).to(self.weight.dtype).contiguous()
        new_weight = self.weight.float()[:, kernel_offset_index] * kernel_valid[None, :, :]
        # (hidden, readout_dv, input_dv) -> (hidden, input_dv).
        return (new_weight * self.read.float()[None, :, None]).sum(dim=1).to(self.weight.dtype).contiguous()

    def _should_use_triton(self, x: torch.Tensor) -> bool:
        if self._implementation == "torch":
            return False
        if not x.is_cuda:
            if self._implementation == "triton_required":
                raise RuntimeError(f"{self._module_name} requires CUDA tensors for Triton shortconv.")
            return False
        if not _TRITON_AVAILABLE:
            if self._implementation == "triton_required":
                raise RuntimeError(f"{self._module_name} requires Triton for accelerated shortconv.")
            return False
        return True

    def _run_triton_shortconv(self, x_flat: torch.Tensor, w_eff: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, _ShortConvFn.apply(x_flat.contiguous(), w_eff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected residual with shape (B, T, d, d_v), got {tuple(x.shape)}.")
        batch_size, seq_len, hidden_size, value_channels = (int(dim) for dim in x.shape)
        if hidden_size != self.hidden_size:
            raise ValueError(f"Expected residual d={self.hidden_size}, got {hidden_size}.")
        if value_channels != self.value_channels:
            raise ValueError(f"Expected residual d_v={self.value_channels}, got {value_channels}.")

        w_eff = self._get_w_eff(batch_size=batch_size)
        x_flat = x.reshape(batch_size * seq_len, hidden_size, value_channels)
        expected_shape = (batch_size, hidden_size, value_channels) if batch_size > 1 else (hidden_size, value_channels)
        if tuple(w_eff.shape) != expected_shape:
            raise RuntimeError(
                f"{self._module_name} effective CC shortconv weight must have shape "
                f"({hidden_size}, {value_channels}), got {tuple(w_eff.shape)}."
            )
        if self._should_use_triton(x):
            out = self._run_triton_shortconv(x_flat, w_eff)
            return out.reshape(batch_size, seq_len, hidden_size)
        run_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        # Explicit fp32 products/reductions also keep FP16 backward independent
        # of the batch-dependent GEMM reduction layout. Preserve AMP output dtype.
        with torch.autocast(x.device.type, enabled=False):
            weights = w_eff if batch_size > 1 else w_eff.unsqueeze(0)
            output = (x.to(run_dtype).float() * weights.to(run_dtype).float()[:, None]).sum(dim=-1)
        return output.to(run_dtype)


class TemporalResidualShortConvCompressor(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        implementation: TemporalShortConvImplementation = "torch",
        module_name: str = "DDL",
    ) -> None:
        super().__init__()
        spec = _TemporalShortConvSpec.from_config(config)
        self.hidden_size = spec.hidden_size
        self.value_channels = spec.value_channels
        self.residual_size = self.hidden_size * self.value_channels
        self.kernel_size = spec.kernel_size
        self.read_init = spec.read_init
        self._implementation = implementation
        self._module_name = module_name

        # Direct construction and meta materialization use a neutral average
        # readout. Full DDL models preserve this state while upstream projections
        # provide channel-specific symmetry breaking.
        self.weight = nn.Parameter(
            torch.full(
                (self.hidden_size, self.value_channels, self.kernel_size),
                1.0 / float(self.kernel_size),
            )
        )
        self.shortconv = _TemporalShortConvCompatView(self)
        self.read = nn.Parameter(torch.full((self.value_channels,), spec.read_init))

    def reset_deterministic_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0 / float(self.kernel_size))
            self.read.fill_(self.read_init)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        was_meta = self.weight.device.type == "meta" or self.read.device.type == "meta"
        result = super()._apply(fn, recurse=recurse)
        if was_meta and self.weight.device.type != "meta":
            self.reset_deterministic_parameters()
        return result

    def _get_w_eff(self, *, batch_size: int = 1) -> torch.Tensor:
        # With a configured activation this is only the linear folded readout
        # for inspection/debugging. Runtime paths avoid this folded weight when
        # the activation must run before the learned readout.
        if batch_size > 1:
            weight = self.weight.float().unsqueeze(0).expand(batch_size, -1, -1, -1)
            read = self.read.float().unsqueeze(0).expand(batch_size, -1)
            return (weight * read[:, None, :, None]).to(self.weight.dtype).contiguous()
        return (self.weight.float() * self.read.float()[None, :, None]).to(self.weight.dtype).contiguous()

    def _should_use_triton(self, x: torch.Tensor) -> bool:
        if self._implementation == "torch":
            return False
        if _has_configured_activation(self.shortconv.activation):
            if self._implementation == "triton_required":
                raise RuntimeError(f"{self._module_name} Triton temporal shortconv does not support activations.")
            return False
        if not x.is_cuda:
            if self._implementation == "triton_required":
                raise RuntimeError(f"{self._module_name} requires CUDA tensors for Triton temporal shortconv.")
            return False
        if not _TRITON_AVAILABLE:
            if self._implementation == "triton_required":
                raise RuntimeError(f"{self._module_name} requires Triton for accelerated temporal shortconv.")
            return False
        return True

    def _validate_x(self, x: torch.Tensor) -> tuple[int, int, int, int]:
        if x.ndim != 4:
            raise ValueError(f"Expected residual with shape (B, T, d, d_v), got {tuple(x.shape)}.")
        batch_size, seq_len, hidden_size, value_channels = (int(dim) for dim in x.shape)
        if hidden_size != self.hidden_size:
            raise ValueError(f"Expected residual d={self.hidden_size}, got {hidden_size}.")
        if value_channels != self.value_channels:
            raise ValueError(f"Expected residual d_v={self.value_channels}, got {value_channels}.")
        return batch_size, seq_len, hidden_size, value_channels

    def _folded_torch_conv1d(self, x_channels: torch.Tensor) -> torch.Tensor:
        batch_size, channels, padded_length = x_channels.shape
        if batch_size > 1:
            weight = self._get_w_eff(batch_size=batch_size)
            output = F.conv1d(
                x_channels.reshape(1, batch_size * channels, padded_length),
                weight.reshape(batch_size * self.hidden_size, self.value_channels, self.kernel_size),
                groups=batch_size * self.hidden_size,
            )
            return output.reshape(batch_size, self.hidden_size, output.shape[-1])
        return F.conv1d(
            x_channels,
            self._get_w_eff(),
            bias=None,
            stride=1,
            padding=0,
            groups=self.hidden_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size, value_channels = self._validate_x(x)
        del value_channels
        if seq_len == 0:
            return x.new_empty(batch_size, seq_len, hidden_size)
        if self._should_use_triton(x):
            # The Triton kernel applies the same virtual left padding as the
            # Torch `F.pad(..., (kernel_size - 1, 0))` path by bounds-checking
            # temporal loads against the unpadded sequence length.
            return cast(
                torch.Tensor, _TemporalShortConvReadFn.apply(x.contiguous(), self._get_w_eff(batch_size=batch_size))
            )

        x_flat = x.reshape(batch_size, seq_len, self.residual_size).transpose(1, 2)
        x_flat = F.pad(x_flat, (self.kernel_size - 1, 0)).contiguous()
        folds_read = not _has_configured_activation(self.shortconv.activation)
        if folds_read:
            y = self._folded_torch_conv1d(x_flat)
            return y.transpose(1, 2).contiguous()

        weight = self.weight
        weight = weight.reshape(self.residual_size, 1, self.kernel_size)
        y = F.conv1d(x_flat, weight, bias=None, stride=1, padding=0, groups=self.residual_size)
        y_bt = y.transpose(1, 2).contiguous()
        y_bt = apply_activation(y_bt, self.shortconv.activation)
        x_conv = y_bt.reshape(batch_size, seq_len, hidden_size, self.value_channels)
        if not folds_read:
            x_conv = x_conv * self.read[None, None, None, :]
        return torch.sum(x_conv, dim=-1)

    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, hidden_size, value_channels = self._validate_x(x)
        del value_channels
        if self.kernel_size <= 1:
            return self.forward(x), None
        if bool(getattr(self.shortconv, "shift_right1", False)):
            raise NotImplementedError("shift_right1 shortconv is not supported in DDL kv-cache mode.")

        past_len = self.kernel_size - 1
        if past is None:
            past = torch.zeros((batch_size, self.residual_size, past_len), device=x.device, dtype=x.dtype)
        if (
            past.ndim != 3
            or int(past.shape[0]) != batch_size
            or int(past.shape[1]) != self.residual_size
            or int(past.shape[2]) != past_len
        ):
            raise ValueError(f"Expected past with shape (B, C, {past_len}), got {tuple(past.shape)}")
        if seq_len == 0:
            past_out = past.to(device=x.device, dtype=x.dtype).contiguous()
            return x.new_empty(batch_size, seq_len, hidden_size), past_out

        x_flat = x.reshape(batch_size, seq_len, self.residual_size)
        x_for_conv = x_flat
        current_mask: torch.Tensor | None = None
        active_ranks: torch.Tensor | None = None
        if token_mask is not None:
            x_for_conv, current_mask, active_ranks = compact_current_tokens(x_flat, token_mask=token_mask)

        if self._should_use_triton(x):
            past_tokens = (
                past.to(device=x.device, dtype=x.dtype)
                .transpose(1, 2)
                .reshape(
                    batch_size,
                    past_len,
                    hidden_size,
                    self.value_channels,
                )
            )
            x_for_conv_4d = x_for_conv.reshape(batch_size, seq_len, hidden_size, self.value_channels)
            x_cat = torch.cat([past_tokens, x_for_conv_4d], dim=1).contiguous()
            out_full = cast(torch.Tensor, _TemporalShortConvReadFn.apply(x_cat, self._get_w_eff(batch_size=batch_size)))
            out = out_full[:, past_len:, :]
            if current_mask is not None and active_ranks is not None:
                out = gather_compacted_current_slots(
                    out,
                    current_mask=current_mask,
                    active_ranks=active_ranks,
                )
            past_out_tokens = compact_shortconv_history(
                x_flat,
                history=past.to(device=x.device, dtype=x.dtype).transpose(1, 2),
                token_mask=token_mask,
                history_size=past_len,
            )
            past_out = past_out_tokens.transpose(1, 2).contiguous()
            return out, past_out

        x_t = x_for_conv.transpose(1, 2).contiguous()
        x_cat = torch.cat([past.to(device=x.device, dtype=x.dtype), x_t], dim=-1)
        folds_read = not _has_configured_activation(self.shortconv.activation)
        if folds_read:
            y = self._folded_torch_conv1d(x_cat)
            out = y.transpose(1, 2).contiguous()
            if current_mask is not None and active_ranks is not None:
                out = gather_compacted_current_slots(
                    out,
                    current_mask=current_mask,
                    active_ranks=active_ranks,
                )
            past_out = (
                compact_shortconv_history(
                    x_flat,
                    history=past.to(device=x.device, dtype=x.dtype).transpose(1, 2),
                    token_mask=token_mask,
                    history_size=past_len,
                )
                .transpose(1, 2)
                .contiguous()
            )
            return out, past_out

        weight = self.weight
        weight = weight.reshape(self.residual_size, 1, self.kernel_size)
        y = F.conv1d(x_cat, weight, bias=None, stride=1, padding=0, groups=self.residual_size)
        y_bt = y.transpose(1, 2).contiguous()
        y_bt = apply_activation(y_bt, self.shortconv.activation)
        y = y_bt.reshape(batch_size, seq_len, hidden_size, self.value_channels)
        if not folds_read:
            y = y * self.read[None, None, None, :]
        out = torch.sum(y, dim=-1)
        if current_mask is not None and active_ranks is not None:
            out = gather_compacted_current_slots(
                out,
                current_mask=current_mask,
                active_ranks=active_ranks,
            )
        past_out = (
            compact_shortconv_history(
                x_flat,
                history=past.to(device=x.device, dtype=x.dtype).transpose(1, 2),
                token_mask=token_mask,
                history_size=past_len,
            )
            .transpose(1, 2)
            .contiguous()
        )
        return out, past_out


__all__ = [
    "CcResidualShortConvCompressor",
    "CcShortConvImplementation",
    "TemporalResidualShortConvCompressor",
    "TemporalShortConvImplementation",
]

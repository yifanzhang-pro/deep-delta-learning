# pyright: reportMissingImports=false, reportInvalidTypeForm=false
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, cast

import torch
import torch.nn as nn

from .kv_cache import is_cache_len_tensor
from .kv_shift_torch import token_shift_torch
from ..utils.triton_import_utils import import_triton_modules

if TYPE_CHECKING:
    import triton
    import triton.language as tl

_triton_module, _tl_module, _TRITON_AVAILABLE = import_triton_modules()
if _TRITON_AVAILABLE and _triton_module is not None and _tl_module is not None:
    triton = cast(Any, _triton_module)
    tl = cast(Any, _tl_module)


def maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous() if x.stride(-1) != 1 else x


if _TRITON_AVAILABLE:

    @triton.jit
    def shift_fwd_kernel(
        X_PTR,
        PREV_WEIGHT_PTR,
        CURR_WEIGHT_PTR,
        OUT_PTR,
        stride_x_b,
        stride_x_t,
        stride_x_h,
        stride_x_d,
        stride_weight_b,
        stride_weight_t,
        stride_weight_h,
        T: tl.constexpr,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        b_offset = tl.program_id(axis=0).to(tl.int64)
        t_offset = tl.program_id(axis=1).to(tl.int64) * BLOCK_T
        h_offset = tl.program_id(axis=2).to(tl.int64)

        x_ptr_offset = b_offset * stride_x_b + t_offset * stride_x_t + h_offset * stride_x_h
        X_PTR += x_ptr_offset
        OUT_PTR += x_ptr_offset

        weight_ptr_offset = b_offset * stride_weight_b + t_offset * stride_weight_t + h_offset * stride_weight_h
        CURR_WEIGHT_PTR += weight_ptr_offset
        PREV_WEIGHT_PTR += weight_ptr_offset

        x_ptr = X_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
        t_offset_block = t_offset + tl.arange(0, BLOCK_T)[:, None]
        x_mask = t_offset_block < T

        x_prev_ptr = x_ptr - stride_x_t
        t_prev_offset_block = t_offset_block - 1
        x_prev_mask = (t_prev_offset_block < T) & (t_prev_offset_block >= 0)

        curr_weight_ptr = CURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
        prev_weight_ptr = PREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t

        x = tl.load(x_ptr, mask=x_mask, other=0.0)
        x_prev = tl.load(x_prev_ptr, mask=x_prev_mask, other=0.0)
        curr_weight = tl.load(curr_weight_ptr, mask=x_mask, other=0.0)
        prev_weight = tl.load(prev_weight_ptr, mask=x_mask, other=0.0)

        result = x * curr_weight.to(tl.float32) + x_prev * prev_weight.to(tl.float32)
        result = result.to(x.dtype)

        out_ptr = OUT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
        tl.store(out_ptr, result, mask=x_mask)

    @triton.jit
    def shift_bwd_kernel(
        X_PTR,
        PREV_WEIGHT_PTR,
        CURR_WEIGHT_PTR,
        DOUT_PTR,
        DX_PTR,
        DPREV_WEIGHT_PTR,
        DCURR_WEIGHT_PTR,
        stride_x_b,
        stride_x_t,
        stride_x_h,
        stride_x_d,
        stride_weight_b,
        stride_weight_t,
        stride_weight_h,
        T: tl.constexpr,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        b_offset = tl.program_id(axis=0).to(tl.int64)
        t_offset = tl.program_id(axis=1).to(tl.int64) * BLOCK_T
        h_offset = tl.program_id(axis=2).to(tl.int64)

        x_ptr_offset = b_offset * stride_x_b + t_offset * stride_x_t + h_offset * stride_x_h
        X_PTR += x_ptr_offset
        DX_PTR += x_ptr_offset
        DOUT_PTR += x_ptr_offset

        weight_ptr_offset = b_offset * stride_weight_b + t_offset * stride_weight_t + h_offset * stride_weight_h
        CURR_WEIGHT_PTR += weight_ptr_offset
        PREV_WEIGHT_PTR += weight_ptr_offset
        DCURR_WEIGHT_PTR += weight_ptr_offset
        DPREV_WEIGHT_PTR += weight_ptr_offset

        x_ptr = X_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
        t_offset_block = t_offset + tl.arange(0, BLOCK_T)[:, None]
        x_mask = t_offset_block < T

        dout_ptr = DOUT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
        dout_next_ptr = dout_ptr + stride_x_t
        t_next_offset_block = t_offset_block + 1
        x_next_mask = t_next_offset_block < T

        x_prev_ptr = x_ptr - stride_x_t
        t_prev_offset_block = t_offset_block - 1
        x_prev_mask = (t_prev_offset_block < T) & (t_prev_offset_block >= 0)

        curr_weight_ptr = CURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
        prev_weight_ptr = PREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
        next_prev_weight_ptr = prev_weight_ptr + stride_weight_t

        x = tl.load(x_ptr, mask=x_mask, other=0.0)
        x_prev = tl.load(x_prev_ptr, mask=x_prev_mask, other=0.0)
        dout = tl.load(dout_ptr, mask=x_mask, other=0.0)
        dout_next = tl.load(dout_next_ptr, mask=x_next_mask, other=0.0)

        curr_weight = tl.load(curr_weight_ptr, mask=x_mask, other=0.0)
        next_prev_weight = tl.load(next_prev_weight_ptr, mask=x_next_mask, other=0.0)

        dx = dout * curr_weight.to(tl.float32) + dout_next * next_prev_weight.to(tl.float32)
        dx = dx.to(x.dtype)

        dcurr_weight = tl.sum(dout.to(tl.float32) * x, axis=1, keep_dims=True)
        dprev_weight = tl.sum(dout.to(tl.float32) * x_prev, axis=1, keep_dims=True)

        dx_ptr = DX_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
        tl.store(dx_ptr, dx, mask=x_mask)

        dcurr_weight_ptr = DCURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
        tl.store(dcurr_weight_ptr, dcurr_weight, mask=x_mask)

        dprev_weight_ptr = DPREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_weight_t
        tl.store(dprev_weight_ptr, dprev_weight, mask=x_mask)

    class TokenShift(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, prev_weight: torch.Tensor, curr_weight: torch.Tensor):
            if not x.is_cuda:
                raise RuntimeError("TokenShift (triton) requires CUDA tensors.")

            batch_size, seq_len, num_heads, head_dim = x.size()
            if head_dim not in {16, 32, 64, 128}:
                raise ValueError("TokenShift head_dim must be one of {16, 32, 64, 128}.")
            if prev_weight.size() != (batch_size, seq_len, num_heads) or curr_weight.size() != (
                batch_size,
                seq_len,
                num_heads,
            ):
                raise ValueError("prev_weight/curr_weight must have shape (B, T, H).")
            if prev_weight.stride() != curr_weight.stride():
                raise ValueError("prev_weight and curr_weight must have identical strides.")

            x = maybe_contiguous(x)
            out = torch.empty_like(x)

            block_t = triton.next_power_of_2(min(64, seq_len))

            def grid(meta):
                return (batch_size, triton.cdiv(seq_len, meta["BLOCK_T"]), num_heads)

            fwd_kernel = cast(Any, shift_fwd_kernel)
            fwd_kernel[grid](
                x,
                prev_weight,
                curr_weight,
                out,
                *x.stride(),
                *curr_weight.stride(),
                T=seq_len,
                D=head_dim,
                BLOCK_T=block_t,
            )
            ctx.save_for_backward(x, prev_weight, curr_weight)
            return out

        @staticmethod
        def backward(ctx, dout: torch.Tensor):
            x, prev_weight, curr_weight = ctx.saved_tensors
            batch_size, seq_len, num_heads, head_dim = x.size()
            if head_dim not in {16, 32, 64, 128}:
                raise ValueError("TokenShift head_dim must be one of {16, 32, 64, 128}.")

            x = maybe_contiguous(x)
            if dout.stride() != x.stride():
                dout = dout.contiguous()

            dx = torch.empty_like(x)
            dcurr_weight = torch.empty_like(curr_weight)
            dprev_weight = torch.empty_like(prev_weight)

            block_t = triton.next_power_of_2(min(64, seq_len))

            def grid(meta):
                return (batch_size, triton.cdiv(seq_len, meta["BLOCK_T"]), num_heads)

            bwd_kernel = cast(Any, shift_bwd_kernel)
            bwd_kernel[grid](
                x,
                prev_weight,
                curr_weight,
                dout,
                dx,
                dprev_weight,
                dcurr_weight,
                *x.stride(),
                *curr_weight.stride(),
                T=seq_len,
                D=head_dim,
                BLOCK_T=block_t,
            )
            return dx, dprev_weight, dcurr_weight


def token_shift(x: torch.Tensor, prev_weight: torch.Tensor, curr_weight: torch.Tensor) -> torch.Tensor:
    if _TRITON_AVAILABLE and x.is_cuda and x.size(-1) in {16, 32, 64, 128}:
        return cast(torch.Tensor, TokenShift.apply(x, prev_weight, curr_weight))
    return token_shift_torch(x, prev_weight, curr_weight)


class ShiftLinear(nn.Module):
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
        result, next_shift_state = self.forward_with_shift_state(x, shift_state)
        if shift_state is not None:
            shift_state.copy_(next_shift_state)
        return result

    def forward_with_shift_state(
        self,
        x: torch.Tensor,
        shift_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"expected x of shape (B, T, D), got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.size()
        out = self.linear(x)

        alpha = torch.sigmoid(self.shift_proj(x).float()).float()
        out_per_head = out.view(batch_size, seq_len, self.num_heads, -1)

        if seq_len > 1:
            prev_weight = alpha
            curr_weight = 1.0 - alpha
            if shift_state is None:
                result_per_head = token_shift(out_per_head, prev_weight, curr_weight)
            else:
                result_per_head = token_shift_torch(
                    out_per_head,
                    prev_weight,
                    curr_weight,
                    initial_state=shift_state,
                )
        else:
            if shift_state is None:
                result_per_head = out_per_head
            else:
                shift_state_per_head = shift_state.view(batch_size, 1, self.num_heads, -1)
                result_per_head = (
                    alpha.unsqueeze(-1) * shift_state_per_head + (1.0 - alpha).unsqueeze(-1) * out_per_head
                )

        result_per_head = result_per_head.to(out.dtype)
        return result_per_head.reshape(batch_size, seq_len, self.output_dim), out[:, -1, :]


def _shape_matches(tensor: torch.Tensor, expected_shape: tuple[int, int]) -> bool:
    return tuple(int(dim) for dim in tensor.shape) == expected_shape


def get_kv_shift_states(
    past_key_value: Sequence[torch.Tensor] | None,
    *,
    use_k_shift: bool,
    use_v_shift: bool,
    k_state_shape: tuple[int, int],
    v_state_shape: tuple[int, int],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if past_key_value is None or (not use_k_shift and not use_v_shift):
        return None, None

    batch_size = int(k_state_shape[0])
    end = len(past_key_value)
    if end > 0 and is_cache_len_tensor(past_key_value[-1], batch_size=batch_size):
        end -= 1

    cursor = end
    v_shift_state: torch.Tensor | None = None
    if use_v_shift:
        cursor -= 1
        if cursor < 2 or not _shape_matches(past_key_value[cursor], v_state_shape):
            raise ValueError("past_key_value is missing the cached V shift state required by use_v_shift=True.")
        v_shift_state = past_key_value[cursor]

    k_shift_state: torch.Tensor | None = None
    if use_k_shift:
        cursor -= 1
        if cursor < 2 or not _shape_matches(past_key_value[cursor], k_state_shape):
            raise ValueError("past_key_value is missing the cached K shift state required by use_k_shift=True.")
        k_shift_state = past_key_value[cursor]

    return k_shift_state, v_shift_state


def append_kv_shift_states(
    items: list[torch.Tensor],
    *,
    k_shift_state: torch.Tensor | None,
    v_shift_state: torch.Tensor | None,
    cache_len: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if k_shift_state is not None:
        items.append(k_shift_state)
    if v_shift_state is not None:
        items.append(v_shift_state)
    items.append(cache_len)
    return tuple(items)


def project_with_optional_shift_state(
    projection: nn.Module,
    x: torch.Tensor,
    *,
    use_shift: bool,
    shift_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not use_shift:
        return projection(x), None
    shift_projection = cast(ShiftLinear, projection)
    return shift_projection.forward_with_shift_state(x, shift_state)


def split_kv_cache_prefix_states(
    past_key_value: Sequence[torch.Tensor] | None,
    *,
    prefix_state_count: int,
    batch_size: int,
) -> tuple[tuple[torch.Tensor, ...] | None, tuple[torch.Tensor | None, ...]]:
    if prefix_state_count < 0:
        raise ValueError(f"prefix_state_count must be >= 0, got {prefix_state_count}.")
    empty_states = tuple(None for _ in range(prefix_state_count))
    if past_key_value is None:
        return None, empty_states

    end = len(past_key_value)
    cache_len: torch.Tensor | None = None
    if end > 0 and is_cache_len_tensor(past_key_value[-1], batch_size=batch_size):
        cache_len = past_key_value[-1]
        end -= 1

    if end < 2 + prefix_state_count:
        return tuple(past_key_value), empty_states

    prefix_states = tuple(past_key_value[2 : 2 + prefix_state_count])
    if any(state.ndim < 3 or int(state.shape[0]) != batch_size for state in prefix_states):
        return tuple(past_key_value), empty_states

    attention_items = [past_key_value[0], past_key_value[1]]
    attention_items.extend(past_key_value[2 + prefix_state_count : end])
    if cache_len is not None:
        attention_items.append(cache_len)
    return tuple(attention_items), prefix_states


def insert_kv_cache_prefix_states(
    attention_present: Sequence[torch.Tensor],
    prefix_states: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if len(attention_present) < 2:
        raise ValueError("attention_present must contain at least key and value tensors.")
    return (
        attention_present[0],
        attention_present[1],
        *prefix_states,
        *attention_present[2:],
    )


def find_kv_cache_state_by_shape(
    past_key_value: Sequence[torch.Tensor] | None,
    *,
    expected_shape: tuple[int, ...],
    batch_size: int,
) -> torch.Tensor | None:
    if past_key_value is None:
        return None
    end = len(past_key_value)
    if end > 0 and is_cache_len_tensor(past_key_value[-1], batch_size=batch_size):
        end -= 1
    for tensor in reversed(past_key_value[2:end]):
        if tuple(int(dim) for dim in tensor.shape) == expected_shape:
            return tensor
    return None


def get_kv_cache_trailing_state_by_shape(
    past_key_value: Sequence[torch.Tensor] | None,
    *,
    expected_shape: tuple[int, ...],
    batch_size: int,
    prefix_state_count: int = 0,
    skip_trailing_states: int = 0,
) -> torch.Tensor | None:
    if prefix_state_count < 0:
        raise ValueError(f"prefix_state_count must be >= 0, got {prefix_state_count}.")
    if skip_trailing_states < 0:
        raise ValueError(f"skip_trailing_states must be >= 0, got {skip_trailing_states}.")
    if past_key_value is None:
        return None
    end = len(past_key_value)
    if end > 0 and is_cache_len_tensor(past_key_value[-1], batch_size=batch_size):
        end -= 1
    index = end - 1 - skip_trailing_states
    if index < 2 + prefix_state_count:
        return None
    tensor = past_key_value[index]
    if tuple(int(dim) for dim in tensor.shape) != expected_shape:
        return None
    return tensor


def count_kv_cache_trailing_states(
    past_key_value: Sequence[torch.Tensor] | None,
    *,
    batch_size: int,
    prefix_state_count: int = 0,
) -> int:
    if prefix_state_count < 0:
        raise ValueError(f"prefix_state_count must be >= 0, got {prefix_state_count}.")
    if past_key_value is None:
        return 0
    end = len(past_key_value)
    if end > 0 and is_cache_len_tensor(past_key_value[-1], batch_size=batch_size):
        end -= 1
    return max(0, end - 2 - prefix_state_count)


def keep_only_kv_shift_cache_states(
    past_key_value: Sequence[torch.Tensor] | None,
    *,
    use_k_shift: bool,
    use_v_shift: bool,
    k_state_shape: tuple[int, int],
    v_state_shape: tuple[int, int],
) -> tuple[torch.Tensor, ...] | None:
    if past_key_value is None:
        return None
    if len(past_key_value) < 2:
        return tuple(past_key_value)

    batch_size = int(k_state_shape[0])
    end = len(past_key_value)
    cache_len: torch.Tensor | None = None
    if end > 0 and is_cache_len_tensor(past_key_value[-1], batch_size=batch_size):
        cache_len = past_key_value[-1]
        end -= 1

    middle = list(past_key_value[2:end])

    def pop_rightmost_shape(expected_shape: tuple[int, int]) -> torch.Tensor | None:
        for index in range(len(middle) - 1, -1, -1):
            if _shape_matches(middle[index], expected_shape):
                return middle.pop(index)
        return None

    v_shift_state = pop_rightmost_shape(v_state_shape) if use_v_shift else None
    k_shift_state = pop_rightmost_shape(k_state_shape) if use_k_shift else None

    result = [past_key_value[0], past_key_value[1]]
    if k_shift_state is not None:
        result.append(k_shift_state)
    if v_shift_state is not None:
        result.append(v_shift_state)
    if cache_len is not None:
        result.append(cache_len)
    return tuple(result)


__all__ = [
    "ShiftLinear",
    "get_kv_shift_states",
    "append_kv_shift_states",
    "project_with_optional_shift_state",
    "find_kv_cache_state_by_shape",
    "get_kv_cache_trailing_state_by_shape",
    "count_kv_cache_trailing_states",
    "keep_only_kv_shift_cache_states",
    "split_kv_cache_prefix_states",
    "insert_kv_cache_prefix_states",
]

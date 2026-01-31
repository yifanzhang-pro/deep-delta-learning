from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .kv_shift_torch import token_shift_torch

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    _TRITON_AVAILABLE = False
else:
    _TRITON_AVAILABLE = True


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

            shift_fwd_kernel[grid](
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

            shift_bwd_kernel[grid](
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
        return TokenShift.apply(x, prev_weight, curr_weight)
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
        if x.ndim != 3:
            raise ValueError(f"expected x of shape (B, T, D), got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.size()
        out = self.linear(x)

        alpha = torch.sigmoid(self.shift_proj(x).float()).float()
        out_per_head = out.view(batch_size, seq_len, self.num_heads, -1)

        if seq_len > 1:
            prev_weight = alpha
            curr_weight = 1.0 - alpha
            result_per_head = token_shift(out_per_head, prev_weight, curr_weight)
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

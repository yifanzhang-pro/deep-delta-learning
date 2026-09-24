"""Batched projections and native CUDA varlen attention for rotary models."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.varlen import varlen_attn

from .activations import apply_activation
from .rotary import Rotary, apply_rotary_emb
from .segmented_attention import run_equal_length_sequences


def native_packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """Use the configured Torch varlen provider without expanding KV heads."""
    return cast(
        torch.Tensor,
        varlen_attn(
            query=q,
            key=k,
            value=v,
            cu_seq_q=cu_seqlens,
            cu_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
            window_size=(-1, 0),
            enable_gqa=q.shape[1] != k.shape[1],
        ),
    )


def packed_rotary_attention(
    x: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    boundaries: tuple[tuple[int, int], ...],
    q_proj: nn.Module,
    k_proj: nn.Module,
    v_proj: nn.Module,
    o_proj: nn.Module,
    rotary: Rotary,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_activation: str | None,
    k_activation: str | None,
    v_activation: str | None,
    q_norm: nn.Module | None,
    k_norm: nn.Module | None,
    norm_before_rotary: bool,
    output_norm: nn.Module | None,
    gate_proj: nn.Module | None,
) -> torch.Tensor:
    """Project each token once and preserve each model's norm/RoPE ordering.

    CUDA fp16/bf16 uses native varlen attention. CPU, MPS and fp32 retain
    scaled-dot-product attention, batched by exact sequence length. Kernel
    failures propagate; backend selection never retries a different path.
    """
    batch_size, length, _ = x.shape
    total_tokens = batch_size * length
    q = apply_activation(q_proj(x).reshape(1, total_tokens, num_heads, head_dim), q_activation)
    k = apply_activation(k_proj(x).reshape(1, total_tokens, num_kv_heads, head_dim), k_activation)
    v = apply_activation(v_proj(x).reshape(total_tokens, num_kv_heads, head_dim), v_activation)
    if norm_before_rotary:
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)
    starts = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens.diff(), output_size=total_tokens)
    positions = (torch.arange(total_tokens, device=x.device) - starts).view(1, total_tokens)
    cos, sin = rotary(q, position_ids=positions)
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    if not norm_before_rotary:
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)
    q = q.reshape(total_tokens, num_heads, head_dim)
    k = k.reshape(total_tokens, num_kv_heads, head_dim)
    if q.device.type == "cuda" and q.dtype in (torch.float16, torch.bfloat16):
        y = native_packed_attention(
            q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max(end - start for start, end in boundaries)
        )
    else:
        q_width = num_heads * head_dim
        kv_width = num_kv_heads * head_dim
        projected = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1).unsqueeze(0)

        def attend(group: torch.Tensor) -> torch.Tensor:
            group_size, group_length, _ = group.shape
            q_group, k_group, v_group = group.split((q_width, kv_width, kv_width), dim=-1)
            q_group = q_group.reshape(group_size, group_length, num_heads, head_dim).transpose(1, 2)
            k_group = k_group.reshape(group_size, group_length, num_kv_heads, head_dim).transpose(1, 2)
            v_group = v_group.reshape(group_size, group_length, num_kv_heads, head_dim).transpose(1, 2)
            enable_gqa = num_heads != num_kv_heads
            if enable_gqa and group.device.type == "mps":
                k_group = k_group.repeat_interleave(num_heads // num_kv_heads, dim=1)
                v_group = v_group.repeat_interleave(num_heads // num_kv_heads, dim=1)
                enable_gqa = False
            output = F.scaled_dot_product_attention(q_group, k_group, v_group, is_causal=True, enable_gqa=enable_gqa)
            return output.transpose(1, 2).reshape(group_size, group_length, num_heads, head_dim)

        y = run_equal_length_sequences(attend, projected, boundaries).reshape(total_tokens, num_heads, head_dim)
    if output_norm is not None:
        y = output_norm(y)
    if gate_proj is not None:
        y = y * F.silu(gate_proj(x).reshape(total_tokens, num_heads, head_dim))
    return cast(torch.Tensor, o_proj(y.reshape(batch_size, length, num_heads * head_dim)))

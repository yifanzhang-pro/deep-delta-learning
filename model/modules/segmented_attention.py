"""Explicit packed execution for attention with a fixed-sequence backend."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import cast

import torch
from torch import nn


def packed_sequence_boundaries(
    x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int | None
) -> tuple[tuple[int, int], ...]:
    """Validate offsets over the flattened batch and sequence dimensions.

    Reading offsets synchronizes the device once per segmented invocation.
    Native varlen kernels do not use this execution path.
    """
    validator = cast(
        Callable[[torch.Tensor, torch.Tensor, int | None], tuple[tuple[int, int], ...]],
        _packed_sequence_boundaries_eager,
    )
    return validator(x, cu_seqlens, max_seqlen)


@torch.compiler.disable
def _packed_sequence_boundaries_eager(
    x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int | None
) -> tuple[tuple[int, int], ...]:
    if x.ndim < 2:
        raise ValueError("Packed inputs require batch and sequence dimensions.")
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be rank 1 with at least two offsets.")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must use torch.int32.")
    if cu_seqlens.device != x.device or x.device.type == "meta":
        raise ValueError("cu_seqlens and inputs must use the same non-meta device.")
    offsets = [int(value) for value in cu_seqlens.tolist()]
    if offsets[0] != 0 or offsets[-1] != x.shape[0] * x.shape[1]:
        raise ValueError("cu_seqlens must start at 0 and end at the flattened token count.")
    boundaries = tuple(zip(offsets[:-1], offsets[1:], strict=True))
    if any(end <= start for start, end in boundaries):
        raise ValueError("cu_seqlens offsets must be strictly increasing.")
    if max_seqlen is not None and max_seqlen != max(end - start for start, end in boundaries):
        raise ValueError("max_seqlen must equal the longest packed sequence length.")
    return boundaries


class SegmentedSelfAttention(nn.Module):
    """Reset sequence-local computation while retaining the original backend.

    Models explicitly inherit this class when their backend accepts one
    rectangular sequence batch. Parameters and state-dict keys are unchanged;
    autograd, including Triton backward kernels, runs through each segment.
    """

    @abstractmethod
    def _forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        _packed_boundaries: tuple[tuple[int, int], ...] | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is None:
            if max_seqlen is not None:
                raise ValueError("max_seqlen requires cu_seqlens.")
            return self._forward_sequence(x)
        if x.ndim != 3:
            raise ValueError("Packed attention expects inputs with shape (B, T, C).")
        boundaries = (
            packed_sequence_boundaries(x, cu_seqlens, max_seqlen) if _packed_boundaries is None else _packed_boundaries
        )
        return self._forward_packed_validated(x, cu_seqlens=cu_seqlens, boundaries=boundaries)

    def _forward_packed_validated(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        boundaries: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        return run_equal_length_sequences(self._forward_sequence, x, boundaries)


def run_equal_length_sequences(
    forward: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    boundaries: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Batch equal-length documents without padding or changing their backend.

    Different lengths retain independent execution. Group order is sorted and
    outputs return to original token order with a single differentiable scatter.
    No cached metadata outlives this forward, so mutable offsets cannot go stale.
    """
    flattened = x.flatten(0, 1)
    groups: dict[int, list[int]] = {}
    for start, end in boundaries:
        groups.setdefault(end - start, []).append(start)
    if len(groups) == len(boundaries):
        outputs = [forward(flattened[start:end].unsqueeze(0)).squeeze(0) for start, end in boundaries]
        combined = torch.cat(outputs, dim=0)
    else:
        outputs: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        for length, starts in sorted(groups.items()):
            token_indices = (
                torch.tensor(starts, device=x.device, dtype=torch.long)[:, None]
                + torch.arange(length, device=x.device)[None, :]
            ).flatten()
            inputs = flattened.index_select(0, token_indices).reshape(len(starts), length, *x.shape[2:])
            outputs.append(forward(inputs).flatten(0, 1))
            indices.append(token_indices)
        grouped_output = torch.cat(outputs, dim=0)
        combined = torch.empty_like(grouped_output).index_copy(0, torch.cat(indices), grouped_output)
    return combined.reshape(*x.shape[:2], *combined.shape[1:])

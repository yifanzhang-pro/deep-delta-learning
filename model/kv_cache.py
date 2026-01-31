from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import torch


_INT_DTYPES: Final[tuple[torch.dtype, ...]] = (torch.int32, torch.int64)


def is_cache_len_tensor(tensor: torch.Tensor, *, batch_size: int) -> bool:
    if tensor.dtype not in _INT_DTYPES:
        return False
    if tensor.ndim != 1:
        return False
    if int(tensor.shape[0]) != batch_size:
        return False
    return True


def maybe_get_cache_len(past_key_value: Sequence[torch.Tensor] | None, *, batch_size: int) -> torch.Tensor | None:
    if past_key_value is None or len(past_key_value) < 3:
        return None
    candidate = past_key_value[-1]
    if is_cache_len_tensor(candidate, batch_size=batch_size):
        return candidate
    return None


def read_cache_len(*, cache_len: torch.Tensor) -> int:
    if cache_len.ndim != 1:
        raise ValueError(f"cache_len must be a 1D tensor, got shape={tuple(cache_len.shape)}")
    if cache_len.dtype not in _INT_DTYPES:
        raise ValueError(f"cache_len must be an integer tensor, got dtype={cache_len.dtype}")
    first = int(cache_len[0].item())
    if int(cache_len.min().item()) != first or int(cache_len.max().item()) != first:
        raise ValueError("cache_len must be the same for all batch elements.")
    if first < 0:
        raise ValueError(f"cache_len must be >= 0, got {first}.")
    return first


def get_past_len(*, past_k: torch.Tensor, cache_len: torch.Tensor | None) -> int:
    if cache_len is None:
        return int(past_k.shape[-2])
    past_len = read_cache_len(cache_len=cache_len)
    capacity = int(past_k.shape[-2])
    if past_len > capacity:
        raise ValueError(f"cache_len={past_len} exceeds cache capacity={capacity}.")
    return past_len


def _resolve_target_capacity(*, current_capacity: int, required_len: int, slot_size: int) -> int:
    if slot_size <= 0:
        raise ValueError(f"slot_size must be positive, got {slot_size}.")
    capacity = int(current_capacity) if current_capacity > 0 else int(slot_size)
    if capacity < slot_size:
        capacity = int(slot_size)
    # Grow when the cache is more than half full to avoid frequent reallocations.
    # This maintains a <= 0.5 load factor, which trades memory for fewer copies.
    while required_len > capacity // 2:
        capacity *= 2
    return capacity


def append_preallocated(
    *,
    past: Sequence[torch.Tensor] | None,
    cache_len: torch.Tensor | None,
    past_len: int,
    new: Sequence[torch.Tensor],
    slot_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """
    Append `new` (each shaped (B, H, T, D_i)) onto `past` caches.

    This avoids `torch.cat` by writing into growable preallocated buffers. The returned
    `views` narrow to the logical length and should be used for attention; `caches`
    are the full-capacity tensors to store in `past_key_value`.

    Note: When `cache_len` is None, `past` is treated as an old-style exact-length cache.
    """

    if not new:
        raise ValueError("new must contain at least one tensor.")
    batch_size = int(new[0].shape[0])
    new_len = int(new[0].shape[-2])
    required_len = int(past_len) + new_len

    if past is not None and len(past) != len(new):
        raise ValueError(f"Expected past to have {len(new)} tensors, got {len(past)}.")

    current_capacity = 0
    can_reuse = False
    if past is not None:
        if cache_len is not None:
            current_capacity = int(past[0].shape[-2])
            can_reuse = True
        else:
            current_capacity = int(past_len)

    target_capacity = _resolve_target_capacity(
        current_capacity=current_capacity,
        required_len=required_len,
        slot_size=int(slot_size),
    )

    caches: list[torch.Tensor]
    if can_reuse and target_capacity == current_capacity:
        caches = list(past) if past is not None else []
    else:
        caches = []
        for tensor in new:
            if tensor.ndim != 4:
                raise ValueError(f"Expected new tensor to have shape (B, H, T, D), got {tuple(tensor.shape)}")
            b, h, _, d = tensor.shape
            caches.append(torch.empty((b, h, target_capacity, d), device=tensor.device, dtype=tensor.dtype))

        if past is not None:
            for i, past_tensor in enumerate(past):
                past_used = past_tensor[:, :, :past_len, :] if cache_len is not None else past_tensor
                caches[i][:, :, :past_len, :] = past_used

    # Append new tensors in-place.
    for i, tensor in enumerate(new):
        caches[i][:, :, past_len:required_len, :] = tensor

    views = [cache[:, :, :required_len, :] for cache in caches]

    if cache_len is None:
        cache_len_out = torch.full((batch_size,), required_len, device=new[0].device, dtype=torch.int64)
    else:
        cache_len.fill_(required_len)
        cache_len_out = cache_len

    return views, caches, cache_len_out

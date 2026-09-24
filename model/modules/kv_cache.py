from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import torch


_CACHE_LEN_DTYPE: Final[torch.dtype] = torch.int64
_PAST_LEN_DTYPES: Final[tuple[torch.dtype, ...]] = (torch.int32, torch.int64)
_CPU_CACHE_LEN_MESSAGE: Final[str] = "cache_len must be CPU metadata to avoid accelerator sync."
_CPU: Final[torch.device] = torch.device("cpu")


def is_cache_len_tensor(tensor: torch.Tensor, *, batch_size: int) -> bool:
    if tensor.ndim != 1:
        return False
    if int(tensor.shape[0]) != batch_size:
        return False
    if tensor.dtype != _CACHE_LEN_DTYPE:
        return False
    return True


def maybe_get_cache_len(past_key_value: Sequence[torch.Tensor] | None, *, batch_size: int) -> torch.Tensor | None:
    if past_key_value is None or len(past_key_value) < 3:
        return None
    candidate = past_key_value[-1]
    if is_cache_len_tensor(candidate, batch_size=batch_size):
        _require_cpu_cache_len(candidate)
        return candidate
    if candidate.ndim == 1 and int(candidate.shape[0]) == batch_size:
        raise ValueError(f"cache_len must be an int64 integer tensor, got dtype={candidate.dtype}")
    return None


def _require_cpu_cache_len(cache_len: torch.Tensor) -> None:
    if cache_len.device.type != "cpu":
        raise ValueError(_CPU_CACHE_LEN_MESSAGE)


def read_cache_len(*, cache_len: torch.Tensor) -> int:
    """Read the uniform logical cache length used by preallocated KV caches.

    The current cache layout stores one shared logical length for the whole
    batch. Per-sequence cache lengths require separate masks or a future cache
    layout that can narrow each batch row independently.
    """

    if cache_len.ndim != 1:
        raise ValueError(f"cache_len must be a 1D tensor, got shape={tuple(cache_len.shape)}")
    if cache_len.numel() == 0:
        raise ValueError("cache_len must have at least one element.")
    if cache_len.dtype != _CACHE_LEN_DTYPE:
        raise ValueError(f"cache_len must be an int64 integer tensor, got dtype={cache_len.dtype}")
    _require_cpu_cache_len(cache_len)
    first_tensor = cache_len[0]
    if bool((cache_len != first_tensor).any().item()):
        raise ValueError("cache_len must be the same for all batch elements.")
    first = int(first_tensor.item())
    if first < 0:
        raise ValueError(f"cache_len must be >= 0, got {first}.")
    return first


def get_past_len(*, past_k: torch.Tensor | None, cache_len: torch.Tensor | None) -> int:
    if cache_len is None:
        if past_k is None:
            return 0
        return int(past_k.shape[-2])
    past_len = read_cache_len(cache_len=cache_len)
    if past_k is None:
        return past_len
    capacity = int(past_k.shape[-2])
    if past_len > capacity:
        raise ValueError(f"cache_len={past_len} exceeds cache capacity={capacity}.")
    return past_len


def _logical_past_len(past_len: int | torch.Tensor) -> int:
    if isinstance(past_len, torch.Tensor):
        if past_len.ndim != 1:
            raise ValueError(f"past_len tensor must have shape (B,), got {tuple(past_len.shape)}")
        if past_len.numel() == 0:
            raise ValueError("past_len tensor must have at least one element.")
        if past_len.dtype not in _PAST_LEN_DTYPES:
            raise ValueError(f"past_len tensor must be an integer tensor, got dtype={past_len.dtype}")
        if past_len.device.type != "cpu":
            raise ValueError("tensor past_len must be CPU metadata; pass an int to avoid accelerator sync.")
        # CPU metadata is the only tensor form allowed here; accelerator
        # tensors are rejected above before any scalar read can synchronize.
        first_tensor = past_len[0]
        if bool((past_len != first_tensor).any().item()):
            raise ValueError("past_len tensor must be the same for all batch elements.")
        first = int(first_tensor.item())
        past_len_value = first
    else:
        past_len_value = past_len
    if past_len_value < 0:
        raise ValueError(f"past_len must be non-negative, got {past_len_value}.")
    return past_len_value


def narrow_cache_to_past_len(tensor: torch.Tensor, *, past_len: int | torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 3:
        raise ValueError(f"Expected cache tensor with a sequence dimension, got shape={tuple(tensor.shape)}")
    past_len_value = _logical_past_len(past_len)
    capacity = int(tensor.shape[-2])
    if past_len_value > capacity:
        raise ValueError(f"past_len={past_len_value} exceeds cache capacity={capacity}.")
    return tensor.narrow(dim=-2, start=0, length=past_len_value)


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


def _can_reuse_cache_storage(*, past: Sequence[torch.Tensor] | None, new: Sequence[torch.Tensor]) -> bool:
    if past is None:
        return False
    return all(
        past_tensor.device == new_tensor.device and past_tensor.dtype == new_tensor.dtype
        for past_tensor, new_tensor in zip(past, new, strict=True)
    )


def _new_cache_len(*, batch_size: int, required_len: int) -> torch.Tensor:
    cache_len = torch.empty((batch_size,), device=_CPU, dtype=_CACHE_LEN_DTYPE)
    cache_len.fill_(required_len)
    return cache_len


def _updated_cache_len(
    *,
    cache_len: torch.Tensor | None,
    batch_size: int,
    required_len: int,
    copy_past: bool,
) -> torch.Tensor:
    if cache_len is not None and not copy_past:
        cache_len.fill_(required_len)
        return cache_len
    return _new_cache_len(batch_size=batch_size, required_len=required_len)


def append_preallocated(
    *,
    past: Sequence[torch.Tensor] | None,
    cache_len: torch.Tensor | None,
    past_len: int,
    new: Sequence[torch.Tensor],
    slot_size: int,
    copy_past: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """
    Append `new` (each shaped (B, H, T, D_i)) onto `past` caches.

    This avoids `torch.cat` by writing into growable preallocated buffers. The returned
    `views` narrow to the logical length and should be used for attention; `caches`
    are the full-capacity tensors to store in `past_key_value`. Set `copy_past=True`
    when a reusable prefix cache must remain immutable for branching.
    Returned `cache_len` is CPU metadata so Python slice bounds do not synchronize
    an accelerator stream during autoregressive decoding.

    Note: When `cache_len` is None, `past` is treated as an old-style exact-length cache.
    """

    if not new:
        raise ValueError("new must contain at least one tensor.")
    batch_size = int(new[0].shape[0])
    new_len = int(new[0].shape[-2])
    if past_len < 0:
        raise ValueError(f"past_len must be non-negative, got {past_len}.")
    required_len = int(past_len) + new_len
    if cache_len is not None:
        cache_len_value = read_cache_len(cache_len=cache_len)
        if int(cache_len.shape[0]) != batch_size:
            raise ValueError(f"cache_len must have shape ({batch_size},), got shape={tuple(cache_len.shape)}.")
        if cache_len_value != past_len:
            raise ValueError(f"cache_len={cache_len_value} must match past_len={past_len}.")

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
    reuse_existing = (
        can_reuse
        and target_capacity == current_capacity
        and not copy_past
        and _can_reuse_cache_storage(past=past, new=new)
    )
    if reuse_existing:
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
                past_used = (
                    narrow_cache_to_past_len(past_tensor, past_len=past_len) if cache_len is not None else past_tensor
                )
                # Only logically empty meta prefixes can materialize from `new`.
                # Non-empty history must keep the same storage contract even if a
                # zero-sized feature dimension means there are no values to copy.
                if past_len > 0:
                    if past_used.device != caches[i].device:
                        if past_used.device.type == "meta" and caches[i].device.type != "meta":
                            raise ValueError(
                                "non-empty past cache tensor on meta device cannot be copied into a materialized "
                                "cache; materialize caches before calling forward or start from an empty logical "
                                "prefix."
                            )
                        raise ValueError(
                            "past cache tensor device must match new cache device for non-empty preallocated caches; "
                            f"got past={past_used.device} and new={caches[i].device}."
                        )
                    if past_used.dtype != caches[i].dtype:
                        raise ValueError(
                            "past cache tensor dtype must match new cache dtype for non-empty preallocated caches; "
                            f"got past={past_used.dtype} and new={caches[i].dtype}."
                        )
                if past_used.numel() > 0:
                    caches[i][:, :, :past_len, :] = past_used

    # Append new tensors in-place.
    for i, tensor in enumerate(new):
        caches[i][:, :, past_len:required_len, :] = tensor

    views = [cache[:, :, :required_len, :] for cache in caches]

    # `cache_len` is CPU metadata by design: generation needs Python slice
    # bounds, so storing this scalar state on accelerators would synchronize the
    # stream when we read it back.
    # Allocate CPU metadata only when there is no reusable buffer or when
    # copy_past=True must preserve the caller's cache_len for branching. The
    # normal autoregressive path updates the existing CPU tensor in-place.
    cache_len_out = _updated_cache_len(
        cache_len=cache_len,
        batch_size=batch_size,
        required_len=required_len,
        copy_past=copy_past,
    )

    return views, caches, cache_len_out


__all__ = [
    "get_past_len",
    "maybe_get_cache_len",
    "narrow_cache_to_past_len",
    "append_preallocated",
]

from __future__ import annotations

import torch


def _validate_token_mask(token_mask: torch.Tensor, *, batch_size: int, seq_len: int) -> None:
    if token_mask.ndim != 2 or tuple(token_mask.shape) != (batch_size, seq_len):
        raise ValueError(
            f"shortconv token_mask must have shape ({batch_size}, {seq_len}), got {tuple(token_mask.shape)}."
        )


def compact_current_tokens(
    x: torch.Tensor,
    *,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim < 2:
        raise ValueError(f"shortconv current tokens must have rank >= 2, got {tuple(x.shape)}.")
    batch_size = int(x.shape[0])
    seq_len = int(x.shape[1])
    _validate_token_mask(token_mask, batch_size=batch_size, seq_len=seq_len)

    current_mask = token_mask.to(device=x.device, dtype=torch.bool, non_blocking=True)
    active_ranks = current_mask.to(dtype=torch.long).cumsum(dim=1) - 1
    flat = x.reshape(batch_size, seq_len, -1)
    compact_flat = torch.zeros_like(flat)
    values = torch.where(current_mask[:, :, None], flat, torch.zeros_like(flat))
    compact_flat.scatter_add_(
        dim=1,
        index=active_ranks.clamp_min(0)[:, :, None].expand_as(flat),
        src=values,
    )
    return compact_flat.reshape_as(x), current_mask, active_ranks


def gather_compacted_current_slots(
    slots: torch.Tensor,
    *,
    current_mask: torch.Tensor,
    active_ranks: torch.Tensor,
) -> torch.Tensor:
    if slots.ndim < 2:
        raise ValueError(f"shortconv slots must have rank >= 2, got {tuple(slots.shape)}.")
    batch_size = int(slots.shape[0])
    seq_len = int(slots.shape[1])
    if tuple(current_mask.shape) != (batch_size, seq_len):
        raise ValueError(
            f"shortconv current_mask must have shape ({batch_size}, {seq_len}), got {tuple(current_mask.shape)}."
        )
    if tuple(active_ranks.shape) != (batch_size, seq_len):
        raise ValueError(
            f"shortconv active_ranks must have shape ({batch_size}, {seq_len}), got {tuple(active_ranks.shape)}."
        )

    flat = slots.reshape(batch_size, seq_len, -1)
    gathered = flat.gather(
        dim=1,
        index=active_ranks.clamp_min(0)[:, :, None].expand_as(flat),
    ).reshape_as(slots)
    mask = current_mask
    while mask.ndim < gathered.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, gathered, torch.zeros_like(gathered)).contiguous()


def compact_shortconv_history(
    x: torch.Tensor,
    *,
    history: torch.Tensor,
    token_mask: torch.Tensor | None,
    history_size: int,
) -> torch.Tensor:
    if history_size < 0:
        raise ValueError(f"shortconv history_size must be non-negative, got {history_size}.")
    if x.ndim < 2:
        raise ValueError(f"shortconv history input must have rank >= 2, got {tuple(x.shape)}.")
    if history.ndim != x.ndim:
        raise ValueError(f"shortconv history rank {history.ndim} must match input rank {x.ndim}.")
    batch_size = int(x.shape[0])
    seq_len = int(x.shape[1])
    trailing_shape = tuple(int(dim) for dim in x.shape[2:])
    expected_history_shape = (batch_size, int(history.shape[1]), *trailing_shape)
    if tuple(history.shape) != expected_history_shape:
        raise ValueError(
            "shortconv history must have shape "
            f"(batch={batch_size}, history_tokens, trailing={trailing_shape}), got {tuple(history.shape)}."
        )
    if history_size == 0:
        return history[:, :0, ...].contiguous()

    history_on_device = history.to(device=x.device, dtype=x.dtype)
    if token_mask is None:
        source = torch.cat((history_on_device, x), dim=1) if int(history_on_device.shape[1]) > 0 else x
        return source[:, -history_size:, ...].contiguous()

    _validate_token_mask(token_mask, batch_size=batch_size, seq_len=seq_len)
    history_len = int(history_on_device.shape[1])
    source = torch.cat((history_on_device, x), dim=1) if history_len > 0 else x
    source_len = int(source.shape[1])
    if source_len == 0:
        return x.new_zeros((batch_size, 0, *trailing_shape))

    history_mask = torch.ones((batch_size, history_len), device=x.device, dtype=torch.bool)
    current_mask = token_mask.to(device=x.device, dtype=torch.bool, non_blocking=True)
    source_mask = torch.cat((history_mask, current_mask), dim=1) if history_len > 0 else current_mask
    active_ranks = source_mask.to(dtype=torch.long).cumsum(dim=1)
    total_active = active_ranks[:, -1]
    history_offsets = torch.arange(history_size, device=x.device, dtype=torch.long)
    target_ranks = total_active[:, None] - history_size + 1 + history_offsets[None, :]
    target_valid = target_ranks > 0

    lookup_ranks = target_ranks.clamp_min(1)
    gather_indices = torch.searchsorted(active_ranks, lookup_ranks, right=False)
    out_of_bounds = target_valid & (gather_indices >= source_len)
    if bool(out_of_bounds.any().item()):
        raise ValueError("shortconv active rank lookup produced out-of-bounds gather indices.")
    gather_indices = torch.where(target_valid, gather_indices, torch.zeros_like(gather_indices))
    flat_source = source.reshape(batch_size, source_len, -1)
    gathered = flat_source.gather(
        dim=1,
        index=gather_indices[:, :, None].expand(batch_size, history_size, int(flat_source.shape[2])),
    ).reshape(batch_size, history_size, *trailing_shape)
    valid_mask = target_valid
    while valid_mask.ndim < gathered.ndim:
        valid_mask = valid_mask.unsqueeze(-1)
    return torch.where(valid_mask, gathered, torch.zeros_like(gathered)).contiguous()


__all__ = [
    "compact_current_tokens",
    "compact_shortconv_history",
    "gather_compacted_current_slots",
]

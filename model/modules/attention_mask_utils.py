"""Attention-mask classification helpers shared across model paths.

Model entrypoints treat 2D numeric ``0/1`` attention masks as token padding
masks: one keeps a token and zero pads it, including all-zero rows. Callers that
need an additive all-zero no-op should pass ``None`` or a query-specific 3D/4D
additive mask instead. Non-binary 2D numeric rows keep additive-mask semantics.
"""

from __future__ import annotations

import torch


def numeric_mask_binary_flags(
    mask: torch.Tensor,
    *,
    require_nonzero: bool = False,
    allow_zero_rows_when_mixed_binary: bool = False,
    keepdim: bool = False,
) -> torch.Tensor:
    """Return which rows contain only numeric binary mask values.

    The helper only classifies values. Callers choose whether all-zero rows are
    valid binary padding rows with ``require_nonzero``.
    """

    if mask.ndim == 0:
        is_binary = torch.logical_or(mask == 0, mask == 1)
        if require_nonzero:
            is_binary = is_binary & (mask != 0)
        return is_binary.to(dtype=torch.bool)

    row_is_binary = torch.all(torch.logical_or(mask == 0, mask == 1), dim=-1, keepdim=True)
    is_binary = row_is_binary
    if require_nonzero:
        has_nonzero = torch.any(mask != 0, dim=-1, keepdim=True)
        is_binary = row_is_binary & has_nonzero
        if allow_zero_rows_when_mixed_binary:
            if mask.ndim >= 3:
                has_binary_nonzero_row = torch.any(is_binary, dim=-2, keepdim=True)
                all_rows_are_binary = torch.all(row_is_binary, dim=-2, keepdim=True)
            else:
                has_binary_nonzero_row = is_binary
                all_rows_are_binary = row_is_binary
            is_binary = is_binary | (row_is_binary & ~has_nonzero & has_binary_nonzero_row & all_rows_are_binary)
    if not keepdim:
        is_binary = is_binary.squeeze(-1)
    return is_binary.to(dtype=torch.bool)

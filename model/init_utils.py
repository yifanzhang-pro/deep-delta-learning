from __future__ import annotations

import math

import torch
import torch.nn as nn

_DEFAULT_EXCLUDE_SUFFIXES = (
    "attn.c_proj.weight",
    "attn.core.c_proj.weight",
    "mlp.c_proj.weight",
    # Positional-encoding params that have their own init (e.g., GRAPE freq spectra).
    "grape.log_freq",
)
_DEFAULT_EXCLUDE_NAMES = (
    "transformer.wte.weight",
    "lm_head.weight",
)


def init_gpt_weights(
    model: nn.Module,
    config,
    *,
    exclude_suffixes: tuple[str, ...] = (),
    exclude_names: tuple[str, ...] = (),
) -> None:
    init_std = float(getattr(config, "embedding_init_std", 0.02))
    hidden_init_std_factor = float(getattr(config, "hidden_init_std_factor", 0.5))
    hidden_size = int(getattr(config, "hidden_size"))
    hidden_std = hidden_init_std_factor / math.sqrt(hidden_size)

    exclude_suffixes = _DEFAULT_EXCLUDE_SUFFIXES + tuple(exclude_suffixes)
    exclude_names = set(_DEFAULT_EXCLUDE_NAMES + tuple(exclude_names))

    with torch.no_grad():
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and getattr(lm_head, "weight", None) is not None:
            lm_head.weight.normal_(mean=0.0, std=init_std)

        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            if name in exclude_names:
                continue
            if any(name.endswith(suffix) for suffix in exclude_suffixes):
                continue
            param.normal_(mean=0.0, std=hidden_std)

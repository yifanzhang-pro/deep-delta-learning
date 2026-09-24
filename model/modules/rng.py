"""RNG helpers for shared modules."""

from __future__ import annotations

from itertools import chain

import torch.nn as nn


def cuda_rng_devices_for_module(module: nn.Module) -> list[int]:
    devices: set[int] = set()
    for tensor in chain(module.parameters(), module.buffers()):
        if not tensor.is_cuda:
            continue
        devices.add(tensor.get_device())
    return sorted(devices)


__all__ = ["cuda_rng_devices_for_module"]

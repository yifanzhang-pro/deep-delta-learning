"""Shared feed-forward blocks."""

from __future__ import annotations

import math
from typing import Any, Callable, Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_utils import optional_float, required_int
from .initialization import residual_output_projection_depth_divisor, residual_output_projection_std
from .rng import cuda_rng_devices_for_module


class MLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size: int = required_int(config, "hidden_size")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}.")
        hidden_dim = math.floor(8 / 3 * self.hidden_size)
        self.c_fc1 = nn.Linear(self.hidden_size, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(self.hidden_size, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, self.hidden_size, bias=False)
        self.hidden_init_std_factor: float = optional_float(config, "hidden_init_std_factor", 0.5)
        self.num_hidden_layers: int = required_int(config, "num_hidden_layers")
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}.")
        self.residual_output_projection_depth_divisor: float = residual_output_projection_depth_divisor(
            config,
            num_layers=self.num_hidden_layers,
        )
        self._reset_output_projection_parameters_preserving_rng()

    def reset_output_projection_parameters(self) -> None:
        if self.o_proj.weight.is_meta:
            return
        with torch.no_grad():
            std = residual_output_projection_std(
                self.o_proj,
                hidden_init_std_factor=self.hidden_init_std_factor,
                depth_divisor=self.residual_output_projection_depth_divisor,
            )
            self.o_proj.weight.normal_(mean=0.0, std=std)

    def _reset_output_projection_parameters_preserving_rng(self) -> None:
        with torch.random.fork_rng(devices=cuda_rng_devices_for_module(self)):
            self.reset_output_projection_parameters()

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        was_meta = self.o_proj.weight.device.type == "meta"
        result = super()._apply(fn, recurse=recurse)
        if was_meta and self.o_proj.weight.device.type != "meta":
            self._reset_output_projection_parameters_preserving_rng()
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)
        x = F.silu(x1) * x2
        x = self.o_proj(x)
        return x


__all__ = ["MLP"]

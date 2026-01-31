# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _apply_residual(x: torch.Tensor, residual: Optional[torch.Tensor], residual_in_fp32: bool) -> torch.Tensor:
    if residual is None:
        return x
    if residual_in_fp32:
        return x.float() + residual.float()
    return x + residual


def rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float = 1e-5,
    residual: Optional[torch.Tensor] = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    x_resid = _apply_residual(x, residual, residual_in_fp32)
    x_float = x_resid.float()
    rstd = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x_float * rstd
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    y = y.to(orig_dtype)
    return y if not prenorm else (y, x_resid)


def group_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    num_groups: int,
    eps: float = 1e-5,
    is_rms_norm: bool = False,
    residual: Optional[torch.Tensor] = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % num_groups != 0:
        raise ValueError("num_channels must be divisible by num_groups")

    orig_dtype = x.dtype
    x_resid = _apply_residual(x, residual, residual_in_fp32)
    x_float = x_resid.float()
    group_size = x_float.shape[-1] // num_groups
    x_group = x_float.reshape(*x_float.shape[:-1], num_groups, group_size)
    if not is_rms_norm:
        x_group = x_group - x_group.mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(x_group.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x_group * rstd
    y = y.reshape(*x_float.shape)
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    y = y.to(orig_dtype)
    return y if not prenorm else (y, x_resid)


class RMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, elementwise_affine: bool = True, bias: bool = False, eps: float = 1e-5
    ) -> RMSNorm:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        return rms_norm(
            x,
            self.weight,
            self.bias,
            eps=self.eps,
            residual=residual,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class GroupNorm(nn.Module):
    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        is_rms_norm: bool = False,
    ) -> GroupNorm:
        super().__init__()

        if hidden_size % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.is_rms_norm = is_rms_norm

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.num_groups}, {self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        if self.is_rms_norm:
            s += f", is_rms_norm={self.is_rms_norm}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        return group_norm(
            x,
            self.weight,
            self.bias,
            num_groups=self.num_groups,
            eps=self.eps,
            is_rms_norm=self.is_rms_norm,
            residual=residual,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )

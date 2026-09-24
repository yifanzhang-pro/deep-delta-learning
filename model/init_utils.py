from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable
from typing import Any, Protocol, cast, runtime_checkable

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Shard

from .modules.config_utils import optional_float
from .modules.initialization import residual_output_projection_depth_divisor, spectral_std

_DEFAULT_EXCLUDE_SUFFIXES = (
    # Positional-encoding params that have their own init (e.g., GRAPE freq spectra).
    "grape.log_freq",
)
_DEFAULT_EXACT_EXCLUDE_NAMES = (
    # Root-level tied-token weights. Layer-local patterns such as
    # `attn.o_proj.weight` intentionally live in suffix lists below.
    "transformer.wte.weight",
    "lm_head.weight",
)
_RESIDUAL_OUTPUT_PROJ_SUFFIXES = (
    "attn.o_proj.weight",
    "mlp.o_proj.weight",
    "mlp.up_proj.weight",
)


@runtime_checkable
class _DeterministicInitModule(Protocol):
    def reset_deterministic_parameters(self) -> None: ...


@runtime_checkable
class _OutputProjectionInitModule(Protocol):
    def reset_output_projection_parameters(self) -> None: ...


def _should_exclude_parameter(name: str, *, exclude_name_set: set[str], exclude_suffixes: tuple[str, ...]) -> bool:
    if name in exclude_name_set:
        return True
    return any(name.endswith(suffix) for suffix in exclude_suffixes)


def _tensor_fans(param: torch.Tensor) -> tuple[int, int]:
    if param.dim() < 2:
        raise ValueError(f"Expected tensor with rank >= 2, got rank {param.dim()}.")
    # Treat the first axis as n_out and all remaining axes as n_in.
    # This matches linear weights (out, in) and Conv1d weights
    # (out_channels, in_channels/groups, kernel_size).
    n_out = int(param.shape[0])
    if n_out <= 0:
        raise ValueError(f"Expected first dimension to be positive, got {n_out}.")
    n_in = int(param.numel()) // n_out
    if n_in <= 0:
        raise ValueError(f"Computed n_in must be positive, got {n_in}.")
    return n_in, n_out


def _is_residual_output_projection(
    name: str,
    *,
    non_residual_output_names: set[str],
) -> bool:
    if name in non_residual_output_names:
        return False
    if any(name.endswith(suffix) for suffix in _RESIDUAL_OUTPUT_PROJ_SUFFIXES):
        return True
    return _is_routed_expert_parameter(name) and name.endswith(".o_proj.weight")


def _has_meta_parameters(model: nn.Module) -> bool:
    return any(param.is_meta for param in model.parameters())


def _reset_module_owned_parameters(model: nn.Module) -> None:
    for module_name, module in model.named_modules():
        if isinstance(module, _DeterministicInitModule):
            module.reset_deterministic_parameters()
        if isinstance(module, _OutputProjectionInitModule) and ".experts." not in (f".{module_name}."):
            module.reset_output_projection_parameters()


def _is_routed_expert_parameter(name: str) -> bool:
    return ".mlp.experts." in f".{name}"


def _stable_parameter_seed(base_seed: int, name: str) -> int:
    payload = f"nanogptpro.deepseek-moe-expert.v1:{base_seed}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _normal_with_stable_seed(
    parameter: torch.Tensor,
    *,
    mean: float,
    std: float,
    seed: int,
) -> None:
    device_index = parameter.device.index
    devices: list[int] = []
    if parameter.device.type == "cuda":
        devices = [torch.cuda.current_device() if device_index is None else device_index]
    with torch.random.fork_rng(devices=devices):
        if parameter.device.type == "cuda":
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)
        parameter.normal_(mean=mean, std=std)


def _deterministic_reset_parameter_names(model: nn.Module) -> set[str]:
    parameter_names: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, _DeterministicInitModule):
            continue
        prefix = f"{module_name}." if module_name else ""
        nn_module = cast(nn.Module, module)
        for parameter_name, _parameter in nn_module.named_parameters(recurse=True):
            parameter_names.add(f"{prefix}{parameter_name}")
    return parameter_names


def _output_projection_reset_parameter_names(model: nn.Module) -> set[str]:
    output_projection_names: set[str] = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, _OutputProjectionInitModule):
            continue
        output_projection = getattr(module, "o_proj", None)
        if not isinstance(output_projection, nn.Module):
            continue
        prefix = f"{module_name}.o_proj" if module_name else "o_proj"
        for parameter_name, _parameter in output_projection.named_parameters(recurse=True):
            output_projection_names.add(f"{prefix}.{parameter_name}")
    return output_projection_names


def _non_residual_output_projection_parameter_names(model: nn.Module) -> set[str]:
    parameter_names: set[str] = set()
    for module_name, module in model.named_modules():
        if getattr(module, "output_is_residual", True) is not False:
            continue
        output_projection = getattr(module, "o_proj", None)
        if not isinstance(output_projection, nn.Module):
            raise TypeError(f"Module {module_name!r} marks its output as non-residual but has no o_proj.")
        prefix = f"{module_name}.o_proj" if module_name else "o_proj"
        for parameter_name, _parameter in output_projection.named_parameters(recurse=True):
            parameter_names.add(f"{prefix}.{parameter_name}")
    return parameter_names


def init_gpt_weights(
    model: nn.Module,
    config: object,
    *,
    exclude_suffixes: tuple[str, ...] = (),
    exclude_names: tuple[str, ...] = (),
) -> None:
    init_std = float(getattr(config, "embedding_init_std", 0.02))
    hidden_init_std_factor = optional_float(config, "hidden_init_std_factor", 0.5)
    num_layers = int(getattr(config, "num_hidden_layers"))
    if num_layers <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {num_layers}.")
    residual_output_proj_std_divisor = residual_output_projection_depth_divisor(config, num_layers=num_layers)
    initialization_seed = torch.initial_seed()

    exclude_suffixes = _DEFAULT_EXCLUDE_SUFFIXES + tuple(exclude_suffixes)
    exclude_name_set = set(_DEFAULT_EXACT_EXCLUDE_NAMES + tuple(exclude_names))
    module_owned_reset_names = _deterministic_reset_parameter_names(model) | _output_projection_reset_parameter_names(
        model
    )
    non_residual_output_names = _non_residual_output_projection_parameter_names(model)

    if _has_meta_parameters(model):
        warnings.warn(
            "init_gpt_weights skipped because model parameters are on the meta device; "
            "materialize the model first and call init_gpt_weights again.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    with torch.no_grad():
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and getattr(lm_head, "weight", None) is not None:
            lm_head.weight.normal_(mean=0.0, std=init_std)

        for name, param in model.named_parameters():
            if param.dim() < 2:
                continue
            if _should_exclude_parameter(name, exclude_name_set=exclude_name_set, exclude_suffixes=exclude_suffixes):
                continue
            routed_expert_parameter = _is_routed_expert_parameter(name)
            if name in module_owned_reset_names and not routed_expert_parameter:
                continue
            if routed_expert_parameter and param.ndim == 3:
                prefix, weight_name = name.rsplit(".", 1)
                projections = {"gate_up_PID": ("gate_proj", "up_proj"), "down_EDI": ("o_proj",)}[weight_name]
                std = spectral_std(
                    n_in=int(param.shape[-1]), n_out=int(param.shape[-2]), hidden_init_std_factor=hidden_init_std_factor
                )
                if projections == ("o_proj",):
                    std /= residual_output_proj_std_divisor
                local = param
                row_start = 0
                if isinstance(param, DTensor):
                    if param.device_mesh.ndim != 1 or param.placements != (Shard(0),):
                        raise ValueError("Grouped expert initialization requires one shard on the expert-row axis.")
                    local = param.to_local()
                    shard_rows = (int(param.shape[0]) + param.device_mesh.size() - 1) // param.device_mesh.size()
                    row_start = shard_rows * param.device_mesh.get_local_rank()
                for row, matrix in enumerate(local.unbind(0), start=row_start):
                    expert_index, projection_index = divmod(row, len(projections))
                    _normal_with_stable_seed(
                        matrix,
                        mean=0.0,
                        std=std,
                        seed=_stable_parameter_seed(
                            initialization_seed, f"{prefix}.{expert_index}.{projections[projection_index]}.weight"
                        ),
                    )
                continue
            n_in, n_out = _tensor_fans(param)
            std = spectral_std(
                n_in=n_in,
                n_out=n_out,
                hidden_init_std_factor=hidden_init_std_factor,
            )
            if _is_residual_output_projection(
                name,
                non_residual_output_names=non_residual_output_names,
            ):
                std = std / residual_output_proj_std_divisor
            if routed_expert_parameter:
                _normal_with_stable_seed(
                    param,
                    mean=0.0,
                    std=std,
                    seed=_stable_parameter_seed(initialization_seed, name),
                )
            else:
                param.normal_(mean=0.0, std=std)

        _reset_module_owned_parameters(model)


def init_test_randn(
    base_randn: Callable[..., torch.Tensor],
    *size: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Route test normal initialization through `model.init_utils`.

    This indirection lets tests use a centralized initialization entrypoint
    without changing the distribution semantics of `torch.randn`.
    """

    return base_randn(*size, **kwargs)


def init_test_rand(
    base_rand: Callable[..., torch.Tensor],
    *size: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """Route test uniform initialization through `model.init_utils`.

    This indirection lets tests use a centralized initialization entrypoint
    without changing the distribution semantics of `torch.rand`.
    """

    return base_rand(*size, **kwargs)

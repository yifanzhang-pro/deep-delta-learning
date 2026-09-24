from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import sys
import types
from typing import Any, Annotated, Union, cast, get_args, get_origin, get_type_hints

import torch
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr, create_model
from transformers import configuration_utils as hf_configuration_utils

_L2NORM_KEY_ALIASES: tuple[tuple[str, str], ...] = (
    ("ql2norm", "q_l2norm"),
    ("kl2norm", "k_l2norm"),
    ("vl2norm", "v_l2norm"),
)
_L2NORM_KEYS: tuple[str, ...] = tuple(canonical for _, canonical in _L2NORM_KEY_ALIASES)
_L2NORM_INPUT_KEYS: tuple[str, ...] = _L2NORM_KEYS + tuple(alias for alias, _ in _L2NORM_KEY_ALIASES)
_RMSNORM_KEYS: tuple[str, ...] = ("q_rmsnorm", "k_rmsnorm", "v_rmsnorm")
_PRESERVED_PRETRAINED_CONFIG_KWARGS: tuple[str, ...] = ("use_cache",)
_PRESERVED_PRETRAINED_CONFIG_PREFIX = "_nanogpt_preserved_"
_PRESERVED_SERIALIZATION_PATCH_FLAG = "_nanogpt_preserved_serialization_installed"
_PRESERVED_DEFAULT_PREFIX = "_nanogpt_preserved_default_"
_PRESERVED_PROPERTY_PATCH_PREFIX = "_nanogpt_preserved_property_installed_"
_PRESERVED_PRETRAINED_CONFIG_DEFAULTS: Mapping[str, object] = {"use_cache": True}
_MISSING = object()


def _install_preserved_pretrained_config_serialization(config_cls: type[object]) -> None:
    if config_cls.__dict__.get(_PRESERVED_SERIALIZATION_PATCH_FLAG) is True:
        return

    original_remove_keys = getattr(config_cls, "_remove_keys_not_serialized")

    def _remove_keys_not_serialized(self: object, data: dict[str, Any]) -> None:
        original_remove_keys(self, data)
        for public_key in _PRESERVED_PRETRAINED_CONFIG_KWARGS:
            private_key = f"{_PRESERVED_PRETRAINED_CONFIG_PREFIX}{public_key}"
            if private_key in data:
                data[public_key] = data.pop(private_key)

    setattr(config_cls, "_remove_keys_not_serialized", _remove_keys_not_serialized)
    setattr(config_cls, _PRESERVED_SERIALIZATION_PATCH_FLAG, True)


def _preserved_pretrained_config_default(config_cls: type[object], key: str) -> object:
    for base_cls in config_cls.__mro__:
        default = base_cls.__dict__.get(key, _MISSING)
        if default is _MISSING:
            continue
        if isinstance(default, property):
            return _MISSING
        return default
    return _PRESERVED_PRETRAINED_CONFIG_DEFAULTS.get(key, _MISSING)


def _install_preserved_pretrained_config_property(config_cls: type[object], key: str) -> None:
    _install_preserved_pretrained_config_serialization(config_cls)
    property_flag = f"{_PRESERVED_PROPERTY_PATCH_PREFIX}{key}"
    if config_cls.__dict__.get(property_flag) is True:
        return
    default = _preserved_pretrained_config_default(config_cls, key)
    if isinstance(default, property):
        return
    private_key = f"{_PRESERVED_PRETRAINED_CONFIG_PREFIX}{key}"
    default_key = f"{_PRESERVED_DEFAULT_PREFIX}{key}"
    if default is not _MISSING:
        setattr(config_cls, default_key, default)

    def _get_preserved_value(self: object) -> Any:
        values = getattr(self, "__dict__", {})
        if key in values:
            return values[key]
        if private_key in values:
            return values[private_key]
        return getattr(type(self), default_key, None)

    def _set_preserved_value(self: object, value: Any) -> None:
        values = getattr(self, "__dict__")
        values[key] = value
        values[private_key] = value

    setattr(config_cls, key, property(_get_preserved_value, _set_preserved_value))
    setattr(config_cls, property_flag, True)


def _to_pydantic_type(py_type: Any) -> Any:
    origin = get_origin(py_type)
    if origin is None:
        if py_type is bool:
            return StrictBool
        if py_type is int:
            return StrictInt
        if py_type is str:
            return StrictStr
        if py_type is float:
            # Int -> float coercion is convenient in configs (e.g. dropout=0).
            return float
        return py_type

    args = get_args(py_type)
    if not args:
        return py_type

    if origin is Annotated:
        return _to_pydantic_type(args[0])

    if origin in (Union, types.UnionType):
        mapped_args = tuple(_to_pydantic_type(arg) for arg in args)
        return Union[mapped_args]

    if origin is list:
        return list[_to_pydantic_type(args[0])]
    if origin is set:
        return set[_to_pydantic_type(args[0])]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple[_to_pydantic_type(args[0]), ...]
        return tuple[tuple(_to_pydantic_type(arg) for arg in args)]
    if origin is dict:
        return dict[_to_pydantic_type(args[0]), _to_pydantic_type(args[1])]

    return py_type


@lru_cache(maxsize=None)
def _build_config_model(config_cls: type[object]) -> type[BaseModel]:
    create_dynamic_model = cast(Any, create_model)
    raw_annotations = config_cls.__dict__.get("__annotations__", {})
    if not raw_annotations:
        return cast(
            type[BaseModel],
            create_dynamic_model(
                f"{config_cls.__module__}.{config_cls.__qualname__}ConfigModel",
                __config__=ConfigDict(extra="allow"),
            ),
        )

    module = sys.modules.get(config_cls.__module__)
    globalns: dict[str, Any] = dict(vars(hf_configuration_utils))
    if module is not None:
        globalns.update(vars(module))
    globalns.setdefault("torch", torch)
    resolved = get_type_hints(config_cls, globalns=globalns, include_extras=True)
    fields: dict[str, tuple[Any, object]] = {}
    for name in raw_annotations:
        if name not in resolved:
            continue
        default = getattr(config_cls, name, ...)
        preserved_default = getattr(config_cls, f"{_PRESERVED_DEFAULT_PREFIX}{name}", _MISSING)
        if isinstance(default, property) and preserved_default is not _MISSING:
            default = preserved_default
        fields[name] = (_to_pydantic_type(resolved[name]), default)

    return cast(
        type[BaseModel],
        create_dynamic_model(
            f"{config_cls.__module__}.{config_cls.__qualname__}ConfigModel",
            __config__=ConfigDict(extra="allow"),
            **fields,
        ),
    )


def _apply_l2norm_key_aliases(config_cls: type[object], values: dict[str, Any]) -> None:
    raw_annotations = config_cls.__dict__.get("__annotations__", {})
    if not isinstance(raw_annotations, dict):
        return

    for alias, canonical in _L2NORM_KEY_ALIASES:
        if canonical not in raw_annotations:
            continue
        if alias not in values:
            continue
        if canonical in values:
            raise ValueError(
                f"Conflicting GPT config keys: both {alias!r} and {canonical!r} were provided; set only one."
            )
        values[canonical] = values.pop(alias)


def _config_module_stem(config_cls: type[object]) -> str:
    module = getattr(config_cls, "__module__", "")
    if not isinstance(module, str):
        return ""
    return module.rsplit(".", maxsplit=1)[-1]


def _is_mha_deltanet_module(config_cls: type[object]) -> bool:
    module_stem = _config_module_stem(config_cls).replace("-", "_")
    if module_stem.startswith("gpt_mha_falcon_1_scaled_"):
        return False
    return module_stem.startswith(("gpt_mha_delta_net", "gpt_mha_falcon_1_", "gpt_mha_falcon_2_"))


def _is_mha_gated_deltanet_module(config_cls: type[object]) -> bool:
    return _config_module_stem(config_cls).startswith("gpt-mha-gated_deltanet")


def _is_fla_module(config_cls: type[object]) -> bool:
    return "_fla" in _config_module_stem(config_cls)


def _config_declares_any_key(config_cls: type[object], keys: tuple[str, ...]) -> bool:
    raw_annotations = config_cls.__dict__.get("__annotations__", {})
    if not isinstance(raw_annotations, dict):
        return False
    return any(key in raw_annotations for key in keys)


def _raise_unsupported_renamed_config_keys(
    values: Mapping[str, Any],
    renamed_keys: Mapping[str, str],
) -> None:
    forbidden_keys = [key for key in renamed_keys if key in values]
    if not forbidden_keys:
        return

    if len(forbidden_keys) == 1:
        legacy_key = forbidden_keys[0]
        replacement = renamed_keys[legacy_key]
        raise ValueError(f"Unsupported config key: {legacy_key!r}. Use {replacement!r} instead.")

    replacements = ", ".join(f"{key!r} -> {renamed_keys[key]!r}" for key in forbidden_keys)
    raise ValueError(
        f"Unsupported GPT config key(s) {forbidden_keys}; use the canonical names instead: {replacements}."
    )


def validate_pretrained_config_kwargs(config_cls: type[object], values: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate a Hugging Face `PretrainedConfig` kwarg dict using a generated Pydantic model.

    Notes:
    - Known config fields (declared via type annotations on `config_cls`) are validated.
    - Extra keys are allowed and passed through unchanged, mirroring `PretrainedConfig` behavior.
    """
    values_dict = dict(values)
    module_stem = _config_module_stem(config_cls)

    if "mup" in values_dict:
        raise ValueError("Unsupported GPT config key(s) ['mup']; use 'parameterization'.")
    if values_dict.get("parameterization") in ("muP", "mup"):
        raise ValueError("Unsupported GPT config parameterization 'muP'; use parameterization='widthmuP'.")

    if _config_declares_any_key(config_cls, _RMSNORM_KEYS):
        forbidden_l2norm_keys = [key for key in _L2NORM_INPUT_KEYS if key in values_dict]
        if len(forbidden_l2norm_keys) == 1:
            legacy_key = forbidden_l2norm_keys[0]
            raise ValueError(f"Unsupported config key: {legacy_key}. Use q_rmsnorm/k_rmsnorm/v_rmsnorm instead.")
        if forbidden_l2norm_keys:
            raise ValueError(
                f"Unsupported config key(s) {forbidden_l2norm_keys}. Use q_rmsnorm/k_rmsnorm/v_rmsnorm instead."
            )

    _apply_l2norm_key_aliases(config_cls, values_dict)

    deprecated_beta_scale_keys = ("delta_allow_neg_eigval", "gated_deltanet_allow_neg_eigval")
    forbidden_deprecated_beta_scale_keys = [key for key in deprecated_beta_scale_keys if key in values_dict]
    if forbidden_deprecated_beta_scale_keys:
        raise ValueError(
            f"Unsupported GPT config key(s) {forbidden_deprecated_beta_scale_keys}; use 'beta_scale_by_2'."
        )

    if _is_mha_gated_deltanet_module(config_cls):
        _raise_unsupported_renamed_config_keys(
            values_dict,
            {
                "delta_use_beta": "gated_deltanet_use_beta",
                "gated_delta_rule_scale": "delta_rule_scale",
            },
        )

    if module_stem == "gpt-mha-gated_deltanet_chunk_triton_fla":
        _raise_unsupported_renamed_config_keys(
            values_dict,
            {
                "gated_delta_rule_triton_dtype": "delta_rule_triton_dtype",
                "delta_rule_chunk_bwd_impl": "gated_delta_rule_chunk_bwd_impl",
            },
        )

    if module_stem == "gpt-mha-grape-m-ctx":
        _raise_unsupported_renamed_config_keys(
            values_dict,
            {"grape_log_freq_scale": "grape_ctx_omega_scale"},
        )

    if _is_mha_deltanet_module(config_cls):
        forbidden_rmsnorm_keys = [key for key in _RMSNORM_KEYS if key in values_dict]
        if forbidden_rmsnorm_keys:
            raise ValueError(
                f"Unsupported GPT config key(s) {forbidden_rmsnorm_keys} for DeltaNet L2-norm variants; "
                "use q_l2norm/k_l2norm/v_l2norm (legacy: ql2norm/kl2norm/vl2norm) instead."
            )

    size_alias_keys = ("n_embd", "n_embd_base", "d_ff_factor")
    forbidden_size_aliases = [key for key in size_alias_keys if key in values_dict]
    if forbidden_size_aliases:
        raise ValueError(
            f"Unsupported GPT config key(s) {forbidden_size_aliases}; "
            "use 'hidden_size' / 'hidden_size_base' / 'mlp_hidden_mult'."
        )

    deprecated_delta_keys = (
        "delta_beta_init",
        "delta_use_short_conv",
        "delta_conv_size",
        "delta_norm_eps",
        "delta_qk_activation",
        "delta_qk_norm",
    )
    forbidden_deprecated_delta_keys = [key for key in deprecated_delta_keys if key in values_dict]
    if forbidden_deprecated_delta_keys:
        raise ValueError(
            f"Unsupported GPT config key(s) {forbidden_deprecated_delta_keys}; "
            "the deprecated DeltaNet `delta_*` knobs were removed. "
            "Use `q_activation`/`k_activation`/`v_activation`, `q_l2norm`/`k_l2norm`/`v_l2norm`, "
            "the shared `shortconv_*` knobs, and `attention_norm_eps` instead."
        )

    legacy_rank_keys = ("q_rank", "rank")
    forbidden_legacy_rank_keys = [key for key in legacy_rank_keys if key in values_dict]
    if forbidden_legacy_rank_keys:
        raise ValueError(
            f"Unsupported GPT config key(s) {forbidden_legacy_rank_keys}; the legacy attention rank knobs were removed."
        )

    if _is_mha_deltanet_module(config_cls) and "delta_use_gate" in values_dict:
        raise ValueError("Unsupported GPT config key(s) ['delta_use_gate']; use 'use_output_gate'.")

    if "fast_weight_lambda" in values_dict and not _is_fla_module(config_cls):
        raise ValueError(
            "Unsupported GPT config key(s) ['fast_weight_lambda']; "
            "use 'fast_weight_lambda_min' and 'fast_weight_lambda_max'."
        )

    _raise_unsupported_renamed_config_keys(
        values_dict,
        {"lambda_scale": "fast_weight_lambda_scale"},
    )

    if "fast_weight_alpha" in values_dict:
        raise ValueError(
            "Unsupported GPT config key(s) ['fast_weight_alpha']; "
            "the FALCON-3A sliding-state *_v2 variants that used this key were removed."
        )

    if _is_fla_module(config_cls):
        forbidden_fla_lambda_keys = [
            key
            for key in (
                "fast_weight_lambda",
                "fast_weight_lambda_min",
                "fast_weight_lambda_max",
                "fast_weight_lambda_scale",
            )
            if key in values_dict
        ]
        if forbidden_fla_lambda_keys:
            raise ValueError(
                f"Unsupported GPT config key(s) {forbidden_fla_lambda_keys} for FLA variants; "
                "FLA models do not implement lambda decay."
            )

    model = _build_config_model(config_cls)
    validated = model.model_validate(values_dict)
    dumped = validated.model_dump()
    for key in _PRESERVED_PRETRAINED_CONFIG_KWARGS:
        _install_preserved_pretrained_config_property(config_cls, key)
    for key in _PRESERVED_PRETRAINED_CONFIG_KWARGS:
        private_key = f"{_PRESERVED_PRETRAINED_CONFIG_PREFIX}{key}"
        if key in values_dict or private_key in values_dict:
            dumped[private_key] = values_dict.get(key, values_dict.get(private_key))
            dumped.pop(key, None)
    return dumped

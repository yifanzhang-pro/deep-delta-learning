from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import types
from typing import Any, Annotated, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr, create_model


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
    raw_annotations = config_cls.__dict__.get("__annotations__", {})
    if not raw_annotations:
        return create_model(
            f"{config_cls.__module__}.{config_cls.__qualname__}ConfigModel",
            __config__=ConfigDict(extra="allow"),
        )

    resolved = get_type_hints(config_cls, include_extras=True)
    fields: dict[str, tuple[Any, object]] = {}
    for name in raw_annotations:
        if name not in resolved:
            continue
        default = getattr(config_cls, name, ...)
        fields[name] = (_to_pydantic_type(resolved[name]), default)

    return create_model(
        f"{config_cls.__module__}.{config_cls.__qualname__}ConfigModel",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


def validate_pretrained_config_kwargs(config_cls: type[object], values: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate a Hugging Face `PretrainedConfig` kwarg dict using a generated Pydantic model.

    Notes:
    - Known config fields (declared via type annotations on `config_cls`) are validated.
    - Extra keys are allowed and passed through unchanged, mirroring `PretrainedConfig` behavior.
    """
    size_alias_keys = ("n_embd", "n_embd_base", "d_ff_factor")
    forbidden_size_aliases = [key for key in size_alias_keys if key in values]
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
    )
    forbidden_deprecated_delta_keys = [key for key in deprecated_delta_keys if key in values]
    if forbidden_deprecated_delta_keys:
        raise ValueError(
            f"Unsupported GPT config key(s) {forbidden_deprecated_delta_keys}; "
            "the deprecated DeltaNet `delta_*` knobs were removed. "
            "Use the shared `shortconv_*` knobs and `attention_norm_eps`, or switch to a newer DeltaNet variant."
        )

    legacy_rank_keys = ("q_rank", "rank")
    forbidden_legacy_rank_keys = [key for key in legacy_rank_keys if key in values]
    if forbidden_legacy_rank_keys:
        raise ValueError(
            f"Unsupported GPT config key(s) {forbidden_legacy_rank_keys}; the legacy attention rank knobs were removed."
        )
    model = _build_config_model(config_cls)
    validated = model.model_validate(dict(values))
    return validated.model_dump()

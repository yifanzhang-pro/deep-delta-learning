"""Small config parsing helpers shared by model modules."""

from __future__ import annotations

from numbers import Integral
from typing import Any


def optional_bool(config: Any, field_name: str, default: bool) -> bool:
    value = getattr(config, field_name, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean, got {value!r}.")
    return value


def optional_float(config: Any, field_name: str, default: float) -> float:
    value = getattr(config, field_name, default)
    if value is None:
        return default
    return float(value)


def optional_int(config: Any, field_name: str, default: int) -> int:
    value = getattr(config, field_name, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be an integer, got {value!r}.")
    return int(value)


def required_int(config: Any, field_name: str) -> int:
    value = getattr(config, field_name, None)
    if value is None:
        raise ValueError(f"{field_name} must be set.")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be an integer, got {value!r}.")
    return int(value)


__all__ = ["optional_bool", "optional_float", "optional_int", "required_int"]

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Final

import torch

_TRITON_MODULE_PREFIX: Final[str] = "triton"
_TRITON_UNAVAILABILITY_MARKERS: Final[tuple[str, ...]] = (
    "triton is not available",
    "no module named 'triton'",
    "requires cuda",
    "not implemented for 'cpu'",
    "found no nvidia driver",
    "cuda driver version is insufficient",
    "cuda driver initialization failed",
    "failed call to cuinit",
    "no cuda gpus are available",
    "no cuda-capable device is detected",
    "0 active drivers",
    "there should only be one",
    "no active triton target available",
    "libcuda.so cannot found",
    "cannot locate libamdhip64.so",
    "failed to open libcuda.so.1",
    "failed to retrieve culaunchkernelex from libcuda.so.1",
)


def is_missing_triton_module(exc: ModuleNotFoundError) -> bool:
    missing_name = exc.name
    if missing_name is not None:
        return missing_name == _TRITON_MODULE_PREFIX or missing_name.startswith(f"{_TRITON_MODULE_PREFIX}.")
    return "no module named 'triton'" in str(exc).lower()


def is_triton_unavailable_message(message: str) -> bool:
    msg = message.lower()
    return any(marker in msg for marker in _TRITON_UNAVAILABILITY_MARKERS)


def is_triton_unavailable_error(exc: Exception) -> bool:
    return is_triton_unavailable_message(str(exc))


def is_triton_driver_unavailable_error(exc: Exception) -> bool:
    return is_triton_unavailable_error(exc)


def _torch_cuda_is_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except AssertionError, RuntimeError:
        return False


def _probe_triton_driver_availability(triton_module: ModuleType) -> bool:
    runtime = getattr(triton_module, "runtime", None)
    driver = getattr(runtime, "driver", None) if runtime is not None else None
    active_driver = getattr(driver, "active", None) if driver is not None else None
    if active_driver is None:
        return True

    probe = getattr(active_driver, "get_current_target", None)
    if not callable(probe):
        probe = getattr(active_driver, "get_benchmarker", None)
    if not callable(probe):
        return True

    result = probe()
    return result is not None


def import_triton_modules(*, require_cuda: bool = True) -> tuple[ModuleType | None, ModuleType | None, bool]:
    if require_cuda and not _torch_cuda_is_available():
        return None, None, False

    try:
        triton_module = importlib.import_module("triton")
        tl_module = importlib.import_module("triton.language")
    except ModuleNotFoundError as exc:
        if not is_missing_triton_module(exc):
            raise
        return None, None, False
    try:
        if not _probe_triton_driver_availability(triton_module):
            return None, None, False
    except Exception as exc:
        if not is_triton_driver_unavailable_error(exc):
            raise
        return None, None, False
    return triton_module, tl_module, True


__all__ = [
    "import_triton_modules",
    "is_missing_triton_module",
    "is_triton_driver_unavailable_error",
    "is_triton_unavailable_error",
    "is_triton_unavailable_message",
]

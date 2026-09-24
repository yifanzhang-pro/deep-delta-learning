# pyright: reportMissingImports=false, reportInvalidTypeForm=false
from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, TypeAlias, cast

import torch
from .utils.triton_import_utils import import_triton_modules

if TYPE_CHECKING:
    import triton
    import triton.language as tl


def _next_power_of_2(value: int) -> int:
    value_int = int(value)
    if value_int <= 1:
        return 1
    return 1 << (value_int - 1).bit_length()


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-int(numerator) // int(denominator))


class _MissingTritonKernel:
    def __init__(self, name: str) -> None:
        self._name = name

    def __getitem__(self, _grid: object) -> Callable[..., None]:
        def _launch(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(f"Triton is not available; kernel {self._name!r} cannot run in this environment.")

        return _launch


class _MissingTritonModule:
    @staticmethod
    def jit(fn: Callable[..., Any]) -> _MissingTritonKernel:
        return _MissingTritonKernel(fn.__name__)

    @staticmethod
    def next_power_of_2(value: int) -> int:
        return _next_power_of_2(value)

    @staticmethod
    def cdiv(numerator: int, denominator: int) -> int:
        return _ceil_div(numerator, denominator)


class _MissingTritonLanguage:
    constexpr = Any
    float32 = float
    int64 = int

    @staticmethod
    def range(*args: Any) -> range:
        return range(*[int(arg) for arg in args])

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"Triton is not available; attempted to access triton.language.{name}.")


_TRITON_AVAILABLE = False
_triton_module, _tl_module, _TRITON_AVAILABLE = import_triton_modules()
if _TRITON_AVAILABLE and not torch.cuda.is_available():
    _TRITON_AVAILABLE = False
if not _TRITON_AVAILABLE or _triton_module is None or _tl_module is None:
    triton = cast(Any, _MissingTritonModule())
    tl = cast(Any, _MissingTritonLanguage())
else:
    triton = cast(Any, _triton_module)
    tl = cast(Any, _tl_module)

_ShortConvCacheKey = tuple[int, int, bool, str, int]
_SHORTCONV_KERNEL_INDEX_CACHE: dict[_ShortConvCacheKey, tuple[torch.Tensor, torch.Tensor]] = {}
_FusedDeltaTuningProfile: TypeAlias = Literal["hopper_h200", "generic"]
_HOPPER_CUDA_MAJOR: Final[int] = 9


def shortconv_kernel_size(owner: object | None) -> int:
    if owner is None:
        return 0
    shortconv = getattr(owner, "shortconv", None)
    if shortconv is None:
        return 0
    kernel_size = getattr(shortconv, "kernel_size", 0)
    if isinstance(kernel_size, (list, tuple)):
        return int(kernel_size[0]) if len(kernel_size) > 0 else 0
    return int(kernel_size)


def _resolve_shortconv_cache_device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    if device.type == "cuda":
        try:
            return int(torch.cuda.current_device())
        except AssertionError, RuntimeError:
            return -1
    return -1


def _matmul_with_output_dtype(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    compute_dtype = torch.promote_types(lhs.dtype, rhs.dtype)
    out = torch.matmul(lhs.to(dtype=compute_dtype), rhs.to(dtype=compute_dtype))
    if out.dtype != output_dtype:
        out = out.to(dtype=output_dtype)
    return out


def validate_expanded_delta_inputs(
    *,
    module_name: str,
    x: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    context: torch.Tensor,
    hidden_size: int,
    value_channels: int,
) -> None:
    if x.ndim != 4:
        raise ValueError(f"{module_name} expected x with shape (B, T, d, d_v), got {tuple(x.shape)}.")
    if int(x.size(-2)) != hidden_size:
        raise ValueError(f"{module_name} expected x feature dim {hidden_size}, got {int(x.size(-2))}.")
    if int(x.size(-1)) != value_channels:
        raise ValueError(f"{module_name} expected x value channels {value_channels}, got {int(x.size(-1))}.")

    expected_shape = (int(x.size(0)), int(x.size(1)), hidden_size)
    for name, tensor in (("k_in", k_in), ("v_in", v_in), ("context", context)):
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{module_name} expected {name} with shape {expected_shape}, got {tuple(tensor.shape)}.")
        if tensor.device != x.device:
            raise ValueError(f"{module_name} expected {name} on device {x.device}, got {tensor.device}.")


def validate_vdim1_delta_inputs(
    *,
    module_name: str,
    x: torch.Tensor,
    k_in: torch.Tensor,
    context: torch.Tensor,
    hidden_size: int,
) -> None:
    if x.ndim != 3:
        raise ValueError(f"{module_name} expected x with shape (B, T, d), got {tuple(x.shape)}.")
    if int(x.size(-1)) != hidden_size:
        raise ValueError(f"{module_name} expected x feature dim {hidden_size}, got {int(x.size(-1))}.")

    expected_shape = (int(x.size(0)), int(x.size(1)), hidden_size)
    for name, tensor in (("k_in", k_in), ("context", context)):
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{module_name} expected {name} with shape {expected_shape}, got {tuple(tensor.shape)}.")
        if tensor.device != x.device:
            raise ValueError(f"{module_name} expected {name} on device {x.device}, got {tensor.device}.")


def build_shortconv_kernel_index_and_mask(
    *,
    value_channels: int,
    kernel_size: int,
    causal: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dv = int(value_channels)
    ks = int(kernel_size)
    if ks <= 0:
        raise ValueError(f"kernel_size must be positive, got {ks}.")
    if ks > dv:
        raise ValueError(f"kernel_size must be <= value_channels ({dv}), got {ks}.")

    device_index = _resolve_shortconv_cache_device_index(device)
    cache_key: _ShortConvCacheKey = (dv, ks, bool(causal), device.type, device_index)
    cached = _SHORTCONV_KERNEL_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    pad_total = ks - 1
    pad_left = pad_total if causal else pad_total // 2
    input_positions = torch.arange(dv, device=device)
    output_positions = torch.arange(dv, device=device)
    kernel_positions = input_positions[None, :] + pad_left - output_positions[:, None]
    valid = ((kernel_positions >= 0) & (kernel_positions < ks)).contiguous()
    kernel_index = kernel_positions.clamp(0, ks - 1).to(dtype=torch.long).contiguous()
    cached = (kernel_index, valid)
    _SHORTCONV_KERNEL_INDEX_CACHE[cache_key] = cached
    return cached


def _choose_fused_delta_block_d(hidden_size: int) -> int:
    d = int(hidden_size)
    if d <= 0:
        raise ValueError(f"hidden_size must be positive, got {d}.")
    return min(256, int(triton.next_power_of_2(d)))


def _choose_fused_delta_block_dv(value_channels: int) -> int:
    dv = int(value_channels)
    if dv <= 0:
        raise ValueError(f"value_channels must be positive, got {dv}.")
    return int(triton.next_power_of_2(dv))


def _choose_fused_delta_num_warps(block_d: int) -> int:
    block_d_int = int(block_d)
    if block_d_int <= 256:
        return 4
    return 8


def _fused_delta_tuning_profile(device: torch.device | None = None) -> _FusedDeltaTuningProfile:
    if device is None or device.type != "cuda":
        # Non-CUDA calls are dispatch unit tests that monkeypatch Triton kernels;
        # keep them pinned to the H200 profile used by the benchmark sweeps.
        return "hopper_h200"
    if not torch.cuda.is_available():
        return "generic"
    major, _minor = torch.cuda.get_device_capability(device)
    if int(major) >= _HOPPER_CUDA_MAJOR:
        return "hopper_h200"
    return "generic"


def _use_blocked_rank1_fused_delta(
    hidden_size: int,
    block_dv: int | None = None,
    num_tokens: int | None = None,
    device: torch.device | None = None,
) -> bool:
    d = int(hidden_size)
    tuning_profile = _fused_delta_tuning_profile(device)
    blocked_min_hidden_size = 1024 if tuning_profile == "hopper_h200" else 2048
    if d <= blocked_min_hidden_size:
        return False
    block_dv_int = None if block_dv is None else int(block_dv)
    use_h200_tuning = tuning_profile == "hopper_h200"
    # H200 sweeps found the full-hidden kernel still wins for DV=4 at these
    # power-of-two widths, while the blocked path helps nearby non-powers.
    # Keep the exceptions explicit so future retuning can update them in one
    # place instead of inferring them from the broader D > 1024 rule.
    if d == 2048:
        if use_h200_tuning and block_dv_int is not None and block_dv_int <= 4:
            return num_tokens is not None and int(num_tokens) >= 2048
        return False
    if use_h200_tuning and d == 4096 and block_dv_int is not None and block_dv_int <= 4:
        return False
    if d <= 2048 and d == int(triton.next_power_of_2(d)):
        return False
    return True


def _choose_rank1_fused_delta_block_d(
    hidden_size: int,
    block_dv: int,
    num_tokens: int | None = None,
    device: torch.device | None = None,
) -> int:
    d = int(hidden_size)
    if d <= 0:
        raise ValueError(f"hidden_size must be positive, got {d}.")
    block_dv_int = int(block_dv)
    if not _use_blocked_rank1_fused_delta(d, block_dv_int, num_tokens, device):
        return int(triton.next_power_of_2(d))
    if block_dv_int <= 4:
        n = None if num_tokens is None else int(num_tokens)
        if d == 1280:
            if n is not None and n <= 2048:
                return 512
            return 256
        if d == 1536:
            return 512
        if d == 2048:
            return 512
        if 3072 <= d < 4096:
            if n is not None and n >= 2048:
                return 512
            return 1024
        if d >= 4096:
            if n is not None:
                if n >= 4096:
                    return 128
                if n >= 3072:
                    return 256
                if n >= 2048:
                    return 512
            return 1024
        return 512
    if block_dv_int <= 8:
        if num_tokens is not None:
            n = int(num_tokens)
            if n >= 3072 and n < 4096:
                if d >= 3072:
                    return 64
                if d >= 1280:
                    return 256
            if n >= 2048:
                return 128
            if n <= 512:
                if d == 1536 or d >= 4096:
                    return 512
                if d >= 3072:
                    return 256
                return 128
            if n >= 1536 and d >= 1280:
                return 128
            if d >= 1280:
                return 256
        return 128
    return 256


def _choose_rank1_fused_delta_num_warps(
    block_d: int,
    block_dv: int,
    hidden_size: int | None = None,
    num_tokens: int | None = None,
) -> int:
    block_d_int = int(block_d)
    block_dv_int = int(block_dv)
    if block_dv_int <= 4:
        hidden_size_int = None if hidden_size is None else int(hidden_size)
        num_tokens_int = None if num_tokens is None else int(num_tokens)
        if block_d_int <= 256:
            if hidden_size_int == 1280 and num_tokens_int == 1024:
                return 4
            return 1
        if block_d_int == 512:
            if hidden_size_int == 1280 and num_tokens_int is not None and num_tokens_int <= 2048:
                return 8
            if hidden_size_int == 1536 and num_tokens_int is not None:
                if num_tokens_int >= 3072:
                    return 1
                if num_tokens_int >= 2048:
                    return 4
                return 8
            if hidden_size_int is not None and hidden_size_int >= 5120:
                if num_tokens_int is not None and num_tokens_int >= 2048:
                    return 1
            return 4
        if block_d_int == 1024:
            if num_tokens_int is not None and num_tokens_int <= 512 and hidden_size_int is not None:
                if hidden_size_int >= 5120:
                    return 4
            return 8
    if block_dv_int == 8:
        if block_d_int <= 128:
            return 1
        if block_d_int <= 512:
            return 4
    return _choose_fused_delta_num_warps(block_d)


def _build_new_weight_ShortConvFn(
    weight: torch.Tensor, kernel_size: int, value_channels: int, causal: bool
) -> torch.Tensor:
    dv = int(value_channels)
    ks = int(kernel_size)
    kernel_index, kernel_valid = build_shortconv_kernel_index_and_mask(
        value_channels=dv,
        kernel_size=ks,
        causal=causal,
        device=weight.device,
    )
    output_to_input_weight = weight[:, kernel_index] * kernel_valid[None, :, :]
    expected_shape = (weight.shape[0], dv, dv)
    if output_to_input_weight.shape != expected_shape:
        raise ValueError(
            f"Unexpected shortconv weight shape {output_to_input_weight.shape}, expected {expected_shape}."
        )
    return output_to_input_weight


def _build_w_eff_ShortConvFn(
    weight: torch.Tensor, read: torch.Tensor, kernel_size: int, value_channels: int, causal: bool
) -> torch.Tensor:
    output_to_input_weight = _build_new_weight_ShortConvFn(weight, kernel_size, value_channels, causal)
    w_eff = (output_to_input_weight * read[None, :, None]).sum(dim=1)  # (d, dv)
    return w_eff.contiguous()


@triton.jit
def _fwd_kernel_ShortConvFn(
    X_ptr,
    W_ptr,
    OUT_ptr,
    BT: tl.constexpr,
    T: tl.constexpr,
    BATCHED_W: tl.constexpr,
    D: tl.constexpr,
    DV,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_d = tl.program_id(1)
    weight_offset = (pid_bt // T) * D * DV if BATCHED_W else 0

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dv_offs = tl.arange(0, BLOCK_DV)
    d_mask = d_offs < D
    dv_mask = dv_offs < DV
    mask = d_mask[:, None] & dv_mask[None, :]

    w = tl.load(W_ptr + weight_offset + d_offs[:, None] * DV + dv_offs[None, :], mask=mask, other=0.0).to(tl.float32)
    x = tl.load(
        X_ptr + pid_bt * D * DV + d_offs[:, None] * DV + dv_offs[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    out = tl.sum(x * w, axis=1)
    tl.store(OUT_ptr + pid_bt * D + d_offs, out, mask=d_mask)


@triton.jit
def _bwd_kernel_ShortConvFn_dx(
    GRAD_OUT_ptr,
    W_ptr,
    DX_ptr,
    BT: tl.constexpr,
    T: tl.constexpr,
    BATCHED_W: tl.constexpr,
    D: tl.constexpr,
    DV,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_d = tl.program_id(1)
    weight_offset = (pid_bt // T) * D * DV if BATCHED_W else 0

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dv_offs = tl.arange(0, BLOCK_DV)
    d_mask = d_offs < D
    dv_mask = dv_offs < DV
    mask = d_mask[:, None] & dv_mask[None, :]

    grad_out = tl.load(GRAD_OUT_ptr + pid_bt * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + weight_offset + d_offs[:, None] * DV + dv_offs[None, :], mask=mask, other=0.0).to(tl.float32)
    dx = grad_out[:, None] * w
    tl.store(
        DX_ptr + pid_bt * D * DV + d_offs[:, None] * DV + dv_offs[None, :],
        dx,
        mask=mask,
    )


@triton.jit
def _bwd_kernel_ShortConvFn_dw(
    GRAD_OUT_ptr,
    X_ptr,
    DW_ptr,
    BT: tl.constexpr,
    T: tl.constexpr,
    BATCHED_W: tl.constexpr,
    D: tl.constexpr,
    DV,
    BLOCK_BT: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_d = tl.program_id(0) % D
    pid_dv = tl.program_id(1)
    pid_bt = tl.program_id(0) // D

    dv_offs = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    dv_mask = dv_offs < DV
    reduction_length = T if BATCHED_W else BT
    tiles = tl.cdiv(reduction_length, BLOCK_BT)
    sequence = pid_bt // tiles
    local_tokens = (pid_bt % tiles) * BLOCK_BT + tl.arange(0, BLOCK_BT)
    bt_offs = sequence * reduction_length + local_tokens
    bt_mask = local_tokens < reduction_length
    grad_out = tl.load(GRAD_OUT_ptr + bt_offs * D + pid_d, mask=bt_mask, other=0.0).to(tl.float32)
    x = tl.load(
        X_ptr + bt_offs[:, None] * D * DV + pid_d * DV + dv_offs[None, :],
        mask=bt_mask[:, None] & dv_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(grad_out[:, None] * x, axis=0)
    tl.store(DW_ptr + pid_bt * D * DV + pid_d * DV + dv_offs, acc, mask=dv_mask)


# Fixed time tiles and one reduction tree per sequence/parameter avoid atomic
# accumulation and keep read-gradient cancellation local to each sequence.
@triton.jit
def _reduce_shortconv_weight_partials(Partial, DW, N: tl.constexpr, P: tl.constexpr, BN: tl.constexpr):
    # Keep parameter count on grid X; valid wide models exceed the grid Y limit.
    sequence = tl.program_id(0) // P
    parameter = tl.program_id(0) % P
    tiles = tl.arange(0, BN)
    values = tl.load(Partial + (sequence * N + tiles) * P + parameter, tiles < N, other=0.0)
    tl.store(DW + sequence * P + parameter, tl.sum(values, axis=0))


class _ShortConvContext(Protocol):
    saved_tensors: tuple[torch.Tensor, ...]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _ShortConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _ShortConvContext, x_flat: torch.Tensor, w_eff: torch.Tensor) -> torch.Tensor:
        BT, D, DV = x_flat.shape
        batched_weights = w_eff.ndim == 3
        batch_size = w_eff.shape[0] if batched_weights else 1
        if BT % batch_size:
            raise ValueError("CC shortconv batched weights require equal-length sequences.")
        seq_len = BT // batch_size
        out = torch.empty(BT, D, dtype=x_flat.dtype, device=x_flat.device)
        BLOCK_D = 128
        BLOCK_DV = _choose_fused_delta_block_dv(DV)
        grid = (BT, triton.cdiv(D, BLOCK_D))
        if BT == 0:
            ctx.save_for_backward(x_flat, w_eff)
            return out
        cast(Any, _fwd_kernel_ShortConvFn)[grid](
            x_flat,
            w_eff,
            out,
            BT=BT,
            T=seq_len,
            BATCHED_W=batched_weights,
            D=D,
            DV=DV,
            BLOCK_D=BLOCK_D,
            BLOCK_DV=BLOCK_DV,
        )
        ctx.save_for_backward(x_flat, w_eff)
        return out

    @staticmethod
    def backward(ctx: _ShortConvContext, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_flat, w_eff = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        BT, D, DV = x_flat.shape
        batched_weights = w_eff.ndim == 3
        batch_size = w_eff.shape[0] if batched_weights else 1
        if BT % batch_size:
            raise ValueError("CC shortconv batched weights require equal-length sequences.")
        seq_len = BT // batch_size
        if BT == 0:
            return torch.zeros_like(x_flat), torch.zeros_like(w_eff)
        dx_flat = torch.empty_like(x_flat)
        dw_eff = torch.empty_like(w_eff, dtype=torch.float32)
        BLOCK_D = 128
        BLOCK_DV = _choose_fused_delta_block_dv(DV)
        BLOCK_BT = 64
        dx_grid = (BT, triton.cdiv(D, BLOCK_D))
        tiles = triton.cdiv(seq_len, BLOCK_BT)
        partial = torch.empty((batch_size, tiles, D * DV), device=w_eff.device, dtype=torch.float32)
        dw_grid = (batch_size * tiles * D, triton.cdiv(DV, BLOCK_DV))
        cast(Any, _bwd_kernel_ShortConvFn_dx)[dx_grid](
            grad_out,
            w_eff,
            dx_flat,
            BT=BT,
            T=seq_len,
            BATCHED_W=batched_weights,
            D=D,
            DV=DV,
            BLOCK_D=BLOCK_D,
            BLOCK_DV=BLOCK_DV,
        )
        cast(Any, _bwd_kernel_ShortConvFn_dw)[dw_grid](
            grad_out,
            x_flat,
            partial,
            BT=BT,
            T=seq_len,
            BATCHED_W=batched_weights,
            D=D,
            DV=DV,
            BLOCK_BT=BLOCK_BT,
            BLOCK_DV=BLOCK_DV,
        )
        cast(Any, _reduce_shortconv_weight_partials)[(batch_size * D * DV,)](
            partial, dw_eff, N=tiles, P=D * DV, BN=triton.next_power_of_2(tiles)
        )
        return dx_flat, dw_eff


@triton.jit
def _fwd_kernel_TemporalShortConvReadFn(
    X_ptr,
    W_ptr,
    OUT_ptr,
    B: tl.constexpr,
    BATCHED_W: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_d = tl.program_id(1)
    b = pid_bt // T
    t = pid_bt - b * T
    weight_offset = b * D * DV * K if BATCHED_W else 0

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dv_offs = tl.arange(0, BLOCK_DV)
    d_mask = d_offs < D
    dv_mask = dv_offs < DV
    matrix_mask = d_mask[:, None] & dv_mask[None, :]

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k in range(0, K):
        input_t = t + k - (K - 1)
        time_valid = (input_t >= 0) & (input_t < T)
        x = tl.load(
            X_ptr + ((b * T + input_t) * D + d_offs[:, None]) * DV + dv_offs[None, :],
            mask=matrix_mask & time_valid,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            W_ptr + weight_offset + (d_offs[:, None] * DV + dv_offs[None, :]) * K + k,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(x * w, axis=1)

    tl.store(OUT_ptr + (b * T + t) * D + d_offs, acc, mask=d_mask)


@triton.jit
def _bwd_kernel_TemporalShortConvReadFn_dx(
    GRAD_OUT_ptr,
    W_ptr,
    DX_ptr,
    B: tl.constexpr,
    BATCHED_W: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_d = tl.program_id(1)
    b = pid_bt // T
    t = pid_bt - b * T
    weight_offset = b * D * DV * K if BATCHED_W else 0

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dv_offs = tl.arange(0, BLOCK_DV)
    d_mask = d_offs < D
    dv_mask = dv_offs < DV
    matrix_mask = d_mask[:, None] & dv_mask[None, :]

    acc = tl.zeros((BLOCK_D, BLOCK_DV), dtype=tl.float32)
    for k in range(0, K):
        out_t = t - k + (K - 1)
        time_valid = (out_t >= 0) & (out_t < T)
        grad_out = tl.load(
            GRAD_OUT_ptr + (b * T + out_t) * D + d_offs,
            mask=d_mask & time_valid,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            W_ptr + weight_offset + (d_offs[:, None] * DV + dv_offs[None, :]) * K + k,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        acc += grad_out[:, None] * w

    tl.store(
        DX_ptr + ((b * T + t) * D + d_offs[:, None]) * DV + dv_offs[None, :],
        acc,
        mask=matrix_mask,
    )


@triton.jit
def _bwd_kernel_TemporalShortConvReadFn_dw(
    GRAD_OUT_ptr,
    X_ptr,
    DW_ptr,
    B: tl.constexpr,
    BATCHED_W: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    K: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_dk = tl.program_id(0) % (D * K)
    pid_d = pid_dk // K
    k = pid_dk - pid_d * K
    pid_dv = tl.program_id(1)
    pid_bt = tl.program_id(0) // (D * K)

    dv_offs = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    dv_mask = dv_offs < DV
    reduction_length = T if BATCHED_W else B * T
    tiles = tl.cdiv(reduction_length, BLOCK_BT)
    sequence = pid_bt // tiles
    local_tokens = (pid_bt % tiles) * BLOCK_BT + tl.arange(0, BLOCK_BT)
    bt_offs = sequence * reduction_length + local_tokens
    bt_mask = local_tokens < reduction_length
    b = bt_offs // T
    out_t = bt_offs - b * T
    grad_out = tl.load(GRAD_OUT_ptr + bt_offs * D + pid_d, mask=bt_mask, other=0.0).to(tl.float32)
    input_t = out_t + k - (K - 1)
    time_valid = (input_t >= 0) & (input_t < T)
    x = tl.load(
        X_ptr + ((b[:, None] * T + input_t[:, None]) * D + pid_d) * DV + dv_offs[None, :],
        mask=bt_mask[:, None] & time_valid[:, None] & dv_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    dw_acc = tl.sum(grad_out[:, None] * x, axis=0)

    tl.store(
        DW_ptr + pid_bt * D * DV * K + (pid_d * DV + dv_offs) * K + k,
        dw_acc,
        mask=dv_mask,
    )


class _TemporalShortConvReadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: _ShortConvContext, x: torch.Tensor, w_eff: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size, value_channels = x.shape
        kernel_size = int(w_eff.shape[-1])
        batched_weights = w_eff.ndim == 4
        x_contiguous = x.contiguous()
        w_contiguous = w_eff.contiguous()
        out = torch.empty((batch_size, seq_len, hidden_size), dtype=x.dtype, device=x.device)
        block_d = 64
        block_dv = _choose_fused_delta_block_dv(value_channels)
        grid = (batch_size * seq_len, triton.cdiv(hidden_size, block_d))
        cast(Any, _fwd_kernel_TemporalShortConvReadFn)[grid](
            x_contiguous,
            w_contiguous,
            out,
            B=batch_size,
            BATCHED_W=batched_weights,
            T=seq_len,
            D=hidden_size,
            DV=value_channels,
            K=kernel_size,
            BLOCK_D=block_d,
            BLOCK_DV=block_dv,
        )
        ctx.save_for_backward(x_contiguous, w_contiguous)
        return out

    @staticmethod
    def backward(ctx: _ShortConvContext, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, w_eff = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        batch_size, seq_len, hidden_size, value_channels = x.shape
        kernel_size = int(w_eff.shape[-1])
        batched_weights = w_eff.ndim == 4
        dx = torch.empty_like(x)
        dw_eff = torch.empty_like(w_eff, dtype=torch.float32)
        block_d = 64
        block_dv = _choose_fused_delta_block_dv(value_channels)
        block_bt = 64
        dx_grid = (batch_size * seq_len, triton.cdiv(hidden_size, block_d))
        groups = batch_size if batched_weights else 1
        tiles = triton.cdiv(seq_len if batched_weights else batch_size * seq_len, block_bt)
        parameters = hidden_size * value_channels * kernel_size
        partial = torch.empty((groups, tiles, parameters), device=x.device, dtype=torch.float32)
        dw_grid = (groups * tiles * hidden_size * kernel_size, triton.cdiv(value_channels, block_dv))
        cast(Any, _bwd_kernel_TemporalShortConvReadFn_dx)[dx_grid](
            grad_out,
            w_eff,
            dx,
            B=batch_size,
            BATCHED_W=batched_weights,
            T=seq_len,
            D=hidden_size,
            DV=value_channels,
            K=kernel_size,
            BLOCK_D=block_d,
            BLOCK_DV=block_dv,
        )
        cast(Any, _bwd_kernel_TemporalShortConvReadFn_dw)[dw_grid](
            grad_out,
            x,
            partial,
            B=batch_size,
            BATCHED_W=batched_weights,
            T=seq_len,
            D=hidden_size,
            DV=value_channels,
            K=kernel_size,
            BLOCK_BT=block_bt,
            BLOCK_DV=block_dv,
        )
        cast(Any, _reduce_shortconv_weight_partials)[(groups * parameters,)](
            partial, dw_eff, N=tiles, P=parameters, BN=triton.next_power_of_2(tiles)
        )
        return dx, dw_eff


@triton.jit
def _fwd_kernel_TemporalShortConvExpandFn(
    X_ptr,
    W_ptr,
    OUT_ptr,
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_dv = tl.program_id(2)
    b = pid_bt // T
    t = pid_bt - b * T

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dv_offs = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    d_mask = d_offs < D
    dv_mask = dv_offs < DV
    matrix_mask = d_mask[:, None] & dv_mask[None, :]

    acc = tl.zeros((BLOCK_D, BLOCK_DV), dtype=tl.float32)
    for k in range(0, K):
        input_t = t + k - (K - 1)
        time_valid = (input_t >= 0) & (input_t < T)
        x = tl.load(
            X_ptr + (b * T + input_t) * D + d_offs,
            mask=d_mask & time_valid,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            W_ptr + (d_offs[:, None] * DV + dv_offs[None, :]) * K + k,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        acc += x[:, None] * w

    tl.store(
        OUT_ptr + ((b * T + t) * D + d_offs[:, None]) * DV + dv_offs[None, :],
        acc,
        mask=matrix_mask,
    )


@triton.jit
def _bwd_kernel_TemporalShortConvExpandFn_dx(
    GRAD_OUT_ptr,
    W_ptr,
    DX_ptr,
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_d = tl.program_id(1)
    b = pid_bt // T
    t = pid_bt - b * T

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    dv_offs = tl.arange(0, BLOCK_DV)
    d_mask = d_offs < D
    dv_mask = dv_offs < DV
    matrix_mask = d_mask[:, None] & dv_mask[None, :]

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k in range(0, K):
        out_t = t - k + (K - 1)
        time_valid = (out_t >= 0) & (out_t < T)
        grad_out = tl.load(
            GRAD_OUT_ptr + ((b * T + out_t) * D + d_offs[:, None]) * DV + dv_offs[None, :],
            mask=matrix_mask & time_valid,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            W_ptr + (d_offs[:, None] * DV + dv_offs[None, :]) * K + k,
            mask=matrix_mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(grad_out * w, axis=1)

    tl.store(DX_ptr + (b * T + t) * D + d_offs, acc, mask=d_mask)


@triton.jit
def _bwd_kernel_TemporalShortConvExpandFn_dw(
    GRAD_OUT_ptr,
    X_ptr,
    DW_ptr,
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    K: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_dk = tl.program_id(0)
    pid_d = pid_dk // K
    k = pid_dk - pid_d * K
    pid_dv = tl.program_id(1)
    pid_bt = tl.program_id(2)

    dv_offs = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    dv_mask = dv_offs < DV
    bt_offs = pid_bt * BLOCK_BT + tl.arange(0, BLOCK_BT)
    bt_mask = bt_offs < (B * T)
    b = bt_offs // T
    out_t = bt_offs - b * T
    grad_out = tl.load(
        GRAD_OUT_ptr + ((bt_offs[:, None] * D + pid_d) * DV + dv_offs[None, :]),
        mask=bt_mask[:, None] & dv_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    input_t = out_t + k - (K - 1)
    time_valid = (input_t >= 0) & (input_t < T)
    x = tl.load(
        X_ptr + (b * T + input_t) * D + pid_d,
        mask=bt_mask & time_valid,
        other=0.0,
    ).to(tl.float32)
    dw_acc = tl.sum(x[:, None] * grad_out, axis=0)

    tl.atomic_add(
        DW_ptr + (pid_d * DV + dv_offs) * K + k,
        dw_acc,
        sem="relaxed",
        mask=dv_mask,
    )


class _TemporalShortConvExpandFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        batch_size, seq_len, hidden_size = x.shape
        value_channels = int(weight.shape[-2])
        kernel_size = int(weight.shape[-1])
        x_contiguous = x.contiguous()
        weight_contiguous = weight.contiguous()
        out = torch.empty((batch_size, seq_len, hidden_size, value_channels), dtype=x.dtype, device=x.device)
        block_d = 64
        block_dv = _choose_fused_delta_block_dv(value_channels)
        grid = (batch_size * seq_len, triton.cdiv(hidden_size, block_d), triton.cdiv(value_channels, block_dv))
        cast(Any, _fwd_kernel_TemporalShortConvExpandFn)[grid](
            x_contiguous,
            weight_contiguous,
            out,
            B=batch_size,
            T=seq_len,
            D=hidden_size,
            DV=value_channels,
            K=kernel_size,
            BLOCK_D=block_d,
            BLOCK_DV=block_dv,
        )
        ctx.save_for_backward(x_contiguous, weight_contiguous)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        batch_size, seq_len, hidden_size = x.shape
        value_channels = int(weight.shape[-2])
        kernel_size = int(weight.shape[-1])
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight, dtype=torch.float32)
        dw.zero_()
        block_d = 64
        block_dv = _choose_fused_delta_block_dv(value_channels)
        block_bt = 64
        dx_grid = (batch_size * seq_len, triton.cdiv(hidden_size, block_d))
        dw_grid = (
            hidden_size * kernel_size,
            triton.cdiv(value_channels, block_dv),
            triton.cdiv(batch_size * seq_len, block_bt),
        )
        cast(Any, _bwd_kernel_TemporalShortConvExpandFn_dx)[dx_grid](
            grad_out,
            weight,
            dx,
            B=batch_size,
            T=seq_len,
            D=hidden_size,
            DV=value_channels,
            K=kernel_size,
            BLOCK_D=block_d,
            BLOCK_DV=block_dv,
        )
        cast(Any, _bwd_kernel_TemporalShortConvExpandFn_dw)[dw_grid](
            grad_out,
            x,
            dw,
            B=batch_size,
            T=seq_len,
            D=hidden_size,
            DV=value_channels,
            K=kernel_size,
            BLOCK_BT=block_bt,
            BLOCK_DV=block_dv,
        )
        return dx, dw


@triton.jit
def fwd_delta_kernel(
    X_ptr,
    K_ptr,
    V_in_ptr,
    C_in_ptr,
    W_v_ptr,
    B_v_ptr,
    W_beta_ptr,
    B_beta_ptr,
    X_new_ptr,
    Khat_ptr,
    Delta_v_ptr,
    Inv_norm_ptr,
    V_ptr,
    Beta_ptr,
    N,
    D,
    DV,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    EPS: tl.constexpr,
    V_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    off_d = tl.arange(0, BLOCK_D)
    off_dv = tl.arange(0, BLOCK_DV)
    d_mask = off_d < D
    dv_mask = off_dv < DV
    matrix_mask = d_mask[:, None] & dv_mask[None, :]

    x_ptrs = X_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
    k_ptrs = K_ptr + pid * D + off_d
    v_in_ptrs = V_in_ptr + pid * D + off_d
    c_in_ptrs = C_in_ptr + pid * D + off_d

    # Full-hidden path invariant: dispatch only calls this kernel when
    # BLOCK_D >= D, so this single reduction covers the full hidden dimension.
    # The tiled D > BLOCK_D case uses `fwd_delta_blocked_kernel` below.
    k = tl.load(k_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    norm_sq = tl.sum(k * k) + EPS * EPS
    inv_norm = 1.0 / tl.sqrt(norm_sq)
    k_hat = k * inv_norm
    tl.store(Inv_norm_ptr + pid, inv_norm)
    tl.store(Khat_ptr + pid * D + off_d, k_hat, mask=d_mask)

    c_in = tl.load(c_in_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    w_beta = tl.load(W_beta_ptr + off_d, mask=d_mask, other=0.0).to(tl.float32)
    b_beta = tl.load(B_beta_ptr).to(tl.float32)
    beta_logit = tl.sum(c_in * w_beta) + b_beta
    beta = 2.0 * tl.sigmoid(beta_logit)
    tl.store(Beta_ptr + pid, beta)

    v_in = tl.load(v_in_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    w_v_ptrs = W_v_ptr + off_dv[:, None] * D + off_d[None, :]
    w_v = tl.load(w_v_ptrs, mask=dv_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    b_v = tl.load(B_v_ptr + off_dv, mask=dv_mask, other=0.0).to(tl.float32)
    v_logits = tl.sum(w_v * v_in[None, :], axis=1) + b_v
    v = tl.sigmoid(v_logits) * V_SCALE
    tl.store(V_ptr + pid * DV + off_dv, v, mask=dv_mask)

    x = tl.load(x_ptrs, mask=matrix_mask, other=0.0).to(tl.float32)
    proj = tl.sum(k_hat[:, None] * x, axis=0)
    delta_v = v - proj
    tl.store(Delta_v_ptr + pid * DV + off_dv, delta_v, mask=dv_mask)

    update = beta * k_hat[:, None] * delta_v[None, :]
    x_new = x + update
    tl.store(X_new_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :], x_new, mask=matrix_mask)


@triton.jit
def bwd_delta_kernel(
    dY_ptr,
    X_ptr,
    Khat_ptr,
    Delta_v_ptr,
    Inv_norm_ptr,
    V_ptr,
    Beta_ptr,
    dX_ptr,
    dK_ptr,
    dV_logits_ptr,
    dBeta_logits_ptr,
    N,
    D,
    DV,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    V_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    off_d = tl.arange(0, BLOCK_D)
    off_dv = tl.arange(0, BLOCK_DV)
    d_mask = off_d < D
    dv_mask = off_dv < DV
    matrix_mask = d_mask[:, None] & dv_mask[None, :]

    dy_ptrs = dY_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
    x_ptrs = X_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
    khat_ptrs = Khat_ptr + pid * D + off_d

    dy = tl.load(dy_ptrs, mask=matrix_mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptrs, mask=matrix_mask, other=0.0).to(tl.float32)
    khat = tl.load(khat_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    delta_v = tl.load(Delta_v_ptr + pid * DV + off_dv, mask=dv_mask, other=0.0).to(tl.float32)
    inv_norm = tl.load(Inv_norm_ptr + pid).to(tl.float32)
    v = tl.load(V_ptr + pid * DV + off_dv, mask=dv_mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + pid).to(tl.float32)

    # Full-hidden path invariant: BLOCK_D >= D, matching the forward kernel
    # that stored a full-hidden k-hat and inv_norm for this token.
    dk_hat_proj = tl.sum(dy * khat[:, None], axis=0)

    dv = beta * dk_hat_proj
    v_sig = v / V_SCALE
    dv_logits = dv * v_sig * (1.0 - v_sig) * V_SCALE
    tl.store(dV_logits_ptr + pid * DV + off_dv, dv_logits, mask=dv_mask)

    dbeta = tl.sum(dk_hat_proj * delta_v)
    beta_sig = beta / 2.0
    dbeta_logits = dbeta * beta_sig * (1.0 - beta_sig) * 2.0
    tl.store(dBeta_logits_ptr + pid, dbeta_logits)

    dx = dy - beta * khat[:, None] * dk_hat_proj[None, :]
    tl.store(dX_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :], dx, mask=matrix_mask)

    dy_dot_delta_v = tl.sum(dy * delta_v[None, :], axis=1)
    x_dot_dk_hat_proj = tl.sum(x * dk_hat_proj[None, :], axis=1)
    dkhat = beta * (dy_dot_delta_v - x_dot_dk_hat_proj)
    khat_dot_dkhat = tl.sum(khat * dkhat)
    dk = inv_norm * (dkhat - khat * khat_dot_dkhat)
    tl.store(dK_ptr + pid * D + off_d, dk, mask=d_mask)


@triton.jit
def fwd_delta_blocked_kernel(
    X_ptr,
    K_ptr,
    V_in_ptr,
    C_in_ptr,
    W_v_ptr,
    B_v_ptr,
    W_beta_ptr,
    B_beta_ptr,
    X_new_ptr,
    Khat_ptr,
    Delta_v_ptr,
    Inv_norm_ptr,
    V_ptr,
    Beta_ptr,
    N,
    D,
    DV,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    EPS: tl.constexpr,
    V_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    off_dv = tl.arange(0, BLOCK_DV)
    dv_mask = off_dv < DV
    b_v = tl.load(B_v_ptr + off_dv, mask=dv_mask, other=0.0).to(tl.float32)
    b_beta = tl.load(B_beta_ptr).to(tl.float32)

    norm_sq = tl.zeros((), dtype=tl.float32) + EPS * EPS
    beta_logit = b_beta
    v_logits = b_v

    # Blocked path: accumulate normalization and projection logits across every
    # hidden tile before deriving inv_norm, beta, and v.
    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D

        k = tl.load(K_ptr + pid * D + off_d, mask=d_mask, other=0.0).to(tl.float32)
        norm_sq += tl.sum(k * k)

        c_in = tl.load(C_in_ptr + pid * D + off_d, mask=d_mask, other=0.0).to(tl.float32)
        w_beta = tl.load(W_beta_ptr + off_d, mask=d_mask, other=0.0).to(tl.float32)
        beta_logit += tl.sum(c_in * w_beta)

        v_in = tl.load(V_in_ptr + pid * D + off_d, mask=d_mask, other=0.0).to(tl.float32)
        w_v_ptrs = W_v_ptr + off_dv[:, None] * D + off_d[None, :]
        w_v = tl.load(w_v_ptrs, mask=dv_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_logits += tl.sum(w_v * v_in[None, :], axis=1)

    inv_norm = 1.0 / tl.sqrt(norm_sq)
    tl.store(Inv_norm_ptr + pid, inv_norm)

    beta = 2.0 * tl.sigmoid(beta_logit)
    tl.store(Beta_ptr + pid, beta)

    v = tl.sigmoid(v_logits) * V_SCALE
    tl.store(V_ptr + pid * DV + off_dv, v, mask=dv_mask)

    proj = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D
        x_ptrs = X_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        k = tl.load(K_ptr + pid * D + off_d, mask=d_mask, other=0.0).to(tl.float32)
        k_hat = k * inv_norm
        tl.store(Khat_ptr + pid * D + off_d, k_hat, mask=d_mask)
        x = tl.load(x_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        proj += tl.sum(k_hat[:, None] * x, axis=0)

    delta_v = v - proj
    tl.store(Delta_v_ptr + pid * DV + off_dv, delta_v, mask=dv_mask)

    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D
        x_ptrs = X_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        x_new_ptrs = X_new_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        k = tl.load(K_ptr + pid * D + off_d, mask=d_mask, other=0.0).to(tl.float32)
        khat = k * inv_norm
        x = tl.load(x_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        update = beta * khat[:, None] * delta_v[None, :]
        x_new = x + update
        tl.store(x_new_ptrs, x_new, mask=d_mask[:, None] & dv_mask[None, :])


@triton.jit
def bwd_delta_blocked_kernel(
    dY_ptr,
    X_ptr,
    Khat_ptr,
    Delta_v_ptr,
    Inv_norm_ptr,
    V_ptr,
    Beta_ptr,
    dX_ptr,
    dK_ptr,
    dV_logits_ptr,
    dBeta_logits_ptr,
    N,
    D,
    DV,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    V_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    off_dv = tl.arange(0, BLOCK_DV)
    dv_mask = off_dv < DV
    delta_v = tl.load(Delta_v_ptr + pid * DV + off_dv, mask=dv_mask, other=0.0).to(tl.float32)
    inv_norm = tl.load(Inv_norm_ptr + pid).to(tl.float32)
    v = tl.load(V_ptr + pid * DV + off_dv, mask=dv_mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + pid).to(tl.float32)

    dk_hat_proj = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D
        dy_ptrs = dY_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        khat_ptrs = Khat_ptr + pid * D + off_d

        dy = tl.load(dy_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        khat = tl.load(khat_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        dk_hat_proj += tl.sum(dy * khat[:, None], axis=0)

    dv = beta * dk_hat_proj
    v_sig = v / V_SCALE
    dv_logits = dv * v_sig * (1.0 - v_sig) * V_SCALE
    tl.store(dV_logits_ptr + pid * DV + off_dv, dv_logits, mask=dv_mask)

    dbeta = tl.sum(dk_hat_proj * delta_v)
    beta_sig = beta / 2.0
    dbeta_logits = dbeta * beta_sig * (1.0 - beta_sig) * 2.0
    tl.store(dBeta_logits_ptr + pid, dbeta_logits)

    khat_dot_dkhat = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D
        dy_ptrs = dY_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        x_ptrs = X_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        khat_ptrs = Khat_ptr + pid * D + off_d

        dy = tl.load(dy_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        khat = tl.load(khat_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        dx = dy - beta * khat[:, None] * dk_hat_proj[None, :]
        tl.store(
            dX_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :],
            dx,
            mask=d_mask[:, None] & dv_mask[None, :],
        )

        dy_dot_delta_v = tl.sum(dy * delta_v[None, :], axis=1)
        x_dot_dk_hat_proj = tl.sum(x * dk_hat_proj[None, :], axis=1)
        dkhat = beta * (dy_dot_delta_v - x_dot_dk_hat_proj)
        khat_dot_dkhat += tl.sum(khat * dkhat)

    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D
        dy_ptrs = dY_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        x_ptrs = X_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        khat_ptrs = Khat_ptr + pid * D + off_d

        dy = tl.load(dy_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        khat = tl.load(khat_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        dy_dot_delta_v = tl.sum(dy * delta_v[None, :], axis=1)
        x_dot_dk_hat_proj = tl.sum(x * dk_hat_proj[None, :], axis=1)
        dkhat = beta * (dy_dot_delta_v - x_dot_dk_hat_proj)
        dk = inv_norm * (dkhat - khat * khat_dot_dkhat)
        tl.store(dK_ptr + pid * D + off_d, dk, mask=d_mask)


class FusedDeepDeltaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k_in, v_in, context, W_v, b_v, W_beta, b_beta, config_k_eps, config_v_scale):
        B, T, D, DV = x.shape
        N = B * T
        BLOCK_DV = _choose_fused_delta_block_dv(DV)
        BLOCK_D = _choose_rank1_fused_delta_block_d(D, BLOCK_DV, N, x.device)
        use_blocked_delta = _use_blocked_rank1_fused_delta(D, BLOCK_DV, N, x.device)
        num_warps = (
            _choose_rank1_fused_delta_num_warps(BLOCK_D, BLOCK_DV, D, N)
            if use_blocked_delta
            else _choose_fused_delta_num_warps(BLOCK_D)
        )

        x_flat = x.reshape(N, D, DV).contiguous()
        k_flat = k_in.reshape(N, D).contiguous()
        v_in_flat = v_in.reshape(N, D).contiguous()
        c_in_flat = context.reshape(N, D).contiguous()

        x_new = torch.empty_like(x_flat)
        khat = torch.empty_like(k_flat)
        delta_v = torch.empty((N, DV), device=x.device, dtype=torch.float32)
        inv_norm = torch.empty((N,), device=x.device, dtype=torch.float32)
        v_val = torch.empty((N, DV), device=x.device, dtype=torch.float32)
        beta_val = torch.empty((N,), device=x.device, dtype=torch.float32)

        eps_adj = float(config_k_eps)

        fwd_kernel = fwd_delta_blocked_kernel if use_blocked_delta else fwd_delta_kernel
        cast(Any, fwd_kernel)[(N,)](
            x_flat,
            k_flat,
            v_in_flat,
            c_in_flat,
            W_v,
            b_v,
            W_beta,
            b_beta,
            x_new,
            khat,
            delta_v,
            inv_norm,
            v_val,
            beta_val,
            N,
            D,
            DV,
            BLOCK_D,
            BLOCK_DV,
            EPS=eps_adj,
            V_SCALE=config_v_scale,
            num_warps=num_warps,
        )

        ctx.save_for_backward(
            x_flat,
            khat,
            delta_v,
            inv_norm,
            v_val,
            beta_val,
            v_in_flat,
            c_in_flat,
            W_v,
            b_v,
            W_beta,
            b_beta,
        )
        ctx.D = D
        ctx.DV = DV
        ctx.BLOCK_D = BLOCK_D
        ctx.BLOCK_DV = BLOCK_DV
        ctx.V_SCALE = config_v_scale
        ctx.USE_BLOCKED_DELTA = use_blocked_delta
        ctx.B, ctx.T = B, T

        return x_new.view(B, T, D, DV)

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, khat, delta_v, inv_norm, v_val, beta_val, v_in_flat, c_in_flat, W_v, b_v, W_beta, b_beta = (
            ctx.saved_tensors
        )
        N = x_flat.shape[0]

        grad_output_flat = grad_output.reshape(N, ctx.D, ctx.DV).contiguous()

        dx = torch.empty_like(x_flat)
        dk = torch.empty_like(khat)
        dv_logits = torch.empty((N, ctx.DV), device=x_flat.device, dtype=torch.float32)
        dbeta_logits = torch.empty((N,), device=x_flat.device, dtype=torch.float32)
        use_blocked_delta = getattr(ctx, "USE_BLOCKED_DELTA", False)
        num_warps = (
            _choose_rank1_fused_delta_num_warps(ctx.BLOCK_D, ctx.BLOCK_DV, ctx.D, N)
            if use_blocked_delta
            else _choose_fused_delta_num_warps(ctx.BLOCK_D)
        )

        bwd_kernel = bwd_delta_blocked_kernel if use_blocked_delta else bwd_delta_kernel
        cast(Any, bwd_kernel)[(N,)](
            grad_output_flat,
            x_flat,
            khat,
            delta_v,
            inv_norm,
            v_val,
            beta_val,
            dx,
            dk,
            dv_logits,
            dbeta_logits,
            N,
            ctx.D,
            ctx.DV,
            ctx.BLOCK_D,
            ctx.BLOCK_DV,
            V_SCALE=ctx.V_SCALE,
            num_warps=num_warps,
        )

        dW_v = _matmul_with_output_dtype(dv_logits.t(), v_in_flat, output_dtype=W_v.dtype)
        db_v = dv_logits.sum(dim=0).to(dtype=b_v.dtype)
        dv_in = _matmul_with_output_dtype(dv_logits, W_v, output_dtype=v_in_flat.dtype).view(ctx.B, ctx.T, ctx.D)

        dbeta_logits_unsqueezed = dbeta_logits.unsqueeze(1)
        dW_beta = _matmul_with_output_dtype(dbeta_logits_unsqueezed.t(), c_in_flat, output_dtype=W_beta.dtype)
        db_beta = dbeta_logits.sum(dim=0, keepdim=True).to(dtype=b_beta.dtype)
        dcontext = _matmul_with_output_dtype(
            dbeta_logits_unsqueezed,
            W_beta,
            output_dtype=c_in_flat.dtype,
        ).view(ctx.B, ctx.T, ctx.D)

        return (
            dx.view(ctx.B, ctx.T, ctx.D, ctx.DV),
            dk.view(ctx.B, ctx.T, ctx.D),
            dv_in,
            dcontext,
            dW_v,
            db_v,
            dW_beta,
            db_beta,
            None,
            None,
        )


@triton.jit
def fwd_delta_lambda_kernel(
    X_ptr,
    K_ptr,
    V_in_ptr,
    C_in_ptr,
    W_v_ptr,
    B_v_ptr,
    W_beta_ptr,
    B_beta_ptr,
    W_lambda_ptr,
    B_lambda_ptr,
    X_new_ptr,
    Khat_ptr,
    Resid_ptr,
    Inv_norm_ptr,
    V_ptr,
    Beta_ptr,
    LambdaBar_ptr,
    KNorm2_ptr,
    Eta_ptr,
    Gamma_ptr,
    RawAlpha_ptr,
    N,
    D,
    DV,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    EPS_NORM: tl.constexpr,
    K_EPS: tl.constexpr,
    V_SCALE: tl.constexpr,
    LAMBDA_SCALE: tl.constexpr,
    GAMMA_EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    off_dv = tl.arange(0, BLOCK_DV)
    dv_mask = off_dv < DV
    b_v = tl.load(B_v_ptr + off_dv, mask=dv_mask, other=0.0).to(tl.float32)
    b_beta = tl.load(B_beta_ptr).to(tl.float32)
    b_lambda = tl.load(B_lambda_ptr).to(tl.float32)

    norm_sq = tl.zeros((), dtype=tl.float32) + EPS_NORM * EPS_NORM
    beta_logit = b_beta
    lambda_logit = b_lambda
    v_logits = b_v

    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D

        k = tl.load(K_ptr + pid * D + off_d, mask=d_mask, other=0.0).to(tl.float32)
        norm_sq += tl.sum(k * k)

        c_in = tl.load(C_in_ptr + pid * D + off_d, mask=d_mask, other=0.0).to(tl.float32)
        w_beta = tl.load(W_beta_ptr + off_d, mask=d_mask, other=0.0).to(tl.float32)
        beta_logit += tl.sum(c_in * w_beta)

        w_lambda = tl.load(W_lambda_ptr + off_d, mask=d_mask, other=0.0).to(tl.float32)
        lambda_logit += tl.sum(c_in * w_lambda)

        v_in = tl.load(V_in_ptr + pid * D + off_d, mask=d_mask, other=0.0).to(tl.float32)
        w_v_ptrs = W_v_ptr + off_dv[:, None] * D + off_d[None, :]
        w_v = tl.load(w_v_ptrs, mask=dv_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_logits += tl.sum(w_v * v_in[None, :], axis=1)

    inv_norm = 1.0 / tl.sqrt(norm_sq)
    tl.store(Inv_norm_ptr + pid, inv_norm)

    beta = 2.0 / (1.0 + tl.exp(-beta_logit))
    tl.store(Beta_ptr + pid, beta)

    if LAMBDA_SCALE > 0.0:
        lambda_bar = (1.0 / (1.0 + tl.exp(-lambda_logit))) * LAMBDA_SCALE
    else:
        lambda_bar = tl.zeros((), dtype=tl.float32)
    tl.store(LambdaBar_ptr + pid, lambda_bar)

    v = (1.0 / (1.0 + tl.exp(-v_logits))) * V_SCALE
    tl.store(V_ptr + pid * DV + off_dv, v, mask=dv_mask)

    proj = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    k_norm2 = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D
        x_ptrs = X_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        k = tl.load(K_ptr + pid * D + off_d, mask=d_mask, other=0.0).to(tl.float32)
        k_hat = k * inv_norm
        k_norm2 += tl.sum(k_hat * k_hat, axis=0)
        tl.store(Khat_ptr + pid * D + off_d, k_hat, mask=d_mask)
        x = tl.load(x_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        proj += tl.sum(k_hat[:, None] * x, axis=0)

    tl.store(KNorm2_ptr + pid, k_norm2)

    lambda_eff = lambda_bar * k_norm2
    eta = beta / (k_norm2 + lambda_eff + K_EPS)
    raw_alpha = eta * lambda_eff
    if raw_alpha <= 1.0 - GAMMA_EPS:
        gamma = 1.0 - raw_alpha
    else:
        gamma = GAMMA_EPS
    eta_hat = eta / gamma

    tl.store(Eta_ptr + pid, eta)
    tl.store(Gamma_ptr + pid, gamma)
    tl.store(RawAlpha_ptr + pid, raw_alpha)

    resid = v - proj
    tl.store(Resid_ptr + pid * DV + off_dv, resid, mask=dv_mask)

    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D
        x_ptrs = X_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        x_new_ptrs = X_new_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        khat = tl.load(Khat_ptr + pid * D + off_d, mask=d_mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        update = eta_hat * khat[:, None] * resid[None, :]
        x_new = gamma * (x + update)
        tl.store(x_new_ptrs, x_new, mask=d_mask[:, None] & dv_mask[None, :])


@triton.jit
def bwd_delta_lambda_kernel(
    dY_ptr,
    X_ptr,
    Khat_ptr,
    Resid_ptr,
    Inv_norm_ptr,
    V_ptr,
    Beta_ptr,
    LambdaBar_ptr,
    KNorm2_ptr,
    Eta_ptr,
    Gamma_ptr,
    RawAlpha_ptr,
    dX_ptr,
    dK_ptr,
    dV_logits_ptr,
    dBeta_logits_ptr,
    dLambda_logits_ptr,
    N,
    D,
    DV,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    K_EPS: tl.constexpr,
    V_SCALE: tl.constexpr,
    LAMBDA_SCALE: tl.constexpr,
    GAMMA_EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    off_dv = tl.arange(0, BLOCK_DV)
    dv_mask = off_dv < DV
    resid = tl.load(Resid_ptr + pid * DV + off_dv, mask=dv_mask, other=0.0).to(tl.float32)
    inv_norm = tl.load(Inv_norm_ptr + pid).to(tl.float32)
    v = tl.load(V_ptr + pid * DV + off_dv, mask=dv_mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + pid).to(tl.float32)
    lambda_bar = tl.load(LambdaBar_ptr + pid).to(tl.float32)
    k_norm2 = tl.load(KNorm2_ptr + pid).to(tl.float32)
    eta = tl.load(Eta_ptr + pid).to(tl.float32)
    gamma = tl.load(Gamma_ptr + pid).to(tl.float32)
    raw_alpha = tl.load(RawAlpha_ptr + pid).to(tl.float32)

    eta_hat = eta / gamma
    lambda_eff = lambda_bar * k_norm2
    denom = k_norm2 + lambda_eff + K_EPS

    q = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    x_dot = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D
        dy_ptrs = dY_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        x_ptrs = X_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        khat_ptrs = Khat_ptr + pid * D + off_d

        dy = tl.load(dy_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        khat = tl.load(khat_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        q += tl.sum(dy * khat[:, None], axis=0)
        x_dot += tl.sum(dy * x)

    qr = tl.sum(q * resid)
    dgamma = x_dot + eta_hat * qr
    deta_hat = gamma * qr

    if raw_alpha <= 1.0 - GAMMA_EPS:
        deta = (deta_hat / (gamma * gamma)) - dgamma * lambda_eff
        dlambda_eff = (deta_hat * eta_hat * eta_hat) - dgamma * eta
    else:
        deta = deta_hat / gamma
        dlambda_eff = tl.zeros((), dtype=tl.float32)

    dbeta = deta / denom
    dden = -deta * eta / denom
    dlambda_eff_total = dlambda_eff + dden
    dlambda_bar = dlambda_eff_total * k_norm2
    dknorm2 = dden + dlambda_eff_total * lambda_bar

    dv = eta * q
    v_sig = v / V_SCALE
    dv_logits = dv * v_sig * (1.0 - v_sig) * V_SCALE
    tl.store(dV_logits_ptr + pid * DV + off_dv, dv_logits, mask=dv_mask)

    beta_sig = beta / 2.0
    dbeta_logits = dbeta * beta_sig * (1.0 - beta_sig) * 2.0
    tl.store(dBeta_logits_ptr + pid, dbeta_logits)

    if LAMBDA_SCALE > 0.0:
        lambda_sig = lambda_bar / LAMBDA_SCALE
        dlambda_logits = dlambda_bar * lambda_sig * (1.0 - lambda_sig) * LAMBDA_SCALE
    else:
        dlambda_logits = tl.zeros((), dtype=tl.float32)
    tl.store(dLambda_logits_ptr + pid, dlambda_logits)

    khat_dot_dkhat = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D
        dy_ptrs = dY_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        x_ptrs = X_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        khat_ptrs = Khat_ptr + pid * D + off_d

        dy = tl.load(dy_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        khat = tl.load(khat_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        dx = gamma * (dy - eta_hat * khat[:, None] * q[None, :])
        tl.store(
            dX_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :],
            dx,
            mask=d_mask[:, None] & dv_mask[None, :],
        )

        u = tl.sum(dy * resid[None, :], axis=1)
        xq = tl.sum(x * q[None, :], axis=1)
        dkhat = gamma * eta_hat * (u - xq) + (2.0 * dknorm2) * khat
        khat_dot_dkhat += tl.sum(khat * dkhat)

    for d_start in range(0, D, BLOCK_D):
        off_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = off_d < D
        dy_ptrs = dY_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        x_ptrs = X_ptr + pid * D * DV + off_d[:, None] * DV + off_dv[None, :]
        khat_ptrs = Khat_ptr + pid * D + off_d

        dy = tl.load(dy_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=d_mask[:, None] & dv_mask[None, :], other=0.0).to(tl.float32)
        khat = tl.load(khat_ptrs, mask=d_mask, other=0.0).to(tl.float32)

        u = tl.sum(dy * resid[None, :], axis=1)
        xq = tl.sum(x * q[None, :], axis=1)
        dkhat = gamma * eta_hat * (u - xq) + (2.0 * dknorm2) * khat
        dk = inv_norm * (dkhat - khat * khat_dot_dkhat)
        tl.store(dK_ptr + pid * D + off_d, dk, mask=d_mask)


class FusedDeepDeltaLambdaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        context: torch.Tensor,
        W_v: torch.Tensor,
        b_v: torch.Tensor,
        W_beta: torch.Tensor,
        b_beta: torch.Tensor,
        W_lambda: torch.Tensor,
        b_lambda: torch.Tensor,
        config_k_eps: float,
        config_v_scale: float,
        config_lambda_scale: float,
        config_gamma_eps: float,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size, value_channels = x.shape
        num_tokens = batch_size * seq_len
        block_d = _choose_fused_delta_block_d(hidden_size)
        block_dv = _choose_fused_delta_block_dv(value_channels)
        num_warps = _choose_fused_delta_num_warps(block_d)

        x_flat = x.reshape(num_tokens, hidden_size, value_channels).contiguous()
        k_flat = k_in.reshape(num_tokens, hidden_size).contiguous()
        v_in_flat = v_in.reshape(num_tokens, hidden_size).contiguous()
        c_in_flat = context.reshape(num_tokens, hidden_size).contiguous()

        x_new = torch.empty_like(x_flat)
        khat = torch.empty_like(k_flat)
        resid = torch.empty((num_tokens, value_channels), device=x.device, dtype=torch.float32)
        inv_norm = torch.empty((num_tokens,), device=x.device, dtype=torch.float32)
        v_val = torch.empty((num_tokens, value_channels), device=x.device, dtype=torch.float32)
        beta_val = torch.empty((num_tokens,), device=x.device, dtype=torch.float32)
        lambda_bar_val = torch.empty((num_tokens,), device=x.device, dtype=torch.float32)
        k_norm2_val = torch.empty((num_tokens,), device=x.device, dtype=torch.float32)
        eta_val = torch.empty((num_tokens,), device=x.device, dtype=torch.float32)
        gamma_val = torch.empty((num_tokens,), device=x.device, dtype=torch.float32)
        raw_alpha_val = torch.empty((num_tokens,), device=x.device, dtype=torch.float32)

        eps_norm = float(config_k_eps)

        cast(Any, fwd_delta_lambda_kernel)[(num_tokens,)](
            x_flat,
            k_flat,
            v_in_flat,
            c_in_flat,
            W_v,
            b_v,
            W_beta,
            b_beta,
            W_lambda,
            b_lambda,
            x_new,
            khat,
            resid,
            inv_norm,
            v_val,
            beta_val,
            lambda_bar_val,
            k_norm2_val,
            eta_val,
            gamma_val,
            raw_alpha_val,
            num_tokens,
            hidden_size,
            value_channels,
            block_d,
            block_dv,
            EPS_NORM=eps_norm,
            K_EPS=float(config_k_eps),
            V_SCALE=float(config_v_scale),
            LAMBDA_SCALE=float(config_lambda_scale),
            GAMMA_EPS=float(config_gamma_eps),
            num_warps=num_warps,
        )

        ctx.save_for_backward(
            x_flat,
            khat,
            resid,
            inv_norm,
            v_val,
            beta_val,
            lambda_bar_val,
            k_norm2_val,
            eta_val,
            gamma_val,
            raw_alpha_val,
            v_in_flat,
            c_in_flat,
            W_v,
            b_v,
            W_beta,
            b_beta,
            W_lambda,
            b_lambda,
        )
        ctx.D = hidden_size
        ctx.DV = value_channels
        ctx.BLOCK_D = block_d
        ctx.BLOCK_DV = block_dv
        ctx.K_EPS = float(config_k_eps)
        ctx.V_SCALE = float(config_v_scale)
        ctx.LAMBDA_SCALE = float(config_lambda_scale)
        ctx.GAMMA_EPS = float(config_gamma_eps)
        ctx.B = batch_size
        ctx.T = seq_len

        return x_new.view(batch_size, seq_len, hidden_size, value_channels)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (
            x_flat,
            khat,
            resid,
            inv_norm,
            v_val,
            beta_val,
            lambda_bar_val,
            k_norm2_val,
            eta_val,
            gamma_val,
            raw_alpha_val,
            v_in_flat,
            c_in_flat,
            W_v,
            b_v,
            W_beta,
            b_beta,
            W_lambda,
            b_lambda,
        ) = ctx.saved_tensors
        num_tokens = x_flat.shape[0]

        grad_output_flat = grad_output.reshape(num_tokens, ctx.D, ctx.DV).contiguous()

        dx = torch.empty_like(x_flat)
        dk = torch.empty_like(khat)
        dv_logits = torch.empty((num_tokens, ctx.DV), device=x_flat.device, dtype=torch.float32)
        dbeta_logits = torch.empty((num_tokens,), device=x_flat.device, dtype=torch.float32)
        dlambda_logits = torch.empty((num_tokens,), device=x_flat.device, dtype=torch.float32)
        num_warps = _choose_fused_delta_num_warps(ctx.BLOCK_D)

        cast(Any, bwd_delta_lambda_kernel)[(num_tokens,)](
            grad_output_flat,
            x_flat,
            khat,
            resid,
            inv_norm,
            v_val,
            beta_val,
            lambda_bar_val,
            k_norm2_val,
            eta_val,
            gamma_val,
            raw_alpha_val,
            dx,
            dk,
            dv_logits,
            dbeta_logits,
            dlambda_logits,
            num_tokens,
            ctx.D,
            ctx.DV,
            ctx.BLOCK_D,
            ctx.BLOCK_DV,
            K_EPS=ctx.K_EPS,
            V_SCALE=ctx.V_SCALE,
            LAMBDA_SCALE=ctx.LAMBDA_SCALE,
            GAMMA_EPS=ctx.GAMMA_EPS,
            num_warps=num_warps,
        )

        dW_v = _matmul_with_output_dtype(dv_logits.t(), v_in_flat, output_dtype=W_v.dtype)
        db_v = dv_logits.sum(dim=0).to(dtype=b_v.dtype)
        dv_in = _matmul_with_output_dtype(dv_logits, W_v, output_dtype=v_in_flat.dtype).view(ctx.B, ctx.T, ctx.D)

        dbeta_logits_unsqueezed = dbeta_logits.unsqueeze(1)
        dW_beta = _matmul_with_output_dtype(dbeta_logits_unsqueezed.t(), c_in_flat, output_dtype=W_beta.dtype)
        db_beta = dbeta_logits.sum(dim=0, keepdim=True).to(dtype=b_beta.dtype)

        dlambda_logits_unsqueezed = dlambda_logits.unsqueeze(1)
        dW_lambda = _matmul_with_output_dtype(dlambda_logits_unsqueezed.t(), c_in_flat, output_dtype=W_lambda.dtype)
        db_lambda = dlambda_logits.sum(dim=0, keepdim=True).to(dtype=b_lambda.dtype)

        dcontext = (
            _matmul_with_output_dtype(dbeta_logits_unsqueezed, W_beta, output_dtype=c_in_flat.dtype)
            + _matmul_with_output_dtype(dlambda_logits_unsqueezed, W_lambda, output_dtype=c_in_flat.dtype)
        ).view(ctx.B, ctx.T, ctx.D)

        return (
            dx.view(ctx.B, ctx.T, ctx.D, ctx.DV),
            dk.view(ctx.B, ctx.T, ctx.D),
            dv_in,
            dcontext,
            dW_v,
            db_v,
            dW_beta,
            db_beta,
            dW_lambda,
            db_lambda,
            None,
            None,
            None,
            None,
        )

"""Shared MHA+RoPE attention block."""

from __future__ import annotations

from typing import Any, Callable, Final, Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segmented_attention import SegmentedSelfAttention
from .packed_rotary_attention import packed_rotary_attention
from ..gpt_base import PastKeyValue
from .activations import ActivationName, apply_activation, validate_activation_name
from .attention_mask_utils import numeric_mask_binary_flags
from .config_utils import optional_bool, optional_float, optional_int, required_int
from .initialization import residual_output_projection_depth_divisor, residual_output_projection_std
from .kv_cache import append_preallocated, get_past_len, maybe_get_cache_len, narrow_cache_to_past_len
from .rmsnorm import RMSNorm
from .rng import cuda_rng_devices_for_module
from .rotary import Rotary, apply_rotary_emb


_POSITION_ID_DTYPES: Final[tuple[torch.dtype, ...]] = (torch.int32, torch.int64)
_ADDITIVE_MASK_SENTINEL: Final[int] = -10000


def _make_shift_linear(*args: Any, **kwargs: Any) -> nn.Module:
    from .kv_shift import ShiftLinear

    return ShiftLinear(*args, **kwargs)


def _get_kv_shift_states_or_empty(
    past_key_value: PastKeyValue | None,
    *,
    use_k_shift: bool,
    use_v_shift: bool,
    k_state_shape: tuple[int, int],
    v_state_shape: tuple[int, int],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not use_k_shift and not use_v_shift:
        return None, None
    from .kv_shift import get_kv_shift_states

    return get_kv_shift_states(
        past_key_value,
        use_k_shift=use_k_shift,
        use_v_shift=use_v_shift,
        k_state_shape=k_state_shape,
        v_state_shape=v_state_shape,
    )


def _project_with_optional_shift_state(
    projection: nn.Module,
    x: torch.Tensor,
    *,
    use_shift: bool,
    shift_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not use_shift:
        return projection(x), None
    from .kv_shift import project_with_optional_shift_state

    return project_with_optional_shift_state(projection, x, use_shift=use_shift, shift_state=shift_state)


def _append_kv_shift_states_or_cache_len(
    items: list[torch.Tensor],
    *,
    k_shift_state: torch.Tensor | None,
    v_shift_state: torch.Tensor | None,
    cache_len: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if k_shift_state is None and v_shift_state is None:
        items.append(cache_len)
        return tuple(items)
    from .kv_shift import append_kv_shift_states

    return append_kv_shift_states(
        items,
        k_shift_state=k_shift_state,
        v_shift_state=v_shift_state,
        cache_len=cache_len,
    )


def _past_len_all_zero(past_len: int | torch.Tensor) -> bool:
    if isinstance(past_len, torch.Tensor):
        if past_len.ndim != 1:
            raise ValueError(f"past_len tensor must have shape (B,), got {tuple(past_len.shape)}")
        if past_len.device.type != "cpu":
            return False
        return bool(torch.all(past_len == 0).item())
    return past_len == 0


def _past_len_any_positive(past_len: int | torch.Tensor, *, allow_sync: bool = False) -> bool:
    if isinstance(past_len, torch.Tensor):
        if past_len.ndim != 1:
            raise ValueError(f"past_len tensor must have shape (B,), got {tuple(past_len.shape)}")
        if past_len.device.type != "cpu" and not allow_sync:
            return False
        return bool(torch.any(past_len > 0).item())
    return past_len > 0


def _uniform_past_len_int(past_len: int | torch.Tensor) -> int:
    if isinstance(past_len, torch.Tensor):
        if past_len.ndim != 1:
            raise ValueError(f"past_len tensor must have shape (B,), got {tuple(past_len.shape)}")
        if past_len.numel() == 0:
            raise ValueError("past_len tensor must have at least one element.")
        if past_len.device.type != "cpu":
            raise ValueError("past_len tensor must be CPU metadata for query-specific position validation.")
        first_tensor = past_len[0]
        if bool((past_len != first_tensor).any().item()):
            raise ValueError("past_len tensor must be uniform for query-specific position validation.")
        past_len_value = int(first_tensor.item())
    else:
        past_len_value = past_len
    if past_len_value < 0:
        raise ValueError(f"past_len must be non-negative, got {past_len_value}.")
    return past_len_value


def _validate_position_ids(
    position_ids: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    allow_value_sync: bool = False,
) -> None:
    if position_ids.shape != (batch_size, seq_len):
        raise ValueError(f"position_ids must have shape (B, Q), got {tuple(position_ids.shape)}")
    if position_ids.dtype not in _POSITION_ID_DTYPES:
        raise ValueError(f"position_ids must be an int32 or int64 tensor, got dtype={position_ids.dtype}.")
    if position_ids.device.type != "cpu" and not allow_value_sync:
        return
    if position_ids.device.type == "meta":
        raise ValueError("position_ids on meta device cannot be materialized.")
    if position_ids.numel() > 0 and bool((position_ids < 0).any().item()):
        raise ValueError("position_ids must be non-negative.")


def _is_binary_mask_on_cpu(
    mask: torch.Tensor,
    *,
    require_nonzero: bool = False,
    allow_zero_rows_when_mixed_binary: bool = False,
) -> bool:
    if mask.device.type != "cpu":
        return False
    is_binary = numeric_mask_binary_flags(
        mask,
        require_nonzero=require_nonzero,
        allow_zero_rows_when_mixed_binary=allow_zero_rows_when_mixed_binary,
        keepdim=True,
    )
    return bool(is_binary.all().item())


def _numeric_attention_mask_to_bias(
    mask: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    require_nonzero_binary_mask: bool = False,
    allow_zero_rows_when_mixed_binary: bool = False,
) -> torch.Tensor:
    attention_bias = mask.to(dtype=target_dtype)
    binary_bias = torch.zeros_like(attention_bias)
    # Use -inf so numeric binary masks match boolean SDPA masking exactly.
    # Finite sentinels can leak probability mass on long or fully masked rows.
    binary_bias = binary_bias.masked_fill(mask == 0, -torch.inf)
    # Keep the predicate as a tensor so accelerator masks stay device-side and
    # so mixed numeric batches can preserve additive rows independently.
    is_binary_mask = numeric_mask_binary_flags(
        mask,
        require_nonzero=require_nonzero_binary_mask,
        allow_zero_rows_when_mixed_binary=allow_zero_rows_when_mixed_binary,
        keepdim=True,
    )
    return torch.where(is_binary_mask, binary_bias, attention_bias)


def _query_positions_from_past_len(
    past_len: int | torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    offsets = torch.arange(seq_len, device=device)
    if isinstance(past_len, torch.Tensor):
        if past_len.ndim != 1 or int(past_len.shape[0]) != batch_size:
            raise ValueError(f"past_len tensor must have shape ({batch_size},), got {tuple(past_len.shape)}")
        if past_len.device != device:
            if past_len.device.type == "meta":
                raise ValueError("past_len tensor on meta device cannot be materialized.")
            past_len = past_len.to(device=device)
        return past_len[:, None] + offsets[None, :]
    if past_len < 0:
        raise ValueError(f"past_len must be non-negative, got {past_len}.")
    return (past_len + offsets).unsqueeze(0).expand(batch_size, seq_len)


def _query_specific_position_boundary_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    total_len: int,
    num_heads: int,
    past_len: int | torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor | None:
    if attention_mask is None or attention_mask.ndim not in (3, 4):
        return None
    past_len_value = _uniform_past_len_int(past_len)
    if torch.is_complex(attention_mask):
        return None
    if int(attention_mask.shape[0]) != batch_size:
        raise ValueError(f"attention_mask batch mismatch: expected {batch_size}, got {int(attention_mask.shape[0])}")
    if int(attention_mask.shape[-1]) != total_len:
        raise ValueError(
            f"attention_mask key sequence mismatch: expected {total_len}, got {int(attention_mask.shape[-1])}"
        )
    if attention_mask.ndim == 3:
        if int(attention_mask.shape[1]) != seq_len:
            raise ValueError(
                f"attention_mask query sequence mismatch: expected {seq_len}, got {int(attention_mask.shape[1])}"
            )
        mask = attention_mask
        head_dim: int | None = None
    else:
        if int(attention_mask.shape[-2]) != seq_len:
            raise ValueError(
                f"attention_mask query sequence mismatch: expected {seq_len}, got {int(attention_mask.shape[-2])}"
            )
        if int(attention_mask.shape[1]) not in (1, num_heads):
            raise ValueError(
                f"attention_mask head dimension must be 1 or {num_heads}, got {int(attention_mask.shape[1])}"
            )
        mask = attention_mask
        head_dim = 1

    previous_position_ids = torch.empty_like(position_ids)
    previous_position_ids[:, :1] = past_len_value - 1
    if seq_len > 1:
        previous_position_ids[:, 1:] = position_ids[:, :-1]
    segment_starts = position_ids != previous_position_ids + 1
    query_key_positions = past_len_value + torch.arange(seq_len, device=position_ids.device, dtype=position_ids.dtype)
    start_candidates = torch.where(
        segment_starts,
        query_key_positions[None, :].expand(batch_size, seq_len),
        torch.zeros((batch_size, seq_len), device=position_ids.device, dtype=position_ids.dtype),
    )
    segment_start_keys = torch.cummax(start_candidates, dim=1).values.to(device=attention_mask.device)
    key_positions = torch.arange(total_len, device=attention_mask.device, dtype=segment_start_keys.dtype)
    prefix_selector = key_positions[None, None, :] < segment_start_keys[:, :, None]
    if head_dim is not None:
        prefix_selector = prefix_selector[:, None, :, :]

    if mask.dtype == torch.bool:
        per_head_blocks = ~torch.any(mask & prefix_selector, dim=-1)
        if head_dim is None:
            return per_head_blocks
        return torch.all(per_head_blocks, dim=head_dim)

    binary_rows = numeric_mask_binary_flags(
        mask,
        require_nonzero=True,
        allow_zero_rows_when_mixed_binary=True,
        keepdim=False,
    )
    binary_blocks = binary_rows & ~torch.any((mask != 0) & prefix_selector, dim=-1)
    # Query-specific additive masks can also express a packed boundary when
    # every prior key in the current packed segment receives the same large
    # negative sentinel used elsewhere in this codebase for additive masking.
    supports_additive_sentinel = mask.is_floating_point() or mask.dtype in (torch.int16, torch.int32, torch.int64)
    additive_blocks = torch.zeros_like(binary_blocks)
    if supports_additive_sentinel:
        additive_blocks = ~binary_rows & torch.all(
            torch.where(prefix_selector, mask <= _ADDITIVE_MASK_SENTINEL, True),
            dim=-1,
        )
    per_head_blocks = binary_blocks | additive_blocks
    if head_dim is None:
        return per_head_blocks
    return torch.all(per_head_blocks, dim=head_dim)


def _canonicalize_attention_mask(
    attention_mask: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    total_len: int,
    num_heads: int,
    target_device: torch.device,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    if attention_mask.ndim not in (2, 3, 4):
        raise ValueError(
            f"attention_mask must have shape (B, S), (B, Q, S), or (B, H, Q, S), got {tuple(attention_mask.shape)}"
        )
    if torch.is_complex(attention_mask):
        raise ValueError(f"attention_mask must be bool, integer, or floating, got dtype={attention_mask.dtype}.")
    if int(attention_mask.shape[0]) != batch_size:
        raise ValueError(f"attention_mask batch mismatch: expected {batch_size}, got {int(attention_mask.shape[0])}")
    if int(attention_mask.shape[-1]) != total_len:
        raise ValueError(
            f"attention_mask key sequence mismatch: expected {total_len}, got {int(attention_mask.shape[-1])}"
        )
    if attention_mask.ndim == 2:
        mask = attention_mask[:, None, None, :]
    elif attention_mask.ndim == 3:
        if int(attention_mask.shape[1]) != seq_len:
            raise ValueError(
                f"attention_mask query sequence mismatch: expected {seq_len}, got {int(attention_mask.shape[1])}"
            )
        mask = attention_mask[:, None, :, :]
    else:
        if int(attention_mask.shape[-2]) != seq_len:
            raise ValueError(
                f"attention_mask query sequence mismatch: expected {seq_len}, got {int(attention_mask.shape[-2])}"
            )
        if int(attention_mask.shape[1]) not in (1, num_heads):
            raise ValueError(
                f"attention_mask head dimension must be 1 or {num_heads}, got {int(attention_mask.shape[1])}"
            )
        mask = attention_mask

    # By contract, 2D numeric 0/1 masks are token padding masks, so an all-zero
    # row represents a fully padded sequence. Query-specific 3D/4D numeric masks
    # may be additive biases, so they require a nonzero binary row before using
    # boolean-mask semantics.
    require_nonzero_binary_mask = attention_mask.ndim > 2
    allow_zero_rows_when_mixed_binary = attention_mask.ndim > 2
    is_binary_cpu_mask = _is_binary_mask_on_cpu(
        mask,
        require_nonzero=require_nonzero_binary_mask,
        allow_zero_rows_when_mixed_binary=allow_zero_rows_when_mixed_binary,
    )

    if mask.is_floating_point():
        if is_binary_cpu_mask:
            return mask.to(device=target_device, dtype=torch.bool, non_blocking=True)
        if mask.device != target_device:
            if mask.device.type == "meta":
                raise ValueError("attention_mask on meta device cannot be materialized.")
            mask = mask.to(device=target_device, non_blocking=True)
        return _numeric_attention_mask_to_bias(
            mask,
            target_dtype=target_dtype,
            require_nonzero_binary_mask=require_nonzero_binary_mask,
            allow_zero_rows_when_mixed_binary=allow_zero_rows_when_mixed_binary,
        )
    if mask.dtype == torch.bool:
        if mask.device != target_device:
            if mask.device.type == "meta":
                raise ValueError("attention_mask on meta device cannot be materialized.")
            mask = mask.to(device=target_device, non_blocking=True)
        return mask
    if is_binary_cpu_mask:
        return mask.to(device=target_device, dtype=torch.bool, non_blocking=True)
    if mask.device != target_device:
        if mask.device.type == "meta":
            raise ValueError("attention_mask on meta device cannot be materialized.")
        mask = mask.to(device=target_device, non_blocking=True)
    return _numeric_attention_mask_to_bias(
        mask,
        target_dtype=target_dtype,
        require_nonzero_binary_mask=require_nonzero_binary_mask,
        allow_zero_rows_when_mixed_binary=allow_zero_rows_when_mixed_binary,
    )


def _causal_attention_mask(
    past_len: int | torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    total_len: int,
    device: torch.device,
) -> torch.Tensor:
    key_positions = torch.arange(total_len, device=device)
    query_positions = _query_positions_from_past_len(
        past_len,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )
    causal_mask = key_positions <= query_positions[:, :, None]
    return causal_mask[:, None, :, :]


def _merge_attention_and_causal_masks(
    attn_mask: torch.Tensor | None,
    causal_mask: torch.Tensor,
    *,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    if attn_mask is None:
        return causal_mask
    if attn_mask.dtype == torch.bool:
        return attn_mask & causal_mask
    attn_bias = attn_mask.to(dtype=target_dtype)
    if attn_bias.is_floating_point():
        # Preserve additive `-inf` masks on causal-allowed keys; only finite-ize
        # `+inf` scores because SDPA can otherwise produce NaNs from inf - inf.
        attn_bias = _finite_positive_infinity_bias(attn_bias, target_dtype=target_dtype)
    causal_bias = torch.full(causal_mask.shape, -torch.inf, device=causal_mask.device, dtype=target_dtype)
    return torch.where(causal_mask, attn_bias, causal_bias)


def _finite_positive_infinity_bias(attn_bias: torch.Tensor, *, target_dtype: torch.dtype) -> torch.Tensor:
    finite_max = torch.full((), torch.finfo(target_dtype).max, device=attn_bias.device, dtype=target_dtype)
    return torch.where(torch.isposinf(attn_bias), finite_max, attn_bias)


class CausalSelfAttention(SegmentedSelfAttention):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.n_head: int = required_int(config, "num_attention_heads")
        self.hidden_size: int = required_int(config, "hidden_size")
        self.head_dim: int = required_int(config, "head_dim")
        if self.n_head <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.n_head}.")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}.")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}.")
        self.q_activation: ActivationName | None = validate_activation_name(
            getattr(config, "q_activation", None),
            field_name="q_activation",
        )
        self.k_activation: ActivationName | None = validate_activation_name(
            getattr(config, "k_activation", None),
            field_name="k_activation",
        )
        self.v_activation: ActivationName | None = validate_activation_name(
            getattr(config, "v_activation", None),
            field_name="v_activation",
        )
        self.use_k_shift: bool = bool(getattr(config, "use_k_shift", False))
        self.use_v_shift: bool = bool(getattr(config, "use_v_shift", False))
        self.use_output_gate: bool = bool(getattr(config, "use_output_gate", False))
        self.c_q = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        if self.use_k_shift:
            self.c_k = _make_shift_linear(self.hidden_size, self.n_head * self.head_dim, self.n_head, bias=False)
        else:
            self.c_k = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        if self.use_v_shift:
            self.c_v = _make_shift_linear(self.hidden_size, self.n_head * self.head_dim, self.n_head, bias=False)
        else:
            self.c_v = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, self.hidden_size, bias=False)
        self.hidden_init_std_factor: float = optional_float(config, "hidden_init_std_factor", 0.5)
        self.num_hidden_layers: int = required_int(config, "num_hidden_layers")
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}.")
        self.residual_output_projection_depth_divisor: float = residual_output_projection_depth_divisor(
            config,
            num_layers=self.num_hidden_layers,
        )
        # Initialize standalone attention modules immediately. Full GPT
        # entrypoints run `init_gpt_weights` after construction and may
        # overwrite this projection with the same residual scaling contract.
        self._reset_output_projection_parameters_preserving_rng()
        rope_ratio = optional_float(config, "rope_ratio", 1.0)
        rope_base = optional_float(config, "rope_base", 10000.0)
        self.rotary = Rotary(self.head_dim, base=rope_base, rope_ratio=rope_ratio)
        self.using_groupnorm: bool = bool(getattr(config, "using_groupnorm", False))
        self.use_qk_rmsnorm: bool = bool(getattr(config, "use_qk_rmsnorm", True))
        if self.use_qk_rmsnorm:
            self.q_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
            self.k_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        if self.using_groupnorm:
            self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        if self.use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
            if not self.using_groupnorm:
                self.o_norm = RMSNorm(
                    self.head_dim,
                    eps=optional_float(config, "rms_norm_eps", 1e-5),
                    elementwise_affine=True,
                )

        # This strict cache-continuity check may synchronize accelerator
        # tensors, so keep it opt-in for debugging or validation runs.
        self.validate_cached_position_ids: bool = optional_bool(config, "validate_cached_position_ids", False)

        kv_cache_slot_size = optional_int(config, "kv_cache_slot_size", 128)
        if kv_cache_slot_size <= 0:
            raise ValueError(f"kv_cache_slot_size must be positive, got {kv_cache_slot_size}.")
        self.kv_cache_slot_size: int = kv_cache_slot_size

    def reset_output_projection_parameters(self) -> None:
        if self.o_proj.weight.is_meta:
            # Meta tensors have no storage to initialize. Materialization via
            # _apply reruns this reset once the projection has real storage;
            # direct calls after to_empty work because weight.is_meta is false.
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

    def _forward_packed_validated(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        boundaries: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        if self.use_k_shift or self.use_v_shift:
            return super()._forward_packed_validated(x, cu_seqlens=cu_seqlens, boundaries=boundaries)
        return packed_rotary_attention(
            x,
            cu_seqlens=cu_seqlens,
            boundaries=boundaries,
            q_proj=self.c_q,
            k_proj=self.c_k,
            v_proj=self.c_v,
            o_proj=self.o_proj,
            rotary=self.rotary,
            num_heads=self.n_head,
            num_kv_heads=self.n_head,
            head_dim=self.head_dim,
            q_activation=self.q_activation,
            k_activation=self.k_activation,
            v_activation=self.v_activation,
            q_norm=self.q_rms if self.use_qk_rmsnorm else None,
            k_norm=self.k_rms if self.use_qk_rmsnorm else None,
            norm_before_rotary=False,
            output_norm=self.subln if self.using_groupnorm else (self.o_norm if self.use_output_gate else None),
            gate_proj=self.g_proj if self.use_output_gate else None,
        )

    def _forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_with_past(x)
        return y

    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        batch_size, seq_len, _ = x.size()
        past_len = 0
        past_k: torch.Tensor | None = None
        past_v: torch.Tensor | None = None
        cache_len = maybe_get_cache_len(past_key_value, batch_size=batch_size)
        past_k_used: torch.Tensor | None = None
        past_v_used: torch.Tensor | None = None
        if past_key_value is not None:
            if len(past_key_value) < 2:
                raise ValueError("past_key_value must have at least 2 tensors: (key, value).")
            past_k = past_key_value[0]
            past_v = past_key_value[1]
            past_len = get_past_len(past_k=past_k, cache_len=cache_len)
            if cache_len is None:
                past_k_used = past_k
                past_v_used = past_v
            else:
                past_k_used = narrow_cache_to_past_len(past_k, past_len=past_len)
                past_v_used = narrow_cache_to_past_len(past_v, past_len=past_len)

        k_shift_state_in, v_shift_state_in = _get_kv_shift_states_or_empty(
            past_key_value,
            use_k_shift=self.use_k_shift,
            use_v_shift=self.use_v_shift,
            k_state_shape=(batch_size, self.n_head * self.head_dim),
            v_state_shape=(batch_size, self.n_head * self.head_dim),
        )

        if position_ids is not None:
            _validate_position_ids(
                position_ids,
                batch_size=batch_size,
                seq_len=seq_len,
                allow_value_sync=self.validate_cached_position_ids,
            )
            if self.validate_cached_position_ids and _past_len_any_positive(past_len):
                expected_position_ids = _query_positions_from_past_len(
                    past_len,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=position_ids.device,
                )
                # This Python bool may synchronize accelerator tensors. The
                # check is intentionally gated by validate_cached_position_ids
                # for debug/validation runs and should stay disabled in
                # throughput-sensitive decoding.
                matching_position_ids = position_ids == expected_position_ids
                if not bool(torch.all(matching_position_ids).item()):
                    boundary_mask = _query_specific_position_boundary_mask(
                        attention_mask,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        # Preallocated KV cache currently exposes one uniform
                        # logical prefix length per batch; variable row lengths
                        # must be represented by the query-specific mask.
                        total_len=_uniform_past_len_int(past_len) + seq_len,
                        num_heads=self.n_head,
                        past_len=past_len,
                        position_ids=position_ids,
                    )
                    allowed_position_ids = matching_position_ids
                    if boundary_mask is not None:
                        allowed_position_ids = allowed_position_ids | boundary_mask.to(device=position_ids.device)
                    if not bool(torch.all(allowed_position_ids).item()):
                        raise ValueError(
                            "position_ids that do not continue cached positions require an attention_mask "
                            "to define packed-sequence boundaries."
                        )

        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k_proj, k_shift_state_out = _project_with_optional_shift_state(
            self.c_k,
            x,
            use_shift=self.use_k_shift,
            shift_state=k_shift_state_in,
        )
        k = k_proj.view(batch_size, seq_len, self.n_head, self.head_dim)
        v_proj, v_shift_state_out = _project_with_optional_shift_state(
            self.c_v,
            x,
            use_shift=self.use_v_shift,
            shift_state=v_shift_state_in,
        )
        v = v_proj.view(batch_size, seq_len, self.n_head, self.head_dim)

        q = apply_activation(q, self.q_activation)
        k = apply_activation(k, self.k_activation)
        v = apply_activation(v, self.v_activation)

        cos, sin = self.rotary(
            q,
            seq_len_offset=past_len,
            position_ids=position_ids,
            validate_position_ids=self.validate_cached_position_ids,
        )
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if self.use_qk_rmsnorm:
            q = self.q_rms(q)
            k = self.k_rms(k)

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        cache_tensors: list[torch.Tensor] | None = None
        cache_len_out: torch.Tensor | None = None
        if use_cache:
            views, caches, cache_len_out = append_preallocated(
                past=[past_k, past_v] if past_k is not None and past_v is not None else None,
                cache_len=cache_len,
                past_len=past_len,
                new=[k_t, v_t],
                slot_size=self.kv_cache_slot_size,
            )
            k_t, v_t = views
            cache_tensors = caches
        elif past_k_used is not None and past_v_used is not None:
            k_t = torch.cat([past_k_used, k_t], dim=-2)
            v_t = torch.cat([past_v_used, v_t], dim=-2)

        total_len = int(k_t.shape[-2])
        attn_mask: torch.Tensor | None = None
        if attention_mask is not None:
            attn_mask = _canonicalize_attention_mask(
                attention_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                total_len=total_len,
                num_heads=self.n_head,
                target_device=x.device,
                target_dtype=q_t.dtype,
            )

        use_is_causal = _past_len_all_zero(past_len) and attn_mask is None
        needs_explicit_causal_mask = not use_is_causal and seq_len > 1
        if needs_explicit_causal_mask:
            causal_mask = _causal_attention_mask(
                past_len,
                batch_size=batch_size,
                seq_len=seq_len,
                total_len=total_len,
                device=x.device,
            )
            attn_mask = _merge_attention_and_causal_masks(attn_mask, causal_mask, target_dtype=q_t.dtype)

        y = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, is_causal=use_is_causal)

        if self.using_groupnorm:
            y = self.subln(y)
        elif self.use_output_gate:
            y = self.o_norm(y)

        if self.use_output_gate:
            gate = self.g_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
            y = y * F.silu(gate)

        y = y.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.n_head * self.head_dim)
        y = self.o_proj(y)
        present: PastKeyValue | None = None
        if use_cache:
            if cache_tensors is None or cache_len_out is None:
                raise RuntimeError("KV cache append did not return expected cache tensors.")
            present = _append_kv_shift_states_or_cache_len(
                [cache_tensors[0], cache_tensors[1]],
                k_shift_state=k_shift_state_out,
                v_shift_state=v_shift_state_out,
                cache_len=cache_len_out,
            )
        return y, present


__all__ = ["CausalSelfAttention"]

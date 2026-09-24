from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Protocol, cast, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.loading_report import LoadStateDictInfo

from .init_utils import init_gpt_weights
from .modules.attention_dtype import attention_dtype_to_torch_dtype, validate_attention_dtype
from .modules.attention_mask_utils import numeric_mask_binary_flags
from .modules.kv_cache import maybe_get_cache_len
from .modules.rmsnorm import RMSNorm
from .modules.segmented_attention import SegmentedSelfAttention

# Each layer cache is at least (key, value). Some attention implementations may append
# additional tensors (e.g., contextual state) after (key, value).
PastKeyValue = tuple[torch.Tensor, ...]
CausalLMForwardTupleItem = torch.Tensor | tuple[PastKeyValue, ...] | tuple[torch.Tensor, ...] | None
CausalLMForwardTuple = tuple[CausalLMForwardTupleItem, ...]
_PARAMETERIZATIONS_WITH_WIDTHMUP_LOGITS = (
    "widthmuP",
    "widthmup",
    "completeP",
    "completep",
    "spectralP",
    "SpectralP",
    "spectralp",
)
_LEGACY_MUP_PARAMETERIZATIONS = ("muP", "mup")


def _raise_if_legacy_mup_parameterization(parameterization: object) -> None:
    if parameterization in _LEGACY_MUP_PARAMETERIZATIONS:
        raise ValueError("Unsupported legacy parameterization='muP'; use parameterization='widthmuP' instead.")


def parameterization_uses_widthmup_logits(config: Any) -> bool:
    parameterization = getattr(config, "parameterization", None)
    _raise_if_legacy_mup_parameterization(parameterization)
    if isinstance(parameterization, str):
        return parameterization in _PARAMETERIZATIONS_WITH_WIDTHMUP_LOGITS
    return False


def resolve_residual_branch_mult(config: Any) -> float:
    parameterization = getattr(config, "parameterization", None)
    _raise_if_legacy_mup_parameterization(parameterization)
    explicit_mult = getattr(config, "residual_branch_mult", None)
    if explicit_mult is not None:
        residual_branch_mult = float(explicit_mult)
    elif parameterization in ("completeP", "completep", "spectralP", "SpectralP", "spectralp"):
        num_layers = float(getattr(config, "num_hidden_layers"))
        num_layers_base = float(getattr(config, "num_hidden_layers_base", num_layers))
        residual_branch_mult = num_layers_base / num_layers
    else:
        residual_branch_mult = 1.0
    if residual_branch_mult <= 0.0:
        raise ValueError(f"residual_branch_mult must be positive, got {residual_branch_mult}.")
    return residual_branch_mult


def _get_alibi_slopes(n_heads: int) -> list[float]:
    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    slopes = get_slopes_power_of_2(closest_power_of_2)
    extra = _get_alibi_slopes(2 * closest_power_of_2)
    slopes += extra[0::2][: n_heads - closest_power_of_2]
    return slopes


def _refresh_alibi_slopes(module: nn.Module) -> bool:
    slopes = module._buffers.get("slopes")
    if not isinstance(slopes, torch.Tensor) or "slopes" not in module._non_persistent_buffers_set:
        return False
    if slopes.ndim != 4 or slopes.shape[0] != 1 or slopes.shape[2:] != (1, 1):
        return False

    n_heads = int(slopes.shape[1])
    module.slopes = torch.tensor(
        _get_alibi_slopes(n_heads),
        dtype=torch.float32,
        device=slopes.device,
    ).view(1, n_heads, 1, 1)

    for cache_name in ("seq_len_cached", "bias_cached", "dist_cached"):
        if hasattr(module, cache_name):
            setattr(module, cache_name, None)
    return True


def apply_residual_branch_update(
    previous_state: torch.Tensor,
    updated_state: torch.Tensor,
    residual_branch_mult: float,
) -> torch.Tensor:
    residual_branch_mult_float = float(residual_branch_mult)
    if residual_branch_mult_float == 1.0:
        return updated_state
    return previous_state + apply_float32_multiplier(updated_state - previous_state, residual_branch_mult_float)


def apply_float32_multiplier(tensor: torch.Tensor, multiplier: float) -> torch.Tensor:
    multiplier_float = float(multiplier)
    if multiplier_float == 1.0:
        return tensor
    if tensor.is_floating_point() and tensor.dtype in (torch.float16, torch.bfloat16):
        return (tensor.float() * multiplier_float).to(dtype=tensor.dtype)
    return tensor * multiplier_float


def logits_scale_for_config(config: Any) -> float:
    if parameterization_uses_widthmup_logits(config):
        return float(getattr(config, "hidden_size_base", 1024)) / float(config.hidden_size)
    return 1.0


def configure_decoder_only_model_config(config: PretrainedConfig) -> None:
    """Normalize custom GPT configs to the decoder-only HF contract."""
    if not hasattr(config, "_attn_implementation_internal"):
        setattr(
            config,
            "_attn_implementation_internal",
            getattr(config, "_attn_implementation", "eager"),
        )

    if hasattr(config, "_nanogpt_preserved_use_cache"):
        preserved_use_cache = getattr(config, "_nanogpt_preserved_use_cache")
        if preserved_use_cache is not None:
            setattr(config, "use_cache", bool(preserved_use_cache))
        delattr(config, "_nanogpt_preserved_use_cache")

    setattr(config, "is_encoder_decoder", False)
    setattr(config, "is_decoder", True)
    setattr(config, "add_cross_attention", False)
    setattr(config, "tie_word_embeddings", True)

    if not hasattr(config, "model_type"):
        setattr(config, "model_type", "nanogpt")
    if not hasattr(config, "torchscript"):
        setattr(config, "torchscript", False)


def validate_past_key_values_length(
    past_key_values: tuple[PastKeyValue, ...] | None,
    *,
    expected_num_layers: int,
) -> None:
    if past_key_values is not None and len(past_key_values) != expected_num_layers:
        raise ValueError(f"past_key_values must have length {expected_num_layers}, got {len(past_key_values)}.")


def causal_lm_output_to_tuple(
    *,
    loss: torch.Tensor | None,
    logits: torch.Tensor | None,
    past_key_values: tuple[PastKeyValue, ...] | None,
    hidden_states: tuple[torch.Tensor, ...] | None,
    attentions: tuple[torch.Tensor, ...] | None,
) -> CausalLMForwardTuple:
    # Keep the legacy local tuple contract stable for training/inference helpers:
    # `(logits, loss, ...)`, with the first two slots always present.
    items: list[CausalLMForwardTupleItem] = [logits, loss]
    if past_key_values is not None:
        items.append(past_key_values)
    if hidden_states is not None:
        items.append(hidden_states)
    if attentions is not None:
        items.append(attentions)
    return tuple(items)


def current_token_mask(
    attention_mask: torch.Tensor | None,
    *,
    current_length: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape (B, S), got {tuple(attention_mask.shape)}.")
    if current_length < 0:
        raise ValueError(f"current_length must be non-negative, got {current_length}.")
    total_length = int(attention_mask.shape[1])
    if total_length < current_length:
        raise ValueError(f"attention_mask length {total_length} is smaller than current_length={current_length}.")
    return attention_mask[:, total_length - current_length :].to(dtype=torch.bool)


def _is_binary_padding_attention_mask(attention_mask: torch.Tensor) -> bool:
    if torch.is_complex(attention_mask):
        return False
    if attention_mask.dtype == torch.bool:
        return True
    return _is_cpu_binary_attention_mask(
        attention_mask,
        require_nonzero=False,
    )


def _is_cpu_binary_attention_mask(attention_mask: torch.Tensor, *, require_nonzero: bool = False) -> bool:
    if attention_mask.device.type != "cpu":
        return False
    return bool(_numeric_binary_row_flags(attention_mask, require_nonzero=require_nonzero).all().item())


def _numeric_binary_row_flags(attention_mask: torch.Tensor, *, require_nonzero: bool = False) -> torch.Tensor:
    return numeric_mask_binary_flags(attention_mask, require_nonzero=require_nonzero)


def _numeric_padding_token_mask_no_sync(
    attention_mask: torch.Tensor,
    *,
    current_length: int,
) -> torch.Tensor:
    total_length = int(attention_mask.shape[1])
    current_mask = attention_mask[:, total_length - current_length :]
    padding_token_mask = current_mask != 0
    noop_token_mask = torch.ones_like(padding_token_mask, dtype=torch.bool)
    is_binary_row = _numeric_binary_row_flags(
        attention_mask,
        require_nonzero=False,
    )
    return torch.where(is_binary_row[:, None], padding_token_mask, noop_token_mask)


def _contiguous_position_ids_from_attention_mask(
    attention_mask: torch.Tensor,
    *,
    current_length: int,
) -> torch.Tensor:
    total_length = int(attention_mask.shape[1])
    if total_length < current_length:
        raise ValueError(f"attention_mask length {total_length} is smaller than current_length={current_length}.")
    positions = torch.arange(
        total_length - current_length,
        total_length,
        device=attention_mask.device,
        dtype=torch.long,
    )
    return positions.unsqueeze(0).expand(int(attention_mask.shape[0]), current_length)


def _binary_position_ids_from_token_mask(token_mask: torch.Tensor, *, current_length: int) -> torch.Tensor:
    position_ids = token_mask.to(dtype=torch.long).cumsum(-1) - 1
    position_ids = position_ids.masked_fill(~token_mask, 0)
    return position_ids[:, -current_length:]


def _position_ids_from_attention_mask(
    attention_mask: torch.Tensor,
    *,
    current_length: int,
) -> torch.Tensor | None:
    if current_length < 0:
        raise ValueError(f"current_length must be non-negative, got {current_length}.")
    if attention_mask.ndim != 2 or torch.is_complex(attention_mask):
        return None

    contiguous_position_ids = _contiguous_position_ids_from_attention_mask(
        attention_mask,
        current_length=current_length,
    )
    if current_length == 0:
        return contiguous_position_ids
    if attention_mask.dtype == torch.bool:
        return _binary_position_ids_from_token_mask(attention_mask, current_length=current_length)
    if attention_mask.device.type == "cpu":
        binary_rows = _numeric_binary_row_flags(
            attention_mask,
            require_nonzero=False,
        )
        if not bool(binary_rows.any().item()):
            return contiguous_position_ids
        position_ids = _binary_position_ids_from_token_mask(attention_mask != 0, current_length=current_length)
        return torch.where(binary_rows[:, None], position_ids, contiguous_position_ids)

    binary_rows = _numeric_binary_row_flags(
        attention_mask,
        require_nonzero=False,
    )
    position_ids = _binary_position_ids_from_token_mask(attention_mask != 0, current_length=current_length)
    return torch.where(binary_rows[:, None], position_ids, contiguous_position_ids)


def _slice_query_specific_attention_mask_for_generation(
    attention_mask: torch.Tensor,
    *,
    current_length: int,
) -> torch.Tensor:
    if current_length < 0:
        raise ValueError(f"current_length must be non-negative, got {current_length}.")
    if attention_mask.ndim == 3:
        if current_length == 0:
            return attention_mask[:, :0, :]
        return attention_mask[:, -current_length:, :]
    if attention_mask.ndim == 4:
        if current_length == 0:
            return attention_mask[:, :, :0, :]
        return attention_mask[:, :, -current_length:, :]
    return attention_mask


def _is_cpu_all_ones_binary_attention_mask(attention_mask: torch.Tensor) -> bool:
    # Avoid host-device sync from scalarizing accelerator masks just to detect the
    # trivial all-ones case. Keep the small CPU-only fast path.
    if attention_mask.device.type != "cpu":
        return False
    if attention_mask.dtype == torch.bool:
        return bool(attention_mask.all().item())
    if torch.is_complex(attention_mask):
        return False
    return bool((attention_mask == 1).all().item())


def _maybe_strip_noop_padding_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor | None:
    if _is_cpu_all_ones_binary_attention_mask(attention_mask):
        return None
    return attention_mask


def _is_cpu_default_position_ids_for_attention_mask(
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    current_length: int,
) -> bool:
    if position_ids.device.type != "cpu" or attention_mask.device.type != "cpu":
        return False
    if attention_mask.ndim != 2 or position_ids.ndim != 2:
        return False
    if int(position_ids.shape[0]) != int(attention_mask.shape[0]) or int(position_ids.shape[1]) != current_length:
        return False
    expected_position_ids = _contiguous_position_ids_from_attention_mask(
        attention_mask,
        current_length=current_length,
    )
    return torch.equal(position_ids, expected_position_ids)


def _is_cpu_default_position_ids_for_cache_position(
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    *,
    batch_size: int,
    current_length: int,
) -> bool:
    if position_ids.device.type != "cpu" or cache_position.device.type != "cpu":
        return False
    if position_ids.ndim != 2 or cache_position.ndim != 1:
        return False
    if int(position_ids.shape[0]) != batch_size or int(position_ids.shape[1]) != current_length:
        return False
    if int(cache_position.shape[0]) < current_length:
        return False
    if current_length == 0:
        expected_positions = cache_position[:0].unsqueeze(0).expand(batch_size, 0)
        return torch.equal(position_ids, expected_positions)
    expected_positions = cache_position[-current_length:].unsqueeze(0).expand(batch_size, current_length)
    return torch.equal(position_ids, expected_positions)


def prepare_ddl_attention_masks(
    attention_mask: torch.Tensor | None,
    *,
    current_length: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Split DDL attention semantics from padding-state masking.

    DDL state cleanup only has a well-defined token-level interpretation for
    2D padding masks. Numeric 2D ``0/1`` masks follow the model-wide padding
    contract: one keeps a token and zero pads it, including all-zero rows.
    Callers that need additive no-op semantics should pass ``None`` or a
    query-specific 3D/4D additive mask. Query-specific, packed, and additive
    masks stay attached to attention and do not imply state zeroing.
    """
    if attention_mask is None:
        return None, None
    if current_length < 0:
        raise ValueError(f"current_length must be non-negative, got {current_length}.")
    if attention_mask.ndim != 2:
        return attention_mask, None
    if torch.is_complex(attention_mask):
        return attention_mask, None

    total_length = int(attention_mask.shape[1])
    if total_length < current_length:
        raise ValueError(f"attention_mask length {total_length} is smaller than current_length={current_length}.")
    if attention_mask.dtype == torch.bool:
        token_mask = attention_mask[:, total_length - current_length :]
    elif attention_mask.device.type == "cpu":
        binary_rows = _numeric_binary_row_flags(
            attention_mask,
            require_nonzero=False,
        )
        if not bool(binary_rows.any().item()):
            return attention_mask, None
        current_mask = attention_mask[:, total_length - current_length :]
        padding_token_mask = current_mask.to(dtype=torch.bool)
        noop_token_mask = torch.ones_like(padding_token_mask, dtype=torch.bool)
        token_mask = torch.where(binary_rows[:, None], padding_token_mask, noop_token_mask)
    else:
        token_mask = _numeric_padding_token_mask_no_sync(attention_mask, current_length=current_length)
    return _maybe_strip_noop_padding_attention_mask(attention_mask), token_mask


def maybe_strip_full_attention_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if _is_cpu_all_ones_binary_attention_mask(attention_mask):
        return None
    return attention_mask


def apply_token_mask(x: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    if token_mask is None:
        return x
    if x.ndim < 2:
        raise ValueError(f"Expected rank >= 2 tensor for token masking, got shape {tuple(x.shape)}.")
    if token_mask.ndim != 2:
        raise ValueError(f"token_mask must have shape (B, T), got {tuple(token_mask.shape)}.")
    expected_shape = (int(x.shape[0]), int(x.shape[1]))
    actual_shape = (int(token_mask.shape[0]), int(token_mask.shape[1]))
    if actual_shape != expected_shape:
        raise ValueError(f"token_mask shape {actual_shape} does not match tensor prefix shape {expected_shape}.")
    mask = token_mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    return x * mask


def resolve_input_ids_and_embeds(
    idx: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int]:
    provided_inputs = int(idx is not None) + int(input_ids is not None) + int(inputs_embeds is not None)
    if provided_inputs != 1:
        raise ValueError("Exactly one of `idx`, `input_ids`, or `inputs_embeds` must be provided.")

    token_ids = idx if idx is not None else input_ids
    if token_ids is not None:
        if token_ids.ndim != 2:
            raise ValueError(f"input token ids must have shape (B, T), got {tuple(token_ids.shape)}.")
        return token_ids, None, int(token_ids.shape[0]), int(token_ids.shape[1])

    assert inputs_embeds is not None
    if inputs_embeds.ndim != 3:
        raise ValueError(f"inputs_embeds must have shape (B, T, C), got {tuple(inputs_embeds.shape)}.")
    return None, inputs_embeds, int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])


def token_embeddings_or_inputs_embeds(
    token_embedding: object,
    *,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
) -> torch.Tensor:
    if inputs_embeds is not None:
        return inputs_embeds
    if input_ids is None:
        raise ValueError("input_ids must be provided when inputs_embeds is None.")
    if not isinstance(token_embedding, nn.Embedding):
        raise TypeError(f"Expected token embedding to be nn.Embedding, got {type(token_embedding).__name__}.")
    return token_embedding(input_ids)


def past_key_values_have_history(past_key_values: Any | None) -> bool:
    if past_key_values is None:
        return False

    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            seq_length = int(cast(Any, get_seq_length)())
        except TypeError:  # older HF Cache: get_seq_length requires a layer index
            seq_length = int(cast(Any, get_seq_length)(0))
        return seq_length > 0

    try:
        num_layers = len(past_key_values)
    except TypeError:  # no __len__: an opaque Cache object; assume it has history
        return True
    if num_layers == 0:
        return False

    first_layer = past_key_values[0]
    if first_layer is None:
        return False

    try:
        first_item = first_layer[0]
    except IndexError, TypeError:  # empty or non-indexable layer entry: no history
        return False

    if isinstance(first_item, torch.Tensor):
        batch_size = int(first_item.shape[0]) if first_item.ndim >= 1 else 0
        cache_len = None
        if batch_size > 0:
            try:
                cache_len = maybe_get_cache_len(cast(tuple[torch.Tensor, ...], first_layer), batch_size=batch_size)
            except TypeError:  # layer tuple does not carry cache lengths
                cache_len = None
        if cache_len is not None:
            return int(cache_len.max().item()) > 0
        if first_item.ndim < 2:
            return False
        return int(first_item.shape[-2]) > 0

    return first_item is not None


def should_treat_past_key_values_as_empty_prefill(past_key_values: Any | None) -> bool:
    if past_key_values is None or past_key_values_have_history(past_key_values):
        return False
    if isinstance(past_key_values, tuple):
        return False
    return isinstance(past_key_values, Cache)


class _SupportsPastKeyValue(Protocol):
    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue | None]: ...


def _attention_supports_forward_parameter(attention: nn.Module, parameter: str) -> bool:
    try:
        signature = inspect.signature(attention.forward)
    except TypeError, ValueError:  # builtins and some wrappers have no signature
        return False
    return parameter in signature.parameters


def _validate_torch_varlen_attention_metadata(
    x: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> tuple[tuple[int, int], ...]:
    validator = cast(
        Callable[[torch.Tensor, torch.Tensor, int], tuple[tuple[int, int], ...]],
        _validate_torch_varlen_attention_metadata_eager,
    )
    return validator(x, cu_seqlens, max_seqlen)


@torch.compiler.disable
def _validate_torch_varlen_attention_metadata_eager(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> tuple[tuple[int, int], ...]:
    """Validate the cumulative-offset contract required by Torch varlen attention."""
    if x.ndim != 3:
        raise ValueError(f"Variable-length attention expects x with shape (B, T, C), got {tuple(x.shape)}.")
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be a rank-1 tensor.")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least two offsets.")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must use torch.int32.")
    if cu_seqlens.device != x.device:
        raise ValueError("cu_seqlens and x must use the same device.")
    if cu_seqlens.device.type == "meta":
        raise ValueError("cu_seqlens on the meta device cannot be validated.")

    total_tokens = int(x.shape[0]) * int(x.shape[1])
    boundaries = [int(offset) for offset in cu_seqlens.tolist()]
    if boundaries[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {boundaries[0]}.")
    if boundaries[-1] != total_tokens:
        raise ValueError(f"cu_seqlens must end at the flattened token count {total_tokens}, got {boundaries[-1]}.")
    sequence_lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    if any(length <= 0 for length in sequence_lengths):
        raise ValueError("cu_seqlens offsets must be strictly increasing.")
    actual_max_seqlen = max(sequence_lengths)
    if max_seqlen != actual_max_seqlen:
        raise ValueError(
            f"max_seqlen must equal the longest packed sequence length {actual_max_seqlen}, got {max_seqlen}."
        )
    return tuple(zip(boundaries[:-1], boundaries[1:], strict=True))


def _attention_forward_with_past_supports_position_ids(attention: nn.Module) -> bool:
    forward_with_past = getattr(attention, "forward_with_past", None)
    if forward_with_past is None:
        return False
    try:
        signature = inspect.signature(forward_with_past)
    except TypeError, ValueError:  # builtins and some wrappers have no signature
        return False
    return "position_ids" in signature.parameters


class CausalLMGenerationMixin(GenerationMixin):
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs

        if past_key_values_have_history(past_key_values):
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = _slice_query_specific_attention_mask_for_generation(
                    attention_mask,
                    current_length=int(input_ids.shape[1]),
                )
            if cache_position is not None:
                cache_position = cache_position[..., -1:]

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = maybe_strip_full_attention_mask(attention_mask)
        if position_ids is None and attention_mask is not None:
            position_ids = _position_ids_from_attention_mask(
                attention_mask,
                current_length=int(input_ids.shape[1]),
            )

        return dict(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    def _reorder_cache(
        self, past_key_values: tuple[PastKeyValue, ...], beam_idx: torch.Tensor
    ) -> tuple[PastKeyValue, ...]:
        reordered: list[PastKeyValue] = []
        for layer_past in past_key_values:
            reordered_layer: list[torch.Tensor] = []
            for tensor in layer_past:
                tensor_beam_idx = beam_idx if beam_idx.device == tensor.device else beam_idx.to(device=tensor.device)
                reordered_layer.append(tensor.index_select(0, tensor_beam_idx))
            reordered.append(tuple(reordered_layer))
        return tuple(reordered)


class DecoderOnlyCausalLMPreTrainedModel(PreTrainedModel, CausalLMGenerationMixin):
    base_model_prefix = "nanogptpro"
    config_class: ClassVar[type[PretrainedConfig]] = PretrainedConfig
    supports_gradient_checkpointing = True
    _tied_weights_keys: ClassVar[dict[str, str]] = {"lm_head.weight": "transformer.wte.weight"}
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = ["lm_head.weight"]

    @classmethod
    def can_generate(cls) -> bool:
        return True

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return False

    def __init__(self, config: PretrainedConfig) -> None:
        configure_decoder_only_model_config(config)
        super().__init__(config)
        self.config = config

    @staticmethod
    def _model_from_pretrained_result(result: Any) -> DecoderOnlyCausalLMPreTrainedModel | None:
        candidate = result[0] if isinstance(result, tuple) and len(result) > 0 else result
        if isinstance(candidate, DecoderOnlyCausalLMPreTrainedModel):
            return candidate
        return None

    @overload
    def _adjust_missing_and_unexpected_keys(self, loading_info: LoadStateDictInfo) -> None: ...

    @overload
    def _adjust_missing_and_unexpected_keys(
        self,
        loading_info: Sequence[str],
        unexpected_keys: Sequence[str],
        loading_task_model_from_base_state_dict: bool = False,
    ) -> tuple[list[str], list[str]]: ...

    def _adjust_missing_and_unexpected_keys(
        self,
        loading_info: LoadStateDictInfo | Sequence[str],
        unexpected_keys: Sequence[str] | None = None,
        loading_task_model_from_base_state_dict: bool = False,
    ) -> tuple[list[str], list[str]] | None:
        if isinstance(loading_info, LoadStateDictInfo):
            super()._adjust_missing_and_unexpected_keys(loading_info)
            return None
        if unexpected_keys is None:
            raise TypeError("unexpected_keys must be provided when using the legacy list API.")

        missing_keys = list(loading_info)
        unexpected_keys_list = list(unexpected_keys)
        state_dict_info = LoadStateDictInfo(
            missing_keys=set(missing_keys),
            unexpected_keys=set(unexpected_keys_list),
            mismatched_keys=set(),
            error_msgs=[],
            conversion_errors={},
        )
        super()._adjust_missing_and_unexpected_keys(state_dict_info)
        adjusted_missing_keys = [key for key in missing_keys if key in state_dict_info.missing_keys]
        adjusted_unexpected_keys = [key for key in unexpected_keys_list if key in state_dict_info.unexpected_keys]
        return adjusted_missing_keys, adjusted_unexpected_keys

    @classmethod
    def get_init_context(
        cls,
        dtype: torch.dtype,
        is_quantized: bool,
        _is_ds_init_called: bool,
        allow_all_kernels: bool | None,
    ) -> list[Any]:
        return list(super().get_init_context(dtype, is_quantized, _is_ds_init_called, allow_all_kernels))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> Any:
        result = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = cls._model_from_pretrained_result(result)
        if model is not None:
            model.refresh_derived_buffers()
        return result

    def refresh_derived_buffers(self) -> None:
        from .modules.rotary import Rotary

        for module in self.modules():
            if isinstance(module, Rotary):
                module.refresh_derived_buffers()
                continue
            if _refresh_alibi_slopes(module):
                continue
            if not (
                hasattr(module, "_kernel_index")
                and hasattr(module, "_kernel_valid")
                and hasattr(module, "value_channels")
                and hasattr(module, "kernel_size")
            ):
                continue
            from .DDL_utils import build_shortconv_kernel_index_and_mask

            weight = getattr(module, "weight", None)
            device = weight.device if isinstance(weight, torch.Tensor) else torch.device("cpu")
            kernel_index, kernel_valid = build_shortconv_kernel_index_and_mask(
                value_channels=int(getattr(module, "value_channels")),
                kernel_size=int(getattr(module, "kernel_size")),
                causal=bool(getattr(module, "dv_shortconv_causal", False)),
                device=device,
            )
            module._kernel_index = kernel_index
            module._kernel_valid = kernel_valid

    def get_input_embeddings(self) -> nn.Embedding:
        owner = cast(Any, self)
        return cast(nn.Embedding, owner.transformer.wte)

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(f"Expected nn.Embedding, got {type(value).__name__}")
        owner = cast(Any, self)
        owner.transformer.wte = value
        owner.tie_weights()

    def get_output_embeddings(self) -> nn.Linear:
        return cast(nn.Linear, cast(Any, self).lm_head)

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if not isinstance(new_embeddings, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(new_embeddings).__name__}")
        owner = cast(Any, self)
        owner.lm_head = new_embeddings
        owner.tie_weights()

    def tie_weights(self, missing_keys: set[str] | None = None, recompute_mapping: bool = True) -> None:
        if recompute_mapping or not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=True)
        super().tie_weights(missing_keys=missing_keys, recompute_mapping=False)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            transformer = getattr(self, "transformer", None)
            wpe = getattr(transformer, "wpe", None)
            if isinstance(wpe, nn.Embedding):
                n_params -= wpe.weight.numel()
        return n_params


class SwiGLUMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden_size = int(getattr(config, "hidden_size"))
        intermediate_size = getattr(config, "intermediate_size", None)
        if intermediate_size is not None:
            hidden_dim = int(intermediate_size)
        else:
            mlp_hidden_mult = float(getattr(config, "mlp_hidden_mult", 8 / 3))
            if mlp_hidden_mult <= 0:
                raise ValueError(f"mlp_hidden_mult must be positive, got {mlp_hidden_mult}.")
            hidden_dim = int(math.floor(mlp_hidden_mult * float(hidden_size)))
        if hidden_dim <= 0:
            raise ValueError(f"mlp_hidden_dim must be positive, got {hidden_dim}.")

        self.c_fc1 = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)
        x = F.silu(x1) * x2
        return self.o_proj(x)


class GPTBlock(nn.Module):
    mlp_cls: ClassVar[type[nn.Module]] = SwiGLUMLP

    def __init__(self, config: Any, attention_cls: type[nn.Module]) -> None:
        super().__init__()
        hidden_size = int(getattr(config, "hidden_size"))
        norm_eps = float(getattr(config, "rms_norm_eps", 1e-5))

        self.attn = attention_cls(config)
        self.attn_forward_with_past_supports_position_ids: bool = _attention_forward_with_past_supports_position_ids(
            self.attn
        )
        attention_dtype = validate_attention_dtype(getattr(config, "attention_dtype", None))
        self.attn_compute_dtype = attention_dtype_to_torch_dtype(attention_dtype)
        if self.attn_compute_dtype is not None:
            self.attn.to(dtype=self.attn_compute_dtype)
        self.mlp = type(self).mlp_cls(config)
        self.ln_1 = RMSNorm(hidden_size, eps=norm_eps)
        self.ln_2 = RMSNorm(hidden_size, eps=norm_eps)
        self.residual_branch_mult = resolve_residual_branch_mult(config)

    def _cast_attention_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.attn_compute_dtype is None or x.dtype == self.attn_compute_dtype:
            return x
        return x.to(dtype=self.attn_compute_dtype)

    @staticmethod
    def _cast_attention_output(x: torch.Tensor, *, target_dtype: torch.dtype) -> torch.Tensor:
        if x.dtype == target_dtype:
            return x
        return x.to(dtype=target_dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        _varlen_metadata_validated: bool = False,
        _packed_boundaries: tuple[tuple[int, int], ...] | None = None,
    ) -> torch.Tensor:
        attn_in = self._cast_attention_input(self.ln_1(x))
        if cu_seqlens is None:
            if max_seqlen is not None:
                raise ValueError("max_seqlen requires cu_seqlens.")
            attn_raw = self.attn(attn_in)
        elif (
            isinstance(self.attn, SegmentedSelfAttention)
            and type(self.attn).forward is SegmentedSelfAttention.forward
            and _packed_boundaries is not None
        ):
            # Keep module hooks active while reusing the model-level metadata validation.
            attn_raw = self.attn(
                attn_in,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                _packed_boundaries=_packed_boundaries,
            )
        else:
            if not _attention_supports_forward_parameter(self.attn, "cu_seqlens"):
                raise RuntimeError(f"{type(self.attn).__name__} does not support `cu_seqlens`.")
            if _attention_supports_forward_parameter(self.attn, "max_seqlen"):
                validated_forward = getattr(self.attn, "_forward_varlen_validated", None)
                if _varlen_metadata_validated and callable(validated_forward):
                    attn_raw = cast(Any, validated_forward)(
                        attn_in,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )
                else:
                    attn_raw = cast(Any, self.attn)(
                        attn_in,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                    )
            else:
                if max_seqlen is not None:
                    if not _varlen_metadata_validated:
                        _validate_torch_varlen_attention_metadata(
                            attn_in,
                            cu_seqlens=cu_seqlens,
                            max_seqlen=max_seqlen,
                        )
                    # Fast-weight packed attention uses one flattened token axis.
                    packed_input = attn_in.reshape(1, -1, attn_in.shape[-1])
                    packed_output = cast(torch.Tensor, self.attn(packed_input, cu_seqlens=cu_seqlens))
                    attn_raw = packed_output.reshape_as(attn_in)
                else:
                    attn_raw = cast(Any, self.attn)(attn_in, cu_seqlens=cu_seqlens)
        attn_out = self._cast_attention_output(cast(torch.Tensor, attn_raw), target_dtype=x.dtype)
        x = x + apply_float32_multiplier(attn_out, self.residual_branch_mult)
        x = x + apply_float32_multiplier(self.mlp(self.ln_2(x)), self.residual_branch_mult)
        return x

    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        attn_in = self._cast_attention_input(self.ln_1(x))
        present: PastKeyValue | None = None
        if getattr(self.attn, "forward_with_past", None) is not None:
            attn_impl = cast(_SupportsPastKeyValue, self.attn)
            if position_ids is not None and self.attn_forward_with_past_supports_position_ids:
                attn_out, present = attn_impl.forward_with_past(
                    attn_in,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            else:
                attn_out, present = attn_impl.forward_with_past(
                    attn_in,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )
        else:
            attn_out = self.attn(attn_in)
        attn_out = self._cast_attention_output(attn_out, target_dtype=x.dtype)

        x = x + apply_float32_multiplier(attn_out, self.residual_branch_mult)
        x = x + apply_float32_multiplier(self.mlp(self.ln_2(x)), self.residual_branch_mult)
        return x, present


def dense_training_flops(model: nn.Module, batch_size: int, sequence_length: int) -> int:
    """`model_flops_v1` training FLOPs for a dense-attention transformer.

    Counts every instantiated `nn.Linear` (forward plus both backward matmuls,
    2 FLOPs per multiply-add) and full-square SDPA for each `GPTBlock`. Valid
    only for models whose non-linear compute is standard quadratic attention;
    see `nanogptpro.utils.model_flops` for the full convention.
    """
    tokens = batch_size * sequence_length
    flops = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            flops += 6 * tokens * module.in_features * module.out_features
        elif isinstance(module, GPTBlock):
            num_heads = int(getattr(module.attn, "n_head"))
            head_dim = int(getattr(module.attn, "head_dim"))
            flops += 12 * batch_size * num_heads * head_dim * sequence_length**2
    return flops


def attention_flops_per_squared_token(model: nn.Module) -> int:
    """Return the quadratic-attention FLOPs coefficient for one batch."""
    return sum(
        12 * int(getattr(module.attn, "n_head")) * int(getattr(module.attn, "head_dim"))
        for module in model.modules()
        if isinstance(module, GPTBlock)
    )


def sparse_moe_training_flops(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
    *,
    sparse_linear_parameter_count: int,
) -> int:
    """Return training FLOPs for standard attention and sparse routed experts."""
    if sparse_linear_parameter_count <= 0:
        raise ValueError(f"sparse_linear_parameter_count must be positive, got {sparse_linear_parameter_count}.")
    tokens = batch_size * sequence_length
    flops = 6 * tokens * sparse_linear_parameter_count
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ".mlp.experts." not in f".{name}.":
            flops += 6 * tokens * module.in_features * module.out_features
        elif isinstance(module, GPTBlock):
            num_heads = int(getattr(module.attn, "n_head"))
            head_dim = int(getattr(module.attn, "head_dim"))
            flops += 12 * batch_size * num_heads * head_dim * sequence_length**2
    return flops


class GPTBase(DecoderOnlyCausalLMPreTrainedModel):
    config_class: ClassVar[type[PretrainedConfig]] = PretrainedConfig

    attention_cls: ClassVar[type[nn.Module]]
    block_cls: ClassVar[type[GPTBlock]] = GPTBlock
    init_exclude_suffixes: ClassVar[tuple[str, ...]] = ()
    init_exclude_names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)

        attention_cls = getattr(type(self), "attention_cls", None)
        if attention_cls is None:  # pragma: no cover
            raise TypeError(f"{type(self).__name__} must define attention_cls.")

        vocab_size = int(getattr(config, "vocab_size"))
        hidden_size = int(getattr(config, "hidden_size"))
        num_layers = int(getattr(config, "num_hidden_layers"))
        token_embedding = nn.Embedding(vocab_size, hidden_size)
        blocks = nn.ModuleList([type(self).block_cls(config, attention_cls) for _ in range(num_layers)])
        self.transformer = nn.ModuleDict(
            dict(
                wte=token_embedding,
                h=blocks,
            )
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = token_embedding.weight
        norm_eps = float(getattr(config, "rms_norm_eps", 1e-5))
        self.ln_f = RMSNorm(hidden_size, eps=norm_eps)
        self.initialize_parameters()
        self.tie_weights()

    def initialize_parameters(self) -> None:
        """Initialize all stochastic parameters after storage materialization."""
        init_gpt_weights(
            self,
            self.config,
            exclude_suffixes=tuple(getattr(type(self), "init_exclude_suffixes", ())),
            exclude_names=tuple(getattr(type(self), "init_exclude_names", ())),
        )

    def training_auxiliary_loss(self) -> torch.Tensor | None:
        """Return an architecture-specific loss for a completed training forward pass."""
        return None

    def training_supervised_loss(
        self,
        input_ids: torch.Tensor | None,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        varlen_metadata_validated: bool = False,
    ) -> torch.Tensor | None:
        """Return an architecture-specific objective over supervised targets."""
        return None

    def _forward_training_tail(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        *,
        return_logits: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Compute training loss and return full logits only when requested."""
        x = self.ln_f(x)
        logits_scale = logits_scale_for_config(self.config)
        logits = apply_float32_multiplier(self.lm_head(x).float(), logits_scale)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return (logits if return_logits else None), loss

    def forward(
        self,
        idx: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
        output_all_seq: bool = False,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[PastKeyValue, ...] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | CausalLMForwardTuple:
        del kwargs

        hf_style_call = input_ids is not None or inputs_embeds is not None
        supervised_call = labels is not None or targets is not None
        return_dict_requested = return_dict is not None

        idx, inputs_embeds, batch_size, current_length = resolve_input_ids_and_embeds(
            idx,
            input_ids,
            inputs_embeds,
        )

        if labels is not None and targets is not None:
            raise ValueError("Only one of `labels` or `targets` can be provided.")
        if targets is None:
            targets = labels

        use_cache = (
            bool(use_cache)
            if use_cache is not None
            else (bool(getattr(self.config, "use_cache", False)) if hf_style_call and not supervised_call else False)
        )
        return_dict = (
            bool(return_dict)
            if return_dict is not None
            else (bool(getattr(self.config, "return_dict", False)) if hf_style_call else False)
        )
        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False
        output_attentions = bool(output_attentions) if output_attentions is not None else False
        if should_treat_past_key_values_as_empty_prefill(past_key_values):
            past_key_values = None

        if output_attentions:
            raise NotImplementedError("output_attentions=True is not currently supported for GPTBase models.")

        original_attention_mask = attention_mask
        attention_mask = maybe_strip_full_attention_mask(attention_mask)
        if (
            attention_mask is None
            and original_attention_mask is not None
            and position_ids is not None
            and _is_cpu_default_position_ids_for_attention_mask(
                position_ids,
                original_attention_mask,
                current_length=current_length,
            )
        ):
            position_ids = None
        if (
            attention_mask is None
            and position_ids is not None
            and cache_position is not None
            and _is_cpu_default_position_ids_for_cache_position(
                position_ids,
                cache_position,
                batch_size=batch_size,
                current_length=current_length,
            )
        ):
            position_ids = None
        if cu_seqlens is None and max_seqlen is not None:
            raise ValueError("max_seqlen requires cu_seqlens.")
        if cu_seqlens is not None and (
            use_cache
            or past_key_values is not None
            or attention_mask is not None
            or position_ids is not None
            or cache_position is not None
        ):
            raise ValueError(
                "cu_seqlens cannot be combined with KV cache, attention_mask, position_ids, "
                "or cache_position in GPTBase."
            )

        token_embedding = self.transformer["wte"]
        if not isinstance(token_embedding, nn.Embedding):
            raise RuntimeError("transformer.wte must be an nn.Embedding.")
        blocks = self.transformer["h"]
        if not isinstance(blocks, nn.ModuleList):
            raise RuntimeError("transformer.h must be an nn.ModuleList.")

        x = token_embeddings_or_inputs_embeds(token_embedding, input_ids=idx, inputs_embeds=inputs_embeds)
        varlen_metadata_validated = False
        packed_boundaries: tuple[tuple[int, int], ...] | None = None
        if cu_seqlens is not None and max_seqlen is not None:
            packed_boundaries = _validate_torch_varlen_attention_metadata(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            varlen_metadata_validated = True
        hidden_states: tuple[torch.Tensor, ...] | None = (x,) if output_hidden_states else None

        present_key_values: list[PastKeyValue] | None = [] if use_cache else None
        validate_past_key_values_length(past_key_values, expected_num_layers=len(blocks))

        for layer_idx, block_module in enumerate(blocks):
            if not isinstance(block_module, GPTBlock):
                raise RuntimeError("transformer.h entries must be GPTBlock instances.")
            block = block_module
            if use_cache or past_key_values is not None or attention_mask is not None or position_ids is not None:
                past = past_key_values[layer_idx] if past_key_values is not None else None
                x, present = block.forward_with_past(
                    x,
                    past_key_value=past,
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                if use_cache:
                    if present is None:
                        raise NotImplementedError(
                            f"{type(block.attn).__name__} does not implement `forward_with_past` needed for KV cache."
                        )
                    assert present_key_values is not None
                    present_key_values.append(present)
            else:
                x = block(
                    x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    _varlen_metadata_validated=varlen_metadata_validated,
                    _packed_boundaries=packed_boundaries,
                )

            if output_hidden_states:
                assert hidden_states is not None
                hidden_states = (*hidden_states, x)

        if targets is not None:
            logits, loss = self._forward_training_tail(x, targets, return_logits=return_logits)
            if self.training:
                auxiliary_loss = self.training_auxiliary_loss()
                if auxiliary_loss is not None:
                    loss = loss + auxiliary_loss
                supervised_loss = self.training_supervised_loss(
                    idx,
                    x,
                    targets,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    varlen_metadata_validated=varlen_metadata_validated,
                )
                if supervised_loss is not None:
                    loss = loss + supervised_loss
        else:
            x = self.ln_f(x)
            logits_scale = logits_scale_for_config(self.config)
            loss = None
            if output_all_seq or return_dict_requested or hf_style_call:
                logits = apply_float32_multiplier(self.lm_head(x).float(), logits_scale)
            else:
                logits = apply_float32_multiplier(self.lm_head(x[:, [-1], :]).float(), logits_scale)

        past_out: tuple[PastKeyValue, ...] | None = None
        if use_cache:
            assert present_key_values is not None
            past_out = tuple(present_key_values)
        if not return_logits:
            logits = None
        if not return_dict:
            return causal_lm_output_to_tuple(
                loss=loss,
                logits=logits,
                past_key_values=past_out,
                hidden_states=hidden_states,
                attentions=None,
            )
        return CausalLMOutputWithPast(
            loss=cast(torch.FloatTensor | None, loss),
            logits=cast(torch.FloatTensor | None, logits),
            past_key_values=cast(Any, past_out),
            hidden_states=cast(tuple[torch.FloatTensor, ...] | None, hidden_states),
            attentions=None,
        )

    def crop_block_size(self, block_size: int) -> None:
        block_size_int = int(block_size)
        if block_size_int <= 0:
            raise ValueError(f"block_size must be a positive integer, got {block_size_int}.")

        current = getattr(self.config, "block_size", None)
        if isinstance(current, int):
            current_int = int(current)
            if block_size_int > current_int:
                raise ValueError(f"block_size must be <= {current_int} to crop, got {block_size_int}.")

        setattr(self.config, "block_size", block_size_int)

    def attention_flops_per_squared_token(self) -> int:
        """Return the quadratic-attention FLOPs coefficient for one batch."""
        return attention_flops_per_squared_token(self)

    def training_flops(self, batch_size: int, sequence_length: int) -> int | None:
        """`model_flops_v1` FLOPs for one forward+backward at this batch shape.

        Architectures opt in with a validated formula (see
        `dense_training_flops`). The default is None: the trainer omits MFU
        instead of guessing from parameter counts.
        """
        return None

    def save_pretrained(self, save_directory: str, *args: Any, **kwargs: Any) -> None:
        super().save_pretrained(save_directory, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> Any:
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

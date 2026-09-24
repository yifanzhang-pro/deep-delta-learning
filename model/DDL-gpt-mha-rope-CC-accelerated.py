"""
Deep Delta Learning (DDL), expanded state (d_v > 1), on top of GPT (MHA + RoPE).

Implements the Delta update:
    X_{l+1} = X_l + beta_l * k_l * (v_l^T - k_l^T X_l)

The hidden state is treated as a matrix X in R^{d x d_v} (flattened in memory),
where d is the backbone width and d_v is a small value-channel expansion (default: 4).

To interface with standard Transformer sublayers expecting inputs in R^d, we:
- Start by replicating token embeddings across d_v channels.
- Before each sublayer, compress the expanded residual with a depthwise short convolution along
  `d_v` (optionally causal) and a learned read vector to produce a d-dimensional hidden.
- Run pre-norm + sublayer to obtain k (used as the update direction).
- Project v in R^{d_v} and apply the rank-1 write k v^T, synchronized with erasure k^T X.
"""

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modules.segmented_causal_lm import forward_packed_ddl
from .modules.activations import ActivationName
from .modules.ddl_shortconv import CcResidualShortConvCompressor as _CcResidualShortConvCompressor
from .modules.attention_dtype import AttentionDType
from .modules.attention import CausalSelfAttention
from .modules.mlp import MLP
from .modules.kv_shift import keep_only_kv_shift_cache_states
from .DDL_utils import (
    FusedDeepDeltaFunction,
    validate_expanded_delta_inputs,
)
from .gpt_base import (
    CausalLMForwardTuple,
    DecoderOnlyCausalLMPreTrainedModel,
    PastKeyValue,
    apply_float32_multiplier,
    apply_residual_branch_update,
    apply_token_mask,
    causal_lm_output_to_tuple,
    prepare_ddl_attention_masks,
    resolve_input_ids_and_embeds,
    token_embeddings_or_inputs_embeds,
    configure_decoder_only_model_config,
    logits_scale_for_config,
    should_treat_past_key_values_as_empty_prefill,
    resolve_residual_branch_mult,
    validate_past_key_values_length,
)
from .modules.rmsnorm import RMSNorm
from .init_utils import init_gpt_weights
from .pydantic_config import validate_pretrained_config_kwargs


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p) - math.log(1.0 - p)


class ResidualShortConvCompressor(_CcResidualShortConvCompressor):
    def __init__(self, config: Any) -> None:
        super().__init__(config, implementation="triton_required", module_name="DDL-gpt-mha-rope-CC-accelerated")


class DeepDeltaResidualExpanded(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        hidden_size = int(config.hidden_size)
        value_channels = int(getattr(config, "ddl_value_channels", 4))
        if value_channels <= 1:
            raise ValueError("ddl_value_channels must be > 1 for expanded-state DDL.")
        self.hidden_size = hidden_size
        self.value_channels = value_channels

        self.residual_branch_mult = resolve_residual_branch_mult(config)
        self.k_eps = float(getattr(config, "ddl_k_eps", 1e-5))
        self.v_sigmoid = bool(getattr(config, "ddl_v_sigmoid", True))
        self.v_sigmoid_scale = float(getattr(config, "ddl_v_sigmoid_scale", 4.0))
        self.v_constant = bool(getattr(config, "ddl_v_constant", False))

        self.beta_single_linear = bool(getattr(config, "ddl_beta_single_linear", True))
        if not self.beta_single_linear or self.v_constant or not self.v_sigmoid:
            raise ValueError(
                "DDL-gpt-mha-rope-CC-accelerated requires ddl_beta_single_linear=True, "
                "ddl_v_constant=False, and ddl_v_sigmoid=True."
            )

        self.beta = nn.Linear(hidden_size, 1, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.value_channels, bias=True)

        beta_init = float(getattr(config, "ddl_beta_init", 0.0))
        beta_init = min(max(beta_init, 0.0), 2.0)
        beta_init_p = beta_init / 2.0
        with torch.no_grad():
            self.beta.bias.fill_(_logit(beta_init_p))

    def forward(
        self,
        x: torch.Tensor,
        *,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        validate_expanded_delta_inputs(
            module_name="DDL-gpt-mha-rope-CC-accelerated",
            x=x,
            k_in=k_in,
            v_in=v_in,
            context=context,
            hidden_size=self.hidden_size,
            value_channels=self.value_channels,
        )
        if not (x.is_cuda and k_in.is_cuda and v_in.is_cuda and context.is_cuda):
            raise RuntimeError("DDL-gpt-mha-rope-CC-accelerated requires CUDA tensors for fused Triton updates.")

        updated = cast(
            torch.Tensor,
            FusedDeepDeltaFunction.apply(
                x,
                k_in,
                v_in,
                context,
                self.v_proj.weight,
                self.v_proj.bias,
                self.beta.weight,
                self.beta.bias,
                self.k_eps,
                self.v_sigmoid_scale,
            ),
        )
        return apply_residual_branch_update(x, updated, self.residual_branch_mult)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.compress = ResidualShortConvCompressor(config)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ddl_attn = DeepDeltaResidualExpanded(config)
        self.ddl_mlp = DeepDeltaResidualExpanded(config)
        # Define RMSNorm layers once in the module
        self.ln_1 = RMSNorm(config.hidden_size)
        self.ln_2 = RMSNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply pre-norm before sublayers (compress -> prenorm -> sublayer -> DDL update).
        x_in = self.compress(x)
        x_norm = self.ln_1(x_in)
        k_attn = self.attn(x_norm)
        x = self.ddl_attn(x, k_in=k_attn, v_in=x_in, context=x_norm)

        x_in = self.compress(x)
        x_norm = self.ln_2(x_in)
        k_mlp = self.mlp(x_norm)
        x = self.ddl_mlp(x, k_in=k_mlp, v_in=x_in, context=x_norm)
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
        batch_size = int(x.shape[0])
        attention_past_key_value = keep_only_kv_shift_cache_states(
            past_key_value,
            use_k_shift=self.attn.use_k_shift,
            use_v_shift=self.attn.use_v_shift,
            k_state_shape=(batch_size, self.attn.n_head * self.attn.head_dim),
            v_state_shape=(batch_size, self.attn.n_head * self.attn.head_dim),
        )
        x_in = self.compress(x)
        x_norm = self.ln_1(x_in)
        k_attn, present = self.attn.forward_with_past(
            x_norm,
            past_key_value=attention_past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        x = self.ddl_attn(x, k_in=k_attn, v_in=x_in, context=x_norm)

        x_in = self.compress(x)
        x_norm = self.ln_2(x_in)
        k_mlp = self.mlp(x_norm)
        x = self.ddl_mlp(x, k_in=k_mlp, v_in=x_in, context=x_norm)
        return x, present


# -----------------------------------------------------------------------------
# The main GPT-2 model


@dataclass
class GPTConfig(PretrainedConfig):
    ddl_sequence_reduction_version: int = 1
    model_type = "nanogptpro"
    vocab_size: int = 50304
    num_hidden_layers: int = 12
    num_attention_heads: int = 6  # head dim 128 suggested by @Grad62304977
    hidden_size: int = 768
    head_dim: int = 128  # Dimension per head
    block_size: int = 1024  # Maximum sequence length
    bias: bool = False  # Use bias in all linear layers
    dropout: float = 0.0  # Dropout rate
    scale_attn_by_inverse_layer_idx: bool = False  # Scale attention by 1/sqrt(layer_idx)
    using_groupnorm: bool = False  # Whether to use Group Layernorm
    use_output_gate: bool = False
    use_qk_rmsnorm: bool = True  # Apply learnable RMSNorm to Q and K in attention
    use_k_shift: bool = False
    use_v_shift: bool = False

    # QKV activation knobs (applied by attention impls when supported)
    q_activation: ActivationName | None = None
    k_activation: ActivationName | None = None
    v_activation: ActivationName | None = None
    attention_dtype: AttentionDType = "auto"

    rope_ratio: float = 1.0  # Apply RoPE on the first rope_ratio*head_dim dimensions (must be in [0, 1])
    # Embedding init std (normal init for tied token embedding / LM head)
    embedding_init_std: float = 0.02
    # Factor for hidden (>=2D) param init; actual std = factor / sqrt(hidden_size)
    hidden_init_std_factor: float = 0.5
    # DDL expanded-state knobs
    ddl_value_channels: int = 4
    ddl_state_shortconv_kernel_size: int = 4
    ddl_state_read_init: float | None = None
    ddl_state_dv_shortconv_causal: bool = False
    ddl_k_eps: float = 1e-5
    ddl_beta_hidden_size: int = 128
    ddl_beta_single_linear: bool = True
    ddl_v_sigmoid: bool = True
    ddl_v_sigmoid_scale: float = 4.0
    ddl_v_constant: bool = False
    ddl_v_constant_value: float = 2.0
    # Initialize beta; clamped to [0, 2]. Use 1.0 by default for baseline comparability.
    ddl_beta_init: float = 1.0

    def __init__(self, **kwargs: Any) -> None:
        reduction_version = kwargs.get("ddl_sequence_reduction_version", 1)
        if type(reduction_version) is not int or reduction_version != 1:
            raise ValueError("Unsupported ddl_sequence_reduction_version; expected 1.")
        kwargs.setdefault("ddl_sequence_reduction_version", 1)
        raw = dict(kwargs)
        if "ddl_value_channels" in raw and "ddl_state_shortconv_kernel_size" not in raw:
            raw["ddl_state_shortconv_kernel_size"] = raw["ddl_value_channels"]
        super().__init__(**validate_pretrained_config_kwargs(type(self), raw))


class GPT(DecoderOnlyCausalLMPreTrainedModel):
    config_class = GPTConfig
    base_model_prefix = "nanogptpro"
    supports_gradient_checkpointing = True

    def __init__(self, config: Any):
        configure_decoder_only_model_config(config)
        super().__init__(config)
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size),
                h=nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)]),
            )
        )
        self.readout = ResidualShortConvCompressor(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()
        self.ln_f = RMSNorm(config.hidden_size)
        init_gpt_weights(self, config)

    def tie_weights(self, missing_keys: set[str] | None = None, recompute_mapping: bool = True) -> None:
        super().tie_weights(missing_keys=missing_keys, recompute_mapping=recompute_mapping)

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
        if cu_seqlens is None and max_seqlen is not None:
            raise ValueError("max_seqlen requires cu_seqlens.")
        if cu_seqlens is not None:
            return forward_packed_ddl(
                self,
                idx=idx,
                targets=targets,
                return_logits=return_logits,
                output_all_seq=output_all_seq,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
                cache_position=cache_position,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        del cache_position, kwargs

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

        use_cache_flag = (
            bool(use_cache)
            if use_cache is not None
            else (bool(getattr(self.config, "use_cache", False)) if hf_style_call and not supervised_call else False)
        )
        return_dict_flag = (
            bool(return_dict)
            if return_dict is not None
            else (bool(getattr(self.config, "return_dict", False)) if hf_style_call else False)
        )
        output_hidden_states_flag = bool(output_hidden_states) if output_hidden_states is not None else False
        output_attentions_flag = bool(output_attentions) if output_attentions is not None else False
        if should_treat_past_key_values_as_empty_prefill(past_key_values):
            past_key_values = None

        if output_attentions_flag:
            raise NotImplementedError("output_attentions=True is not currently supported for DDL models.")

        attention_mask, token_mask = prepare_ddl_attention_masks(attention_mask, current_length=current_length)

        transformer_blocks = [cast(Block, block) for block in cast(nn.ModuleList, self.transformer.h)]
        validate_past_key_values_length(past_key_values, expected_num_layers=len(transformer_blocks))

        x_emb = apply_token_mask(
            token_embeddings_or_inputs_embeds(self.transformer.wte, input_ids=idx, inputs_embeds=inputs_embeds),
            token_mask,
        )
        value_channels = int(getattr(self.config, "ddl_value_channels", 4))
        x = x_emb.unsqueeze(-1).repeat(1, 1, 1, value_channels)
        hidden_states: tuple[torch.Tensor, ...] | None = (x_emb,) if output_hidden_states_flag else None
        present_key_values: list[PastKeyValue] | None = [] if use_cache_flag else None

        for layer_idx, block in enumerate(transformer_blocks):
            if use_cache_flag or past_key_values is not None or attention_mask is not None or position_ids is not None:
                past = past_key_values[layer_idx] if past_key_values is not None else None
                x, present = block.forward_with_past(
                    x,
                    past_key_value=past,
                    use_cache=use_cache_flag,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                if use_cache_flag:
                    if present is None:
                        raise RuntimeError("Block did not return past_key_value for KV cache.")
                    assert present_key_values is not None
                    present_key_values.append(present)
            else:
                x = block(x)
            x = apply_token_mask(x, token_mask)

            if output_hidden_states_flag:
                assert hidden_states is not None
                hidden_states = (*hidden_states, self.readout(x))

        if output_hidden_states_flag and len(transformer_blocks) > 0:
            assert hidden_states is not None
            x_out = cast(torch.Tensor, hidden_states[-1])
        else:
            x_out = self.readout(x)
        x_out = self.ln_f(x_out)

        logits_scale = logits_scale_for_config(self.config)

        if targets is not None:
            logits = apply_float32_multiplier(self.lm_head(x_out).float(), logits_scale)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            if not return_logits:
                logits = None
        else:
            loss = None
            if not return_logits:
                logits = None
            elif output_all_seq or return_dict_requested or hf_style_call:
                logits = apply_float32_multiplier(self.lm_head(x_out).float(), logits_scale)
            else:
                logits = apply_float32_multiplier(self.lm_head(x_out[:, [-1], :]).float(), logits_scale)

        past_out: tuple[PastKeyValue, ...] | None = None
        if use_cache_flag:
            assert present_key_values is not None
            past_out = tuple(present_key_values)
        if not return_dict_flag:
            return causal_lm_output_to_tuple(
                loss=loss,
                logits=cast(torch.FloatTensor | None, logits),
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

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        n_params = self.get_num_params()
        cfg = self.config
        flops_per_token = (
            6 * n_params
            + 12
            * cfg.num_hidden_layers
            * cfg.num_attention_heads
            * (cfg.hidden_size // cfg.num_attention_heads)
            * cfg.block_size
        )
        flops_per_fwdbwd = flops_per_token * cfg.block_size
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    def get_num_params(self, non_embedding: bool = True) -> int:
        return super().get_num_params(non_embedding=non_embedding)

    def save_pretrained(self, save_directory: str, *args: Any, **kwargs: Any) -> None:
        super().save_pretrained(save_directory, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> Any:
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

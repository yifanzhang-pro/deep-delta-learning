"""
Deep Delta Learning (DDL), expanded state (d_v > 1), on top of GPT (MHA + RoPE).

Implements the Delta update:
    X_{l+1} = X_l + beta_l * k_l * (v_l^T - k_l^T X_l)

The hidden state is treated as a matrix X in R^{d x d_v} (flattened in memory),
where d is the backbone width and d_v is a small value-channel expansion (default: 4).

To interface with standard Transformer sublayers expecting inputs in R^d, we:
- Start by replicating token embeddings across d_v channels.
- Before each sublayer, compress the expanded residual with a short causal conv and a
  learned read vector to produce a d-dimensional hidden.
- Run pre-norm + sublayer to obtain k (used as the update direction).
- Project v in R^{d_v} and apply the rank-1 write k v^T, synchronized with erasure k^T X.
"""

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.segmented_causal_lm import forward_packed_ddl
from .modules.amp_utils import linear_fp32
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modules.activations import ActivationName
from .modules.ddl_shortconv import TemporalResidualShortConvCompressor as _TemporalResidualShortConvCompressor
from .modules.attention_dtype import AttentionDType
from .modules.attention import CausalSelfAttention
from .modules.mlp import MLP
from .gpt_base import (
    CausalLMForwardTuple,
    DecoderOnlyCausalLMPreTrainedModel,
    PastKeyValue,
    apply_float32_multiplier,
    apply_residual_branch_update,
    apply_token_mask,
    causal_lm_output_to_tuple,
    resolve_input_ids_and_embeds,
    token_embeddings_or_inputs_embeds,
    configure_decoder_only_model_config,
    prepare_ddl_attention_masks,
    logits_scale_for_config,
    should_treat_past_key_values_as_empty_prefill,
    resolve_residual_branch_mult,
    validate_past_key_values_length,
)
from .modules.kv_cache import maybe_get_cache_len
from .modules.rmsnorm import RMSNorm
from .modules.kv_shift import (
    get_kv_cache_trailing_state_by_shape,
    insert_kv_cache_prefix_states,
    keep_only_kv_shift_cache_states,
    split_kv_cache_prefix_states,
)
from .DDL_utils import shortconv_kernel_size
from .init_utils import init_gpt_weights
from .pydantic_config import validate_pretrained_config_kwargs


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p) - math.log(1.0 - p)


class ResidualShortConvCompressor(_TemporalResidualShortConvCompressor):
    def __init__(self, config: Any) -> None:
        super().__init__(config, implementation="torch", module_name="DDL-gpt-mha-rope-TC")


class DeepDeltaResidualExpanded(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = int(config.hidden_size)
        value_channels = int(getattr(config, "ddl_value_channels", 4))
        if value_channels <= 1:
            raise ValueError("ddl_value_channels must be > 1 for expanded-state DDL.")
        self.value_channels = value_channels

        self.residual_branch_mult = resolve_residual_branch_mult(config)
        self.k_eps = float(getattr(config, "ddl_k_eps", 1e-5))
        self.v_sigmoid = bool(getattr(config, "ddl_v_sigmoid", True))
        self.v_sigmoid_scale: float = float(getattr(config, "ddl_v_sigmoid_scale", 4.0))
        self.v_constant = bool(getattr(config, "ddl_v_constant", False))
        self.v_constant_value: float = float(getattr(config, "ddl_v_constant_value", 2.0))

        self.beta_single_linear = bool(getattr(config, "ddl_beta_single_linear", True))
        if self.beta_single_linear:
            self.beta = nn.Linear(hidden_size, 1, bias=True)
        else:
            beta_hidden_size = int(getattr(config, "ddl_beta_hidden_size", 128))
            if beta_hidden_size <= 0:
                raise ValueError("ddl_beta_hidden_size must be positive.")

            self.beta_in = nn.Linear(hidden_size, beta_hidden_size, bias=False)
            self.beta_out = nn.Linear(beta_hidden_size, 1, bias=True)

        # v is a vector in R^{d_v} in the expanded-state regime.
        self.v_proj = nn.Linear(hidden_size, self.value_channels, bias=True)

        beta_init = float(getattr(config, "ddl_beta_init", 0.0))
        beta_init = min(max(beta_init, 0.0), 2.0)
        beta_init_p = beta_init / 2.0
        with torch.no_grad():
            if self.beta_single_linear:
                self.beta.bias.fill_(_logit(beta_init_p))
            else:
                self.beta_out.bias.fill_(_logit(beta_init_p))

    def forward(
        self,
        x: torch.Tensor,
        *,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, T, d, d_v), k_in: (B, T, d), v_in: (B, T, d), context: (B, T, d)
        # Keep large tensors in the model dtype; only compute `beta` in fp32 for stability.
        k_dim = int(k_in.size(-1))
        eps_rms = (self.k_eps * self.k_eps) / float(k_dim)
        k_rms = F.rms_norm(k_in, [k_dim], eps=eps_rms)
        k_scale = 1.0 / math.sqrt(k_dim)

        # beta(X) in [0, 2]
        if self.beta_single_linear:
            beta_logits = linear_fp32(self.beta, context)
        else:
            beta_hidden = torch.tanh(linear_fp32(self.beta_in, context))
            beta_logits = linear_fp32(self.beta_out, beta_hidden)
        beta = 2.0 * torch.sigmoid(beta_logits)  # fp32

        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B, T, d, d_v), got {tuple(x.shape)}")
        if int(x.size(-2)) != k_dim:
            raise ValueError(f"Expected x feature dim {k_dim}, got {int(x.size(-2))}.")
        if int(x.size(-1)) != self.value_channels:
            raise ValueError(f"Expected x value channels {self.value_channels}, got {int(x.size(-1))}.")

        # k^T X, row vector projection (B, T, d_v)
        proj_rms = torch.sum(k_rms.unsqueeze(-1) * x, dim=-2, dtype=torch.float32)  # fp32
        proj = proj_rms * k_scale

        if self.v_constant:
            v = torch.full_like(proj, self.v_constant_value)  # (B, T, d_v)
        else:
            v = self.v_proj(v_in)
            if self.v_sigmoid:
                v = torch.sigmoid(v) * self.v_sigmoid_scale

        # X <- X + beta * k * (v^T - k^T X)
        delta_row = (beta * (v - proj)) * k_scale  # fp32 (B, T, d_v)
        update = k_rms.unsqueeze(-1) * delta_row.to(dtype=x.dtype).unsqueeze(-2)  # (B, T, d, d_v)
        return apply_residual_branch_update(x, x + update, self.residual_branch_mult)


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

    def forward(self, x: torch.Tensor, *, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Apply pre-norm before sublayers (compress -> prenorm -> sublayer -> DDL update).
        x_in = self.compress(x)
        x_norm = self.ln_1(x_in)
        k_attn = self.attn(x_norm)
        x = self.ddl_attn(x, k_in=k_attn, v_in=x_in, context=x_norm)
        x = apply_token_mask(x, token_mask)

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
        attention_past_key_value, (conv_pre_attn, conv_pre_mlp) = split_kv_cache_prefix_states(
            past_key_value,
            prefix_state_count=2,
            batch_size=batch_size,
        )
        attention_past_key_value = keep_only_kv_shift_cache_states(
            attention_past_key_value,
            use_k_shift=self.attn.use_k_shift,
            use_v_shift=self.attn.use_v_shift,
            k_state_shape=(batch_size, self.attn.n_head * self.attn.head_dim),
            v_state_shape=(batch_size, self.attn.n_head * self.attn.head_dim),
        )

        _, token_mask = prepare_ddl_attention_masks(attention_mask, current_length=int(x.shape[1]))
        x_in, conv_pre_attn_out = self.compress.forward_with_past(x, past=conv_pre_attn, token_mask=token_mask)
        x_norm = self.ln_1(x_in)
        k_attn, present_attn = self.attn.forward_with_past(
            x_norm,
            past_key_value=attention_past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        x = self.ddl_attn(x, k_in=k_attn, v_in=x_in, context=x_norm)
        x = apply_token_mask(x, token_mask)

        x_in, conv_pre_mlp_out = self.compress.forward_with_past(x, past=conv_pre_mlp, token_mask=token_mask)
        x_norm = self.ln_2(x_in)
        k_mlp = self.mlp(x_norm)
        x = self.ddl_mlp(x, k_in=k_mlp, v_in=x_in, context=x_norm)

        present: PastKeyValue | None = None
        if use_cache:
            if present_attn is None:
                raise RuntimeError("Attention did not return present_key_value for KV cache.")
            if conv_pre_attn_out is not None and conv_pre_mlp_out is not None:
                present = insert_kv_cache_prefix_states(present_attn, (conv_pre_attn_out, conv_pre_mlp_out))
            else:
                present = present_attn
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
        super().__init__(**validate_pretrained_config_kwargs(type(self), kwargs))


class GPT(DecoderOnlyCausalLMPreTrainedModel):
    config_class = GPTConfig
    base_model_prefix = "nanogptpro"
    supports_gradient_checkpointing = True

    def __init__(self, config: GPTConfig):
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
        # weight tying between token embedding and LM head
        self.tie_weights()  # https://paperswithcode.com/method/weight-tying
        # Final RMSNorm defined in the network
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
        readout_has_state = int(getattr(self.readout.shortconv, "kernel_size", 0)) > 1
        layer_readout_states: list[torch.Tensor | None] = [None] * len(transformer_blocks)
        shift_state_count = int(bool(getattr(self.config, "use_k_shift", False))) + int(
            bool(getattr(self.config, "use_v_shift", False))
        )
        if past_key_values is not None and readout_has_state:
            readout_kernel_size = int(getattr(self.readout.shortconv, "kernel_size", 0))
            for layer_idx, layer_past in enumerate(past_key_values):
                if maybe_get_cache_len(layer_past, batch_size=batch_size) is None:
                    continue
                block = transformer_blocks[layer_idx]
                block_prefix_state_count = 2 if shortconv_kernel_size(getattr(block, "compress", None)) > 1 else 0
                layer_readout_states[layer_idx] = get_kv_cache_trailing_state_by_shape(
                    layer_past,
                    expected_shape=(
                        batch_size,
                        int(
                            getattr(
                                self.readout,
                                "residual_size",
                                int(self.config.hidden_size) * int(getattr(self.config, "ddl_value_channels", 4)),
                            )
                        ),
                        readout_kernel_size - 1,
                    ),
                    batch_size=batch_size,
                    prefix_state_count=block_prefix_state_count + shift_state_count,
                )
        if output_hidden_states_flag and past_key_values is not None and readout_has_state:
            if any(state is None for state in layer_readout_states):
                raise NotImplementedError(
                    "output_hidden_states with past_key_values requires cached readout states from a prior "
                    "call with output_hidden_states=True."
                )

        readout_state = layer_readout_states[-1]

        x_emb = apply_token_mask(
            token_embeddings_or_inputs_embeds(self.transformer.wte, input_ids=idx, inputs_embeds=inputs_embeds),
            token_mask,
        )  # (B, T, d)
        value_channels = int(getattr(self.config, "ddl_value_channels", 4))
        x = x_emb.unsqueeze(-1).repeat(1, 1, 1, value_channels)  # (B, T, d, d_v)
        hidden_states: tuple[torch.Tensor, ...] | None = (x_emb,) if output_hidden_states_flag else None
        layer_readout_state_outs: list[torch.Tensor | None] = [None] * len(transformer_blocks)

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
                x_hidden, layer_readout_state_out = self.readout.forward_with_past(
                    x,
                    past=layer_readout_states[layer_idx],
                    token_mask=token_mask,
                )
                hidden_states = (*hidden_states, x_hidden)
                layer_readout_state_outs[layer_idx] = layer_readout_state_out

        if output_hidden_states_flag:
            assert hidden_states is not None
            x_out = cast(torch.Tensor, hidden_states[-1])
            readout_state_out = layer_readout_state_outs[-1]
        else:
            x_out, readout_state_out = self.readout.forward_with_past(x, past=readout_state, token_mask=token_mask)
        x_out = self.ln_f(x_out)

        logits_scale = logits_scale_for_config(self.config)

        if targets is not None:
            logits = apply_float32_multiplier(self.lm_head(x_out).float(), logits_scale)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            loss = None
            if output_all_seq or return_dict_requested or hf_style_call:
                logits = apply_float32_multiplier(self.lm_head(x_out).float(), logits_scale)
            else:
                logits = apply_float32_multiplier(self.lm_head(x_out[:, [-1], :]).float(), logits_scale)

        past_out: tuple[PastKeyValue, ...] | None = None
        if use_cache_flag:
            assert present_key_values is not None
            if present_key_values and readout_state_out is not None:
                if output_hidden_states_flag:
                    for layer_idx, layer_state_out in enumerate(layer_readout_state_outs):
                        if layer_state_out is None:
                            continue
                        layer_items = list(cast(tuple[torch.Tensor, ...], present_key_values[layer_idx]))
                        if len(layer_items) < 2:
                            raise RuntimeError("Invalid past_key_value payload: expected at least key/value tensors.")
                        present_key_values[layer_idx] = tuple([*layer_items[:-1], layer_state_out, layer_items[-1]])
                else:
                    last_items = list(cast(tuple[torch.Tensor, ...], present_key_values[-1]))
                    if len(last_items) < 2:
                        raise RuntimeError("Invalid past_key_value payload: expected at least key/value tensors.")
                    present_key_values[-1] = tuple([*last_items[:-1], readout_state_out, last_items[-1]])
            past_out = tuple(present_key_values)
        if not return_logits:
            logits = None
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

    def crop_block_size(self, block_size):
        block_size_int = int(block_size)
        if block_size_int <= 0:
            raise ValueError(f"block_size must be a positive integer, got {block_size_int}.")

        current = getattr(self.config, "block_size", None)
        if isinstance(current, int):
            current_int = int(current)
            if block_size_int > current_int:
                raise ValueError(f"block_size must be <= {current_int} to crop, got {block_size_int}.")

        setattr(self.config, "block_size", block_size_int)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = (
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.hidden_size // cfg.num_attention_heads,
            cfg.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding: bool = True) -> int:
        return super().get_num_params(non_embedding=non_embedding)

    def save_pretrained(self, save_directory: str, *args: Any, **kwargs: Any) -> None:
        super().save_pretrained(save_directory, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> Any:
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

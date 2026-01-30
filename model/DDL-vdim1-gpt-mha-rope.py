"""
Deep Delta Learning (DDL), scalar value limit (d_v=1), on top of GPT (MHA + RoPE).

Implements the Delta update:
    x_{l+1} = x_l + beta_l * (v_l - k_l^T x_l) * k_l

In this implementation, `k` is the output of the corresponding sublayer
(attention or MLP).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Any
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .activations import ActivationName, apply_activation, validate_activation_name
from .gpt_base import PastKeyValue
from .kv_cache import append_preallocated, get_past_len, maybe_get_cache_len
from .rmsnorm import RMSNorm
from .kv_shift import ShiftLinear
from .init_utils import init_gpt_weights
from .pydantic_config import validate_pretrained_config_kwargs
from .rotary import Rotary, apply_rotary_emb


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p) - math.log(1.0 - p)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.q_activation = validate_activation_name(getattr(config, "q_activation", None), field_name="q_activation")
        self.k_activation = validate_activation_name(getattr(config, "k_activation", None), field_name="k_activation")
        self.v_activation = validate_activation_name(getattr(config, "v_activation", None), field_name="v_activation")
        self.use_k_shift = getattr(config, "use_k_shift", False)
        self.use_v_shift = getattr(config, "use_v_shift", False)
        self.use_output_gate = getattr(config, "use_output_gate", False)
        # projections to per-head dimensions
        self.c_q = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        if self.use_k_shift:
            self.c_k = ShiftLinear(self.hidden_size, self.n_head * self.head_dim, self.n_head, bias=False)
        else:
            self.c_k = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        if self.use_v_shift:
            self.c_v = ShiftLinear(self.hidden_size, self.n_head * self.head_dim, self.n_head, bias=False)
        else:
            self.c_v = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        # output projection maps back to embedding dim
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.hidden_size, bias=False)
        # initialize attn output proj with reduced std: factor/sqrt(hidden_size)/sqrt(layers)
        with torch.no_grad():
            factor = getattr(config, "hidden_init_std_factor", 0.5)
            std = factor / math.sqrt(config.hidden_size) / math.sqrt(config.num_hidden_layers)
            self.c_proj.weight.normal_(mean=0.0, std=std)
        rope_ratio = float(getattr(config, "rope_ratio", 1.0))
        self.rotary = Rotary(self.head_dim, base=getattr(config, "rope_base", 10000.0), rope_ratio=rope_ratio)
        self.using_groupnorm = config.using_groupnorm
        # QK RMSNorm (learnable) flag and layers
        self.use_qk_rmsnorm = getattr(config, "use_qk_rmsnorm", True)
        if self.use_qk_rmsnorm:
            self.q_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
            self.k_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        if self.using_groupnorm:
            # Apply RMSNorm to each head's output dimension
            self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        if self.use_output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
            if not self.using_groupnorm:
                self.o_norm = RMSNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-5), elementwise_affine=True)

        kv_cache_slot_size = int(getattr(config, "kv_cache_slot_size", 128))
        if kv_cache_slot_size <= 0:
            raise ValueError(f"kv_cache_slot_size must be positive, got {kv_cache_slot_size}.")
        self.kv_cache_slot_size = kv_cache_slot_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_with_past(x)
        return y

    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        B, T, _ = x.size()  # batch size, sequence length, embedding dimensionality (hidden_size)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        if self.use_k_shift:
            k = self.c_k(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        if self.use_v_shift:
            v = self.c_v(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        q = apply_activation(q, self.q_activation)
        k = apply_activation(k, self.k_activation)
        v = apply_activation(v, self.v_activation)

        past_len = 0
        past_k: torch.Tensor | None = None
        past_v: torch.Tensor | None = None
        cache_len = maybe_get_cache_len(past_key_value, batch_size=B)
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
                past_k_used = past_k[:, :, :past_len, :]
                past_v_used = past_v[:, :, :past_len, :]

        cos, sin = self.rotary(q, seq_len_offset=past_len)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # Apply learnable RMSNorm to Q and K if enabled
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
            if attention_mask.ndim != 2:
                raise ValueError(f"attention_mask must have shape (B, S), got {tuple(attention_mask.shape)}")
            if int(attention_mask.shape[0]) != B:
                raise ValueError(f"attention_mask batch mismatch: expected {B}, got {int(attention_mask.shape[0])}")
            if int(attention_mask.shape[1]) != total_len:
                raise ValueError(
                    f"attention_mask sequence mismatch: expected {total_len}, got {int(attention_mask.shape[1])}"
                )
            attn_mask = attention_mask.to(dtype=torch.bool)[:, None, None, :]

        use_is_causal = past_len == 0 and attn_mask is None
        if not use_is_causal:
            if T > 1:
                key_positions = torch.arange(total_len, device=x.device)
                query_positions = past_len + torch.arange(T, device=x.device)
                causal_mask = key_positions <= query_positions[:, None]
            else:
                causal_mask = torch.ones((T, total_len), dtype=torch.bool, device=x.device)
            if attn_mask is not None:
                attn_mask = attn_mask & causal_mask[None, None, :, :]
            else:
                attn_mask = causal_mask[None, None, :, :]

        y = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, is_causal=use_is_causal)

        if self.using_groupnorm:
            # Apply RMSNorm directly to each head's output
            y = self.subln(y)
        elif self.use_output_gate:
            y = self.o_norm(y)

        if self.use_output_gate:
            gate = self.g_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            y = y * F.silu(gate)

        y = y.transpose(1, 2).contiguous().reshape(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)
        present: PastKeyValue | None = None
        if use_cache:
            if cache_tensors is None or cache_len_out is None:
                raise RuntimeError("KV cache append did not return expected cache tensors.")
            present = (cache_tensors[0], cache_tensors[1], cache_len_out)
        return y, present


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Calculate the floored hidden dimension size
        hidden_dim = math.floor(8 / 3 * config.hidden_size)

        # Split the linear projection into two parts for SwiGLU
        self.c_fc1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.hidden_size, hidden_dim, bias=False)

        # Output projection
        self.c_proj = nn.Linear(hidden_dim, config.hidden_size, bias=False)
        # initialize MLP output proj with reduced std: factor/sqrt(hidden_size)/sqrt(layers)
        with torch.no_grad():
            factor = getattr(config, "hidden_init_std_factor", 0.5)
            std = factor / math.sqrt(config.hidden_size) / math.sqrt(config.num_hidden_layers)
            self.c_proj.weight.normal_(mean=0.0, std=std)

    def forward(self, x):
        # Apply the first linear layer to produce two projections
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)

        # Apply the SwiGLU gating: SILU on one projection, and gate with the other
        x = F.silu(x1) * x2

        # Apply the final output projection
        x = self.c_proj(x)
        return x


class DeepDeltaResidualVdim1(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = int(config.hidden_size)

        self.k_eps = float(getattr(config, "ddl_k_eps", 1e-5))
        self.v_sigmoid = bool(getattr(config, "ddl_v_sigmoid", True))
        self.v_sigmoid_scale: float = float(getattr(config, "ddl_v_sigmoid_scale", 4.0))
        self.v_constant = bool(getattr(config, "ddl_v_dim1_constant", False))
        self.v_constant_value: float = float(getattr(config, "ddl_v_dim1_constant_value", 2.0))

        self.beta_single_linear = bool(getattr(config, "ddl_beta_single_linear", True))
        if self.beta_single_linear:
            self.beta = nn.Linear(hidden_size, 1, bias=True)
        else:
            beta_hidden_size = int(getattr(config, "ddl_beta_hidden_size", 128))
            if beta_hidden_size <= 0:
                raise ValueError("ddl_beta_hidden_size must be positive.")

            self.beta_in = nn.Linear(hidden_size, beta_hidden_size, bias=False)
            self.beta_out = nn.Linear(beta_hidden_size, 1, bias=True)

        # v is a scalar in the d_v=1 regime; project sublayer output to a scalar.
        self.v_proj = nn.Linear(hidden_size, 1, bias=True)

        beta_init = float(getattr(config, "ddl_beta_init", 0.0))
        beta_init = min(max(beta_init, 0.0), 2.0)
        beta_init_p = beta_init / 2.0
        with torch.no_grad():
            if self.beta_single_linear:
                self.beta.bias.fill_(_logit(beta_init_p))
            else:
                self.beta_out.bias.fill_(_logit(beta_init_p))

    def forward(self, x: torch.Tensor, *, k_in: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), k_in: (B, T, C), context: (B, T, C)
        # Keep large tensors in the model dtype; only compute `beta` in fp32 for stability.
        k_dim = int(k_in.size(-1))
        eps_rms = (self.k_eps * self.k_eps) / float(k_dim)
        k_rms = F.rms_norm(k_in, [k_dim], eps=eps_rms)
        k_scale = 1.0 / math.sqrt(k_dim)

        # beta(X) in [0, 2]
        if self.beta_single_linear:
            beta_logits = self.beta(context).float()
        else:
            beta_logits = self.beta_out(torch.tanh(self.beta_in(context))).float()
        beta = 2.0 * torch.sigmoid(beta_logits)  # fp32

        # k^T x, scalar projection (B, T, 1)
        proj_rms = torch.sum(k_rms * x, dim=-1, keepdim=True, dtype=torch.float32)  # fp32
        proj = proj_rms * k_scale

        if self.v_constant:
            v = torch.full_like(proj, self.v_constant_value)
        else:
            # v(X) is scalar for d_v=1.
            v = self.v_proj(x)
            if self.v_sigmoid:
                v = torch.sigmoid(v) * self.v_sigmoid_scale

        # x <- x + beta * k * (v - k^T x)
        delta_scaled = ((beta * (v - proj)) * k_scale).to(dtype=x.dtype)  # (B, T, 1)
        update = delta_scaled * k_rms
        return x + update


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ddl_attn = DeepDeltaResidualVdim1(config)
        self.ddl_mlp = DeepDeltaResidualVdim1(config)
        # Define RMSNorm layers once in the module
        self.ln_1 = RMSNorm(config.hidden_size)
        self.ln_2 = RMSNorm(config.hidden_size)

    def forward(self, x):
        # Apply pre-norm before sublayers
        x_norm = self.ln_1(x)
        k_attn = self.attn(x_norm)
        x = self.ddl_attn(x, k_in=k_attn, context=x_norm)

        x_norm = self.ln_2(x)
        k_mlp = self.mlp(x_norm)
        x = self.ddl_mlp(x, k_in=k_mlp, context=x_norm)
        return x

    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        x_norm = self.ln_1(x)
        k_attn, present = self.attn.forward_with_past(
            x_norm, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask
        )
        x = self.ddl_attn(x, k_in=k_attn, context=x_norm)

        x_norm = self.ln_2(x)
        k_mlp = self.mlp(x_norm)
        x = self.ddl_mlp(x, k_in=k_mlp, context=x_norm)
        return x, present


# -----------------------------------------------------------------------------
# The main GPT-2 model


@dataclass
class GPTConfig(PretrainedConfig):
    model_type = "nanogpt-pro"
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

    rope_ratio: float = 1.0  # Apply RoPE on the first rope_ratio*head_dim dimensions (must be in [0, 1])
    # Embedding init std (normal init for tied token embedding / LM head)
    embedding_init_std: float = 0.02
    # Factor for hidden (>=2D) param init; actual std = factor / sqrt(hidden_size)
    hidden_init_std_factor: float = 0.5
    # DDL (scalar value limit, d_v=1) knobs
    ddl_k_eps: float = 1e-5
    ddl_beta_hidden_size: int = 128
    ddl_beta_single_linear: bool = True
    ddl_v_sigmoid: bool = True
    ddl_v_sigmoid_scale: float = 4.0
    ddl_v_dim1_constant: bool = False
    ddl_v_dim1_constant_value: float = 2.0
    # Initialize beta; clamped to [0, 2]. Use 1.0 by default for baseline comparability.
    ddl_beta_init: float = 1.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**validate_pretrained_config_kwargs(type(self), kwargs))


class GPT(PreTrainedModel):
    config_class = GPTConfig
    base_model_prefix = "nanogpt-pro"
    supports_gradient_checkpointing = True

    def __init__(self, config):
        # if self is not a subclass of PreTrinedModel, then we need to call super().__init__()
        # else we can just call super().__init__(config) to handle the config argument
        if not isinstance(self, PreTrainedModel):
            super().__init__()
        else:
            super().__init__(config)
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size),
                h=nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)]),
            )
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # weight tying between token embedding and LM head
        self.tie_weights()  # https://paperswithcode.com/method/weight-tying
        # Final RMSNorm defined in the network
        self.ln_f = RMSNorm(config.hidden_size)
        init_gpt_weights(self, config)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        idx: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
        output_all_seq: bool = False,
        *,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[PastKeyValue, ...] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor | None, torch.Tensor | None]:
        del position_ids, cache_position, kwargs

        if (idx is None) == (input_ids is None):
            raise ValueError("Exactly one of `idx` or `input_ids` must be provided.")
        if idx is None:
            idx = input_ids

        if labels is not None and targets is not None:
            raise ValueError("Only one of `labels` or `targets` can be provided.")
        if targets is None:
            targets = labels

        use_cache_flag = bool(use_cache) if use_cache is not None else False
        return_dict_flag = bool(return_dict) if return_dict is not None else False
        output_hidden_states_flag = bool(output_hidden_states) if output_hidden_states is not None else False
        output_attentions_flag = bool(output_attentions) if output_attentions is not None else False

        if output_attentions_flag:
            raise NotImplementedError("output_attentions=True is not currently supported for DDL models.")

        if attention_mask is not None and bool(attention_mask.to(dtype=torch.bool).all().item()):
            attention_mask = None

        x = self.transformer.wte(idx)
        hidden_states: tuple[torch.Tensor, ...] | None = (x,) if output_hidden_states_flag else None

        present_key_values: list[PastKeyValue] | None = [] if use_cache_flag else None
        if past_key_values is not None and len(past_key_values) != len(self.transformer.h):
            raise ValueError(f"past_key_values must have length {len(self.transformer.h)}, got {len(past_key_values)}.")

        for layer_idx, block in enumerate(self.transformer.h):
            if use_cache_flag or past_key_values is not None or attention_mask is not None:
                past = past_key_values[layer_idx] if past_key_values is not None else None
                x, present = block.forward_with_past(
                    x,
                    past_key_value=past,
                    use_cache=use_cache_flag,
                    attention_mask=attention_mask,
                )
                if use_cache_flag:
                    if present is None:
                        raise RuntimeError("Block did not return past_key_value for KV cache.")
                    assert present_key_values is not None
                    present_key_values.append(present)
            else:
                x = block(x)

            if output_hidden_states_flag:
                assert hidden_states is not None
                hidden_states = (*hidden_states, x)

        x = self.ln_f(x)

        logits_scale = 1.0
        if getattr(self.config, "mup", False):
            logits_scale = float(getattr(self.config, "hidden_size_base", 1024)) / float(self.config.hidden_size)

        if targets is not None:
            logits = self.lm_head(x).float() * logits_scale
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            loss = None
            if output_all_seq or return_dict_flag:
                logits = self.lm_head(x) * logits_scale
            else:
                logits = self.lm_head(x[:, [-1], :]).float() * logits_scale

        if not return_logits:
            logits = None
        if not return_dict_flag:
            return logits, loss

        past_out: tuple[PastKeyValue, ...] | None = None
        if use_cache_flag:
            assert present_key_values is not None
            past_out = tuple(present_key_values)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_out,
            hidden_states=hidden_states,
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

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        # return n_params
        return n_params

    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        super().save_pretrained(save_directory, safe_serialization=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> Any:
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        if isinstance(model, GPT):
            model.tie_weights()
        return model

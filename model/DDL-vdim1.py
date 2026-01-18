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
from transformers.modeling_utils import PreTrainedModel

from .gpt_base import PastKeyValue
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
        self.use_k_shift = getattr(config, "use_k_shift", False)
        self.use_v_shift = getattr(config, "use_v_shift", False)
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
        past_len = 0
        past_k: torch.Tensor | None = None
        past_v: torch.Tensor | None = None
        if past_key_value is not None:
            if len(past_key_value) < 2:
                raise ValueError("past_key_value must have at least 2 tensors: (key, value).")
            past_k = past_key_value[0]
            past_v = past_key_value[1]
            past_len = int(past_k.shape[-2])

        cos, sin = self.rotary(q, seq_len_offset=past_len)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # Apply learnable RMSNorm to Q and K if enabled
        if self.use_qk_rmsnorm:
            q = self.q_rms(q)
            k = self.k_rms(k)

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        if past_k is not None and past_v is not None:
            k_t = torch.cat([past_k, k_t], dim=-2)
            v_t = torch.cat([past_v, v_t], dim=-2)

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

        y = y.transpose(1, 2).contiguous().reshape(B, T, self.n_head * self.head_dim)
        y = self.c_proj(y)
        present: PastKeyValue | None = None
        if use_cache:
            present = (k_t, v_t)
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

        self.k_eps = float(getattr(config, "ddl_k_eps", 1e-6))
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
    use_qk_rmsnorm: bool = True  # Apply learnable RMSNorm to Q and K in attention
    use_k_shift: bool = False
    use_v_shift: bool = False
    rope_ratio: float = 1.0  # Apply RoPE on the first rope_ratio*head_dim dimensions (must be in [0, 1])
    # Embedding init std (normal init for tied token embedding / LM head)
    embedding_init_std: float = 0.02
    # Factor for hidden (>=2D) param init; actual std = factor / sqrt(hidden_size)
    hidden_init_std_factor: float = 0.5
    # DDL (scalar value limit, d_v=1) knobs
    ddl_k_eps: float = 1e-6
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
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying
        # Final RMSNorm defined in the network
        self.ln_f = RMSNorm(config.hidden_size)
        init_gpt_weights(self, config)

    def forward(self, idx, targets=None, return_logits=True, output_all_seq=False):
        # forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, hidden_size)
        for block in self.transformer.h:
            x = block(x)
        # Apply final RMSNorm before the LM head
        x = self.ln_f(x)

        logits_scale = 1.0
        if getattr(self.config, "mup", False):
            logits_scale = float(getattr(self.config, "hidden_size_base", 1024)) / float(self.config.hidden_size)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            logits = logits * logits_scale
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        elif output_all_seq:
            logits = self.lm_head(x[:, :, :])  # note: using list [-1] to preserve the time dim
            logits = logits * logits_scale
            loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            logits = logits * logits_scale
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

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
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        return model

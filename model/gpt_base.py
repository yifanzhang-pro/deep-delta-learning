from __future__ import annotations

import math
from typing import Any, ClassVar, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .init_utils import init_gpt_weights
from .rmsnorm import RMSNorm

# Each layer cache is at least (key, value). Some attention implementations may append
# additional tensors (e.g., contextual state) after (key, value).
PastKeyValue = tuple[torch.Tensor, ...]


class _SupportsPastKeyValue(Protocol):
    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue | None]: ...


def _init_residual_proj_weight(weight: torch.Tensor, config: Any) -> None:
    with torch.no_grad():
        factor = float(getattr(config, "hidden_init_std_factor", 0.5))
        hidden_size = float(getattr(config, "hidden_size"))
        num_layers = float(getattr(config, "num_hidden_layers"))
        std = factor / math.sqrt(hidden_size) / math.sqrt(num_layers)
        weight.normal_(mean=0.0, std=std)


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
        self.c_proj = nn.Linear(hidden_dim, hidden_size, bias=False)
        _init_residual_proj_weight(self.c_proj.weight, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)
        x = F.silu(x1) * x2
        return self.c_proj(x)


class GPTBlock(nn.Module):
    def __init__(self, config: Any, attention_cls: type[nn.Module]) -> None:
        super().__init__()
        hidden_size = int(getattr(config, "hidden_size"))
        norm_eps = float(getattr(config, "rms_norm_eps", 1e-5))

        self.attn = attention_cls(config)
        self.mlp = SwiGLUMLP(config)
        self.ln_1 = RMSNorm(hidden_size, eps=norm_eps)
        self.ln_2 = RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        attn_in = self.ln_1(x)
        present: PastKeyValue | None = None
        if getattr(self.attn, "forward_with_past", None) is not None:
            attn_impl = cast(_SupportsPastKeyValue, self.attn)
            attn_out, present = attn_impl.forward_with_past(
                attn_in, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask
            )
        else:
            attn_out = self.attn(attn_in)

        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPTBase(PreTrainedModel, GenerationMixin):
    config_class = PretrainedConfig
    base_model_prefix = "nanogpt-pro"
    supports_gradient_checkpointing = True

    attention_cls: ClassVar[type[nn.Module]]
    # Mark tied embeddings to make `safe_serialization=True` compatible with safetensors.
    # This matches the convention used by Hugging Face GPT2LMHeadModel.
    _tied_weights_keys: ClassVar[list[str]] = ["lm_head.weight"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = ["lm_head.weight"]
    init_exclude_suffixes: ClassVar[tuple[str, ...]] = ()
    init_exclude_names: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: PretrainedConfig) -> None:
        # transformers>=4.48 expects configs to expose an internal attention-impl field.
        # Some of our custom GPTConfig classes are thin PretrainedConfig subclasses and may not
        # define it; set a safe default to keep PreTrainedModel initialization happy.
        if not hasattr(config, "_attn_implementation_internal"):
            # Prefer eager attention unless the config explicitly chose something else.
            setattr(config, "_attn_implementation_internal", getattr(config, "_attn_implementation", "eager"))

        # transformers GenerationConfig and config helpers expect these common PretrainedConfig fields.
        # Our configs are decoder-only models.
        if not hasattr(config, "is_encoder_decoder"):
            setattr(config, "is_encoder_decoder", False)
        if not hasattr(config, "is_decoder"):
            setattr(config, "is_decoder", True)
        if not hasattr(config, "add_cross_attention"):
            setattr(config, "add_cross_attention", False)
        if not hasattr(config, "tie_word_embeddings"):
            setattr(config, "tie_word_embeddings", True)
        if not hasattr(config, "model_type"):
            # Used by HF serialization and GenerationConfig.from_model_config.
            setattr(config, "model_type", "nanogpt")

        # Used by transformers tie_weights / TorchScript paths.
        if not hasattr(config, "torchscript"):
            setattr(config, "torchscript", False)

        super().__init__(config)
        self.config = config

        attention_cls = getattr(type(self), "attention_cls", None)
        if attention_cls is None:  # pragma: no cover
            raise TypeError(f"{type(self).__name__} must define attention_cls.")

        vocab_size = int(getattr(config, "vocab_size"))
        hidden_size = int(getattr(config, "hidden_size"))
        num_layers = int(getattr(config, "num_hidden_layers"))
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, hidden_size),
                h=nn.ModuleList([GPTBlock(config, attention_cls) for _ in range(num_layers)]),
            )
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        norm_eps = float(getattr(config, "rms_norm_eps", 1e-5))
        self.ln_f = RMSNorm(hidden_size, eps=norm_eps)
        init_gpt_weights(
            self,
            config,
            exclude_suffixes=tuple(getattr(type(self), "init_exclude_suffixes", ())),
            exclude_names=tuple(getattr(type(self), "init_exclude_names", ())),
        )
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(self, value: nn.Module) -> None:
        if not isinstance(value, nn.Embedding):
            raise TypeError(f"Expected nn.Embedding, got {type(value).__name__}")
        self.transformer.wte = value
        self.tie_weights()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if not isinstance(new_embeddings, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(new_embeddings).__name__}")
        self.lm_head = new_embeddings
        self.tie_weights()

    def tie_weights(self) -> None:
        super().tie_weights()
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
        del cache_position, kwargs

        if (idx is None) == (input_ids is None):
            raise ValueError("Exactly one of `idx` or `input_ids` must be provided.")
        if idx is None:
            idx = input_ids

        if labels is not None and targets is not None:
            raise ValueError("Only one of `labels` or `targets` can be provided.")
        if targets is None:
            targets = labels

        use_cache = bool(use_cache) if use_cache is not None else False
        return_dict = bool(return_dict) if return_dict is not None else False
        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False
        output_attentions = bool(output_attentions) if output_attentions is not None else False

        if output_attentions:
            raise NotImplementedError("output_attentions=True is not currently supported for GPTBase models.")

        if attention_mask is not None and bool(attention_mask.to(dtype=torch.bool).all().item()):
            attention_mask = None

        x = self.transformer.wte(idx)
        hidden_states: tuple[torch.Tensor, ...] | None = (x,) if output_hidden_states else None

        present_key_values: list[PastKeyValue] | None = [] if use_cache else None
        if past_key_values is not None and len(past_key_values) != len(self.transformer.h):
            raise ValueError(f"past_key_values must have length {len(self.transformer.h)}, got {len(past_key_values)}.")

        for layer_idx, block in enumerate(self.transformer.h):
            if use_cache or past_key_values is not None or attention_mask is not None:
                past = past_key_values[layer_idx] if past_key_values is not None else None
                x, present = block.forward_with_past(
                    x, past_key_value=past, use_cache=use_cache, attention_mask=attention_mask
                )
                if use_cache:
                    if present is None:
                        raise NotImplementedError(
                            f"{type(block.attn).__name__} does not implement `forward_with_past` needed for KV cache."
                        )
                    assert present_key_values is not None
                    present_key_values.append(present)
            else:
                x = block(x)

            if output_hidden_states:
                assert hidden_states is not None
                hidden_states = (*hidden_states, x)

        x = self.ln_f(x)

        logits_scale = 1.0
        if bool(getattr(self.config, "mup", False)):
            logits_scale = float(getattr(self.config, "hidden_size_base", 1024)) / float(self.config.hidden_size)

        if targets is not None:
            logits = self.lm_head(x).float() * logits_scale
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            loss = None
            if output_all_seq or return_dict:
                logits = self.lm_head(x) * logits_scale
            else:
                logits = self.lm_head(x[:, [-1], :]).float() * logits_scale

        if not return_logits:
            logits = None
        if not return_dict:
            return logits, loss

        past_out: tuple[PastKeyValue, ...] | None = None
        if use_cache:
            assert present_key_values is not None
            past_out = tuple(present_key_values)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_out,
            hidden_states=hidden_states,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[PastKeyValue, ...] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.to(dtype=torch.long).cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            position_ids = position_ids[:, -input_ids.shape[1] :]

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
            reordered.append(tuple(tensor.index_select(0, beam_idx) for tensor in layer_past))
        return tuple(reordered)

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
        N = self.get_num_params()
        cfg = self.config
        L = int(getattr(cfg, "num_hidden_layers"))
        H = int(getattr(cfg, "num_attention_heads"))
        head_dim = int(getattr(cfg, "head_dim", int(getattr(cfg, "hidden_size")) // H))
        T = int(getattr(cfg, "block_size"))
        flops_per_token = 6 * N + 12 * L * H * head_dim * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_directory: str, *args: Any, **kwargs: Any) -> None:
        super().save_pretrained(save_directory, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> Any:
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        if isinstance(model, GPTBase):
            model.tie_weights()
        return model

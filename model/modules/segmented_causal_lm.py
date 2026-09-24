"""Packed DDL execution, including sequence-local embedding and readout state."""

from __future__ import annotations

from typing import cast

import torch
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..gpt_base import (
    CausalLMForwardTuple,
    DecoderOnlyCausalLMPreTrainedModel,
    PastKeyValue,
    causal_lm_output_to_tuple,
    resolve_input_ids_and_embeds,
)
from .segmented_attention import packed_sequence_boundaries


def forward_packed_ddl(
    model: DecoderOnlyCausalLMPreTrainedModel,
    *,
    idx: torch.Tensor | None,
    targets: torch.Tensor | None,
    return_logits: bool,
    output_all_seq: bool,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    labels: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    past_key_values: tuple[PastKeyValue, ...] | None,
    use_cache: bool | None,
    output_hidden_states: bool | None,
    output_attentions: bool | None,
    return_dict: bool | None,
    cache_position: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    max_seqlen: int | None,
) -> CausalLMOutputWithPast | CausalLMForwardTuple:
    """Batch equal-length documents through the DDL backend and reduce loss globally."""
    hf_style = input_ids is not None or inputs_embeds is not None
    supervised = targets is not None or labels is not None
    cache_enabled = (
        bool(use_cache)
        if use_cache is not None
        else (bool(getattr(model.config, "use_cache", False)) if hf_style and not supervised else False)
    )
    if cache_enabled or any(
        value is not None for value in (past_key_values, attention_mask, position_ids, cache_position)
    ):
        raise ValueError(
            "cu_seqlens cannot be combined with KV cache, attention_mask, position_ids, or cache_position."
        )
    if output_attentions:
        raise NotImplementedError("output_attentions=True is not currently supported for DDL models.")
    if labels is not None and targets is not None:
        raise ValueError("Only one of `labels` or `targets` can be provided.")
    targets = labels if targets is None else targets
    idx, inputs_embeds, batch_size, sequence_length = resolve_input_ids_and_embeds(idx, input_ids, inputs_embeds)
    inputs = inputs_embeds if inputs_embeds is not None else idx
    assert inputs is not None
    boundaries = packed_sequence_boundaries(inputs, cu_seqlens, max_seqlen)
    flattened = inputs.reshape(batch_size * sequence_length, *inputs.shape[2:])
    logits_parts: list[torch.Tensor] = []
    hidden_parts: list[tuple[torch.Tensor, ...]] = []
    groups: dict[int, list[int]] = {}
    for start, end in boundaries:
        groups.setdefault(end - start, []).append(start)
    token_indices: list[torch.Tensor] = []
    for length, starts in sorted(groups.items()):
        indices = (
            torch.tensor(starts, device=inputs.device, dtype=torch.long)[:, None]
            + torch.arange(length, device=inputs.device)[None, :]
        ).flatten()
        token_indices.append(indices)
        segment = flattened.index_select(0, indices).reshape(len(starts), length, *inputs.shape[2:])
        # The ordinary forward also resets DDL embedding/readout shortconvs,
        # which live outside transformer blocks in several DDL architectures.
        output = model(
            idx=segment if inputs_embeds is None else None,
            inputs_embeds=segment if inputs_embeds is not None else None,
            use_cache=False,
            return_dict=True,
            output_hidden_states=output_hidden_states,
        )
        if not isinstance(output, CausalLMOutputWithPast) or output.logits is None:
            raise RuntimeError("DDL sequence forward must return CausalLMOutputWithPast with logits.")
        logits_parts.append(output.logits.flatten(0, 1))
        if output_hidden_states:
            if output.hidden_states is None:
                raise RuntimeError("DDL sequence forward did not return requested hidden states.")
            hidden_parts.append(tuple(hidden.flatten(0, 1) for hidden in output.hidden_states))
    indices = torch.cat(token_indices)

    def restore_order(parts: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
        grouped = torch.cat(parts, dim=0)
        return torch.empty_like(grouped).index_copy(0, indices, grouped).reshape(batch_size, sequence_length, -1)

    logits = restore_order(logits_parts)
    loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.reshape(-1), ignore_index=-1)
    hidden_states: tuple[torch.Tensor, ...] | None = None
    if hidden_parts:
        hidden_states = tuple(restore_order(parts) for parts in zip(*hidden_parts, strict=True))
    if targets is None and not (output_all_seq or return_dict is not None or hf_style):
        logits = logits[:, [-1], :]
    result_logits = cast(torch.FloatTensor | None, logits if return_logits else None)
    result_hidden = cast(tuple[torch.FloatTensor, ...] | None, hidden_states)
    return_dict_flag = (
        bool(return_dict)
        if return_dict is not None
        else (bool(getattr(model.config, "return_dict", False)) if hf_style else False)
    )
    if return_dict_flag:
        return CausalLMOutputWithPast(loss=loss, logits=result_logits, hidden_states=result_hidden)
    return causal_lm_output_to_tuple(
        loss=loss, logits=result_logits, past_key_values=None, hidden_states=hidden_states, attentions=None
    )

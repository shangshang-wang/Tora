# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch

from torchtune import utils
from torchtune.generation import (
    generate_next_token,
    get_causal_mask_from_padding_mask,
    sample,
)
from torchtune.generation._generation import (
    get_position_ids_from_padding_mask,
    update_stop_tokens_tracker,
)
from torchtune.modules import TransformerDecoder

from tqdm.auto import trange


# NOTE: This is almost the same as torchtune.generation.generate, with a few changes necessary for GRPO.
# Namely:
#   1. The `return_logits` argument - we can optionally omit keeping track of logits during generation, which
#        drastically improves generation speed.
#   2. Stop token-based breaking now communicates across multiple devices in a distributed setting.
# TODO: Figure out the right abstractions to be used in the main repository, and remove this function.
@torch.no_grad()
def generate(
    model: TransformerDecoder,
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Optional[list[int]] = None,
    rng: Optional[torch.Generator] = None,
    custom_generate_next_token: Optional[Callable] = None,
    return_logits: bool = True,
    model_inputs: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generates tokens from a model conditioned on a prompt, and also returns logits for the generations.

    Args:
        model (TransformerDecoder): model used for generation
        prompt (torch.Tensor): tensor with the token IDs associated with the given prompt,
            with shape either [seq_length] or [bsz x seq_length].
        max_generated_tokens (int): number of tokens to be generated
        pad_id (int): token ID to use for padding, default 0.
        temperature (float): value to scale the predicted logits by, default 1.0.
        top_k (Optional[int]): If specified, we prune the sampling to only token ids within the top_k probabilities,
            default None.
        stop_tokens (Optional[list[int]]): If specified, generation is stopped when any of these tokens are generated,
            default None.
        rng (Optional[torch.Generator]): random number generator, default None.
        custom_generate_next_token (Optional[Callable]): If specified, we'll use the
            ``custom_generate_next_token function``. This is generally only useful if
            you want to specify a ``torch.compile`` version of the generate next token for
            performance reasons. If None, we use the default :func:`generate_next_token`.
            Default is None.
        return_logits (bool): whether to return logits associated with the generated tokens, default True.
        model_inputs (Optional[dict[str, torch.Tensor]]): Optional keyword arguments passed to ``model``
            during generation. This is useful for multimodal decoders such as Qwen3-VL that expect
            ``pixel_values``/``image_grid_thw``/``visual_pos_masks`` alongside the token stream.
            When provided, these inputs will be consumed during the initial prompt forward pass and,
            if KV-caching is disabled, on every subsequent decoding step.

    Note:
        This function has only been tested with decoder-only models.

    Examples:
        >>> model = torchtune.models.llama3.llama3_8b()
        >>> tokenizer = torchtune.models.llama3.llama3_tokenizer()
        >>> prompt = tokenizer.encode("Hi my name is")
        >>> rng.manual_seed(42)
        >>> output, logits = generate(model, torch.tensor(prompt), max_generated_tokens=100, pad_id=0)
        >>> print(tokenizer.decode(output[0].tolist()))
        Hi my name is Jeremy and I'm a friendly language model assistant!

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of two tensors:
            - tokens (torch.Tensor): tensor with the generated tokens,
                with shape ``[bsz x seq_len + num_generated_tokens]`` where ``num_generated_tokens``
                may be less than ``max_generated_tokens`` if ``stop_tokens`` are provided.
            - logits (torch.Tensor): tensor with the logits associated with the generated tokens,
                with shape ``[bsz x num_generated_tokens x vocab_size]``.
    """
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    if custom_generate_next_token is None:
        custom_generate_next_token = generate_next_token

    bsz, prompt_length = prompt.size()
    total_response_length = prompt_length + max_generated_tokens

    generated_tokens = prompt.clone()
    incremental_decoding = model.caches_are_enabled()

    model_kwargs_base: dict[str, torch.Tensor] = {}
    visual_pos_masks = None
    if model_inputs is not None:
        model_kwargs_base = dict(model_inputs)
        visual_pos_masks = model_kwargs_base.pop("visual_pos_masks", None)

    multimodal_active = bool(model_kwargs_base) or visual_pos_masks is not None

    def _build_model_kwargs() -> dict[str, torch.Tensor]:
        if not multimodal_active:
            return {}
        kwargs = dict(model_kwargs_base)
        if visual_pos_masks is not None:
            kwargs["visual_pos_masks"] = visual_pos_masks
        return kwargs

    def _generate_with_model_kwargs(
        x: torch.Tensor,
        input_pos: torch.Tensor,
        mask: Optional[torch.Tensor],
        q: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = _build_model_kwargs()
        logits = model(x, input_pos=input_pos, mask=mask, **kwargs)[:, -1]
        next_tokens = sample(
            logits.clone(), temperature=temperature, top_k=top_k, q=q
        )
        return next_tokens, logits.unsqueeze(1)

    def _maybe_generate_next_token(
        x: torch.Tensor,
        input_pos: torch.Tensor,
        mask: Optional[torch.Tensor],
        q: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if multimodal_active:
            return _generate_with_model_kwargs(x, input_pos, mask, q)
        return custom_generate_next_token(
            model,
            input_pos=input_pos,
            mask=mask,
            x=x,
            temperature=temperature,
            top_k=top_k,
            q=q,
        )

    def _extend_visual_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        pad = torch.zeros(
            (mask.shape[0], 1), dtype=mask.dtype, device=mask.device
        )
        return torch.cat([mask, pad], dim=1)

    # grab the correct max_seq_len to generate full causal masks/position ids
    # this is the model's max cache len if incremental decoding, or the sequence
    # length otherwise
    max_seq_len = (
        total_response_length
        if not incremental_decoding
        else model.decoder_max_cache_seq_len
    )

    padding_masks = generated_tokens != pad_id

    if not padding_masks.all():
        # we have padding in the prompt due to varying-length sequences in a batch
        # extend padding masks out to the correct seq len
        padding_masks = torch.nn.functional.pad(
            padding_masks, (0, max_generated_tokens), value=True
        )

        # generate the full causal mask for the whole padding mask with padding ignored
        masks = get_causal_mask_from_padding_mask(
            padding_masks, target_seq_len=max_seq_len
        )

        # right-shift position IDs to account for padding
        input_pos = get_position_ids_from_padding_mask(padding_masks)
    else:
        # just use a regular causal mask if there is no padding
        masks = torch.tril(
            torch.ones(
                total_response_length,
                max_seq_len,
                dtype=torch.bool,
                device=prompt.device,
            )
        ).unsqueeze(0)
        input_pos = torch.arange(
            0, total_response_length, device=generated_tokens.device
        ).unsqueeze(0)

    if incremental_decoding:
        # if KV-caches are enabled, we need a causal mask of shape [bsz, prompt_length, max_cache_len]
        # to match the key/value cache tensor shapes
        curr_masks = masks[:, :prompt_length]
    else:
        # otherwise the causal mask is shape [bsz, prompt_length, prompt_length] because key/value
        # tensors are of identical shape to the prompt
        curr_masks = masks[:, :prompt_length, :prompt_length]

    q = None
    if rng is not None:
        q = torch.empty(
            (bsz, model.tok_embeddings.num_embeddings), device=prompt.device
        ).exponential_(1, generator=rng)
    tokens, generated_logits = _maybe_generate_next_token(
        prompt,
        input_pos=input_pos[:, :prompt_length].squeeze(),
        mask=curr_masks,
        q=q,
    )

    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    if not incremental_decoding and multimodal_active:
        visual_pos_masks = _extend_visual_mask(visual_pos_masks)
    elif incremental_decoding and multimodal_active:
        model_kwargs_base = {}
        visual_pos_masks = None
        multimodal_active = False

    curr_pos = prompt_length

    # keeps track at a high level if we've already hit a stop token in a sequence so we can early stop
    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
    stop_tokens = (
        torch.tensor(stop_tokens, device=prompt.device, dtype=tokens.dtype)
        if stop_tokens
        else None
    )

    # everything in stop_token_mask starts as 1s, and we'll set them to 0 for sequences
    # that already hit a stop token
    stop_token_mask = torch.ones(
        (bsz, prompt_length + 1), dtype=torch.int32, device=prompt.device
    )

    # stop early if we reach a stop token in every seq
    if stop_tokens is not None:
        stop_token_reached = update_stop_tokens_tracker(
            tokens, stop_tokens, stop_token_reached
        )
        if stop_token_reached.all().item():
            return generated_tokens, generated_logits if return_logits else None

    world_size, rank = utils.get_world_size_and_rank()
    for _ in (pbar := trange(max_generated_tokens - 1, leave=False, disable=rank > 0)):
        # update stop_token_mask if we reached a stop token in a previous step
        # by appending the logical not of stop_token_reached to the end of the mask
        # reshaped to be bsz first
        if stop_tokens is not None:
            stop_token_mask = torch.cat(
                [stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1
            )

        # if incremental decoding is enabled, we can use the current position
        # otherwise, we take the whole sequence up to the current position
        if incremental_decoding:
            curr_input_pos = input_pos[:, curr_pos].contiguous()
            curr_masks = masks[:, curr_pos, None, :].contiguous()
        else:
            tokens = generated_tokens.clone()
            curr_input_pos = input_pos[:, : curr_pos + 1]
            curr_masks = masks[:, : curr_pos + 1, : curr_pos + 1]

        q = None
        if rng is not None:
            q = torch.empty(
                (bsz, model.tok_embeddings.num_embeddings), device=prompt.device
            ).exponential_(1, generator=rng)

        tokens, logits = _maybe_generate_next_token(
            tokens.clone(),
            curr_input_pos,
            mask=curr_masks,
            q=q,
        )
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        if return_logits:
            generated_logits = torch.cat([generated_logits, logits], dim=1)
        if not incremental_decoding and multimodal_active:
            visual_pos_masks = _extend_visual_mask(visual_pos_masks)
        curr_pos += 1

        if stop_tokens is not None:
            stop_token_reached = update_stop_tokens_tracker(
                tokens, stop_tokens, stop_token_reached
            )
            if world_size == 1:
                # Single device
                if stop_token_reached.all():
                    break
            else:
                all_done = stop_token_reached.all().int()
                torch.distributed.all_reduce(all_done)
                if all_done == world_size:
                    # Multiple devices
                    break

    # mask out generated tokens in seqs that already hit a stop token
    if stop_tokens is not None:
        generated_tokens *= stop_token_mask
        if return_logits:
            generated_logits *= stop_token_mask[:, -generated_logits.shape[1] :, None]

    return generated_tokens, generated_logits

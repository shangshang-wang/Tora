# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from torchtune.models.qwen3_vl._encoder import Qwen3VLEncoder
from torchtune.models.qwen3_vl.text import Qwen3VLTextModel


@dataclass
class Qwen3VLModelOutput:
    last_hidden_state: Tensor
    visual_pos_masks: Optional[Tensor]
    deepstack_visual_embeds: Optional[Sequence[Tensor]]


class Qwen3VLModel(nn.Module):
    """Lightweight Qwen3-VL container that wires the vision tower to the text decoder."""

    def __init__(self, vision: Qwen3VLEncoder, text: Qwen3VLTextModel) -> None:
        super().__init__()
        self.vision = vision
        self.text = text
        self.decoder_max_cache_seq_len: Optional[int] = None
        self.rope_deltas: Optional[Tensor] = None
        self.image_token_id: Optional[int] = None
        self.video_token_id: Optional[int] = None
        self.vision_start_token_id: Optional[int] = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.text.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.text.embed_tokens = embeddings

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ) -> None:
        _ = encoder_max_seq_len
        if decoder_max_seq_len is not None:
            self.decoder_max_cache_seq_len = decoder_max_seq_len
        else:
            self.decoder_max_cache_seq_len = self.text.config.max_position_embeddings
        self.text.setup_caches(batch_size, dtype, max_seq_len=self.decoder_max_cache_seq_len)

    def caches_are_setup(self) -> bool:
        return self.text.caches_are_setup()

    def caches_are_enabled(self) -> bool:
        return self.text.caches_are_enabled()

    def reset_caches(self) -> None:
        self.text.reset_caches()

    def forward(
        self,
        tokens: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        visual_pos_masks: Optional[Tensor] = None,
        deepstack_visual_embeds: Optional[Sequence[Tensor]] = None,
        **unused_kwargs: object,
    ) -> Qwen3VLModelOutput:

        # Note: Tokens may be provided positionally by generation utilities that follow
        # the TransformerDecoder API; keep remaining arguments keyword-only in downstream code.
        # Some dataloaders pass auxiliary multimodal fields such as encoder_input or encoder_mask.
        # Ignore them since the Qwen3 vision-text wiring consumes pixel_values/image_grid_thw directly.
        unused_kwargs.pop("encoder_input", None)
        unused_kwargs.pop("encoder_mask", None)
        unused_kwargs.pop("timestamps", None)
        unused_kwargs.pop("pixel_values_videos", None)

        video_grid_thw = unused_kwargs.pop("video_grid_thw", None)
        legacy_visual_mask = unused_kwargs.pop("visual_pos_mask", None)

        # if 'input_pos' in unused_kwargs:
        #     position_ids = unused_kwargs.pop("input_pos")
        #     if len(position_ids.shape) == 1:
        #         position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(tokens)
            
        if visual_pos_masks is None:
            visual_pos_masks = legacy_visual_mask

        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw must be supplied when pixel_values are provided.")
            merged, deepstack_feats = self.vision(pixel_values, grid_thw=image_grid_thw)
            deepstack_visual_embeds = deepstack_feats
            if inputs_embeds is None and tokens is not None:
                inputs_embeds = self.text.embed_tokens(tokens)
            if inputs_embeds is None:
                raise ValueError("inputs_embeds must be set before injecting vision features.")
            if visual_pos_masks is None:
                raise ValueError("visual_pos_masks must point to vision placeholder positions.")
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[visual_pos_masks] = merged.to(inputs_embeds.dtype)
        position_ids_provided = position_ids is not None
        if tokens is None and position_ids is None:
            raise ValueError("tokens must be provided to compute rotary position ids.")
        if position_ids is None:
            flat_attention_mask = mask
            if isinstance(mask, torch.Tensor) and mask.ndim != 2:
                flat_attention_mask = None
            position_ids, self.rope_deltas = self.get_rope_index(
                input_ids=tokens,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=flat_attention_mask,
            )
        else:
            batch_dim = position_ids.shape[1]
            self.rope_deltas = torch.zeros(
                (batch_dim, 1), device=position_ids.device, dtype=position_ids.dtype
            )

        kv_cache = self.text.layers[0].self_attn.kv_cache if self.text.layers else None
        cache_active = (
            kv_cache is not None and self.text.layers[0].self_attn.cache_enabled
        )
        if cache_active and kv_cache.size != 0 and not position_ids_provided:
            position_ids = position_ids + kv_cache.size
        hidden_states = self.text(
            tokens=None,
            inputs_embeds=inputs_embeds,
            mask=mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        return Qwen3VLModelOutput(
            last_hidden_state=hidden_states,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def get_rope_index(
        self,
        input_ids: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        video_grid_thw: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if input_ids is None:
            raise ValueError("input_ids must be provided to compute RoPE indices.")

        if image_grid_thw is not None:
            if not torch.is_tensor(image_grid_thw):
                image_grid_thw = torch.as_tensor(image_grid_thw, device=input_ids.device)
            else:
                image_grid_thw = image_grid_thw.to(device=input_ids.device)
            image_grid_thw = image_grid_thw.clone()
        if video_grid_thw is not None:
            if not torch.is_tensor(video_grid_thw):
                video_grid_thw = torch.as_tensor(video_grid_thw, device=input_ids.device)
            else:
                video_grid_thw = video_grid_thw.to(device=input_ids.device)
            video_grid_thw = video_grid_thw.clone()
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = getattr(self.vision.tower, "spatial_merge_size", 1)
        image_token_id = getattr(self, "image_token_id", None)
        video_token_id = getattr(self, "video_token_id", None)
        vision_start_token_id = getattr(self, "vision_start_token_id", None)

        mrope_position_deltas: list[Tensor] = []
        if (
            image_token_id is not None
            and vision_start_token_id is not None
            and (image_grid_thw is not None or video_grid_thw is not None)
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids, dtype=torch.long)

            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index = 0
            video_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for batch_idx, ids in enumerate(total_input_ids):
                valid_ids = ids[attention_mask[batch_idx] == 1]
                vision_start_indices = torch.argwhere(valid_ids == vision_start_token_id).squeeze(1)
                if vision_start_indices.numel() == 0:
                    position_ids[..., batch_idx, attention_mask[batch_idx] == 1] = torch.arange(
                        valid_ids.numel(), device=position_ids.device
                    )
                    mrope_position_deltas.append(torch.tensor(0, device=input_ids.device, dtype=input_ids.dtype))
                    continue

                vision_tokens = valid_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == (video_token_id or -1)).sum()

                tokens_list = valid_ids.tolist()
                llm_pos_ids_list: list[Tensor] = []
                st = 0
                remain_images = int(image_nums)
                remain_videos = int(video_nums)

                for _ in range(remain_images + remain_videos):
                    if remain_images > 0 and image_token_id in tokens_list[st:]:
                        ed_image = tokens_list.index(image_token_id, st)
                    else:
                        ed_image = len(tokens_list) + 1
                    if remain_videos > 0 and video_token_id is not None and video_token_id in tokens_list[st:]:
                        ed_video = tokens_list.index(video_token_id, st)
                    else:
                        ed_video = len(tokens_list) + 1

                    use_image = ed_image < ed_video
                    if use_image:
                        if image_grid_thw is None:
                            raise ValueError("image_grid_thw must be provided when image tokens are present.")
                        t, h, w = image_grid_thw[image_index]
                        image_index += 1
                        remain_images -= 1
                        segment_end = ed_image
                    else:
                        if video_grid_thw is None:
                            raise ValueError("video_grid_thw must be provided when video tokens are present.")
                        t, h, w = video_grid_thw[video_index]
                        video_index += 1
                        remain_videos -= 1
                        segment_end = ed_video

                    llm_grid_t = int(t.item())
                    llm_grid_h = int(h.item() // spatial_merge_size)
                    llm_grid_w = int(w.item() // spatial_merge_size)
                    text_len = segment_end - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                    )

                    # t_index always zero because timestamps encode temporal dimension
                    h_index = torch.arange(llm_grid_h, device=input_ids.device).view(1, -1, 1).expand(
                        llm_grid_t, -1, llm_grid_w
                    )
                    w_index = torch.arange(llm_grid_w, device=input_ids.device).view(1, 1, -1).expand(
                        llm_grid_t, llm_grid_h, -1
                    )
                    t_index = torch.arange(llm_grid_t, device=input_ids.device).view(-1, 1).expand(
                        -1, llm_grid_h * llm_grid_w
                    )
                    stacked = torch.stack(
                        [
                            t_index.flatten(),
                            h_index.flatten(),
                            w_index.flatten(),
                        ]
                    )
                    llm_pos_ids_list.append(stacked + text_len + st_idx)
                    st = segment_end + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(tokens_list):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(tokens_list) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., batch_idx, attention_mask[batch_idx] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - valid_ids.numel())

            mrope_tensor = torch.stack(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_tensor

        if attention_mask is not None:
            mask_positions = attention_mask.long().cumsum(-1) - 1
            mask_positions.masked_fill_(attention_mask == 0, 1)
            position_ids = mask_positions.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                (input_ids.shape[0], 1),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            return position_ids, mrope_position_deltas

        return position_ids, mrope_position_deltas


class Qwen3VLForConditionalGeneration(nn.Module):
    """Minimal conditional generation head on top of Qwen3-VL."""

    def __init__(self, model: Qwen3VLModel, vocab_size: int) -> None:
        super().__init__()
        self.model = model
        self.lm_head = nn.Linear(model.text.config.hidden_size, vocab_size, bias=False)
        self.output = self.lm_head
        self.skip_output_layer = False
        self.decoder_max_cache_seq_len: Optional[int] = None
        self.tie_weights()
        self.tok_embeddings = self.model.get_input_embeddings()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.model.set_input_embeddings(embeddings)
        self.tok_embeddings = embeddings
        self.tie_weights()

    def tie_weights(self) -> None:
        """Share decoder input/output embeddings when vocab matches."""
        self.lm_head.weight = self.model.get_input_embeddings().weight

    def setup_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        *,
        encoder_max_seq_len: Optional[int] = None,
        decoder_max_seq_len: Optional[int] = None,
    ) -> None:
        self.model.setup_caches(
            batch_size,
            dtype,
            encoder_max_seq_len=encoder_max_seq_len,
            decoder_max_seq_len=decoder_max_seq_len,
        )
        self.decoder_max_cache_seq_len = self.model.decoder_max_cache_seq_len

    def caches_are_setup(self) -> bool:
        return self.model.caches_are_setup()

    def caches_are_enabled(self) -> bool:
        return self.model.caches_are_enabled()

    def reset_caches(self) -> None:
        self.model.reset_caches()

    def forward(self, *args, **kwargs) -> Tensor:
        outputs = self.model(*args, **kwargs)
        hidden = outputs.last_hidden_state
        if self.skip_output_layer:
            return hidden
        logits = self.lm_head(hidden)
        return logits

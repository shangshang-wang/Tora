#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

from torchtune.data import Message
from torchtune.data._prompt_templates import _TemplateType
from torchtune.models.qwen3_vl._tokenizer import qwen3_vl_tokenizer
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import ModelTokenizer

try:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
        Qwen2VLImageProcessorFast,
    )
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
except ImportError:  # pragma: no cover - optional dependency
    Qwen2VLImageProcessorFast = None  # type: ignore[assignment]
    Qwen2VLVideoProcessor = None  # type: ignore[assignment]


class Qwen3VLTransform(ModelTokenizer, Transform):
    """Text + vision transform aligned with the Hugging Face Qwen3-VL processor."""

    def __init__(
        self,
        vocab_path: str,
        merges_path: str,
        *,
        spatial_merge_size: int,
        image_processor_config: Optional[dict[str, Any]] = None,
        video_processor_config: Optional[dict[str, Any]] = None,
        special_tokens_path: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[_TemplateType] = None,
        truncation_type: str = "right",
    ) -> None:
        self.spatial_merge_size = spatial_merge_size
        self.tokenizer = qwen3_vl_tokenizer(
            vocab_path,
            merges_path,
            special_tokens_path=special_tokens_path,
            max_seq_len=max_seq_len,
            prompt_template=prompt_template,
            truncation_type=truncation_type,
            spatial_merge_size=spatial_merge_size,
        )
        # Qwen3-VL requires left padding when mixing prompts of different lengths,
        # so surface this preference on both the transform and tokenizer.
        self.padding_side = "left"
        setattr(self.tokenizer, "padding_side", "left")

        if image_processor_config is not None:
            if Qwen2VLImageProcessorFast is None:
                raise ImportError(
                    "transformers is required for Qwen3VL image preprocessing. "
                    "Install transformers>=4.42 to enable this feature."
                )
            self.image_processor = Qwen2VLImageProcessorFast(**image_processor_config)
        else:
            self.image_processor = None

        if video_processor_config is not None:
            if Qwen2VLVideoProcessor is None:
                raise ImportError(
                    "transformers is required for Qwen3VL video preprocessing. "
                    "Install transformers>=4.42 to enable this feature."
                )
            self.video_processor = Qwen2VLVideoProcessor(**video_processor_config)
        else:
            self.video_processor = None

        self.stop_tokens = self.tokenizer.stop_tokens
        self.special_tokens = self.tokenizer.special_tokens
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template
        self.pad_id = self.tokenizer.pad_id

    @property
    def base_vocab_size(self) -> int:
        return self.tokenizer.base_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        return self.tokenizer.encode(text=text, add_bos=add_bos, add_eos=add_eos)

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def tokenize_message(
        self,
        message: Message,
        add_start_tokens: bool = True,
        add_end_tokens: bool = True,
    ) -> list[int]:
        return self.tokenizer.tokenize_message(
            message=message,
            add_start_tokens=add_start_tokens,
            add_end_tokens=add_end_tokens,
        )

    def tokenize_messages(
        self,
        messages: list[Message],
        *,
        add_end_tokens: bool = True,
    ) -> tuple[list[int], list[bool]]:
        return self.tokenizer.tokenize_messages(
            messages=messages,
            add_end_tokens=add_end_tokens,
        )

    def _collect_media(self, messages: Sequence[Message]) -> tuple[list[Any], list[Any]]:
        images: list[Any] = []
        videos: list[Any] = []
        for message in messages:
            for content in message.content:
                if content["type"] == "image":
                    images.append(content["content"])
                elif content["type"] == "video":
                    videos.append(content["content"])
        return images, videos

    def _compute_timestamps(self, grids: torch.Tensor) -> torch.Tensor:
        """Generate temporal indices per merged vision token."""
        if grids.numel() == 0:
            return torch.zeros(0, dtype=torch.long)

        timestamps: list[int] = []
        for grid in grids.tolist():
            t, h, w = (int(x) for x in grid)
            tokens_per_frame = (h * w) // (self.spatial_merge_size**2)
            for frame_idx in range(t):
                timestamps.extend([frame_idx] * tokens_per_frame)
        return torch.tensor(timestamps, dtype=torch.long)

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        messages: list[Message] = sample.pop("messages")
        images, videos = self._collect_media(messages)

        image_outputs = None
        if images:
            if self.image_processor is None:
                raise RuntimeError("Qwen3VLTransform was constructed without an image processor.")
            image_outputs = self.image_processor(images=images, return_tensors="pt")

        video_outputs = None
        if videos:
            if self.video_processor is None:
                raise RuntimeError("Qwen3VLTransform was constructed without a video processor.")
            video_outputs = self.video_processor(videos=videos, return_tensors="pt")

        image_grid_thw = (
            image_outputs["image_grid_thw"].tolist() if image_outputs is not None else None
        )
        video_grid_thw = (
            video_outputs["video_grid_thw"].tolist() if video_outputs is not None else None
        )

        tokens, mask, visual_mask = self.tokenizer.tokenize_messages_with_grids(
            messages=messages,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            add_end_tokens=not inference,
        )

        sample["tokens"] = tokens
        sample["mask"] = mask
        sample["visual_pos_masks"] = visual_mask

        if image_outputs is not None:
            sample["pixel_values_images"] = image_outputs["pixel_values"]
            sample["image_grid_thw"] = image_outputs["image_grid_thw"]

        if video_outputs is not None:
            sample["pixel_values_videos"] = video_outputs["pixel_values"]
            sample["video_grid_thw"] = video_outputs["video_grid_thw"]
        
        return sample

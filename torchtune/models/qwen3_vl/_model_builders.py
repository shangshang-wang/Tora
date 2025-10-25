# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from typing import Optional, Sequence

from torchtune.data._prompt_templates import _TemplateType

from torchtune.models.qwen3_vl._component_builders import (
    lora_qwen3_vl_decoder,
    qwen3_vl_decoder,
    qwen3_vl_encoder,
)
from torchtune.models.qwen3_vl.model import Qwen3VLForConditionalGeneration, Qwen3VLModel
from torchtune.models.qwen3_vl._transform import Qwen3VLTransform


def _load_preprocessor_config(config_path: Path) -> tuple[int, dict[str, object]]:
    with config_path.open("r") as fh:
        data = json.load(fh)

    merge_size = int(data.get("merge_size", 2))
    config: dict[str, object] = {}
    if "size" in data:
        config["size"] = data["size"]
    for key in ("patch_size", "temporal_patch_size", "image_mean", "image_std", "merge_size"):
        if key in data:
            config[key] = data[key]
    return merge_size, config


def qwen3_vl_transform(
    vocab_path: str,
    merges_path: str,
    *,
    max_seq_len: int = 32768,
    special_tokens_path: Optional[str] = None,
    prompt_template: Optional[_TemplateType] = None,
    truncation_type: str = "right",
    preprocessor_config_path: Optional[str] = None,
    video_preprocessor_config_path: Optional[str] = None,
    image_size: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Qwen3VLTransform:
    """Factory for the Qwen3 VL multimodal transform."""

    spatial_merge_size = 2
    image_processor_config: Optional[dict[str, object]] = None
    if preprocessor_config_path is not None:
        spatial_merge_size, image_processor_config = _load_preprocessor_config(Path(preprocessor_config_path))
        if image_processor_config is not None:
            image_processor_config = dict(image_processor_config)
            effective_max_pixels = max_pixels
            if effective_max_pixels is None and image_size is not None:
                effective_max_pixels = int(image_size) * int(image_size)
            if effective_max_pixels is not None:
                image_processor_config["max_pixels"] = effective_max_pixels

    video_processor_config: Optional[dict[str, object]] = None
    if video_preprocessor_config_path is not None:
        _, video_processor_config = _load_preprocessor_config(Path(video_preprocessor_config_path))

    return Qwen3VLTransform(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens_path=special_tokens_path,
        max_seq_len=max_seq_len,
        prompt_template=prompt_template,
        truncation_type=truncation_type,
        spatial_merge_size=spatial_merge_size,
        image_processor_config=image_processor_config,
        video_processor_config=video_processor_config,
    )


def qwen3_vl_4b_instruct(
    decoder_trainable: bool = True,
    encoder_trainable: bool = True,
    image_size: int = 448,
) -> Qwen3VLForConditionalGeneration:
    """Reference Qwen3-VL 4B instruct model."""

    encoder = qwen3_vl_encoder(
        depth=24,
        hidden_size=1024,
        intermediate_size=4096,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=2560,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[5, 11, 17],
        hidden_act="gelu_pytorch_tanh",
        attention_dropout=0.0,
    )

    decoder = qwen3_vl_decoder(
        vocab_size=151_936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2560,
        head_dim=128,
        intermediate_dim=9728,
        max_seq_len=262_144,
        rope_base=5_000_000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        pad_token_id=0,
    )

    model = Qwen3VLModel(vision=encoder, text=decoder)
    if not encoder_trainable:
        for param in model.vision.parameters():
            param.requires_grad = False
    if not decoder_trainable:
        for param in model.text.parameters():
            param.requires_grad = False

    lm = Qwen3VLForConditionalGeneration(model=model, vocab_size=151_936)
    if not decoder_trainable:
        for param in lm.lm_head.parameters():
            param.requires_grad = False
    return lm


def lora_qwen3_vl_4b_instruct(
    lora_attn_modules: Sequence[str],
    apply_lora_to_mlp: bool = True,
    decoder_trainable: bool = True,
    encoder_trainable: bool = False,
    image_size: int = 448,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    lora_type: str = "lora",
    quantize_base: bool = False,
) -> Qwen3VLForConditionalGeneration:
    """Qwen3-VL 4B instruct model with LoRA adapters on the text decoder."""

    encoder = qwen3_vl_encoder(
        depth=24,
        hidden_size=1024,
        intermediate_size=4096,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=2560,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[5, 11, 17],
        hidden_act="gelu_pytorch_tanh",
        attention_dropout=0.0,
    )

    decoder = lora_qwen3_vl_decoder(
        lora_attn_modules=list(lora_attn_modules),
        apply_lora_to_mlp=apply_lora_to_mlp,
        vocab_size=151_936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2560,
        head_dim=128,
        intermediate_dim=9728,
        max_seq_len=262_144,
        rope_base=5_000_000.0,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_act="silu",
        pad_token_id=0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_type=lora_type,
        quantize_base=quantize_base,
    )

    model = Qwen3VLModel(vision=encoder, text=decoder)
    if not encoder_trainable:
        for param in model.vision.parameters():
            param.requires_grad = False
    if not decoder_trainable:
        for param in model.text.parameters():
            param.requires_grad = False

    lm = Qwen3VLForConditionalGeneration(model=model, vocab_size=151_936)
    if not decoder_trainable:
        for param in lm.lm_head.parameters():
            param.requires_grad = False
    return lm

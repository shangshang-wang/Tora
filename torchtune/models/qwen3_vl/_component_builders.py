# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

from torch import nn

from torchtune.models.qwen3_vl._encoder import Qwen3VLEncoder
from torchtune.models.qwen3_vl.vision import (
    Qwen3VLVisionConfig,
    Qwen3VLVisionModel,
)
from torchtune.models.qwen3_vl.text import Qwen3VLTextConfig, Qwen3VLTextModel
from torchtune.modules.peft import (
    DoRALinear,
    DoRALinearCache,
    LORA_ATTN_MODULES,
    LoRALinear,
)


def qwen3_vl_encoder(
    *,
    depth: int,
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    in_channels: int,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    out_hidden_size: int,
    num_position_embeddings: int,
    deepstack_visual_indexes: Sequence[int],
    hidden_act: str = "gelu_pytorch_tanh",
    attention_dropout: float = 0.0,
) -> Qwen3VLEncoder:
    """Build the Qwen3-VL vision encoder aligned with the Hugging Face implementation."""

    config = Qwen3VLVisionConfig(
        depth=depth,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        in_channels=in_channels,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        spatial_merge_size=spatial_merge_size,
        out_hidden_size=out_hidden_size,
        num_position_embeddings=num_position_embeddings,
        deepstack_visual_indexes=deepstack_visual_indexes,
        hidden_act=hidden_act,
        attention_dropout=attention_dropout,
    )
    tower = Qwen3VLVisionModel(config)
    return Qwen3VLEncoder(tower=tower)


def qwen3_vl_decoder(
    *,
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    head_dim: Optional[int] = None,
    intermediate_dim: int,
    max_seq_len: int,
    rope_base: float,
    rms_norm_eps: float,
    attention_dropout: float,
    hidden_act: str,
    pad_token_id: int,
) -> Qwen3VLTextModel:
    """Build the Qwen3-VL text decoder with interleaved MRoPE and DeepStack hooks."""

    config = Qwen3VLTextConfig(
        vocab_size=vocab_size,
        hidden_size=embed_dim,
        intermediate_size=intermediate_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim if head_dim is not None else embed_dim // num_heads,
        rms_norm_eps=rms_norm_eps,
        max_position_embeddings=max_seq_len,
        rope_theta=rope_base,
        rope_scaling=None,
        hidden_act=hidden_act,
        attention_dropout=attention_dropout,
        attention_bias=False,
        pad_token_id=pad_token_id,
    )
    return Qwen3VLTextModel(config)


def _select_lora_linear_cls(lora_type: str) -> type[LoRALinear]:
    lora_type_lower = lora_type.lower()
    if lora_type_lower == "lora":
        return LoRALinear
    if lora_type_lower == "dora":
        return DoRALinear
    if lora_type_lower == "dora_cache":
        return DoRALinearCache
    raise ValueError(
        f"Unsupported lora_type '{lora_type}'. Expected one of ['lora', 'dora', 'dora_cache']."
    )


def _maybe_lora_linear(
    module_name: str,
    *,
    lora_attn_modules: list[LORA_ATTN_MODULES],
    adapter_cls: type[LoRALinear],
    in_dim: int,
    out_dim: int,
    use_bias: bool,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    quantize_base: bool,
) -> nn.Module:
    if module_name in lora_attn_modules:
        return adapter_cls(
            in_dim,
            out_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            use_bias=use_bias,
            quantize_base=quantize_base,
        )
    return nn.Linear(in_dim, out_dim, bias=use_bias)


def lora_qwen3_vl_decoder(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    head_dim: Optional[int] = None,
    intermediate_dim: int,
    max_seq_len: int,
    rope_base: float,
    rms_norm_eps: float,
    attention_dropout: float,
    hidden_act: str,
    pad_token_id: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    lora_type: str = "lora",
    quantize_base: bool = False,
) -> Qwen3VLTextModel:
    """Build the Qwen3-VL text decoder with LoRA adapters on the attention (and optional MLP) blocks."""

    if apply_lora_to_output:
        raise ValueError(
            "apply_lora_to_output is not supported for qwen3_vl because the decoder ties "
            "input and output embeddings."
        )

    adapter_cls = _select_lora_linear_cls(lora_type)
    config = Qwen3VLTextConfig(
        vocab_size=vocab_size,
        hidden_size=embed_dim,
        intermediate_size=intermediate_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim if head_dim is not None else embed_dim // num_heads,
        rms_norm_eps=rms_norm_eps,
        max_position_embeddings=max_seq_len,
        rope_theta=rope_base,
        rope_scaling=None,
        hidden_act=hidden_act,
        attention_dropout=attention_dropout,
        attention_bias=False,
        pad_token_id=pad_token_id,
    )

    decoder = Qwen3VLTextModel(config)
    for layer in decoder.layers:
        attn = layer.self_attn
        hidden_size = config.hidden_size
        attn_dim = attn.num_attention_heads * attn.head_dim
        kv_dim = attn.num_key_value_heads * attn.head_dim

        attn.q_proj = _maybe_lora_linear(
            "q_proj",
            lora_attn_modules=lora_attn_modules,
            adapter_cls=adapter_cls,
            in_dim=hidden_size,
            out_dim=attn_dim,
            use_bias=config.attention_bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        attn.k_proj = _maybe_lora_linear(
            "k_proj",
            lora_attn_modules=lora_attn_modules,
            adapter_cls=adapter_cls,
            in_dim=hidden_size,
            out_dim=kv_dim,
            use_bias=config.attention_bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        attn.v_proj = _maybe_lora_linear(
            "v_proj",
            lora_attn_modules=lora_attn_modules,
            adapter_cls=adapter_cls,
            in_dim=hidden_size,
            out_dim=kv_dim,
            use_bias=config.attention_bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        attn.o_proj = _maybe_lora_linear(
            "output_proj",
            lora_attn_modules=lora_attn_modules,
            adapter_cls=adapter_cls,
            in_dim=attn_dim,
            out_dim=hidden_size,
            use_bias=config.attention_bias,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quantize_base=quantize_base,
        )

        if apply_lora_to_mlp:
            mlp = layer.mlp
            mlp.gate_proj = adapter_cls(
                hidden_size,
                intermediate_dim,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                use_bias=False,
                quantize_base=quantize_base,
            )
            mlp.up_proj = adapter_cls(
                hidden_size,
                intermediate_dim,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                use_bias=False,
                quantize_base=quantize_base,
            )
            mlp.down_proj = adapter_cls(
                intermediate_dim,
                hidden_size,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                use_bias=False,
                quantize_base=quantize_base,
            )
    return decoder

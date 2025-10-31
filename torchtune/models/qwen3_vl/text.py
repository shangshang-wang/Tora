# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torchtune.modules.attention_utils import _MaskType, _sdpa_or_flex_attention
from torchtune.modules.kv_cache import KVCache


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@dataclass
class Qwen3VLTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope_theta: float
    rope_scaling: Optional[dict] = None
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    attention_bias: bool = False
    pad_token_id: int = 0
    use_cache: bool = True


class Qwen3VLTextRMSNorm(nn.Module):
    """Head-wise RMSNorm used in Qwen3-VL attention."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(orig_dtype)
    
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3VLTextRotaryEmbedding(nn.Module):
    """Interleaved multimodal rotary embedding (MRoPE)."""

    inv_freq: Tensor  # type: ignore[assignment]

    def __init__(self, config: Qwen3VLTextConfig, device=None) -> None:
        super().__init__()
        rope_scaling = config.rope_scaling or {}
        self.rope_type = rope_scaling.get("rope_type", "default")
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        dim = config.head_dim
        self.dim = dim
        self.rope_theta = config.rope_theta
        self.register_buffer(
            "inv_freq",
            torch.empty(0, device=device, dtype=torch.float32),
            persistent=False,
        )
        self.rope_init()
        self.original_inv_freq = self.inv_freq.clone()

        self.attention_scaling = 1.0
        self.mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

    def rope_init(self, device: Optional[torch.device] = None) -> None:
        target_device = device if device is not None else self.inv_freq.device
        if target_device.type == "meta":
            if torch.cuda.is_available():
                current_idx = torch.cuda.current_device()
                target_device = torch.device("cuda", current_idx)
            else:
                target_device = torch.device("cpu")
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=target_device)
                / self.dim
            )
        )
        self._buffers["inv_freq"] = inv_freq

    def apply_interleaved_mrope(self, freqs: Tensor, mrope_section: Sequence[int]) -> Tensor:
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        inv_freq = self.inv_freq[None, None, :, None].float()
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ position_ids_expanded).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype, device=x.device), sin.to(dtype=x.dtype, device=x.device)


class Qwen3VLTextAttention(nn.Module):
    """Self-attention with Qwen3-VL specific rotary embedding and head-wise RMSNorm."""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.kv_cache: Optional[KVCache] = None
        self.cache_enabled = False
        self._attention_call = _sdpa_or_flex_attention()

    def setup_cache(self, batch_size: int, dtype: torch.dtype, max_seq_len: int) -> None:
        if self.kv_cache is not None:
            return
        self.kv_cache = KVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            dtype=dtype,
        )
        self.cache_enabled = True

    def reset_cache(self) -> None:
        if self.kv_cache is None:
            raise RuntimeError("KV cache has not been initialised. Call setup_cache first.")
        self.kv_cache.reset()

    def caches_are_setup(self) -> bool:
        return self.kv_cache is not None

    def caches_are_enabled(self) -> bool:
        return self.cache_enabled

    def forward(
        self,
        hidden_states: Tensor,
        *,
        mask: Optional[_MaskType],
        position_embeddings: tuple[Tensor, Tensor],
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # head-wise RMSNorm applied per head
        query_states = self.q_norm(query_states.transpose(1, 2)).transpose(1, 2)
        key_states = self.k_norm(key_states.transpose(1, 2)).transpose(1, 2)

        cos, sin = position_embeddings
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        query_states, key_states = apply_rotary_pos_emb(query_states,key_states,cos,sin,)

        if self.kv_cache is not None and self.cache_enabled:
            key_states, value_states = self.kv_cache.update(key_states, value_states)

        if self.num_attention_heads != self.num_key_value_heads:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        

        use_cache = self.kv_cache is not None and self.cache_enabled
        attn_output = self._attention_call(
            query_states,
            key_states,
            value_states,
            mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None and self.is_causal and not use_cache
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()

        return self.o_proj(attn_output)


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
        super().__init__()
        act = {
            "silu": nn.SiLU(),
            "gelu": nn.GELU(approximate="none"),
            "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
        }.get(config.hidden_act, nn.SiLU())

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = act

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated = self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


class Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        mask: Optional[_MaskType],
        position_embeddings: tuple[Tensor, Tensor],
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            mask=mask,
            position_embeddings=position_embeddings,
            input_pos=input_pos,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def caches_are_setup(self) -> bool:
        return self.self_attn.caches_are_setup()

    def caches_are_enabled(self) -> bool:
        return self.self_attn.caches_are_enabled()


class Qwen3VLTextModel(nn.Module):
    """Minimal text decoder with DeepStack integration."""

    def __init__(self, config: Qwen3VLTextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)
        self.gradient_checkpointing = False

    def setup_caches(self, batch_size: int, dtype: torch.dtype, max_seq_len: int) -> None:
        for layer in self.layers:
            layer.self_attn.setup_cache(batch_size, dtype, max_seq_len)

    def reset_caches(self) -> None:
        for layer in self.layers:
            if layer.self_attn.cache_enabled:
                layer.self_attn.reset_cache()

    def caches_are_setup(self) -> bool:
        if not self.layers:
            return False
        return self.layers[0].caches_are_setup()

    def caches_are_enabled(self) -> bool:
        if not self.layers:
            return False
        return self.layers[0].caches_are_enabled()

    def forward(
        self,
        tokens: Optional[Tensor] = None,
        mask: Optional[_MaskType] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        *,
        visual_pos_masks: Optional[Tensor] = None,
        deepstack_visual_embeds: Optional[Sequence[Tensor]] = None,
    ) -> Tensor:
        if inputs_embeds is None:
            if tokens is None:
                raise ValueError("Either tokens or inputs_embeds must be provided.")
            inputs_embeds = self.embed_tokens(tokens)
        seq_len = inputs_embeds.shape[1]

        # when position_ids is not provided, we need to generate it based on sequence length and kv cache size
        position_ids_provided = position_ids is not None
        if not position_ids_provided:
            base_positions = torch.arange(seq_len, device=inputs_embeds.device)
            position_ids = base_positions.view(1, 1, -1).expand(3, -1, -1)
        kv_cache = self.layers[0].self_attn.kv_cache
        cache_active = kv_cache is not None and self.layers[0].self_attn.cache_enabled
        if cache_active and kv_cache.size != 0 and not position_ids_provided:
            position_ids = position_ids + kv_cache.size

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                mask=mask,
                position_embeddings=position_embeddings,
                input_pos=position_ids[0],
            )

            if (
                deepstack_visual_embeds is not None
                and idx < len(deepstack_visual_embeds)
                and visual_pos_masks is not None
            ):
                ds_embed = deepstack_visual_embeds[idx].to(hidden_states.device, hidden_states.dtype)
                # visual_pos_masks shape: (batch, seq)
                hidden_states = hidden_states.clone()
                hidden_states[visual_pos_masks] = hidden_states[visual_pos_masks] + ds_embed

        hidden_states = self.norm(hidden_states)
        return hidden_states

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the hidden dimensions."""

    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb_vision(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply 2D rotary embeddings to query/key tensors.

    Args:
        q: Query tensor with shape ``(seq_len, num_heads, head_dim)``.
        k: Key tensor with shape ``(seq_len, num_heads, head_dim)``.
        cos: Cosine rotary values broadcastable to ``q`` and ``k``.
        sin: Sine rotary values broadcastable to ``q`` and ``k``.

    Returns:
        Tuple containing rotated query and key tensors with the same shape
        as the inputs.
    """

    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def _gelu_approx(approximation: str) -> nn.Module:
    if approximation == "tanh":
        return nn.GELU(approximate="tanh")
    if approximation == "none":
        return nn.GELU(approximate="none")
    raise ValueError(f"Unsupported GELU approximation: {approximation}")


def _activation_from_name(name: str) -> Callable[[Tensor], Tensor]:
    if name in {"gelu", "gelu_new"}:
        return nn.GELU(approximate="none")
    if name in {"gelu_pytorch_tanh", "gelu_tanh"}:
        return _gelu_approx("tanh")
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation '{name}' for Qwen3-VL vision tower.")


@dataclass
class Qwen3VLVisionConfig:
    """Configuration container for the Qwen3-VL vision tower."""

    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    num_position_embeddings: int
    deepstack_visual_indexes: Sequence[int]
    hidden_act: str = "gelu_pytorch_tanh"
    attention_dropout: float = 0.0
    attn_implementation: str = "eager"


class Qwen3VLVisionPatchEmbed(nn.Module):
    """3D convolutional patch embedding used by the Qwen3-VL vision tower."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        kernel_size = [
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        ]
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.proj = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Patchify input pixels.

        Args:
            pixel_values: A tensor with shape ``(total_chunks, channels, height, width)``
                where ``total_chunks`` equals the total number of temporal patches.

        Returns:
            Flattened patch embeddings with shape
            ``(total_patches, hidden_size)``.
        """
        tgt_dtype = self.proj.weight.dtype
        hidden_states = pixel_values.to(dtype=tgt_dtype)
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = hidden_states.to(memory_format=torch.channels_last_3d)
        hidden_states = self.proj(hidden_states)
        return hidden_states.view(-1, hidden_states.shape[1])


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """Vision rotary embedding table (2D)."""

    def __init__(self, dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        # register an empty buffer that will be materialized on the target device
        self.register_buffer("inv_freq", torch.empty(0), persistent=False)
        self.rope_init()

    def rope_init(self, device: Optional[torch.device] = None) -> None:
        target_device = device if device is not None else self.inv_freq.device
        if target_device.type == "meta":
            if torch.cuda.is_available():
                current_idx = torch.cuda.current_device()
                target_device = torch.device("cuda", current_idx)
            else:
                target_device = torch.device("cpu")
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float32, device=target_device)
                / self.dim
            )
        )
        # replace the existing buffer with the newly materialized values
        self._buffers["inv_freq"] = inv_freq
   
    def forward(self, seq_len: int) -> Tensor:
        seq = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLVisionPatchMerger(nn.Module):
    """Spatial merger used for DeepStack taps and final projection."""

    def __init__(self, config: Qwen3VLVisionConfig, *, use_postshuffle_norm: bool) -> None:
        super().__init__()
        spatial_factor = config.spatial_merge_size**2
        hidden_size = config.hidden_size * spatial_factor
        norm_dim = hidden_size if use_postshuffle_norm else config.hidden_size
        self.hidden_size = hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = _activation_from_name(config.hidden_act)
        self.linear_fc2 = nn.Linear(hidden_size, config.out_hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.use_postshuffle_norm:
            hidden_states = self.norm(hidden_states.view(-1, self.hidden_size))
        else:
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.linear_fc2(self.act(self.linear_fc1(hidden_states)))
        return hidden_states


class Qwen3VLVisionMLP(nn.Module):
    """Feed-forward block used inside each vision transformer layer."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = _activation_from_name(config.hidden_act)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.linear_fc2(self.act(self.linear_fc1(hidden_states)))


class Qwen3VLVisionAttention(nn.Module):
    """Self-attention block for the vision tower."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=True)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attention_dropout = config.attention_dropout

    def _scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.attention_dropout and self.training:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cu_seqlens: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        seq_len, _ = hidden_states.shape
        qkv = self.qkv(hidden_states)
        q, k, v = qkv.view(seq_len, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if len(lengths) == 0:
            lengths = [seq_len]

        outputs: list[Tensor] = []
        offset = 0
        for length in lengths:
            slc = slice(offset, offset + length)
            q_chunk = q[slc].permute(1, 0, 2)  # (num_heads, length, head_dim)
            k_chunk = k[slc].permute(1, 0, 2)
            v_chunk = v[slc].permute(1, 0, 2)
            attn = self._scaled_dot_product_attention(q_chunk, k_chunk, v_chunk)
            attn = attn.permute(1, 0, 2).reshape(length, -1)
            outputs.append(attn)
            offset += length

        attn_output = torch.cat(outputs, dim=0)
        return self.proj(attn_output)


class Qwen3VLVisionBlock(nn.Module):
    """Transformer block for Qwen3-VL vision tower."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        cu_seqlens: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        residual = hidden_states
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionModel(nn.Module):
    """Vision tower that produces merged embeddings and DeepStack taps."""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = config.spatial_merge_size * config.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(math.sqrt(config.num_position_embeddings))

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
                for _ in self.deepstack_visual_indexes
            ]
        )

    def _flatten_visual_inputs(
        self, pixel_values: Tensor, grid_thw: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Align padded vision tensors to the per-image shape expected by the tower."""

        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)
        elif grid_thw.ndim > 2:
            grid_thw = grid_thw.reshape(-1, grid_thw.shape[-1])

        if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise ValueError(
                "grid_thw must be a (num_images, 3) tensor describing (t, h, w) per image."
            )

        if pixel_values.ndim < 5:
            raise ValueError(
                "pixel_values must have at least 5 dims (num_images, C, T, H, W)."
            )

        spatial_shape = pixel_values.shape[-4:]
        pixel_values = pixel_values.reshape(-1, *spatial_shape)

        if grid_thw.shape[0] != pixel_values.shape[0]:
            raise ValueError(
                "pixel_values and grid_thw must agree on the number of images after flattening."
            )

        valid_mask = torch.any(grid_thw != 0, dim=-1)
        if not torch.any(valid_mask):
            raise ValueError("No valid (t, h, w) entries found in grid_thw.")

        pixel_values = pixel_values[valid_mask]
        grid_thw = grid_thw[valid_mask]
        return pixel_values, grid_thw

    def fast_pos_embed_interpolate(self, grid_thw: Tensor) -> Tensor:
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list: list[list[int]] = [[] for _ in range(4)]
        weight_list: list[list[float]] = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=self.pos_embed.weight.device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=self.pos_embed.weight.device)

            h_floor = h_idxs.floor().long()
            w_floor = w_idxs.floor().long()
            h_ceil = (h_floor + 1).clamp(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clamp(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_floor.float()
            dw = w_idxs - w_floor.float()

            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            indices = [
                (base_h[:, None] + w_floor[None, :]).flatten(),
                (base_h[:, None] + w_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
            ]

            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]

            for acc, values in zip(idx_list, indices):
                acc.extend(values.tolist())
            for acc, values in zip(weight_list, weights):
                acc.extend(values.tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds.sum(dim=0)

        merge_size = self.config.spatial_merge_size
        patch_pos_embeds = patch_pos_embeds.split([int(h * w) for h, w in zip(grid_hs, grid_ws)])

        permuted: list[Tensor] = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat_interleave(int(t.item()), dim=0)
            pos_embed = (
                pos_embed.view(int(t), int(h // merge_size), merge_size, int(w // merge_size), merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(-1, pos_embed.shape[-1])
            )
            permuted.append(pos_embed)
        return torch.cat(permuted, dim=0)

    def rot_pos_emb(self, grid_thw: Tensor) -> Tensor:
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h = int(height.item() // merge_size)
            merged_w = int(width.item() // merge_size)

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(int(num_frames.item()), 1)

            length = coords.shape[0]
            pos_ids[offset : offset + length] = coords
            offset += length

        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    def forward(
        self,
        pixel_values: Tensor,
        *,
        grid_thw: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        """Encode image/video patches and return DeepStack taps.

        Args:
            pixel_values: Input tensor containing concatenated frames with shape
                ``(num_patches * temporal_patch_size, channels, patch_h, patch_w)``.
            grid_thw: Tensor describing grid layout per image/video with shape
                ``(num_items, 3)`` where columns are ``(t, h, w)``.

        Returns:
            Tuple ``(merged_embeddings, deepstack_features)`` where:
            - ``merged_embeddings`` has shape ``(total_patches // merge_size^2, out_hidden_size)``
            - ``deepstack_features`` is a list containing tensors aligned with
              ``deepstack_visual_indexes``; each tensor has shape
              ``(total_visual_tokens, out_hidden_size)``.
        """
        # pixel_values, grid_thw = self._flatten_visual_inputs(pixel_values, grid_thw)
        grid_thw = grid_thw.squeeze()
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)

        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.shape
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        lengths = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = lengths.cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_features: list[Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_idx in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_idx)]
                deepstack_features.append(merger(hidden_states))

        merged = self.merger(hidden_states)
        return merged, deepstack_features

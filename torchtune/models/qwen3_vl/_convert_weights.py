#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict

import torch


_HF_TEXT_PREFIX = "model.language_model."
_HF_VISION_PREFIX = "model.visual."
_TT_TEXT_PREFIX = "model.text."
_TT_VISION_PREFIX = "model.vision.tower."
_LM_HEAD_PREFIX = "lm_head."


def _map_key_from_hf(key: str) -> str | None:
    if "rotary_emb.inv_freq" in key or "rotary_pos_emb.inv_freq" in key:
        return None
    if key.startswith(_HF_TEXT_PREFIX):
        return _TT_TEXT_PREFIX + key[len(_HF_TEXT_PREFIX) :]
    if key.startswith(_HF_VISION_PREFIX):
        return _TT_VISION_PREFIX + key[len(_HF_VISION_PREFIX) :]
    if key.startswith(_LM_HEAD_PREFIX):
        return _LM_HEAD_PREFIX + key[len(_LM_HEAD_PREFIX) :]
    return key


def _map_key_to_hf(key: str) -> str | None:
    if "rotary_emb.inv_freq" in key or "rotary_pos_emb.inv_freq" in key:
        return None
    if key.startswith(_TT_TEXT_PREFIX):
        return _HF_TEXT_PREFIX + key[len(_TT_TEXT_PREFIX) :]
    if key.startswith(_TT_VISION_PREFIX):
        return _HF_VISION_PREFIX + key[len(_TT_VISION_PREFIX) :]
    if key.startswith(_LM_HEAD_PREFIX):
        return _LM_HEAD_PREFIX + key[len(_LM_HEAD_PREFIX) :]
    return key


def qwen3_vl_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    *,
    tie_word_embeddings: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert a Hugging Face Qwen3-VL state dict into the torchtune naming convention.
    """
    converted_state_dict: Dict[str, torch.Tensor] = {}
    required_keys = {
        key
        for key in state_dict
        if (mapped := _map_key_from_hf(key)) is not None
        and not (tie_word_embeddings and mapped == f"{_LM_HEAD_PREFIX}weight")
    }
    mapped_keys: set[str] = set()
    for key, value in state_dict.items():
        new_key = _map_key_from_hf(key)
        if new_key is None:
            continue
        if tie_word_embeddings and new_key == f"{_LM_HEAD_PREFIX}weight":
            # Skip loading lm_head when embeddings are tied. We'll reuse the token embedding weights.
            continue
        converted_state_dict[new_key] = value
        mapped_keys.add(key)
        if new_key == f"{_LM_HEAD_PREFIX}weight":
            converted_state_dict["output.weight"] = value
    if (
        tie_word_embeddings
        and f"{_TT_TEXT_PREFIX}embed_tokens.weight" in converted_state_dict
        and f"{_LM_HEAD_PREFIX}weight" not in converted_state_dict
    ):
        embed = converted_state_dict[f"{_TT_TEXT_PREFIX}embed_tokens.weight"]
        converted_state_dict[f"{_LM_HEAD_PREFIX}weight"] = embed
        converted_state_dict["output.weight"] = embed
    if (
        f"{_LM_HEAD_PREFIX}weight" in converted_state_dict
        and "tok_embeddings.weight" not in converted_state_dict
    ):
        converted_state_dict["tok_embeddings.weight"] = converted_state_dict[f"{_LM_HEAD_PREFIX}weight"]
    missing_keys = sorted(set(required_keys) - mapped_keys)
    if missing_keys:
        missing_str = ", ".join(missing_keys)
        raise RuntimeError(f"Failed to convert the following Hugging Face weights: {missing_str}")
    return converted_state_dict


def qwen3_vl_tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
    *,
    tie_word_embeddings: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert a torchtune Qwen3-VL state dict back into Hugging Face naming convention.
    """
    converted_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = _map_key_to_hf(key)
        if new_key is None:
            continue
        converted_state_dict[new_key] = value
        if key == f"{_LM_HEAD_PREFIX}weight":
            converted_state_dict["output.weight"] = value
        if tie_word_embeddings and key == f"{_TT_TEXT_PREFIX}embed_tokens.weight":
            cloned = value.detach().clone()
            converted_state_dict["lm_head.weight"] = cloned
            converted_state_dict["output.weight"] = cloned
    if (
        not tie_word_embeddings
        and "output.weight" not in converted_state_dict
        and f"{_LM_HEAD_PREFIX}weight" in state_dict
    ):
        converted_state_dict["output.weight"] = state_dict[f"{_LM_HEAD_PREFIX}weight"]
    return converted_state_dict

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset utilities for the CLEVR-CoGen-A multimodal reasoning benchmark."""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from torchtune.data import Message
from torchtune.modules.tokenizers import ModelTokenizer

from .data import ReasoningProblem, RLDataset, SYSTEM_MESSAGE

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _strip_tags(text: str) -> str:
    """Utility that normalizes extracted XML-style tag values."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_clevr(sample: dict[str, Any]) -> ReasoningProblem:
    """
    Parses an item from the CLEVR-CoGen-A dataset, extracting the textual question,
    (optional) chain-of-thought, and the target answer while keeping the raw image
    attached for downstream tokenization.
    """
    question = sample["problem"].strip()
    solution = sample.get("solution", "").strip()

    cot_match = THINK_PATTERN.search(solution)
    cot = _strip_tags(cot_match.group(1)) if cot_match else ""

    answer_match = ANSWER_PATTERN.search(solution)
    answer = _strip_tags(answer_match.group(1)) if answer_match else solution

    return {
        "question": question,
        "cot": cot,
        "answer": answer,
        "image": sample.get("image"),
    }


class CLEVRCogenDataset(RLDataset):
    """
    Multimodal RL dataset that prepares CLEVR-CoGen-A problems with associated images.

    Notes:
        - Expects ``tokenizer`` to be a Qwen3-VL style transform that can process multimodal messages.
        - Generates a user message containing the reference image and the question prompt so generation
          can condition on both modalities.
    """

    def _prepare_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        transformed_sample = self._problem_transform(sample)
        if not hasattr(self._tokenizer, "__call__"):
            raise RuntimeError(
                "CLEVRCogenDataset requires a tokenizer/transform that can be called "
                "with Message inputs (e.g., torchtune.models.qwen3_vl.qwen3_vl_transform)."
            )

        question = transformed_sample["question"]
        answer = transformed_sample["answer"]
        image = transformed_sample.get("image")

        # Build a structured conversation matching the requested format.
        system_line = f"system message: {SYSTEM_MESSAGE}"
        user_line = f"<image> {question}"
        assistant_line = ""

        user_content: list[dict[str, Any]] = []
        if image is not None:
            user_content.append({"type": "image", "content": image})
        else:
            user_content.append({"type": "text", "content": "[missing image] "})
        user_content.append({"type": "text", "content": question})

        messages = [
            Message(role="system", content=SYSTEM_MESSAGE),
            Message(role="user", content=user_content),
            Message(role="assistant", content=""),
        ]

        tokenizer_input = {"messages": messages}
        tokenized = self._tokenizer(tokenizer_input, inference=True)

        payload: dict[str, Any] = {
            "question": "\n".join([system_line, user_line, assistant_line]),
            "tokens": tokenized["tokens"],
            "mask": tokenized.get("mask"),
            "answer": answer,
        }

        # Surface optional multimodal metadata produced by the transform for collation.
        for key in ("pixel_values_images", "image_grid_thw", "visual_pos_masks"):
            if key in tokenized:
                payload[key] = tokenized[key]

        return payload


def clevr_cogen_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "leonardPKU/clevr_cogen_a_train",
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    name: Optional[str] = None,
    partition: Optional[str] = None,
    **load_dataset_kwargs: dict[str, Any],
) -> CLEVRCogenDataset:
    """
    CLEVR-CoGen-A dataset prepared for RL with visual reasoning.
    """

    def default_filter_fn(example: dict, idx: int) -> bool:
        if partition is None:
            return True

        match = re.match(r"^(\d+)-(\d+)/(\d+)$", partition)
        if not match:
            raise ValueError(
                f"Invalid partition format: {partition}. Expected format: start-end/total"
            )

        start, end, total = map(int, match.groups())
        current = idx % total
        return start <= current <= end

    filter_fn = filter_fn if filter_fn is not None else default_filter_fn

    dataset_kwargs = {}
    if name is not None:
        dataset_kwargs["name"] = name

    return CLEVRCogenDataset(
        source=source,
        tokenizer=tokenizer,
        problem_transform=normalize_clevr,
        filter_fn=filter_fn,
        filter_kwargs=dict(with_indices=True),
        split=split,
        **dataset_kwargs,
        **load_dataset_kwargs,
    )

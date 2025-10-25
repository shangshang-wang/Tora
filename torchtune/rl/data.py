# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, Mapping, Optional, TypedDict, Union

from typing_extensions import NotRequired

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence
from torchtune.data import CROSS_ENTROPY_IGNORE_IDX, left_pad_sequence
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

SYSTEM_MESSAGE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, "
    "i.e., <think>reasoning process here</think> <answer>answer here</answer>."
)


def build_reasoning_prompt(question: str) -> str:
    """Format a text-only reasoning prompt with explicit conversation roles."""

    return "\n".join(
        [
            f"system message: {SYSTEM_MESSAGE}",
            f"User: {question}",
            "Assistant:",
        ]
    )


class ReasoningProblem(TypedDict):
    question: str
    cot: str
    answer: str
    image: NotRequired[Any]


class RLDataset(Dataset):
    """
    Base class for datasets used in reinforcement learning,
    which provide a reference answer that can be verified to compute rewards.
    """

    def __init__(
        self,
        *,
        source: str,
        problem_transform: Transform,
        tokenizer: ModelTokenizer,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[dict[str, Any]] = None,
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        self._problem_transform = problem_transform
        self._tokenizer = tokenizer

        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            if filter_kwargs is None:
                filter_kwargs = {}
            self._data = self._data.filter(filter_fn, **filter_kwargs)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        transformed_sample = self._problem_transform(
            sample
        )  # keys "question" and "answer"

        question = build_reasoning_prompt(transformed_sample["question"])

        q_tokens = self._tokenizer.encode(question, add_eos=False)
        mask = [1 for _ in q_tokens]
        answer = transformed_sample["answer"]

        return {
            "question": question,
            "tokens": q_tokens,
            "mask": mask,
            "answer": answer,
        }

def padded_collate_rl_left_pad(
    batch: list[dict[str, list[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> dict[str, Union[torch.Tensor, list[str]]]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors. Answers are simply concatenated into a list.

    Args:
        batch (list[dict[str, list[int]]]): A list of dictionaries containing tokens.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        dict[str, Union[torch.Tensor, list[str]]]: Collated input tensors and string answers.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3], "answer": "15"},
        >>>    {"tokens": [7,], "answer": "bromance"},
        >>> ]
        >>> collated = padded_collate_rl(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> collated["tokens"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["answers"]
        >>> ["15", "bromance"]
    """
    input_ids = pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )

    answers = [x["answer"] for x in batch]
    text = [x["question"] for x in batch]

    return {"tokens": input_ids.long(), "answers": answers, "text": text}

def padded_collate_rl(
    batch: list[dict[str, list[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> dict[str, Union[torch.Tensor, list[str]]]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors. Answers are simply concatenated into a list.

    Args:
        batch (list[dict[str, list[int]]]): A list of dictionaries containing tokens.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        dict[str, Union[torch.Tensor, list[str]]]: Collated input tensors and string answers.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3], "answer": "15"},
        >>>    {"tokens": [7,], "answer": "bromance"},
        >>> ]
        >>> collated = padded_collate_rl(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> collated["tokens"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["answers"]
        >>> ["15", "bromance"]
    """
    input_ids = left_pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )

    answers = [x["answer"] for x in batch]
    text = [x["question"] for x in batch]

    return {"tokens": input_ids.long(), "answers": answers, "text": text}


def padded_collate_rl_multimodal(
    batch: list[dict[str, Any]],
    padding_idx: int = 0,
) -> dict[str, Any]:
    """
    Collate function for multimodal RL data that left-pads text tokens and stacks the
    accompanying visual features emitted by vision-language transforms.
    """

    def _pad_bool_sequence(
        sequences: list[list[bool]], pad_value: bool = False
    ) -> torch.Tensor:
        tensors = [torch.tensor(seq, dtype=torch.bool) for seq in sequences]
        return left_pad_sequence(
            tensors,
            batch_first=True,
            padding_value=pad_value,
        )

    def _pad_visual_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
        max_imgs = max(t.shape[0] for t in tensors)
        if max_imgs == 0:
            template = tensors[0]
            return torch.zeros(
                (len(tensors), 0, *template.shape[1:]), dtype=template.dtype
            )
        padded_batch = []
        for tensor in tensors:
            if tensor.shape[0] < max_imgs:
                pad_shape = (max_imgs - tensor.shape[0],) + tensor.shape[1:]
                pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad_tensor], dim=0)
            padded_batch.append(tensor)
        return torch.stack(padded_batch, dim=0)

    def _pad_grid_tensors(grids: list[torch.Tensor]) -> torch.Tensor:
        max_imgs = max(grid.shape[0] for grid in grids)
        if max_imgs == 0:
            return torch.zeros((len(grids), 0, 3), dtype=torch.long)
        padded = []
        for grid in grids:
            if grid.shape[0] < max_imgs:
                pad = torch.zeros(
                    (max_imgs - grid.shape[0], grid.shape[1]), dtype=grid.dtype
                )
                grid = torch.cat([grid, pad], dim=0)
            padded.append(grid)
        return torch.stack(padded, dim=0)

    token_tensors = [torch.tensor(x["tokens"]) for x in batch]
    input_ids = left_pad_sequence(
        token_tensors,
        batch_first=True,
        padding_value=padding_idx,
    )

    collated: dict[str, Any] = {
        "tokens": input_ids.long(),
        "answers": [x["answer"] for x in batch],
        "text": [x["question"] for x in batch],
    }

    if any(x.get("mask") is not None for x in batch):
        masks = [x.get("mask", [True] * len(x["tokens"])) for x in batch]
        collated["mask"] = _pad_bool_sequence(masks, pad_value=True)

    if any(x.get("visual_pos_masks") is not None for x in batch):
        visual_masks = [
            x.get("visual_pos_masks", [False] * len(x["tokens"])) for x in batch
        ]
        collated["visual_pos_masks"] = _pad_bool_sequence(
            visual_masks, pad_value=False
        )

    pixel_batches: list[Optional[torch.Tensor]] = []
    for sample in batch:
        tensor = sample.get("pixel_values_images")
        pixel_batches.append(
            torch.as_tensor(tensor) if tensor is not None else None
        )

    if any(tensor is not None for tensor in pixel_batches):
        template = next(t for t in pixel_batches if t is not None)
        prepared: list[torch.Tensor] = []
        for tensor in pixel_batches:
            if tensor is None:
                prepared.append(
                    torch.zeros(
                        (0,) + template.shape[1:], dtype=template.dtype
                    )
                )
            else:
                prepared.append(tensor)
        collated["pixel_values_images"] = _pad_visual_tensors(prepared)

    image_grids = [
        torch.as_tensor(x["image_grid_thw"], dtype=torch.long)
        if x.get("image_grid_thw") is not None
        else torch.zeros((0, 3), dtype=torch.long)
        for x in batch
    ]
    if any(grid.numel() > 0 for grid in image_grids):
        collated["image_grid_thw"] = _pad_grid_tensors(image_grids)

    return collated
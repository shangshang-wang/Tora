# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Mapping, Optional, Union
from datasets import load_dataset, VerificationMode
import math

from torch.utils.data import Dataset
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._utils import truncate
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer


# For testing use only
class C4Dataset(Dataset):
    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str = "allenai/c4",
        column: str = "text",
        add_eos: bool = True,
        filter_fn: Optional[Callable] = None,
        split: str = "train",
        subset_percent: Optional[float] = None,    # e.g., 0.01 for 1%
        subset_start_shard: int = 0,               # where to start picking shards
        explicit_shards: bool = True,              # switch to JSON builder
        **load_dataset_kwargs: dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._column = column
        self.add_eos = add_eos

        if explicit_shards and subset_percent:
            # C4 train has exactly 1024 shards
            total_shards = 1024
            n = max(1, math.ceil(subset_percent * total_shards))
            start = max(0, min(total_shards - n, subset_start_shard))
            shard_ids = list(range(start, start + n))

            # List only the shards we want using the hf:// filesystem
            files = [
                f"hf://datasets/allenai/c4/en/c4-train.{i:05d}-of-01024.json.gz"
                for i in shard_ids
            ]
            data_files = {"train": files}

            # Use JSON builder so HF does not resolve the C4 manifest
            self._data = load_dataset(
                "json",
                data_files=data_files,
                split="train",
                **load_dataset_kwargs,
            )
        else:
            # Fallback: standard C4 builder with split slicing
            if "name" not in load_dataset_kwargs:
                load_dataset_kwargs["name"] = "en"
            self._data = load_dataset(source, split=split, **load_dataset_kwargs)

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)


    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> dict[str, list[int]]:
        prompt = sample[self._column]
        tokens = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=self.add_eos)
        if self._tokenizer.max_seq_len is not None:
            tokens = truncate(tokens, self._tokenizer.max_seq_len - 1)
        labels = tokens[1:].copy()
        labels.append(CROSS_ENTROPY_IGNORE_IDX)
        return {"tokens": tokens, "labels": labels}


def c4_dataset(
    tokenizer: ModelTokenizer,
    source: str = "allenai/c4",
    column: str = "text",
    add_eos: bool = True,
    packed: bool = False,
    split_across_pack: bool = True,
    split: str = "train",
    subset_percent: Optional[float] = None,
    keep_in_memory: bool = False,
    filter_fn: Optional[Callable] = None,
    **load_dataset_kwargs: dict[str, Any],
) -> Union[C4Dataset, PackedDataset]:
    ds = C4Dataset(
        tokenizer=tokenizer,
        source=source,
        column=column,
        add_eos=add_eos,
        split=split,
        subset_percent=subset_percent,
        keep_in_memory=keep_in_memory,
        filter_fn=filter_fn,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError("PackedDataset requires tokenizer.max_seq_len.")
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=split_across_pack)
    return ds

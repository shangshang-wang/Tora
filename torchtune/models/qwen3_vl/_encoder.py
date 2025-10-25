# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torch import Tensor, nn

from torchtune.models.qwen3_vl.vision import Qwen3VLVisionModel


class Qwen3VLEncoder(nn.Module):
    """Wrapper around the Qwen3-VL vision tower exposing DeepStack taps."""

    def __init__(self, *, tower: Qwen3VLVisionModel) -> None:
        super().__init__()
        self.tower = tower

    def forward(
        self,
        pixel_values: Tensor,
        grid_thw: Tensor,
    ) -> tuple[Tensor, list[Tensor]]:
        """Encode images/videos with the vision tower."""

        if pixel_values is None:
            raise ValueError("`pixel_values` must be provided to run the Qwen3-VL vision encoder.")

        if grid_thw is None:
            raise ValueError("`grid_thw` must be provided to compute Qwen3-VL vision embeddings.")

        return self.tower(pixel_values, grid_thw=grid_thw)

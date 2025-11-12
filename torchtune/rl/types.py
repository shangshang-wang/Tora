# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Sequence as _SequenceABC
from typing import Any, Dict, NamedTuple, Optional, Sequence

import torch


class GRPOTrajectory(NamedTuple):
    """
    Contains a collection of tensors describing a generated trajectory during GRPO training.

    Attributes:
        query_responses (torch.Tensor): (query, response) pairs with shape [B x G, P+L].
        logprobs (torch.Tensor): Log probabilities of the generated responses with shape [B x G, L].
        ref_logprobs (torch.Tensor): Log probabilities of the generated responses using the reference policy with shape [B x G, L].
        rewards (torch.Tensor): Scalar reward values for the generated responses with shape [B x G].
        reward_components (torch.Tensor): Decomposed reward values for the generated responses with shape [B x G, C].
        successes (torch.Tensor): Binary success indicators for the generated responses with
        advantages (torch.Tensor): Advantage estimates for the generated responses with shape [B x G].
        masks (torch.Tensor): Attention masks for input ids-generated responses pairs with shape [B x G, P+L, P+L].
        position_ids (torch.Tensor): Position IDs for input ids-generated responses pairs with shape [B x G, P+L].
        response_padding_masks (torch.Tensor): Padding masks for the truncated and padded generated responses with shape [B x G, L].
        seq_lens (torch.Tensor): Sequence lengths of truncated generated responses.
        answers (str): list of answers for the generated responses. [B x G]
    """

    query_responses: torch.Tensor = None  # [B*G, P+L]
    logprobs: torch.Tensor = None  # [B*G, L]
    ref_logprobs: torch.Tensor = None  # [B*G, L]
    rewards: torch.Tensor = None  # [B*G]
    reward_components: torch.Tensor = None  # [B*G, C]
    successes: torch.Tensor = None  # [B*G]
    advantages: torch.Tensor = None  # [B*G]
    masks: torch.Tensor = None  # [B*G, P+L, P+L]
    position_ids: torch.Tensor = None  # [B*G, P+L]
    response_padding_masks: torch.Tensor = None  # [B*G, L]
    seq_lens: torch.Tensor = None  # [B*G]
    answers: Optional[Sequence[str]] = None  # [B*G]


class GRPOStats(NamedTuple):
    """
    Contains GRPO loss statistics (metrics).

    Attributes:
        loss (torch.Tensor): The total GRPO loss.
        policy_loss (torch.Tensor): The policy function loss.
        kl_loss (torch.Tensor): The KL divergence loss.
        ratios (torch.Tensor): The ratio between the current and old policy probabilities.
        clipfrac (torch.Tensor): The fraction of ratios that were clipped.
        approx_policy_kls (torch.Tensor): Average estimated KL divergence between the policy before and after the optimization step.
        metadata (Optional[dict]): Additional metadata to be logged.
    """

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    ratios: torch.Tensor
    clipfrac: torch.Tensor
    approx_policy_kls: torch.Tensor
    metadata: Optional[Dict[str, Any]] = None


def _first_non_none(values: Sequence[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def concat_grpo_trajectories(
    trajectories: Sequence[GRPOTrajectory],
) -> GRPOTrajectory:
    """
    Concatenate a sequence of ``GRPOTrajectory`` objects along the batch dimension while
    preserving optional non-tensor fields.
    """
    if not trajectories:
        raise ValueError("Cannot concatenate an empty sequence of GRPOTrajectories.")

    concatenated: Dict[str, Any] = {}
    for field in GRPOTrajectory._fields:
        values = [getattr(traj, field) for traj in trajectories]
        exemplar = _first_non_none(values)
        if exemplar is None:
            concatenated[field] = None
            continue

        if isinstance(exemplar, torch.Tensor):
            concatenated[field] = torch.cat(
                [value for value in values if value is not None], dim=0
            )
        elif isinstance(exemplar, _SequenceABC) and not isinstance(
            exemplar, (str, bytes, torch.Tensor)
        ):
            combined = []
            for value in values:
                if value is None:
                    continue
                combined.extend(value)
            concatenated[field] = combined
        else:
            concatenated[field] = exemplar

    return GRPOTrajectory(**concatenated)


def stack_grpo_stats(stats_list: Sequence[GRPOStats]) -> GRPOStats:
    """
    Stack a sequence of ``GRPOStats`` objects along a new dimension, ignoring optional
    metadata fields that cannot be stacked.
    """
    if not stats_list:
        raise ValueError("Cannot stack an empty sequence of GRPOStats.")

    stacked: Dict[str, Any] = {}
    for field in GRPOStats._fields:
        values = [getattr(stats, field) for stats in stats_list]
        exemplar = _first_non_none(values)
        if exemplar is None:
            stacked[field] = None
            continue

        if isinstance(exemplar, torch.Tensor):
            stacked[field] = torch.stack(
                [value for value in values if value is not None], dim=0
            )
        else:
            stacked[field] = exemplar

    return GRPOStats(**stacked)

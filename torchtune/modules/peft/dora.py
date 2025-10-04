# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F

from torch import nn
from torch.distributed.tensor import DTensor

from torchao.dtypes.nf4tensor import linear_nf4, to_nf4
from torchtune.modules.low_precision import _register_nf4_dispatch_ops  # noqa: F401
from torchtune.modules.peft import AdapterModule


class DoRALinear(nn.Module, AdapterModule):
    """DoRA linear layer as introduced in
    `DoRA: Weight-Decomposed Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2402.09353>`_.

    DoRA (Weight-Decomposed Low-Rank Adaptation) fine-tunes a layer by decomposing the pre-trained weights
    into two components: magnitude and direction. The magnitude component is a learnable scalar vector
    that scales each output channel, while the direction component, modified via LoRA, adjusts the orientation
    of weights. By scaling the LoRA update component :math:`BAx` with the `magnitude` vector, DoRA allows the model
    to apply distinct scaling adjustments across different output dimensions.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False
        quantize_base (bool): Whether to quantize base linear weight or not.
            Default: False
        **quantization_kwargs: Keyword arguments to pass to `to_nf4` when quantizing the base linear weight.
            Examples of valid arguments are `block_size` and `scaler_block_size`, which control the granularity of
            weight quantization and scaler quantization respectively. This is only used if `quantize_base` is True.
            Default None

    Raises:
        ValueError: If ``quantize_base`` is False, but quantization kwargs are provided.

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
        **quantization_kwargs,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaling = alpha / rank
        self.use_bias = use_bias
        self._quantize_base = quantize_base

        if not self._quantize_base and any([v for v in quantization_kwargs.values()]):
            raise ValueError(
                f"``quantize_base`` is False, but received the following quantization arguments: {quantization_kwargs}"
            )

        # Setup weight and bias
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=self.use_bias)
        weight = (
            linear.weight
            if not self._quantize_base
            else to_nf4(linear.weight, **quantization_kwargs)
        )
        bias = linear.bias if self.use_bias else None

        # 'self.disabled' is a flag showing whether to turn off DoRA adapters,
        # this can be used in DPO for treating the dora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.magnitude = nn.Parameter(torch.empty(out_dim))
        self.initialize_parameters()

    def to_empty(
        self, *, device: Optional[Union[str, torch.device, int]], recurse: bool = True
    ):
        self.lora_a.to_empty(device=device, recurse=recurse)
        self.lora_b.to_empty(device=device, recurse=recurse)

        magnitude = nn.Parameter(
            torch.empty_like(self.magnitude, device=device),
            requires_grad=self.magnitude.requires_grad,
        )
        torch.utils.swap_tensors(self.magnitude, magnitude)
        return self

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)

    @torch.no_grad()
    def initialize_dora_magnitude(self):
        """
        DoRA initializes the magnitude vector such that its outputs are initially
        identical to standard LoRA's outputs.

        This must be called after loading/initializing base model and LoRA params.

        Raises:
            RuntimeError: If base or LoRA parameters are still on meta device.
        """
        if any(
            [
                self.weight.is_meta,
                self.lora_a.weight.is_meta,
                self.lora_b.weight.is_meta,
            ]
        ):
            raise RuntimeError(
                "Cannot initialize DoRA magnitude if base or LoRA parameters are still on meta device."
            )
        base_weight = self.weight.to(self.lora_a.weight.dtype)
        lora_weight = self.lora_b.weight @ self.lora_a.weight
        weight_norm = self._get_weight_norm(base_weight, lora_weight)

        if isinstance(self.magnitude, DTensor):
            if not isinstance(weight_norm, DTensor):
                device_mesh = self.magnitude.device_mesh
                placements = self.magnitude.placements
                # Convert local tensor to DTensor with same distribution as magnitude
                weight_norm = DTensor.from_local(
                    weight_norm,
                    device_mesh,
                    placements
                )
            self.magnitude.copy_(weight_norm)
        else:
            if isinstance(weight_norm, DTensor):
                weight_norm = weight_norm.to_local()
            self.magnitude.copy_(weight_norm)

    def _get_weight_norm(self, weight, lora_weight):
        # Convert DTensors to local tensors if needed
        if hasattr(weight, 'to_local'):
            weight = weight.to_local()
        if hasattr(lora_weight, 'to_local'):
            lora_weight = lora_weight.to_local()

        weight = weight.to(lora_weight.dtype) + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def adapter_params(self) -> list[str]:
        """
        Return a list of strings corresponding to the names of the ``nn.Parameter`` s in
        the model coming from the adapter.

        For DoRA this means lora_a.weight, lora_b.weight, and magnitude.
        """
        adapter_params = ["lora_a.weight", "lora_b.weight", "magnitude"]
        return adapter_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``
        """
        if self._quantize_base:
            base_out = linear_nf4(input=x, weight=self.weight)
            if self.use_bias:
                base_out = base_out + self.bias
        else:
            base_out = F.linear(x, self.weight, self.bias)
        if self.disabled:
            return base_out

        x = self.dropout(x)

        lora_out = self.lora_b(self.lora_a(x))
        # Can't use raw matmul since FSDP hooks are attached to __call__
        # Instead follow the approach in https://github.com/huggingface/peft/pull/1806
        x_eye = torch.eye(
            self.lora_a.weight.shape[1], device=self.lora_a.weight.device, dtype=x.dtype
        )
        lora_weight = self.lora_b(self.lora_a(x_eye)).T
        magnitude = self.magnitude
        weight = self.weight.to(x.dtype)
        weight_norm = self._get_weight_norm(weight, lora_weight.detach())
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        dora_out = (
            mag_norm_scale - 1
        ) * base_out + mag_norm_scale * lora_out * self.scaling

        return dora_out + base_out


class DoRALinearCache(nn.Module, AdapterModule):
    """Cached DoRA linear layer with optimized forward pass.

    This implementation caches the weight norm and magnitude scale computations
    to avoid redundant calculations during inference or when weights haven't changed.
    The cache is automatically invalidated when parameters are updated during training.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            rank: int,
            alpha: float,
            dropout: float = 0.0,
            use_bias: bool = False,
            quantize_base: bool = False,
            **quantization_kwargs,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scaling = alpha / rank
        self.use_bias = use_bias
        self._quantize_base = quantize_base

        if not self._quantize_base and any([v for v in quantization_kwargs.values()]):
            raise ValueError(
                f"``quantize_base`` is False, but received the following quantization arguments: {quantization_kwargs}"
            )

        # Setup weight and bias
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=self.use_bias)
        weight = (
            linear.weight
            if not self._quantize_base
            else to_nf4(linear.weight, **quantization_kwargs)
        )
        bias = linear.bias if self.use_bias else None

        self.disabled = False
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.magnitude = nn.Parameter(torch.empty(out_dim))

        # Cache-related attributes
        self._cached_weight_norm = None
        self._cached_mag_norm_scale = None
        self._cache_valid = False
        self._version_tracker = {}

        self.initialize_parameters()

        # Register hooks to invalidate cache when parameters change
        self._register_cache_hooks()

    def _register_cache_hooks(self):
        """Register hooks to invalidate cache when parameters are updated."""

        def invalidate_cache(grad):
            self._cache_valid = False
            return grad

        # Register backward hooks on parameters that affect the cache
        if self.lora_a.weight.requires_grad:
            self.lora_a.weight.register_hook(invalidate_cache)
        if self.lora_b.weight.requires_grad:
            self.lora_b.weight.register_hook(invalidate_cache)
        if self.magnitude.requires_grad:
            self.magnitude.register_hook(invalidate_cache)
        if self.weight.requires_grad:
            self.weight.register_hook(invalidate_cache)

    def _check_cache_validity(self):
        """Check if cache is still valid by tracking parameter versions."""
        # For inference mode, we can use a simpler check
        if not self.training:
            return self._cache_valid

        # During training, double-check using version counters
        current_versions = {
            'lora_a': self.lora_a.weight._version,
            'lora_b': self.lora_b.weight._version,
            'magnitude': self.magnitude._version,
            'weight': self.weight._version,
        }

        if current_versions != self._version_tracker:
            self._version_tracker = current_versions.copy()
            return False

        return self._cache_valid

    def to_empty(
            self, *, device: Optional[Union[str, torch.device, int]], recurse: bool = True
    ):
        self.lora_a.to_empty(device=device, recurse=recurse)
        self.lora_b.to_empty(device=device, recurse=recurse)

        magnitude = nn.Parameter(
            torch.empty_like(self.magnitude, device=device),
            requires_grad=self.magnitude.requires_grad,
        )
        torch.utils.swap_tensors(self.magnitude, magnitude)

        # Invalidate cache after device change
        self._invalidate_cache()
        return self

    def initialize_parameters(self):
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)
        self._invalidate_cache()

    @torch.no_grad()
    def initialize_dora_magnitude(self):
        """
        DoRA initializes the magnitude vector such that its outputs are initially
        identical to standard LoRA's outputs.

        This must be called after loading/initializing base model and LoRA params.

        Raises:
            RuntimeError: If base or LoRA parameters are still on meta device.
        """
        if any(
                [
                    self.weight.is_meta,
                    self.lora_a.weight.is_meta,
                    self.lora_b.weight.is_meta,
                ]
        ):
            raise RuntimeError(
                "Cannot initialize DoRA magnitude if base or LoRA parameters are still on meta device."
            )
        base_weight = self.weight.to(self.lora_a.weight.dtype)
        lora_weight = self.lora_b.weight @ self.lora_a.weight
        weight_norm = self._get_weight_norm(base_weight, lora_weight)

        if isinstance(self.magnitude, DTensor):
            if not isinstance(weight_norm, DTensor):
                device_mesh = self.magnitude.device_mesh
                placements = self.magnitude.placements
                # Convert local tensor to DTensor with same distribution as magnitude
                weight_norm = DTensor.from_local(
                    weight_norm,
                    device_mesh,
                    placements
                )
            self.magnitude.copy_(weight_norm)
        else:
            if isinstance(weight_norm, DTensor):
                weight_norm = weight_norm.to_local()
            self.magnitude.copy_(weight_norm)

    def _get_weight_norm(self, weight, lora_weight):
        # Convert DTensors to local tensors if needed
        if hasattr(weight, 'to_local'):
            weight = weight.to_local()
        if hasattr(lora_weight, 'to_local'):
            lora_weight = lora_weight.to_local()

        weight = weight.to(lora_weight.dtype) + self.scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def _invalidate_cache(self):
        """Invalidate the cached values."""
        self._cache_valid = False
        self._cached_weight_norm = None
        self._cached_mag_norm_scale = None

    def _compute_and_cache_norms(self, x_dtype):
        """Compute and cache the weight norm and magnitude scale."""
        with torch.no_grad():
            # Compute lora weight matrix
            x_eye = torch.eye(
                self.lora_a.weight.shape[1],
                device=self.lora_a.weight.device,
                dtype=x_dtype
            )
            lora_weight = self.lora_b(self.lora_a(x_eye)).T

            # Compute weight norm
            weight = self.weight.to(x_dtype)
            weight_norm = self._get_weight_norm(weight, lora_weight)

            # Compute magnitude scale
            mag_norm_scale = (self.magnitude / weight_norm).view(1, -1)

            # Cache the computed values
            self._cached_weight_norm = weight_norm
            self._cached_mag_norm_scale = mag_norm_scale
            self._cache_valid = True

    def adapter_params(self) -> list[str]:
        """Return adapter parameter names."""
        return ["lora_a.weight", "lora_b.weight", "magnitude"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``
        """
        # Compute base output
        if self._quantize_base:
            base_out = linear_nf4(input=x, weight=self.weight)
            if self.use_bias:
                base_out = base_out + self.bias
        else:
            base_out = F.linear(x, self.weight, self.bias)

        if self.disabled:
            return base_out

        # Apply dropout
        x = self.dropout(x)

        # Compute LoRA output
        lora_out = self.lora_b(self.lora_a(x))

        # Check if we can use cached values
        if not self._check_cache_validity() or self._cached_mag_norm_scale is None:
            # Need to recompute and cache
            if self.training:
                # During training, compute norms with gradient tracking
                x_eye = torch.eye(
                    self.lora_a.weight.shape[1],
                    device=self.lora_a.weight.device,
                    dtype=x.dtype
                )
                lora_weight = self.lora_b(self.lora_a(x_eye)).T
                weight = self.weight.to(x.dtype)
                weight_norm = self._get_weight_norm(weight, lora_weight.detach())
                weight_norm = weight_norm.detach()
                mag_norm_scale = (self.magnitude / weight_norm).view(1, -1)

                # Cache for potential reuse within the same forward pass
                with torch.no_grad():
                    self._cached_mag_norm_scale = mag_norm_scale.detach()
                    self._cached_weight_norm = weight_norm
                    self._cache_valid = True
            else:
                # During inference, compute and cache without gradients
                self._compute_and_cache_norms(x.dtype)
                mag_norm_scale = self._cached_mag_norm_scale
        else:
            # Use cached values
            mag_norm_scale = self._cached_mag_norm_scale

            # Ensure correct dtype and device
            if mag_norm_scale.dtype != x.dtype:
                mag_norm_scale = mag_norm_scale.to(x.dtype)
            if mag_norm_scale.device != x.device:
                mag_norm_scale = mag_norm_scale.to(x.device)

        # Compute DoRA output
        dora_out = (mag_norm_scale - 1) * base_out + mag_norm_scale * lora_out * self.scaling

        return dora_out + base_out

    def train(self, mode: bool = True):
        """Override train to invalidate cache when switching modes."""
        if not mode and self.training:
            # Switching from training to eval, invalidate cache
            self._invalidate_cache()
        elif mode and not self.training:
            # Switching from eval to training, invalidate cache
            self._invalidate_cache()
        return super().train(mode)

def _lora_a_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA A weight to Kaiming uniform.
    """
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA B weight to zeros.
    """
    nn.init.zeros_(x.weight)

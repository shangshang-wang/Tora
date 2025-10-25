# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
from functools import partial
from typing import Any, Optional, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, generation, modules, rlhf, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.datasets import ConcatDataset
from torchtune.rl.generation import generate
from torchtune.rl.rewards import batched_rewards
from torchtune.rl.types import GRPOStats, GRPOTrajectory
from torchtune.modules import local_kv_cache
from torchtune.modules.peft import get_adapter_params, set_trainable_params
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import disable_dropout, DummyProfiler, PROFILER_KEY
from torchtune.training.lr_schedulers import get_lr
from tqdm import tqdm

import torch._dynamo.config as dynamo_config
dynamo_config.recompile_limit = 100


class GRPOFullFinetuneRecipeSingleDevice(FTRecipeInterface):
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        self._output_dir = cfg.output_dir

        # Logging attributes
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = utils.get_logger(cfg.log_level)
        if self._log_peak_memory_stats and self._device.type != "cuda":
            self._logger.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Single device attributes
        self.world_size = 1
        self.rank = 0
        self._is_rank_zero = True

        # Training attributes
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._compile = cfg.get("compile", False)

        # Recipe state attributes
        self.seed = training.set_seed(seed=cfg.seed)
        self.total_epochs = cfg.epochs
        self.global_step = 0
        self._steps_run = 0
        self._total_steps = 0
        self._epochs_run = 0
        self._rng = torch.Generator(self._device).manual_seed(self.seed)

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self._epochs_run = ckpt_dict[training.EPOCHS_KEY]
            self._rng.set_state(ckpt_dict[training.RNG_KEY])

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, lr scheduler, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)
        self._metric_logger.log_config(cfg)

        # Handle the MAIN checkpointer. It's stored in `self` to be used for saving later.
        self._checkpointer = config.instantiate(
            cfg.checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        # Handle the REFERENCE checkpointer. It's instantiated locally just to load weights.
        ref_checkpointer_instance = config.instantiate(
            cfg.ref_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        ref_checkpoint_dict = ref_checkpointer_instance.load_checkpoint()

        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            model_sd=checkpoint_dict[training.MODEL_KEY],
        )
        self._ref_model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            model_sd=ref_checkpoint_dict[training.MODEL_KEY],
            eval_mode=True,
        )

        # Utilize the same tokenizer for both models
        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        # Initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if self._compile:
            training.compile_loss(
                self._loss_fn, dynamic=True, verbose=self._is_rank_zero
            )

        collate_name = cfg.get(
            "collate_fn", "torchtune.rl.data.padded_collate_rl"
        )
        self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=cfg.batch_size,
            collate_fn=collate_name,
            dataloader_state_dict=(
                checkpoint_dict[training.DATALOADER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        self._steps_per_epoch = len(self._dataloader)
        self.global_step = self._epochs_run * self._steps_per_epoch

        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        # Set up profiler
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # RL params
        self.grpo_samples = cfg.grpo_samples
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens
        self.batch_size = cfg.batch_size
        self._forward_batch_size = cfg.forward_batch_size

        self._ppo_epochs = cfg.ppo_epochs
        self._save_every_n_epochs = cfg.save_every_n_epochs
        self._total_steps = cfg.num_steps

        # Parse reward function names from the config for named logging
        self.reward_names = []
        if "reward_functions" in cfg and isinstance(
            cfg.reward_functions, (ListConfig, list)
        ):
            for rf_cfg in cfg.reward_functions:
                component_path = rf_cfg.get("_component_", "")
                if component_path:
                    class_name = component_path.split(".")[-1]
                    self.reward_names.append(class_name)

            if self.reward_names:
                self._logger.info(
                    f"Found reward names for logging: {self.reward_names}"
                )

        if cfg.get("stop_token_ids", False):
            stop_token_ids = cfg.stop_token_ids
            if self._tokenizer.eos_id not in stop_token_ids:
                warn(
                    f"tokenizer eos_id ({self._tokenizer.eos_id}) is not in stop_token_ids ({stop_token_ids})."
                    "This may lead to unexpected behaviour."
                )
        else:
            if not hasattr(self._tokenizer, "stop_tokens"):
                warn(
                    "No stop tokens defined in tokenizer, and no stop_token_ids provided. This may lead to unexpected behaviour."
                )
                stop_token_ids = []
            else:
                stop_token_ids = self._tokenizer.stop_tokens
        self._stop_token_ids = torch.tensor(stop_token_ids, device=self._device)

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: Optional[DictConfig],
        num_training_steps: int,
        last_epoch: int,
    ) -> Optional[Optimizer]:
        """
        Set up the learning rate scheduler based on the provided configuration.
        It supports both standard optimization and optimizer-in-backward cases.

        Args:
            cfg_lr_scheduler (Optional[DictConfig]): The learning rate scheduler configuration.
            num_training_steps (int): The total number of training steps.
            last_epoch (int): The index of the last epoch.

        Returns:
            lr_scheduler (Optional[Optimizer]): The learning rate scheduler.
        """
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                self._logger.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        optimizer = self._optimizer

        # Instantiate the learning rate scheduler
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._is_rank_zero:
            self._logger.info("Learning rate scheduler is initialized.")

        return lr_scheduler

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        self._logger.debug(f"Profiler config after instantiation: {profiler_cfg}")
        if self._is_rank_zero:
            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]
                self.profiler_num_cycles = profiler_cfg["num_cycles"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        model_sd: dict[str, Any],
        eval_mode: bool = False,
    ) -> nn.Module:
        """
        Initialize and load a model on a single device.
        """
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)
            for module in model.modules():
                if hasattr(module, "rope_init"):
                    module.rope_init()

        if self._compile:
            training.compile_model(model, dynamic=True, verbose=self._is_rank_zero)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )
            
        incompatible_keys = model.load_state_dict(model_sd, strict=False)
        missing_keys = getattr(incompatible_keys, "missing_keys", [])
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", [])
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys encountered while loading checkpoint: {unexpected_keys}"
            )

        adapter_missing_keys = [
            k for k in missing_keys if "lora" in k or "magnitude" in k
        ]
        non_adapter_missing_keys = [
            k for k in missing_keys if k not in adapter_missing_keys
        ]
        if non_adapter_missing_keys:
            self._logger.warning(
                "Missing keys while loading checkpoint: %s", non_adapter_missing_keys
            )
        if adapter_missing_keys:
            self._logger.debug(
                "LoRA adapter keys missing from checkpoint (expected when initializing new adapters): %s",
                adapter_missing_keys,
            )

        adapter_params = get_adapter_params(model)
        if adapter_params and not eval_mode:
            set_trainable_params(model, adapter_params)
            self._logger.info(
                "Detected %d adapter parameter groups; freezing base model weights.",
                len(adapter_params),
            )

        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )

        disable_dropout(model)

        if eval_mode:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        if self._device.type != "cpu" and self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        self._logger.info(
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs"
        )
        return model

    def _setup_optimizer(
        self,
        cfg_optimizer: DictConfig,
        opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)
        self._logger.info("Optimizer is initialized.")
        return optimizer

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        collate_fn: str,
        dataloader_state_dict: Optional[dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """

        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)

        # Instantiate collate_fn
        collate_fn = _get_component_from_path(collate_fn)

        sampler = StatefulDistributedSampler(
            ds,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            seed=self.seed,
        )
        dataloader = StatefulDataLoader(
            dataset=ds,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=(
                partial(
                    collate_fn,
                    padding_idx=self._tokenizer.pad_id,
                )
            ),
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
        )
        if dataloader_state_dict is not None:
            dataloader.load_state_dict(dataloader_state_dict)
            # B/c we currently only save at epoch boundaries, if we cut the previous epoch short
            # we need to force the dataloader to finish the last iteration before it's actually used
            list(dataloader)
        return dataloader

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Model weights with key training.MODEL_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the model weights and recipe state in
        different checkpoint files. To correctly resume training from an intermediate checkpoint,
        the model weights and recipe state must be provided.
        """
        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs

        self._logger.info(
            "Saving checkpoint. This may take some time. Retrieving full model state dict..."
        )
        start = time.perf_counter()

        cpu_state_dict = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in self._model.state_dict().items()
        }

        self._logger.info(
            f"Getting full model state dict took {time.perf_counter() - start:.2f} secs"
        )

        if intermediate_checkpoint:
            start = time.perf_counter()
            self._logger.info("Getting optimizer state dict...")
            opt_state_dict = self._optimizer.state_dict()
            self._logger.info(
                f"Getting optimizer state dict took {time.perf_counter() - start:.2f} secs"
            )
        else:
            opt_state_dict = None

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file

        start = time.perf_counter()
        checkpoint_dict.update({training.MODEL_KEY: cpu_state_dict})

        if intermediate_checkpoint:
            checkpoint_dict.update(
                {
                    training.OPT_KEY: opt_state_dict,
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self._epochs_run,
                    training.TOTAL_EPOCHS_KEY: self.total_epochs,
                    training.RNG_KEY: self._rng.get_state(),
                    training.DATALOADER_KEY: self._dataloader.state_dict(),
                }
            )

        self._checkpointer.save_checkpoint(
            checkpoint_dict,
            epoch=epoch,
            intermediate_checkpoint=intermediate_checkpoint,
        )
        self._logger.info(
            f"Saving checkpoint took {time.perf_counter() - start:.2f} secs"
        )

    def generate_trajectory(
        self,
        input_ids: torch.Tensor,
        answers: list[str],
        *,
        visual_pos_masks: Optional[torch.Tensor] = None,
        pixel_values_images: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> GRPOTrajectory:
        """
        Generates a trajectory given the current policy model, the reference policy model, the reward function,
        and batch of inputs. This is done over the following steps:

        1: Generate responses, and logits corresponding to the responses using the current policy,
            generating (query, response) pairs.
        2. Estimate logprobs of the generated responses using the current policy.
        3. Compute rewards and successes for the generated responses.
        4. Estimate advantages using GRPO.
        5. Replace any tokens in the response after the first stop token (usually EOS token) with padding,
            producing truncated responses.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]
            answers (list[str]): list of answers corresponding to the input_ids
            visual_pos_masks (Optional[torch.Tensor]): boolean masks indicating which token
                positions should be replaced with visual embeddings. Shape [b, seq_length].
            pixel_values_images (Optional[torch.Tensor]): stacked image pixels produced by the
                Qwen3-VL processor. Shape [b, num_images, C, H, W].
            image_grid_thw (Optional[torch.Tensor]): grid metadata for ``pixel_values_images`` with
                shape [b, num_images, 3].

        Returns:
            Trajectory: An instance of :class:`~torchtune.rlhf.GRPOTrajectory` comprising
                the current trajectory.
        """
        batch_size, context_length = input_ids.shape
        grpo_size = self.grpo_samples

        def _expand_optional(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            return tensor.repeat_interleave(grpo_size, dim=0)

        batch_input_ids = input_ids[:, None, :].expand(-1, grpo_size, -1)
        batch_input_ids = batch_input_ids.reshape(batch_size * grpo_size, -1)

        expanded_visual_masks = _expand_optional(visual_pos_masks)
        expanded_pixel_values = _expand_optional(pixel_values_images)
        expanded_image_grid = _expand_optional(image_grid_thw)

        max_total_len = context_length + self._max_generated_tokens

        model_prompt_kwargs: dict[str, torch.Tensor] = {}
        model_logits_kwargs: dict[str, torch.Tensor] = {}

        if expanded_pixel_values is not None:
            model_prompt_kwargs["pixel_values"] = expanded_pixel_values
            model_logits_kwargs["pixel_values"] = expanded_pixel_values

        if expanded_image_grid is not None:
            model_prompt_kwargs["image_grid_thw"] = expanded_image_grid
            model_logits_kwargs["image_grid_thw"] = expanded_image_grid

        visual_masks_for_logits: Optional[torch.Tensor] = None
        if expanded_visual_masks is not None:
            model_prompt_kwargs["visual_pos_masks"] = expanded_visual_masks
            pad = max_total_len - expanded_visual_masks.shape[-1]
            if pad > 0:
                visual_masks_for_logits = torch.nn.functional.pad(
                    expanded_visual_masks,
                    (0, pad),
                    value=False,
                )
            else:
                visual_masks_for_logits = expanded_visual_masks
            model_logits_kwargs["visual_pos_masks"] = visual_masks_for_logits

        # step 1: generate responses, and logits corresponding to the responses using the current policy
        with local_kv_cache(
            model=self._model,
            batch_size=batch_size * grpo_size,
            device=self._device,
            dtype=self._dtype,
            decoder_max_seq_len=max_total_len,
        ):  
            query_responses, _ = generate(  # [B x G, L], [B x G, L, V]
                model=self._model,
                prompt=batch_input_ids,
                max_generated_tokens=self._max_generated_tokens,
                temperature=self._temperature,
                top_k=self._top_k,
                pad_id=self._tokenizer.pad_id,
                rng=self._rng,
                stop_tokens=self._tokenizer.stop_tokens,
                return_logits=False,
                model_inputs=model_prompt_kwargs if model_prompt_kwargs else None,
            )

        # Truncate if longer than expected and Pad if shorter
        query_responses = query_responses[:, :max_total_len]
        if query_responses.shape[1] < max_total_len:
            pad_len = max_total_len - query_responses.shape[1]
            query_responses = torch.nn.functional.pad(
                query_responses,
                (0, pad_len),
                value=self._tokenizer.pad_id
            )

        responses = query_responses[:, context_length:].clone()
        query_response_padding_masks = query_responses != self._tokenizer.pad_id

        # step 1.1 create attention masks and position IDs for any padding tokens in inputs, used for future forward passes
        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        del query_response_padding_masks

        # step 2. estimate logprobs of the responses using the current policy
        logits = self._model(
            query_responses,
            input_pos=position_ids,
            mask=masks,
            **model_logits_kwargs,
        )
        logits = logits[:, context_length - 1 :]
        logprobs = rlhf.batched_logits_to_logprobs(logits, responses, self._temperature)
        del logits
        torch.cuda.empty_cache()

        # step 2.1 estimate logprobs of the responses using the reference policy
        ref_logits = self._ref_model(
            query_responses,
            input_pos=position_ids,
            mask=masks,
            **model_logits_kwargs,
        )
        ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
        ref_logprobs = rlhf.batched_logits_to_logprobs(
            ref_logits, responses, self._temperature
        )
        del ref_logits
        torch.cuda.empty_cache()

        # step 4. replace any tokens in the responses after the first stop token (usually EOS token) with padding
        # resulting in truncated responses
        (
            response_padding_masks,
            responses,
        ) = rlhf.truncate_sequence_at_first_stop_token(  # [B x G, L]
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )

        # Do some reward modeling
        # responses :: [B x G, L]
        responses = responses.reshape(batch_size, grpo_size, -1)  # [B, G, L]
        rewards, successes, _ = batched_rewards(self._tokenizer, responses, answers, self._device)
        rewards = rewards.to(self._device)  # [B, G]
        successes = successes.to(self._device)  # [B, G]

        # TODO Create equal weights for all reward functions
        reward_components = rewards.clone()

        num_reward_funcs = rewards.shape[-1]
        reward_weights = torch.ones(num_reward_funcs, device=self._device) / num_reward_funcs  # [num_reward_funcs]
        aggregated_rewards = (rewards * reward_weights).sum(dim=-1)  # [B, G]
        successes = successes.mean(dim=-1)  # [B, G]

        # Use the aggregated reward for advantage calculation
        advantages = (aggregated_rewards - aggregated_rewards.mean(1, keepdim=True)) / (
            aggregated_rewards.std(1, keepdim=True) + 1e-4
        )
        advantages = advantages.reshape(batch_size * grpo_size)  # flatten

        del responses
        torch.cuda.empty_cache()

        # step 6. mask out all the invalid values in the trajectory due to padding tokens
        logprobs[response_padding_masks] = 1.0
        ref_logprobs[response_padding_masks] = 1.0

        return GRPOTrajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            rewards=aggregated_rewards.reshape(batch_size * grpo_size),
            reward_components=reward_components.reshape(batch_size * grpo_size, -1),
            successes=successes.reshape(batch_size * grpo_size),
            advantages=advantages,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            seq_lens=training.get_unmasked_sequence_lengths(response_padding_masks),
            visual_pos_masks=visual_masks_for_logits,
            pixel_values=expanded_pixel_values,
            image_grid_thw=expanded_image_grid,
        )

    def generate_trajectory_batched(
        self,
        input_ids: torch.Tensor,
        answers: list[str],
        *,
        visual_pos_masks: Optional[torch.Tensor] = None,
        pixel_values_images: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> GRPOTrajectory:
        """
        Generates a ``self.batch_size`` batch of trajectories using `self._forward_batch_size` batch sizes.
        See ``generate_trajectory`` for more details.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]
            answers: (list[str]): list of answers corresponding to the input_ids

        Returns:
            Trajectory: An instance of :class:`~torchtune.rlhf.Trajectory`, comprising
                the current trajectory.
        """
        trajectories: list[GRPOTrajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[
                    batch_start : batch_start + self._forward_batch_size
                ]
                batch_answers = answers[
                    batch_start : batch_start + self._forward_batch_size
                ]
                batch_visual_masks = (
                    visual_pos_masks[
                        batch_start : batch_start + self._forward_batch_size
                    ]
                    if visual_pos_masks is not None
                    else None
                )
                batch_pixel_values = (
                    pixel_values_images[
                        batch_start : batch_start + self._forward_batch_size
                    ]
                    if pixel_values_images is not None
                    else None
                )
                batch_image_grid = (
                    image_grid_thw[
                        batch_start : batch_start + self._forward_batch_size
                    ]
                    if image_grid_thw is not None
                    else None
                )
                torch.cuda.empty_cache()
                trajectories.append(
                    self.generate_trajectory(
                        batch_input_ids,
                        batch_answers,
                        visual_pos_masks=batch_visual_masks,
                        pixel_values_images=batch_pixel_values,
                        image_grid_thw=batch_image_grid,
                    )
                )
                torch.cuda.empty_cache()
        return self._concat_trajectories(trajectories)

    def _concat_trajectories(
        self, trajectories: list[GRPOTrajectory]
    ) -> GRPOTrajectory:
        def _concat(attr: str) -> Optional[torch.Tensor]:
            values = [getattr(traj, attr) for traj in trajectories]
            if all(value is None for value in values):
                return None
            return torch.cat(values, dim=0)

        return GRPOTrajectory(
            query_responses=_concat("query_responses"),
            logprobs=_concat("logprobs"),
            ref_logprobs=_concat("ref_logprobs"),
            rewards=_concat("rewards"),
            reward_components=_concat("reward_components"),
            successes=_concat("successes"),
            advantages=_concat("advantages"),
            masks=_concat("masks"),
            position_ids=_concat("position_ids"),
            response_padding_masks=_concat("response_padding_masks"),
            seq_lens=_concat("seq_lens"),
            visual_pos_masks=_concat("visual_pos_masks"),
            pixel_values=_concat("pixel_values"),
            image_grid_thw=_concat("image_grid_thw"),
        )

    def grpo_step(
        self,
        trajectory: GRPOTrajectory,
        context_length: int,
    ) -> GRPOStats:
        """
        Perform a single GRPO optimization step over a batch of trajectories and corresponding advantages and returns.

        Args:
            trajectory (Trajectory): a batch of trajectories
            context_length (int): input ids sequence length

        Returns:
            GRPOStats: An instance of :class:`~torchtune.rl.types.GRPOStats`, a NamedTuple containing:
               - loss (torch.Tensor): The total PPO loss.
               - ratios (torch.Tensor): The ratio between the current and old policy probabilities.
               - clipfrac (torch.Tensor): The fraction of ratios that were clipped.
               - approx_policy_kls: Average estimated KL divergence between the policy before and after the optimisation step.

        """
        torch.cuda.empty_cache()

        # estimate logprobs from the policy at the current optimisation step
        model_kwargs: dict[str, torch.Tensor] = {}
        if trajectory.visual_pos_masks is not None:
            model_kwargs["visual_pos_masks"] = trajectory.visual_pos_masks
        if trajectory.pixel_values is not None:
            model_kwargs["pixel_values"] = trajectory.pixel_values
        if trajectory.image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = trajectory.image_grid_thw

        pi_logits = self._model(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.masks,
            **model_kwargs,
        )

        pi_logits = rlhf.truncate_sequence_for_logprobs(pi_logits, context_length)
        pi_logprobs = rlhf.batched_logits_to_logprobs(
            pi_logits,
            trajectory.query_responses[:, context_length:],
            self._temperature,
            chunk_size=1,
        )

        pi_logprobs[trajectory.response_padding_masks] = 1.0

        del pi_logits
        torch.cuda.empty_cache()

        # calculate grpo loss
        loss, policy_loss, kl_loss, ratios, clipfrac = self._loss_fn(
            trajectory.logprobs,
            pi_logprobs,
            trajectory.ref_logprobs,
            trajectory.advantages,
            padding_masks=~trajectory.response_padding_masks,
        )

        torch.cuda.empty_cache()
        loss.backward()

        with torch.no_grad():
            approx_policy_kls = (
                0.5 * (pi_logprobs - trajectory.logprobs).pow(2)
            ).mean()

        return GRPOStats(
            loss.detach(),
            policy_loss.detach(),
            kl_loss.detach(),
            ratios.detach(),
            clipfrac.detach(),
            approx_policy_kls.detach(), # Good practice, though already in no_grad
            # None # TODO do we need metadata?
        )

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        grad_norm = None

        training_completed = False
        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self._epochs_run, self.total_epochs):
            pbar = tqdm(total=self._steps_per_epoch, disable=not self._is_rank_zero)
            self._dataloader.sampler.set_epoch(curr_epoch)
            for idx, batch in enumerate(self._dataloader):
                # Start tracking CUDA memory for active steps for just the first epoch
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history()

                tokens = batch["tokens"]  # type: ignore
                answers = batch["answers"]  # type: ignore

                visual_pos_masks = batch.get("visual_pos_masks")
                pixel_values_images = batch.get("pixel_values_images")
                image_grid_thw = batch.get("image_grid_thw")

                tokens = tokens.to(self._device)  # [B, P]
                if visual_pos_masks is not None:
                    visual_pos_masks = visual_pos_masks.to(self._device)
                if pixel_values_images is not None:
                    pixel_values_images = pixel_values_images.to(self._device)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(self._device)

                _, context_length = tokens.shape

                trajectory = self.generate_trajectory_batched(
                    tokens,
                    answers,
                    visual_pos_masks=visual_pos_masks,
                    pixel_values_images=pixel_values_images,
                    image_grid_thw=image_grid_thw,
                )

                grpo_stats: list[GRPOStats] = []
                for _ in range(self._ppo_epochs):
                    step_stats = self.grpo_step(trajectory, context_length)

                    grpo_stats.append(step_stats)

                    if self._clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(),
                            max_norm=float(self._clip_grad_norm),
                        )
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                # Stop tracking CUDA memory now that active steps are complete
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx
                    == self.profiler_wait_steps
                    + self.profiler_warmup_steps
                    + self.profiler_active_steps
                    and self._device.type == "cuda"
                ):
                    torch.cuda.memory._record_memory_history(enabled=None)

                self._steps_run += 1
                if self._steps_run % self._log_every_n_steps == 0:
                    extra_metrics = {}
                    extra_metrics["lr"] = get_lr(self._optimizer)
                    if grad_norm is not None:
                        extra_metrics["grad_norm"] = grad_norm

                    self.log_metrics(
                        trajectory,
                        GRPOStats(*map(torch.stack, zip(*grpo_stats))),
                        **extra_metrics,
                    )

                self.cleanup_after_step(trajectory, grpo_stats)
                self._profiler.step()

                pbar.update(1)

                if self._steps_run == self._total_steps:
                    training_completed = True
                    break

            self._epochs_run += 1
            if self._epochs_run % self._save_every_n_epochs == 0:
                self.save_checkpoint(curr_epoch)
            if training_completed:
                return

        self._profiler.stop()

    def log_metrics(
        self, trajectory: GRPOTrajectory, grpo_stats: GRPOStats, **extras
    ) -> None:
        """
        Log metrics and statistics for the current step to the metric logger.
        """
        rewards = trajectory.rewards.mean()
        successes = trajectory.successes.mean()
        mean_reward_components = trajectory.reward_components.mean(dim=0)

        log_dict = {
            "rewards": rewards.item(),
            "successes": successes.item(),
            "num_stop_tokens": trajectory.response_padding_masks.any(-1).sum().item(),
            "loss": grpo_stats.loss.mean().item(),
            "policy_loss": grpo_stats.policy_loss.mean().item(),
            "kl_loss": grpo_stats.kl_loss.mean().item(),
            "clipfrac": grpo_stats.clipfrac.mean().item(),
            "ratios": grpo_stats.ratios.mean().item(),
            "approx_policy_kl": grpo_stats.approx_policy_kls.mean().item(),
            "response_lengths": trajectory.seq_lens.float().mean().item(),
            **extras,
        }

        if self._device.type == "cuda" and self._log_peak_memory_stats:
            log_dict.update(training.get_memory_stats(device=self._device))
        if self.reward_names and len(self.reward_names) == len(mean_reward_components):
            for name, reward_comp in zip(self.reward_names, mean_reward_components):
                # Using a "reward/" prefix groups these together in the wandb UI
                log_dict[f"reward/{name}"] = reward_comp.item()
        else:
            for i, reward_comp in enumerate(mean_reward_components):
                log_dict[f"reward/component_{i}"] = reward_comp.item()

        self._metric_logger.log_dict(log_dict, step=self.global_step)

    def cleanup(self) -> None:
        self._metric_logger.close()

    def cleanup_after_step(
        self,
        trajectory: GRPOTrajectory,
        l_grpo_stats: list[GRPOStats],
    ) -> None:
        for v in trajectory:
            del v
        del trajectory
        for g in l_grpo_stats:
            for v in g:
                del v
            del g
        del l_grpo_stats


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """

    recipe = GRPOFullFinetuneRecipeSingleDevice(cfg=cfg)
    config.log_config(recipe_name="GRPOFullFinetuneRecipeSingleDevice", cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

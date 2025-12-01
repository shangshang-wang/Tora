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
import math

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torchtune import config, generation, modules, rlhf, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.datasets import ConcatDataset
from torchtune.rl.generation import generate
from torchtune.rl.rewards import batched_rewards
from torchtune.rl.types import (
    GRPOStats,
    GRPOTrajectory,
    concat_grpo_trajectories,
    stack_grpo_stats,
)
from torchtune.modules import local_kv_cache
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import disable_dropout, DummyProfiler, PROFILER_KEY
from torchtune.training.lr_schedulers import get_lr
from tqdm import tqdm
from torchtune.recipe_support.reward_preview import RewardPreviewer, decode_responses

log = utils.get_logger("DEBUG")

import torch._dynamo.config as dynamo_config

dynamo_config.recompile_limit = 100


class LiftedGRPOFullFinetuneRecipeDistributed(FTRecipeInterface):
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        self._output_dir = cfg.output_dir

        # LIFTED GRPO CONFIG
        # If > 1.0, we expand the inner dimension (K > d).
        # e.g., 2.0 means inner dim is 2x hidden size.
        self.lifted_inner_dim_ratio = cfg.get("lifted_inner_dim_ratio", 0.0)
        self.lifted_lr = cfg.get("lifted_lr", 1e-4)  # Specific LR for A/B matrices

        # Storage for shadow parameters (A, B)
        # Keys: parameter name, Values: {'A': tensor, 'B': tensor}
        self._lifted_params = {}

        if self.lifted_inner_dim_ratio > 0:
            if self.rank == 0:
                log.info(f"Enabled LIFTED GRPO (Full Param). Expansion Ratio: {self.lifted_inner_dim_ratio}")

        # Logging attributes
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # Initialize the distributed environment
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            cfg.device, offload_ops_to_cpu=self.fsdp_cpu_offload
        )
        init_process_group(self.distributed_backend)
        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0
        self._reward_preview = RewardPreviewer(cfg.get("reward_preview"), self._is_rank_zero)

        # Training attributes
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._compile = cfg.get("compile", False)

        self._save_every_n_steps = cfg.get("save_every_n_steps", None)
        if self._save_every_n_steps is not None:
            if self._save_every_n_steps <= 0:
                raise ValueError("save_every_n_steps must be a positive integer.")

        # Recipe state attributes
        self.seed = training.set_seed(seed=cfg.seed)
        self.total_epochs = cfg.epochs
        self.global_step = 0
        self._steps_run = 0
        self._total_steps = 0
        self._epochs_run = 0
        self._rng = torch.Generator(self._device).manual_seed(self.seed)
        self._restored_steps_run: Optional[int] = None
        self._restored_global_step: Optional[int] = None
        self._restored_steps_per_epoch: Optional[int] = None

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> dict[str, Any]:
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        try:
            self._epochs_run = ckpt_dict[training.EPOCHS_KEY]
            self._rng.set_state(ckpt_dict[training.RNG_KEY])
            self._restored_global_step = ckpt_dict.get(training.STEPS_KEY, None)
            self._restored_steps_run = ckpt_dict.get(CHECKPOINT_STEPS_RUN_KEY, None)
            self._restored_steps_per_epoch = ckpt_dict.get(training.MAX_STEPS_KEY, None)

            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]

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
        if self.fsdp_cpu_offload:
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            self._metric_logger.log_config(cfg)

        self._checkpointer = config.instantiate(
            cfg.checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

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
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            model_sd=checkpoint_dict[training.MODEL_KEY],
            reshard_after_forward=False,
        )

        # <<<< LIFTED GRPO INIT
        # Initialize shadow parameters after model load
        if self.lifted_inner_dim_ratio > 0:
            self._init_lifted_parameters()
        # >>>> END LIFTED GRPO INIT

        self._ref_model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            model_sd=ref_checkpoint_dict[training.MODEL_KEY],
            eval_mode=True,
            reshard_after_forward=True,
        )
        torch.distributed.barrier()

        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._reward_preview.set_tokenizer(self._tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )

        self._loss_fn = config.instantiate(cfg.loss)
        if self._compile:
            training.compile_loss(self._loss_fn, dynamic=True, verbose=self._is_rank_zero)

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
        if (
                self._restored_steps_per_epoch is not None
                and self._restored_steps_per_epoch != self._steps_per_epoch
        ):
            warn(
                message=(
                    "Steps per epoch derived from the checkpoint "
                    f"({self._restored_steps_per_epoch}) does not match the current "
                    f"value ({self._steps_per_epoch}). Using the current value."
                )
            )
        if self._restored_steps_run is not None:
            self._steps_run = self._restored_steps_run
        else:
            self._steps_run = self._epochs_run * self._steps_per_epoch
        if self._restored_global_step is not None:
            self.global_step = self._restored_global_step
        else:
            self.global_step = self._epochs_run * self._steps_per_epoch

        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.get("lr_scheduler", None),
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.global_step - 1,
        )

        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # RL params
        self.grpo_samples = cfg.grpo_samples
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens
        self.batch_size = cfg.batch_size
        self._forward_batch_size = cfg.forward_batch_size

        self._ppo_epochs = cfg.ppo_epochs
        self._save_every_n_epochs = cfg.get("save_every_n_epochs", 1)
        if self._save_every_n_epochs <= 0:
            raise ValueError("save_every_n_epochs must be a positive integer.")
        self._total_steps = cfg.get("early_stop_steps", None)

        self.reward_names: list[str] = []
        self.reward_functions = []
        if "reward_functions" in cfg and isinstance(cfg.reward_functions, (ListConfig, list)):
            for rf_cfg in cfg.reward_functions:
                component_path = rf_cfg.get("_component_", "")
                if component_path:
                    class_name = component_path.split(".")[-1]
                    self.reward_names.append(class_name)
                self.reward_functions.append(config.instantiate(rf_cfg))

            utils.log_rank_zero(log, f"Configured reward functions: {self.reward_names}")

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

    # LIFTED GRPO HELPERS
    def _init_lifted_parameters(self):
        """
        Initializes A and B matrices for all Linear layers in the model.
        Stores them on CPU to save GPU memory.
        """
        utils.log_rank_zero(log, "Initializing Lifted Parameters (A, B) on CPU...")

        # We must summon full params to know the real shapes of FSDP wrapped modules
        with FSDP.summon_full_params(self._model, writeback=False, rank0_only=False):
            for name, module in self._model.named_modules():
                if isinstance(module, nn.Linear):
                    # Determine Lifted Rank K
                    d_in = module.in_features
                    d_out = module.out_features
                    K = int(d_in * self.lifted_inner_dim_ratio)

                    # Initialize A (d_out, K) and B (K, d_in)
                    # Note: nn.Linear weights are stored as (out, in)
                    # So W = A @ B

                    # Standard Kaiming init equivalent logic
                    A = torch.empty(d_out, K)
                    B = torch.empty(K, d_in)
                    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(B, a=math.sqrt(5))

                    # Store in dictionary
                    self._lifted_params[name] = {
                        'A': A.cpu(),  # Keep on CPU
                        'B': B.cpu()
                    }

        utils.log_rank_zero(log, f"Initialized Lifted Params for {len(self._lifted_params)} layers.")

    def _step_lifted_parameters(self):
        """
        The Core Lifted Update Step.
        1. Access Full W and Gradient dW (via FSDP summon).
        2. Compute gradients for A and B (Chain Rule).
        3. Update A and B.
        4. Recompute W = A @ B.
        5. Copy W back to model.
        6. Clear dW to prevent double updates.
        """
        if not self._lifted_params:
            return

        # writeback=True needed to modify weights in place
        with FSDP.summon_full_params(self._model, writeback=True, rank0_only=False):
            for name, module in self._model.named_modules():
                if name in self._lifted_params:
                    param = module.weight

                    # Skip if no gradient (e.g. frozen)
                    if param.grad is None:
                        continue

                    # Retrieve Shadow Params (Move to GPU for compute)
                    shadow = self._lifted_params[name]
                    A = shadow['A'].to(param.device).float()  # (d_out, K)
                    B = shadow['B'].to(param.device).float()  # (K, d_in)

                    grad_W = param.grad.float()  # (d_out, d_in)

                    # --- Lifted Gradient Calculation ---
                    # Forward: W = A @ B
                    # dL/dA = dL/dW @ B.T
                    # dL/dB = A.T @ dL/dW

                    grad_A = grad_W @ B.t()
                    grad_B = A.t() @ grad_W

                    # --- Update Shadow Params (Simple SGD) ---
                    # Using self.lifted_lr
                    A = A - (self.lifted_lr * grad_A)
                    B = B - (self.lifted_lr * grad_B)

                    # --- Project Back ---
                    W_new = A @ B

                    # Update Model Weight
                    # Ensure dtype matches original
                    param.data.copy_(W_new.to(param.dtype))

                    # CRITICAL: Zero out the gradient on the model weight
                    # This ensures the standard optimizer (if it steps later) does nothing
                    # or we effectively override it.
                    param.grad = None

                    # Store updated A/B back to CPU cache
                    shadow['A'] = A.cpu()
                    shadow['B'] = B.cpu()

    def _setup_lr_scheduler(
            self,
            cfg_lr_scheduler: Optional[DictConfig],
            num_training_steps: int,
            last_epoch: int,
    ) -> Optional[Optimizer]:
        if cfg_lr_scheduler is None:
            if self._is_rank_zero:
                log.info(
                    "No learning rate scheduler configured. Using constant learning rate."
                )
            return None

        optimizer = self._optimizer

        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        if self._is_rank_zero:
            log.info("Learning rate scheduler is initialized.")

        return lr_scheduler

    def _setup_profiler(
            self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                    cfg_profiler.get("_component_")
                    == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently."

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        utils.log_rank_zero(
            log, f" Profiler config after instantiation: {profiler_cfg}"
        )
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
            fsdp_cpu_offload: bool,
            model_sd: dict[str, Any],
            custom_sharded_layers: Optional[list[str]] = None,
            eval_mode: bool = False,
            reshard_after_forward: bool = True,
    ) -> tuple[nn.Module, nn.Module]:
        utils.log_rank_zero(
            log,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )
        init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        if eval_mode:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

        if self._compile:
            training.compile_model(model, dynamic=True, verbose=self._is_rank_zero)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        fsdp_shard_conditions = [
            partial(
                training.get_shard_conditions,
                names_to_match=custom_sharded_layers,
            )
        ]

        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                if hasattr(m, "rope_init"):
                    m.rope_init()

        training.load_from_full_model_state_dict(
            model,
            model_sd,
            self._device,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        training.validate_no_params_on_meta_device(model)
        utils.log_rank_zero(
            log,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )
        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        disable_dropout(model)

        return model

    def _setup_optimizer(
            self,
            cfg_optimizer: DictConfig,
            opt_state_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[Optimizer]:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model,
                optimizer,
                opt_state_dict,
                self._device,
            )
        utils.log_rank_zero(log, "Optimizer is initialized.")
        return optimizer

    def _setup_data(
            self,
            cfg_dataset: DictConfig,
            shuffle: bool,
            batch_size: int,
            collate_fn: str,
            dataloader_state_dict: Optional[dict[str, Any]] = None,
    ) -> StatefulDataLoader:
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, self._tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, self._tokenizer)

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
            drop_last=True,
        )
        if dataloader_state_dict is not None:
            dataloader.load_state_dict(dataloader_state_dict)
            should_drain = True
            if self._resume_from_checkpoint:
                steps_per_epoch = len(dataloader)
                restored_steps_run = self._restored_steps_run
                if (
                        restored_steps_run is not None
                        and steps_per_epoch > 0
                ):
                    expected_steps_before_current_epoch = self._epochs_run * steps_per_epoch
                    steps_into_epoch = restored_steps_run - expected_steps_before_current_epoch
                    should_drain = steps_into_epoch <= 0
            if should_drain:
                list(dataloader)
        return dataloader

    def save_checkpoint(
            self,
            epoch: int,
            *,
            is_final: bool = False,
    ) -> None:
        checkpoint_dict = {}
        intermediate_checkpoint = not is_final
        epoch = max(epoch, 0)

        utils.log_rank_zero(
            log,
            "Saving checkpoint. This may take some time. Retrieving full model state dict...",
        )
        start = time.perf_counter()

        cpu_state_dict = training.gather_cpu_state_dict(
            self._model,
            self._is_rank_zero,
            device=self._device,
        )

        utils.log_rank_zero(
            log,
            f"Getting full model state dict took {time.perf_counter() - start:.2f} secs",
        )

        if intermediate_checkpoint:
            start = time.perf_counter()
            utils.log_rank_zero(log, "Getting optimizer state dict...")
            opt_state_dict = training.get_full_optimizer_state_dict(
                self._model,
                self._optimizer,
                self._is_rank_zero,
                device=self._device,
            )
            utils.log_rank_zero(
                log,
                f"Getting optimizer state dict took {time.perf_counter() - start:.2f} secs",
            )
        else:
            opt_state_dict = None

        if self._is_rank_zero:
            start = time.perf_counter()
            checkpoint_dict.update({training.MODEL_KEY: cpu_state_dict})

            checkpoint_dict.update(
                {
                    training.SEED_KEY: self.seed,
                    training.EPOCHS_KEY: self._epochs_run,
                    training.TOTAL_EPOCHS_KEY: self.total_epochs,
                    training.RNG_KEY: self._rng.get_state(),
                    training.STEPS_KEY: self.global_step,
                    training.MAX_STEPS_KEY: self._steps_per_epoch,
                    CHECKPOINT_STEPS_RUN_KEY: self._steps_run,
                }
            )

            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        training.OPT_KEY: opt_state_dict,
                        training.DATALOADER_KEY: self._dataloader.state_dict(),
                    }
                )

            dir_prefix = "step" if self._save_every_n_steps is not None else "epoch"
            step_value: Optional[int] = self._steps_run if dir_prefix == "step" else None
            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
                dir_prefix=dir_prefix,
                step=step_value,
            )
            log.info(f"Saving checkpoint took {time.perf_counter() - start:.2f} secs")

        torch.distributed.barrier()

    def generate_trajectory(
            self, input_ids: torch.Tensor, answers: list[str]
    ) -> GRPOTrajectory:
        batch_size, context_length = input_ids.shape
        grpo_size = self.grpo_samples

        batch_input_ids = input_ids[:, None, :].expand(-1, grpo_size, -1)
        batch_input_ids = batch_input_ids.reshape(batch_size * grpo_size, -1)

        max_total_len = context_length + self._max_generated_tokens

        with local_kv_cache(
                model=self._model,
                batch_size=batch_size * grpo_size,
                device=self._device,
                dtype=self._dtype,
                decoder_max_seq_len=max_total_len,
        ):
            query_responses, _ = generate(
                model=self._model,
                prompt=batch_input_ids,
                max_generated_tokens=self._max_generated_tokens,
                temperature=self._temperature,
                top_k=self._top_k,
                pad_id=self._tokenizer.pad_id,
                rng=self._rng,
                stop_tokens=self._tokenizer.stop_tokens,
                return_logits=False,
            )

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

        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        del query_response_padding_masks

        logits = self._model(query_responses, input_pos=position_ids, mask=masks)
        logits = logits[:, context_length - 1:]
        logprobs = rlhf.batched_logits_to_logprobs(logits, responses, self._temperature)
        del logits
        torch.cuda.empty_cache()

        ref_logits = self._ref_model(
            query_responses, input_pos=position_ids, mask=masks
        )
        ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
        ref_logprobs = rlhf.batched_logits_to_logprobs(
            ref_logits, responses, self._temperature
        )
        del ref_logits
        torch.cuda.empty_cache()

        (
            response_padding_masks,
            responses,
        ) = rlhf.truncate_sequence_at_first_stop_token(
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )

        response_ids = responses.reshape(batch_size * grpo_size, -1)
        responses_str: list[str] | None = None
        if self.reward_functions or self._reward_preview.enabled:
            responses_str = decode_responses(self._tokenizer, response_ids)

        self._reward_preview.maybe_preview(input_ids, responses_str, answers, grpo_size)

        if self.reward_functions:
            answers_expanded = [
                answer for answer in answers for _ in range(grpo_size)
            ]
            if len(answers_expanded) != response_ids.shape[0]:
                raise ValueError(
                    "Number of answers does not match the number of generated responses."
                )

            if responses_str is None:
                responses_str = decode_responses(self._tokenizer, response_ids)

            reward_outputs = [
                reward_fn(response_ids, responses_str, answers_expanded)
                for reward_fn in self.reward_functions
            ]

            reward_components_bg = torch.stack(
                [reward_output.total_reward for reward_output in reward_outputs],
                dim=-1,
            ).to(self._device)
            successes_bg = torch.stack(
                [reward_output.successes for reward_output in reward_outputs], dim=-1
            ).to(self._device)
            reward_components_bg = reward_components_bg.reshape(
                batch_size, grpo_size, -1
            )
            successes_bg = successes_bg.reshape(batch_size, grpo_size, -1)
        else:
            responses = response_ids.reshape(batch_size, grpo_size, -1)
            rewards_bg, successes_bg, _ = batched_rewards(
                self._tokenizer, responses, answers, self._device
            )
            reward_components_bg = rewards_bg.to(self._device)
            successes_bg = successes_bg.to(self._device)

        num_reward_funcs = reward_components_bg.shape[-1]
        reward_weights = torch.ones(num_reward_funcs, device=self._device) / max(
            num_reward_funcs, 1
        )
        aggregated_rewards_bg = (reward_components_bg * reward_weights).sum(dim=-1)
        successes_bg = successes_bg.mean(dim=-1)

        advantages = (
                             aggregated_rewards_bg - aggregated_rewards_bg.mean(1, keepdim=True)
                     ) / (aggregated_rewards_bg.std(1, keepdim=True) + 1e-4)
        aggregated_rewards = aggregated_rewards_bg.reshape(batch_size * grpo_size)
        successes = successes_bg.reshape(batch_size * grpo_size)
        reward_components = reward_components_bg.reshape(batch_size * grpo_size, -1)
        advantages = advantages.reshape(batch_size * grpo_size)

        del responses
        torch.cuda.empty_cache()

        logprobs[response_padding_masks] = 1.0
        ref_logprobs[response_padding_masks] = 1.0

        return GRPOTrajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            rewards=aggregated_rewards,
            reward_components=reward_components,
            successes=successes,
            advantages=advantages,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            seq_lens=training.get_unmasked_sequence_lengths(response_padding_masks),
        )

    def generate_trajectory_batched(
            self, input_ids: torch.Tensor, answers: list[str]
    ) -> GRPOTrajectory:
        trajectories: list[GRPOTrajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[
                    batch_start: batch_start + self._forward_batch_size
                ]
                batch_answers = answers[
                    batch_start: batch_start + self._forward_batch_size
                ]
                torch.cuda.empty_cache()
                trajectories.append(
                    self.generate_trajectory(batch_input_ids, batch_answers)
                )
                torch.cuda.empty_cache()
        return concat_grpo_trajectories(trajectories)

    def grpo_step(
            self,
            trajectory: GRPOTrajectory,
            context_length: int,
    ) -> GRPOStats:
        torch.cuda.empty_cache()

        pi_logits = self._model(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.masks,
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
            approx_policy_kls.detach(),
        )

    def train(self) -> None:
        training.cleanup_before_training()

        self._optimizer.zero_grad()

        grad_norm = None
        training_completed = False
        interrupted = False
        self._profiler.start()

        total_target = self.total_epochs * self._steps_per_epoch
        if self._total_steps:
            total_target = min(total_target, self._total_steps)

        pbar = tqdm(total=total_target, disable=not self._is_rank_zero)
        if self._steps_run:
            pbar.update(self._steps_run)

        last_epoch = max(self._epochs_run - 1, 0)
        try:
            for curr_epoch in range(self._epochs_run, self.total_epochs):
                last_epoch = curr_epoch
                self._dataloader.sampler.set_epoch(curr_epoch)
                for idx, batch in enumerate(self._dataloader):
                    if (
                            self._is_rank_zero
                            and curr_epoch == 0
                            and self.profiler_profile_memory
                            and idx == self.profiler_wait_steps + self.profiler_warmup_steps
                            and self._device.type == "cuda"
                    ):
                        torch.cuda.memory._record_memory_history()

                    tokens = batch["tokens"]
                    answers = batch["answers"]
                    tokens = tokens.to(self._device)

                    _, context_length = tokens.shape

                    self._reward_preview.maybe_reset(self.global_step)
                    trajectory = self.generate_trajectory_batched(tokens, answers)
                    torch.distributed.barrier()

                    grpo_stats: list[GRPOStats] = []
                    for _ in range(self._ppo_epochs):
                        step_stats = self.grpo_step(trajectory, context_length)

                        grpo_stats.append(step_stats)

                        if self._clip_grad_norm is not None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self._model.parameters(),
                                max_norm=float(self._clip_grad_norm),
                            )
                        torch.distributed.barrier()

                        # <<<< LIFTED GRPO UPDATE
                        if self.lifted_inner_dim_ratio > 0:
                            # Perform update on Shadow Params A/B and project to W
                            self._step_lifted_parameters()

                            # Optionally step other non-lifted params (like LayerNorm/Embeddings)
                            # Note: _step_lifted_params clears grads for lifted layers
                            # so step() here only affects the rest.
                            self._optimizer.step()
                        else:
                            # Standard update
                            self._optimizer.step()
                        # >>>> END LIFTED GRPO UPDATE

                        self._optimizer.zero_grad(set_to_none=True)
                        torch.distributed.barrier()

                        self.global_step += 1

                        if self._lr_scheduler is not None:
                            self._lr_scheduler.step()

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
                    if self._save_every_n_steps is not None:
                        if self._steps_run % self._save_every_n_steps == 0:
                            self.save_checkpoint(curr_epoch)

                    if self._steps_run % self._log_every_n_steps == 0:
                        extra_metrics = {}
                        extra_metrics["lr"] = get_lr(self._optimizer)
                        if grad_norm is not None:
                            extra_metrics["grad_norm"] = grad_norm

                        self.log_metrics(
                            trajectory,
                            stack_grpo_stats(grpo_stats),
                            **extra_metrics,
                        )

                    self.cleanup_after_step(trajectory, grpo_stats)
                    self._profiler.step()

                    pbar.update(1)

                    if self._total_steps and self._steps_run >= self._total_steps:
                        training_completed = True
                        break

                self._epochs_run += 1
                if (
                        self._save_every_n_steps is None
                        and self._epochs_run % self._save_every_n_epochs == 0
                ):
                    self.save_checkpoint(curr_epoch)
                if training_completed:
                    break
        except KeyboardInterrupt:
            interrupted = True
            utils.log_rank_zero(
                log,
                "Training interrupted by user. Saving final checkpoint before exit.",
            )
        finally:
            self._profiler.stop()
            pbar.close()
            final_epoch = max(self._epochs_run - 1, last_epoch, 0)
            self.save_checkpoint(final_epoch, is_final=True)
            if interrupted:
                return

    # ... (Log metrics and cleanup remain unchanged) ...
    def log_metrics(
            self, trajectory: GRPOTrajectory, grpo_stats: GRPOStats, **extras
    ) -> None:
        rewards = trajectory.rewards.mean()
        torch.distributed.reduce(rewards, dst=0, op=torch.distributed.ReduceOp.AVG)

        successes = trajectory.successes.mean()
        torch.distributed.reduce(successes, dst=0, op=torch.distributed.ReduceOp.AVG)

        mean_reward_components = trajectory.reward_components.mean(dim=0)
        torch.distributed.reduce(mean_reward_components, dst=0, op=torch.distributed.ReduceOp.AVG)

        log_dict = {
            "rewards": rewards,
            "successes": successes,
            "num_stop_tokens": trajectory.response_padding_masks.any(-1).sum(),
            "loss": grpo_stats.loss.mean(),
            "policy_loss": grpo_stats.policy_loss.mean(),
            "kl_loss": grpo_stats.kl_loss.mean(),
            "clipfrac": grpo_stats.clipfrac.mean(),
            "ratios": grpo_stats.ratios.mean(),
            "approx_policy_kl": grpo_stats.approx_policy_kls.mean(),
            "response_lengths": trajectory.seq_lens.float().mean(),
            **extras,
        }

        if self._device.type == "cuda" and self._log_peak_memory_stats:
            log_dict.update(training.get_memory_stats(device=self._device))
        if self._is_rank_zero:
            if self.reward_names and len(self.reward_names) == len(mean_reward_components):
                for name, reward_comp in zip(self.reward_names, mean_reward_components):
                    log_dict[f"reward/{name}"] = reward_comp.item()
            else:
                for i, reward_comp in enumerate(mean_reward_components):
                    log_dict[f"reward/component_{i}"] = reward_comp.item()

            self._metric_logger.log_dict(log_dict, step=self.global_step)

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()

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
    recipe = LiftedGRPOFullFinetuneRecipeDistributed(cfg=cfg)
    config.log_config(recipe_name="LiftedGRPOFullFinetuneRecipeDistributed", cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

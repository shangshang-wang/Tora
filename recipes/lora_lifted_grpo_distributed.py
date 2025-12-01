import sys
import time
from functools import partial
from typing import Any, Optional, Union
from warnings import warn
import copy

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
from torchtune.modules.peft import (
    AdapterModule,
    disable_adapter,
    get_adapter_params,
    get_lora_module_names,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
    print_lora_trainable_parameters
)
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import disable_dropout, DummyProfiler, PROFILER_KEY
from torchtune.training.lr_schedulers import get_lr
from tqdm import tqdm
from torchtune.recipe_support.reward_preview import RewardPreviewer, decode_responses

log = utils.get_logger("DEBUG")

# import torch._dynamo.config as dynamo_config
# dynamo_config.recompile_limit = 1000

CHECKPOINT_STEPS_RUN_KEY = "dataloader_steps_run"


class LoRALiftedGRPORecipeDistributed(FTRecipeInterface):
    """
    Distributed LoRA GRPO recipe with support for Lifted Optimization (Iterative Rank Alignment).

    Mechanism:
    - Trains a High-Rank "Shadow Adapter" (e.g., r=64) to allow better exploration.
    - Periodically synchronizes (SVD Projection) down to a Low-Rank "Target Adapter" (e.g., r=16)
      to consolidate knowledge, then continues training.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        inference_dtype_cfg = cfg.get("inference_dtype", None)
        if inference_dtype_cfg is None:
            self._inference_dtype = self._dtype
        else:
            self._inference_dtype = training.get_dtype(
                inference_dtype_cfg, device=self._device
            )

        if self._dtype == torch.float16 and self._device.type not in ("cuda", "xpu"):
            raise ValueError(
                "full fp16 training requires CUDA or XPU devices. Please switch to bf16/fp32 or set device=cuda."
            )

        if self._inference_dtype == torch.float16 and self._device.type not in ("cuda", "xpu"):
            raise ValueError(
                "full fp16 inference requires CUDA or XPU devices. Please switch to bf16/fp32 or set device=cuda."
            )

        if self._dtype == torch.float16:
            warn(
                message=(
                    "Full fp16 training is experimental and may lead to numerical instability. "
                    "Consider bf16 if your hardware supports it."
                )
            )

        if self._inference_dtype == torch.float16 and self._inference_dtype != self._dtype:
            warn(
                message=(
                    "Full fp16 inference is experimental and may behave differently from the training dtype. "
                    "Monitor for numerical instability."
                )
            )

        # Set up the backend for distributed training (NCCL, GLOO, etc.)
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            cfg.device, offload_ops_to_cpu=self.fsdp_cpu_offload
        )
        init_process_group(self.distributed_backend)

        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0
        self._reward_preview = RewardPreviewer(cfg.get("reward_preview"), self._is_rank_zero)

        # Config for Projected GRPO
        model_cfg = cfg.model
        # cfg.model.lora_rank is the "Shadow Rank" (Training Rank)
        self._shadow_rank = self._parse_int_field(
            model_cfg.get("lora_rank", None), "model.lora_rank"
        )
        self._shadow_alpha = self._parse_float_field(
            model_cfg.get("lora_alpha", None), "model.lora_alpha"
        )
        # cfg.model.target_lora_rank is the "Target Rank" (Inference/Deployment Rank)
        # If not set in config, defaults to shadow_rank (standard behavior)
        target_rank_cfg = model_cfg.get(
            "target_lora_rank", cfg.get("target_lora_rank", None)
        )
        self._target_rank = self._parse_int_field(
            target_rank_cfg, "model.target_lora_rank", default=self._shadow_rank
        )
        target_alpha_cfg = model_cfg.get(
            "target_lora_alpha", cfg.get("target_lora_alpha", None)
        )
        self._target_lora_alpha = self._parse_float_field(
            target_alpha_cfg,
            "model.target_lora_alpha",
            default=self._shadow_alpha,
        )

        self._shadow_sync_every_n_steps = model_cfg.get(
            "shadow_sync_every_n_steps", cfg.get("sync_every_n_steps", None)
        )
        if self._shadow_sync_every_n_steps is not None:
            if (
                    not isinstance(self._shadow_sync_every_n_steps, int)
                    or self._shadow_sync_every_n_steps <= 0
            ):
                raise ValueError("shadow_sync_every_n_steps must be a positive integer when provided.")

        self._use_projection = self._target_rank < self._shadow_rank
        self._shadow_weights_cache = {}  # To store High Rank weights on CPU during inference

        if self._use_projection and self.rank == 0:
            log.info(
                f"Enabled LIFTED GRPO (Iterative Alignment): Shadow Rank={self._shadow_rank}, Target Rank={self._target_rank}, Sync Step={self._shadow_sync_every_n_steps}"
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = utils.get_logger(cfg.log_level)

        utils.log_rank_zero(
            self._logger,
            f"Precision configuration -> training dtype: {self._dtype}, inference dtype: {self._inference_dtype}",
        )

        if self._log_peak_memory_stats and self._device.type != "cuda":
            log.info(
                "log_peak_memory_stats was set to True, however, training does not use cuda. "
                "Setting log_peak_memory_stats=False."
            )
            self._log_peak_memory_stats = False

        # activation checkpointing/offloading
        self._enable_activation_checkpointing = cfg.get(
            "enable_activation_checkpointing", False
        )
        self._enable_activation_offloading = cfg.get(
            "enable_activation_offloading", False
        )
        if self._enable_activation_offloading:
            if self._device.type != "cuda" and self._device.type != "xpu":
                raise RuntimeError(
                    "enable_activation_offloading should only be True when training on CUDA or XPU"
                )
            if not self._enable_activation_checkpointing:
                raise RuntimeError(
                    "enable_activation_offloading should only be True when enable_activation_checkpointing is True"
                )
        elif self._enable_activation_checkpointing:
            utils.log_rank_zero(
                self._logger,
                "Hint: enable_activation_checkpointing is True, but enable_activation_offloading isn't. "
                "Enabling activation offloading should reduce memory further.",
            )

        self._save_every_n_steps = cfg.get("save_every_n_steps", None)
        if self._save_every_n_steps is not None:
            if self._save_every_n_steps <= 0:
                raise ValueError("save_every_n_steps must be a positive integer.")

        # These attributes constitute the recipe state and are updated by ``load_checkpoint``
        # when ``resume_from_checkpoint`` is ``True``
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self._epochs_run = 0
        self.total_epochs = cfg.epochs
        self.global_step = 0
        self._steps_run = 0
        self._total_steps = cfg.get("early_stop_steps", None)
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._compile = cfg.get("compile", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._rng = torch.Generator(self._device).manual_seed(self.seed)
        self._restored_steps_run: Optional[int] = None
        self._restored_global_step: Optional[int] = None
        self._restored_steps_per_epoch: Optional[int] = None

    def _update_recipe_state(self, ckpt_dict: dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
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

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> dict[str, Any]:
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()
        return checkpoint_dict

    def setup(self, cfg: DictConfig) -> None:
        if self.fsdp_cpu_offload:
            training.set_torch_num_threads()

        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)
            self._metric_logger.log_config(cfg)

        utils.log_rank_zero(self._logger, "metric logger is initialized.")

        # Load checkpoints
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

        # Setup model with LoRA
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=self._enable_activation_checkpointing,
            enable_activation_offloading=self._enable_activation_offloading,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", False),
            base_model_state_dict=checkpoint_dict[training.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[training.ADAPTER_KEY]
                if training.ADAPTER_KEY in checkpoint_dict
                else None
            ),
        )

        if self._is_rank_zero:
            print_lora_trainable_parameters(self._model)

        torch.distributed.barrier()

        self._tokenizer = config.instantiate(cfg.tokenizer)
        self._reward_preview.set_tokenizer(self._tokenizer)

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint and training.OPT_KEY in checkpoint_dict
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
                if self._resume_from_checkpoint and training.DATALOADER_KEY in checkpoint_dict
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

    def _setup_model(
            self,
            cfg_model: DictConfig,
            enable_activation_checkpointing: bool,
            enable_activation_offloading: bool,
            fsdp_cpu_offload: bool,
            reshard_after_forward: bool,
            base_model_state_dict: dict[str, Any],
            custom_sharded_layers: Optional[list[str]] = None,
            lora_weights_state_dict: Optional[dict[str, Any]] = None,
    ) -> nn.Module:
        self._lora_rank = self._shadow_rank
        self._lora_alpha = self._shadow_alpha
        self._lora_attn_modules = list(cfg_model.lora_attn_modules)
        self._apply_lora_to_mlp = cfg_model.apply_lora_to_mlp
        self._apply_lora_to_output = getattr(cfg_model, "apply_lora_to_output", False)

        self._adapter_config = {
            "r": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "target_modules": get_lora_module_names(
                self._lora_attn_modules,
                self._apply_lora_to_mlp,
                self._apply_lora_to_output,
            ),
            "peft_type": "LORA",
        }

        init_start = time.perf_counter()
        utils.log_rank_zero(self._logger, "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...")

        # Remove non-constructor keys before instantiating the backbone
        cfg_model_for_instantiation = copy.deepcopy(cfg_model)
        cfg_model_for_instantiation["lora_rank"] = self._lora_rank
        cfg_model_for_instantiation["lora_alpha"] = self._lora_alpha
        if "target_lora_rank" in cfg_model_for_instantiation:
            del cfg_model_for_instantiation["target_lora_rank"]
        if "target_lora_alpha" in cfg_model_for_instantiation:
            del cfg_model_for_instantiation["target_lora_alpha"]
        if "shadow_sync_every_n_steps" in cfg_model_for_instantiation:
            del cfg_model_for_instantiation["shadow_sync_every_n_steps"]

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model_for_instantiation)

        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

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

        if lora_weights_state_dict:
            lora_missing, lora_unexpected = training.load_from_full_model_state_dict(
                model,
                lora_weights_state_dict,
                self._device,
                cpu_offload=fsdp_cpu_offload,
            )
        else:
            lora_missing, lora_unexpected = None, None

        with training.set_default_dtype(self._dtype), self._device:
            lora_device = "cpu" if fsdp_cpu_offload else self._device
            for m in model.modules():
                if (isinstance(m, AdapterModule)) and not lora_weights_state_dict:
                    m.to_empty(device=lora_device)
                    m.initialize_parameters()
                if hasattr(m, "rope_init"):
                    m.rope_init()

        base_missing, base_unexpected = training.load_from_full_model_state_dict(
            model,
            base_model_state_dict,
            self._device,
            cpu_offload=fsdp_cpu_offload,
        )

        is_dora = False
        for m in model.modules():
            if hasattr(m, "initialize_dora_magnitude"):
                is_dora = True
                m.initialize_dora_magnitude()

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=self._lora_attn_modules,
            apply_lora_to_mlp=self._apply_lora_to_mlp,
            apply_lora_to_output=self._apply_lora_to_output,
            state_dict_keys=model.state_dict().keys(),
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )

        self.activations_handling_ctx = training.get_act_offloading_ctx_manager(
            model, enable_activation_offloading
        )

        training.validate_no_params_on_meta_device(model)

        utils.log_rank_zero(
            self._logger,
            f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs",
        )

        if self._is_rank_zero:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        disable_dropout(model)

        return model

    def _setup_optimizer(
            self, cfg_optimizer: DictConfig, opt_state_dict: Optional[dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                self._model,
                optimizer,
                opt_state_dict,
                self._device,
            )
        utils.log_rank_zero(self._logger, "Optimizer and loss are initialized.")
        return optimizer

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

        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )
        utils.log_rank_zero(self._logger, "Learning rate scheduler is initialized.")
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

        return profiler

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

        utils.log_rank_zero(self._logger, "Dataset and Sampler are initialized.")
        return dataloader

    # Helper functions for Projected GRPO
    @torch.no_grad()
    def _project_weights_for_inference(self):
        """
        Compresses the Shadow (High Rank) weights to Target (Low Rank) weights,
        and pads them to fit the model container.

        NOTE: In 'Lifted' mode, this is NOT called every step. It is available
        if you want to save the compressed weights or evaluate the low-rank proxy.
        """
        if not self._use_projection:
            return

        # Summon full params is essential for FSDP to modify weights in-place
        # writeback=True ensures modifications persist
        with FSDP.summon_full_params(self._model, writeback=True, rank0_only=False):

            # Identify LoRA pairs
            state_dict = self._model.state_dict()
            # We find keys ending in lora_a.weight and lora_b.weight
            keys = list(state_dict.keys())
            lora_a_keys = [k for k in keys if "lora_a.weight" in k]

            for key_a in lora_a_keys:
                key_b = key_a.replace("lora_a.weight", "lora_b.weight")
                if key_b not in state_dict:
                    continue

                W_A_high = state_dict[key_a]
                W_B_high = state_dict[key_b]

                # Cache Shadow Weights (Move to CPU to save GPU mem)
                self._shadow_weights_cache[key_a] = W_A_high.clone().cpu()
                self._shadow_weights_cache[key_b] = W_B_high.clone().cpu()

                # --- QR-SVD Projection Logic ---
                # Convert to Float32 for stability during SVD
                A = W_A_high.float().t()  # (d_in, R)
                B = W_B_high.float().t()  # (R, d_out)

                R_dim = A.shape[1]

                # 1. QR Decomposition (Optimized for R > d cases)
                # Q_A: (d_in, R), R_A: (R, R)
                Q_A, R_A = torch.linalg.qr(A)
                # Apply QR to B.T (d_out, R) -> Q_B (d_out, R), R_B (R, R)
                Q_B, R_B = torch.linalg.qr(B.t())

                # 2. Form Core Matrix M = R_A @ R_B.T (R x R)
                M = R_A @ R_B.t()

                # 3. SVD on Core
                try:
                    U, S, Vh = torch.linalg.svd(M)
                except RuntimeError:
                    # Fallback if SVD fails (rare)
                    log.warning(f"SVD failed for {key_a}, using shadow weights.")
                    continue

                # 4. Truncate to Target Rank (r)
                r = self._target_rank
                U_r = U[:, :r]
                S_r = S[:r]
                Vh_r = Vh[:r, :]

                # 5. Project back
                sqrt_S = torch.diag(torch.sqrt(S_r))

                # A_low = Q_A @ U_r @ sqrt_S  --> Shape (d_in, r)
                A_low = Q_A @ U_r @ sqrt_S

                # B_low = sqrt_S @ Vh_r @ Q_B.T --> Shape (r, d_out)
                B_low = sqrt_S @ Vh_r @ Q_B.t()

                # 6. Pad back to Shadow Rank R so it fits in the model container
                # A_pad: (d_in, R)
                A_pad = torch.zeros_like(A)
                A_pad[:, :r] = A_low

                # B_pad: (R, d_out)
                B_pad = torch.zeros_like(B)
                B_pad[:r, :] = B_low

                # 7. Update Model Weights (Transpose back to nn.Linear format)
                state_dict[key_a].copy_(A_pad.t().to(W_A_high.dtype))
                state_dict[key_b].copy_(B_pad.t().to(W_B_high.dtype))

        if self._target_lora_alpha != getattr(self, "_lora_alpha", self._shadow_alpha):
            self._set_adapter_alpha(self._target_lora_alpha)

    @torch.no_grad()
    def _restore_weights_for_training(self):
        """
        Restores the Shadow (High Rank) weights from cache.
        """
        if not self._use_projection:
            return

        if not self._shadow_weights_cache:
            return

        with FSDP.summon_full_params(self._model, writeback=True, rank0_only=False):
            state_dict = self._model.state_dict()
            for key, val_cpu in self._shadow_weights_cache.items():
                if key in state_dict:
                    state_dict[key].copy_(val_cpu.to(state_dict[key].device))

            # Clear cache to free memory
            self._shadow_weights_cache.clear()

        if self._target_lora_alpha != getattr(self, "_lora_alpha", self._shadow_alpha):
            self._set_adapter_alpha(getattr(self, "_lora_alpha", self._shadow_alpha))

    def _set_adapter_alpha(self, alpha: float) -> None:
        """
        Update the alpha scaling factor on all LoRA adapters.
        """
        if not hasattr(self, "_model") or self._model is None:
            return
        for module in self._model.modules():
            if (
                    hasattr(module, "lora_a")
                    and hasattr(module, "lora_b")
                    and hasattr(module, "alpha")
            ):
                module.alpha = alpha

    @torch.no_grad()
    def _synchronize_shadow_weights(self):
        """
        Projects the current Shadow (High Rank) adapters back onto the Target (Low Rank)
        manifold and RE-INITIALIZES the Shadow weights from this projection.

        CRITICAL FIX: Adds small Gaussian noise to the 'null space' (dimensions r to R)
        to break symmetry and prevent the optimizer from ignoring the empty dimensions.
        """
        if not self._use_projection:
            return

        # We use a small epsilon for noise.
        # Too large = destroys the benefit of projection (loss spike).
        # Too small = optimizer cannot find the gradient (dead neurons).
        NOISE_STD = 1e-3 # 1e-3

        with FSDP.summon_full_params(self._model, writeback=True, rank0_only=False):
            state_dict = self._model.state_dict()
            keys = list(state_dict.keys())
            lora_a_keys = [k for k in keys if "lora_a.weight" in k]

            for key_a in lora_a_keys:
                key_b = key_a.replace("lora_a.weight", "lora_b.weight")
                if key_b not in state_dict:
                    continue

                W_A_high = state_dict[key_a]
                W_B_high = state_dict[key_b]

                # Convert to float32 for SVD stability
                A = W_A_high.float().t()  # Shape (d_in, R)
                B = W_B_high.float().t()  # Shape (R, d_out)

                try:
                    # QR-SVD logic as before
                    Q_A, R_A = torch.linalg.qr(A)
                    Q_B, R_B = torch.linalg.qr(B.t())
                    M = R_A @ R_B.t()
                    U, S, Vh = torch.linalg.svd(M)
                except RuntimeError:
                    log.warning(f"SVD failed for {key_a} during synchronization; keeping existing weights.")
                    continue

                r = self._target_rank

                # 1. Extract the Core Low-Rank components
                U_r = U[:, :r]
                S_r = S[:r]
                Vh_r = Vh[:r, :]

                # 2. Reconstruct the Low-Rank Approximation
                sqrt_S = torch.diag(torch.sqrt(S_r))
                A_low = Q_A @ U_r @ sqrt_S  # (d_in, r)
                B_low = sqrt_S @ Vh_r @ Q_B.t()  # (r, d_out)

                # 3. Embed into High-Rank Container & APPLY NOISE BREAKING

                # A_clean: (d_in, R)
                A_clean = torch.zeros_like(A)
                A_clean[:, :r] = A_low
                # Inject noise into columns r through R
                A_clean[:, r:] = torch.randn_like(A_clean[:, r:]) * NOISE_STD

                # B_clean: (R, d_out)
                B_clean = torch.zeros_like(B)
                B_clean[:r, :] = B_low
                # Inject noise into rows r through R
                B_clean[r:, :] = torch.randn_like(B_clean[r:, :]) * NOISE_STD

                # 4. Copy back to model
                state_dict[key_a].copy_(A_clean.t().to(W_A_high.dtype))
                state_dict[key_b].copy_(B_clean.t().to(W_B_high.dtype))

                # Clear cache if it exists
                if key_a in self._shadow_weights_cache:
                    del self._shadow_weights_cache[key_a]
                if key_b in self._shadow_weights_cache:
                    del self._shadow_weights_cache[key_b]

    @staticmethod
    def _parse_int_field(
            value: Any, field_name: str, default: Optional[int] = None
    ) -> int:
        if value is None:
            if default is None:
                raise ValueError(f"{field_name} must be provided and castable to int.")
            return default
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be castable to int, but received value {value!r}."
            ) from exc

    @staticmethod
    def _parse_float_field(
            value: Any, field_name: str, default: Optional[float] = None
    ) -> float:
        if value is None:
            if default is None:
                raise ValueError(f"{field_name} must be provided and castable to float.")
            return default
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be castable to float, but received value {value!r}."
            ) from exc

    def save_checkpoint(self, epoch: int, *, is_final: bool = False) -> None:
        """
        Checkpoint the state of the recipe.
        """
        # final dict passed onto the checkpointer
        checkpoint_dict = {}
        intermediate_checkpoint = not is_final
        epoch = max(epoch, 0)

        utils.log_rank_zero(
            log,
            "Saving checkpoint. This may take some time. Retrieving full model state dict...",
        )
        start = time.perf_counter()

        # NOTE: This saves the current SHADOW weights (rank R).
        # If you want to deploy as rank r, you must project offline or call project() here.
        # We default to saving the training state (Shadow) so training can be resumed.

        cpu_state_dict = training.gather_cpu_state_dict(
            self._model,
            self._is_rank_zero,
            device=self._device,
        )

        utils.log_rank_zero(
            log,
            f"Getting full model state dict took {time.perf_counter() - start:.2f} secs",
        )

        if self._is_rank_zero:
            cleaned_state_dict = {}
            adapter_state_dict = {}

            for key, value in cpu_state_dict.items():
                cleaned_key = key.replace("._checkpoint_wrapped_module", "")
                if "lora_" in cleaned_key:
                    adapter_state_dict[cleaned_key] = value
                else:
                    cleaned_state_dict[cleaned_key] = value

            cpu_state_dict = cleaned_state_dict

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

        if self._save_adapter_weights_only:
            adapter_state_dict = {
                k: v for k, v in cpu_state_dict.items()
                if any(module_name in k for module_name in self._adapter_config["target_modules"])
            }
        else:
            adapter_state_dict = None

        if self._is_rank_zero:
            start = time.perf_counter()

            checkpoint_dict.update({training.MODEL_KEY: cpu_state_dict})

            if adapter_state_dict is not None:
                checkpoint_dict.update({training.ADAPTER_KEY: adapter_state_dict})

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
        """
        Generates a trajectory given the current policy model (Shadow Adapter),
        the reference policy model, the reward function, and batch of inputs.
        """
        batch_size, context_length = input_ids.shape
        grpo_size = self.grpo_samples

        batch_input_ids = input_ids[:, None, :].expand(-1, grpo_size, -1)
        batch_input_ids = batch_input_ids.reshape(batch_size * grpo_size, -1)

        max_total_len = context_length + self._max_generated_tokens

        with training.set_default_dtype(self._inference_dtype):
            with local_kv_cache(
                    model=self._model,
                    batch_size=batch_size * grpo_size,
                    device=self._device,
                    dtype=self._inference_dtype,
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

        with self.activations_handling_ctx:
            logits = self._model(query_responses, input_pos=position_ids, mask=masks)
        logits = logits[:, context_length - 1:]
        logprobs = rlhf.batched_logits_to_logprobs(logits, responses, self._temperature)
        del logits
        torch.cuda.empty_cache()

        with torch.no_grad(), disable_adapter(self._model):
            ref_logits = self._model(
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

        advantages = (aggregated_rewards_bg - aggregated_rewards_bg.mean(1, keepdim=True)) / (
                    aggregated_rewards_bg.std(1, keepdim=True) + 1e-4)
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

        with self.activations_handling_ctx:
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
                    # --- REVISION START ---
                    # DISABLED: Per-step projection causes "SVD Amnesia".
                    # We now sample directly from Shadow (Rank R) to explore high-dim space.
                    # if self._use_projection:
                    #     self._project_weights_for_inference()
                    # ----------------------

                    trajectory = self.generate_trajectory_batched(tokens, answers)
                    torch.distributed.barrier()

                    # --- REVISION START ---
                    # DISABLED: No restoration needed since we didn't project.
                    # if self._use_projection:
                    #     self._restore_weights_for_training()
                    # ----------------------

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
                        self._optimizer.step()
                        torch.distributed.barrier()

                        self.global_step += 1

                        # --- REVISION START ---
                        # ENABLED: This is the core "Iterative Alignment" logic.
                        # Periodically (e.g. every 100 steps), we compress knowledge to Rank r
                        # and then re-expand to Rank R.
                        if (
                                self._use_projection
                                and self._shadow_sync_every_n_steps
                                and self.global_step % self._shadow_sync_every_n_steps == 0
                        ):
                            self._synchronize_shadow_weights()
                        # ----------------------

                        self._optimizer.zero_grad(set_to_none=True)
                        torch.distributed.barrier()

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
                    pbar.set_description(
                        f"{curr_epoch + 1}|{self.global_step}|Loss: {grpo_stats[-1].loss.item():.4f}"
                    )

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
                self._logger,
                "Training interrupted by user. Saving final checkpoint before exit.",
            )
        finally:
            self._profiler.stop()
            pbar.close()
            final_epoch = max(self._epochs_run - 1, last_epoch, 0)
            self.save_checkpoint(final_epoch, is_final=True)
            if interrupted:
                return

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
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    if cfg.get("fsdp_cpu_offload", False):
        training.set_torch_num_threads()

    config.log_config(recipe_name="LoRALiftedGRPORecipeDistributed", cfg=cfg)

    recipe = LoRALiftedGRPORecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())

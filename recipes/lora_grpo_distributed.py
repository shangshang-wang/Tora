import sys
import time
from functools import partial
from typing import Any, Optional, Union
from warnings import warn

import torch
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
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

log = utils.get_logger("DEBUG")

import torch._dynamo.config as dynamo_config
dynamo_config.recompile_limit = 1000


class LoRAGRPORecipeDistributed(FTRecipeInterface):
    """
    Distributed LoRA GRPO recipe for dense transformer-based LLMs such as Llama2. This recipe supports
    distributed training and can be run on a single node (1 to 8 GPUs).

    Features:
        - FSDP. Supported using PyTorch's FSDP APIs. CPU offload of parameters, gradients, and optimizer states
            is supported via ``fsdp_cpu_offload``. Resharding of parameters after the forward pass is
            done by default (corresponding to FULL_SHARD sharding strategy), but can be disabled by setting the config
            ``fsdp_reshard_after_forward`` to False (this corresponds to SHARD_GRAD_OP sharding strategy).
            DDP is currently not supported. Training on CPU is not supported.

        - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Activation Offloading. This can be controlled using the ``enable_activation_offloading``
            flag. Activation offloading is a technique similar to activations checkpointing that helps
            reduce the memory footprint to prevent OOMs on CUDA and enable bigger batches. Where activations
            checkpointing drops the activation in the forward to recompute it later in the backward,
            activations offloading will drop the activation in the forward to the CPU and bring it
            back during the backward pass. As always, there is a tradeoff--these savings in memory can
            come at the cost of training performance and CPU resources.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * number of GPUs * gradient accumulation steps.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Currently we checkpoint both the adapter weights (trainable params only) and the
            complete merged weights (adapter weights added back to the base model).

            Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training. Resuming
            training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        ValueError: If world_size is 1
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.
        RuntimeError: If ``enable_activation_offloading`` is True and device is not CUDA or XPU.
        RuntimeError: If ``enable_activation_offloading`` is True and ``enable_activation_checkpointing`` is False.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        # Set up the backend for distributed training (NCCL, GLOO, etc.)
        self.fsdp_cpu_offload = cfg.get("fsdp_cpu_offload", False)
        self.distributed_backend = training.get_distributed_backend(
            cfg.device, offload_ops_to_cpu=self.fsdp_cpu_offload
        )
        init_process_group(self.distributed_backend)

        self.world_size, self.rank = utils.get_world_size_and_rank()
        self._is_rank_zero = self.rank == 0

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)
        self._logger = utils.get_logger(cfg.log_level)

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

        # These attributes constitute the recipe state and are updated by ``load_checkpoint``
        # when ``resume_from_checkpoint`` is ``True``
        self.seed = training.set_seed(
            seed=cfg.seed, debug_mode=cfg.get("cudnn_deterministic_mode", None)
        )
        self._epochs_run = 0
        self.total_epochs = cfg.epochs
        self.global_step = 0
        self._steps_run = 0
        self._total_steps = cfg.num_steps
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._save_adapter_weights_only = cfg.get("save_adapter_weights_only", False)
        self._compile = cfg.get("compile", False)
        self._clip_grad_norm = cfg.get("clip_grad_norm", None)
        self._rng = torch.Generator(self._device).manual_seed(self.seed)

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

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
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

        # Instantiate the REFERENCE checkpointer locally just to load the weights.
        # It will not be stored in `self` and will not overwrite the main checkpointer.
        ref_checkpointer_instance = config.instantiate(
            cfg.ref_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint, # Should be False for ref model
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

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[training.OPT_KEY]
                if self._resume_from_checkpoint and training.OPT_KEY in checkpoint_dict
                else None
            ),
        )

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)
        if self._compile:
            training.compile_loss(self._loss_fn, dynamic=True, verbose=self._is_rank_zero)

        # sampler and dataloader depend on the tokenizer and loss_fn and should be
        # setup after both of these are initialized
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

        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.
        self._steps_per_epoch = len(self._dataloader)
        self.global_step = self._epochs_run * self._steps_per_epoch

        # Setup lr scheduler
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

        # Parse reward function names from the config for named logging
        self.reward_names = []
        if "reward_functions" in cfg and isinstance(cfg.reward_functions, (ListConfig, list)):
            for rf_cfg in cfg.reward_functions:
                # Extracts the class name, e.g., "FormattedMathCorrectnessReward"
                component_path = rf_cfg.get("_component_", "")
                if component_path:
                    class_name = component_path.split('.')[-1]
                    self.reward_names.append(class_name)

            utils.log_rank_zero(log, f"Found reward names for logging: {self.reward_names}")

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
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
           c. We register (pre-)forward hooks with ``fully_shard`` instead of wrapping `nn.Module`
        """
        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
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

        utils.log_rank_zero(
            self._logger,
            "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...",
        )

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if self._compile:
            training.compile_model(model, dynamic=True, verbose=self._is_rank_zero)

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # For FSDP sharding
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

        # Initialize LoRA params and RoPE buffers
        with training.set_default_dtype(self._dtype), self._device:
            lora_device = "cpu" if fsdp_cpu_offload else self._device
            for m in model.modules():
                if (isinstance(m, AdapterModule)) and not lora_weights_state_dict:
                    # lora may not be covered in state dict
                    # if finetune for the 1st time
                    m.to_empty(device=lora_device)
                    m.initialize_parameters()
                # RoPE is not covered in state dict
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

        # activation offloading
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
        """
        Set up the learning rate scheduler based on the provided configuration.
        """
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
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory.
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

        utils.log_rank_zero(self._logger, "Dataset and Sampler are initialized.")

        return dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe.
        """
        # final dict passed onto the checkpointer
        checkpoint_dict = {}
        intermediate_checkpoint = epoch + 1 < self.total_epochs

        utils.log_rank_zero(
            log,
            "Saving checkpoint. This may take some time. Retrieving full model state dict...",
        )
        start = time.perf_counter()

        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
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
                # Clean the key by removing the activation checkpointing prefix
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

        # Extract and save only the adapter weights if specified
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

            # if training is in-progress, checkpoint the optimizer state and recipe state
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
            log.info(f"Saving checkpoint took {time.perf_counter() - start:.2f} secs")

        torch.distributed.barrier()

    def generate_trajectory(
            self, input_ids: torch.Tensor, answers: list[str]
    ) -> GRPOTrajectory:
        """
        Generates a trajectory given the current policy model (with LoRA adapters),
        the reference policy model (base model without adapters), the reward function,
        and batch of inputs.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]
            answers (list[str]): list of answers corresponding to the input_ids

        Returns:
            Trajectory: An instance of :class:`~torchtune.rlhf.GRPOTrajectory` comprising
                the current trajectory.
        """
        batch_size, context_length = input_ids.shape
        grpo_size = self.grpo_samples

        batch_input_ids = input_ids[:, None, :].expand(-1, grpo_size, -1)  # [B, G, L]
        batch_input_ids = batch_input_ids.reshape(batch_size * grpo_size, -1)

        max_total_len = context_length + self._max_generated_tokens

        # step 1: generate responses using the current policy (with LoRA adapters enabled)
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

        # step 1.1 create attention masks and position IDs for any padding tokens in inputs
        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )
        del query_response_padding_masks

        # step 2. estimate logprobs of the responses using the current policy (with adapters)
        with self.activations_handling_ctx:
            logits = self._model(query_responses, input_pos=position_ids, mask=masks)
        logits = logits[:, context_length - 1:]
        logprobs = rlhf.batched_logits_to_logprobs(logits, responses, self._temperature)
        del logits
        torch.cuda.empty_cache()

        # step 2.1 estimate logprobs of the responses using the reference policy (base model without adapters)
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

        # step 4. replace any tokens in the responses after the first stop token with padding
        (
            response_padding_masks,
            responses,
        ) = rlhf.truncate_sequence_at_first_stop_token(
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )

        # Do some reward modeling
        responses = responses.reshape(batch_size, grpo_size, -1)  # [B, G, L]
        rewards, successes, _ = batched_rewards(self._tokenizer, responses, answers, self._device)
        rewards = rewards.to(self._device)  # [B, G]
        successes = successes.to(self._device)  # [B, G]

        # TODO Create equal weights for all reward functions
        reward_components = rewards.clone()

        # Create equal weights for all reward functions
        num_reward_funcs = rewards.shape[-1]
        reward_weights = torch.ones(num_reward_funcs, device=self._device) / num_reward_funcs
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
        )

    def generate_trajectory_batched(
            self, input_ids: torch.Tensor, answers: list[str]
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
        return GRPOTrajectory(*map(torch.cat, zip(*trajectories)))

    def grpo_step(
            self,
            trajectory: GRPOTrajectory,
            context_length: int,
    ) -> GRPOStats:
        """
        Perform a single GRPO optimization step over a batch of trajectories.

        Args:
            trajectory (Trajectory): a batch of trajectories
            context_length (int): input ids sequence length

        Returns:
            GRPOStats: An instance of :class:`~torchtune.rlhf.GRPOStats`
        """
        torch.cuda.empty_cache()

        # estimate logprobs from the policy at the current optimisation step (with adapters enabled)
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
            approx_policy_kls.detach(),
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
                tokens = tokens.to(self._device)  # [B, P]

                _, context_length = tokens.shape

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
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    torch.distributed.barrier()

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
                pbar.set_description(
                    f"{curr_epoch + 1}|{self.global_step}|Loss: {grpo_stats[-1].loss.item():.4f}"
                )

                if self._steps_run == self._total_steps:
                    training_completed = True
                    break

            self._epochs_run += 1
            if self._epochs_run % self._save_every_n_epochs == 0:
                self.save_checkpoint(curr_epoch)
            if training_completed:
                break

        self._profiler.stop()

        # Save final checkpoint if not already saved
        if not training_completed or self._epochs_run % self._save_every_n_epochs != 0:
            self.save_checkpoint(self._epochs_run - 1)

    def log_metrics(
            self, trajectory: GRPOTrajectory, grpo_stats: GRPOStats, **extras
    ) -> None:
        """
        Log metrics and statistics for the current step to the metric logger.
        """
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
            if self._is_rank_zero:
                if self.reward_names and len(self.reward_names) == len(mean_reward_components):
                    for name, reward_comp in zip(self.reward_names, mean_reward_components):
                        # Using a "reward/" prefix groups these together in the wandb UI
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
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()

    config.log_config(recipe_name="LoRAGRPORecipeDistributed", cfg=cfg)

    recipe = LoRAGRPORecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
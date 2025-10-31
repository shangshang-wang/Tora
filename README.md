# Tora: Torchtune-LoRA for RL

[**Overview**](#overview) | [**Training Dynamics**](#training-dynamics) | [**Benchmarking**](#memory-and-efficiency-benchmarking) | [**Getting Started**](#getting-started) |  [**Behind the Name**](#behind-the-name)

Tora is a project built on [torchtune](https://github.com/meta-pytorch/torchtune) that provides LoRA-based RL methods for post-training.

Building on the torchtune library, Tora extends its functionality for RL post-training.
It integrates PEFT methods like (Q)LoRA and (Q)DoRA into RL, providing an efficient and
memory-friendly framework.
Tora enables researchers to reduce the computational resources required for fine-tuning large language models via RL.

**📚 Key References: LoRA for RL**
* *October 2025*: LoRA Without Regret by Thinking Machines. [link](https://thinkingmachines.ai/blog/lora/)
* *May 2025*: Tina: Tiny Reasoning Models via LoRA. [link](https://github.com/shangshang-wang/Tina)

## Overview

The following table summarizes the key features of RL methods and LoRA-based techniques supported in Tora.

| RL Method | Type of Weight Update | Torch Model Compile | Multiple Devices with One Node |
|-----------|-----------------------|:-------------------:|:------------------------------:|
| GRPO      | Full                  |          ✅          |               ✅                | 
|           | (Q)LoRA               |          ✅          |               ✅                |
|           | (Q)DoRA               |          ✅          |               ✅                |
|           | (Q)DoRA w/ Cache      |          ❌          |               ✅                |

**DoRA w/ Cache:** The standard DoRA layer in torchtune ([link](https://github.com/meta-pytorch/torchtune/blob/main/torchtune/modules/peft/dora.py)) recalculates the weight norm and magnitude scale on every forward pass. This is inefficient for GRPO's completion generation step, as these values remain static between weight updates.
DoRA w/ Cache optimizes this by caching these expensive computations. It computes the values once and reuses them on subsequent forward passes, avoiding redundant calculations and significantly improving performance. However, the current caching implementation is not compatible with torch.compile.

## Getting Started

Clone the repository and install the required packages.

```bash
git clone https://github.com/shangshang-wang/Tora.git && cd Tora
pip install -e .
pip install -e .[async_rl] # For async RL training
pip install torch torchvision torchao --upgrade  # Install the latest PyTorch and TorchAO

pip install wandb
```

Download a model from the Hugging Face Hub.
```bash
MODEL_SIZE=2B  # 2B, 4B, or 8B
tune download "Qwen/Qwen3-VL-${MODEL_SIZE}-Instruct" \
--output-dir "/tmp/Qwen3-VL-${MODEL_SIZE}-Instruct" \
--hf-token <HF_TOKEN>
```

Below are example commands for running single-device and distributed GRPO training.
You can easily switch between LoRA methods by modifying the `lora_type` parameter in the config file or overriding it on the command line.

Full-Parameter RL:
```bash
# single-device
tune run full_grpo_single_device --config qwen3_vl/2B_full_grpo_single_device
# distributed
tune run --nproc_per_node 2 full_grpo_distributed --config qwen3_vl/2B_full_grpo
```

LoRA-Based RL:
```bash
# In the config file, set lora_type to "lora", "dora", or "dora_cache"
# single-device
tune run lora_grpo_single_device --config qwen3_vl/2B_lora_grpo_single_device model.lora_type="lora"
# distributed
tune run --nproc_per_node 2 lora_grpo_distributed --config qwen3_vl/2B_lora_grpo model.lora_type="lora"
```

## Behind the Name

The name Tora (虎) means Tiger in Japanese. It's also a blend of **To**rchTune and Lo**RA**. 
The name is inspired by the film _Crouching Tiger, Hidden Dragon_, which refers to masters with hidden strengths.
This symbolism captures the role of LoRA in RL post-training: by updating only a tiny
fraction of a model's parameters, LoRA unleashes significant performance gains—a
"crouching tiger" of potential within the base model.

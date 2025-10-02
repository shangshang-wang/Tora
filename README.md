# Tora: Torchtune LoRA for RL

## Overview 📚

### Status

| RL Method | Type of Weight Update | Multiple Device |
|-----------|-----------------------|:---------------:|
| GRPO      | Full                  |        ✅        |
|           | LoRA                  |        ✅        |
|           | LoRA-XS               |        ✅        |
|           | DoRA                  |        ✅        |
|           | DoRA w/ Cache         |        ✅        |

### GRPO Post-Training Performance

#### Full-Parameter

|    Model     | GRPO Setting | Runnable On | Peak Memory per GPU | Tokens/sec * |
|:------------:|:------------:|:-----------:|:-------------------:|:------------:|
| Qwen2.5-1.5B |     Full     |   2x A40    |         GB          |              |
|  Qwen2.5-3B  |     Full     |   2x A40    |         GB          |              |

#### LoRA and LoRA-XS

|    Model     |  GRPO Setting   | Runnable On | Peak Memory per GPU | Tokens/sec * |
|:------------:|:---------------:|:-----------:|:-------------------:|:------------:|
| Qwen2.5-1.5B |  LoRA (rank=)   |   2x A40    |         GB          |              |
|  Qwen2.5-3B  |  LoRA (rank=)   |   2x A40    |         GB          |              |
|  Qwen2.5-7B  |  LoRA (rank=)   |   2x A40    |         GB          |              |
| Qwen2.5-1.5B | LoRA-XS (rank=) |   2x A40    |         GB          |              |
|  Qwen2.5-3B  | LoRA-XS (rank=) |   2x A40    |         GB          |              |
|  Qwen2.5-7B  | LoRA-XS (rank=) |   2x A40    |         GB          |              |

#### DoRA w/ and w/o Cache

|    Model     |     GRPO Setting      | Runnable On | Peak Memory per GPU | Tokens/sec * |
|:------------:|:---------------------:|:-----------:|:-------------------:|:------------:|
| Qwen2.5-1.5B |     DoRA (rank=)      |   2x A40    |         GB          |              |
|  Qwen2.5-3B  |     DoRA (rank=)      |   2x A40    |         GB          |              |
|  Qwen2.5-7B  |     DoRA (rank=)      |   2x A40    |         GB          |              |
| Qwen2.5-1.5B | DoRA w/ Cache (rank=) |   2x A40    |         GB          |              |
|  Qwen2.5-3B  | DoRA w/ Cache (rank=) |   2x A40    |         GB          |              |
|  Qwen2.5-7B  | DoRA w/ Cache (rank=) |   2x A40    |         GB          |              |

## Installation 🛠️

```bash
pip install torch torchvision torchao
pip install torchtune
pip install wandb math_verify
```

## Get Started 🚀

### Downloading a model

```bash
MODEL_SIZE=1.5B  # 1.5B, 3B, 7B
tune download "Qwen/Qwen2.5-${MODEL_SIZE}" \
--output-dir "/tmp/Qwen2.5-${MODEL_SIZE}" \
--hf-token <HF_TOKEN>
```

### Running finetuning recipes

Example commands for distributed GRPO training with 2 GPUs.

```bash
tune run --nproc_per_node 2 full_grpo_distributed --config qwen2_5/1.5B_full_grpo
tune run --nproc_per_node 2 lora_grpo_distributed --config qwen2_5/1.5B_lora_grpo
tune run --nproc_per_node 2 lora_grpo_distributed --config qwen2_5/1.5B_dora_grpo
```

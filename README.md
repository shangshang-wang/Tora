# Tora: Torchtune-LoRA for RL

Tora is a Reinforcement Learning (RL) library built on [TorchTune](https://github.com/meta-pytorch/torchtune) that specializes in LoRA-based RL fine-tuning methods.

The name Tora (虎) means Tiger in Japanese. It's also a blend of **To**rchTune and Lo**RA**. 
The name is inspired by the film _Crouching Tiger, Hidden Dragon_, which refers to masters with hidden strengths.
This symbolism perfectly captures the role of LoRA in RL post-training.
By updating only a tiny fraction of a model's parameters, LoRA unleashes significant performance gains—a "hidden tiger" of potential within the base model

## Overview

Tora integrates state-of-the-art Parameter-Efficient Fine-Tuning (PEFT) techniques, such as (Q)LoRA and (Q)DoRA, into the RL workflow.
This allows for efficient training and generation, especially in memory-constrained environments.

### Implementation Status

| RL Method | Type of Weight Update      | Torch Model Compile | Multiple Devices with One Node |
|-----------|----------------------------|:-------------------:|:------------------------------:|
| GRPO      | Full                       |          ✅          |               ✅                | 
|           | (Q)LoRA                    |          ✅          |               ✅                |
|           | (Q)DoRA                    |          ✅          |               ✅                |
|           | (Q)DoRA w/ Inference Cache |          ❌          |               ✅                |

The standard DoRA layer in [TorchTune](https://github.com/meta-pytorch/torchtune) recalculates the weight norm and magnitude scale on every forward pass. This is inefficient for GRPO's completion generation step, as these values remain static between weight updates.
DoRA w/ Inference Cache optimizes this by caching these expensive computations. It computes the values once and reuses them on subsequent forward passes, avoiding redundant calculations and significantly improving performance. However, the current caching implementation is not compatible with torch.compile.

### GRPO Post-Training Performance

Unless otherwise specified in the table, the GRPO settings are as follows:

All experiments were conducted on two NVIDIA RTX A40 GPUs with a per-GPU batch size of 2 and a GRPO generation sequence length of 512.

For all LoRA-based methods, LoRA was applied to all layers with a rank of 1, an alpha of 2, and zero dropout.
The 'Q' prefix in QLoRA and QDoRA means that the base model was quantized to 4-bits.
During training, activation checkpointing was enabled for all experiments, and we utilized Fully Sharded Data Parallelism (FSDP).
The learning rate for LoRA-based methods was set 20 times higher than that used for full-parameter GRPO training.

#### Full-Parameter

<table>
  <thead>
    <tr>
      <th rowspan=2 style="text-align:center;">Model</th>
      <th rowspan=2 style="text-align:center;">GRPO Setting</th>
      <th rowspan=2 style="text-align:center;">Peak Memory per GPU</th>
      <th colspan=2 style="text-align:center;">Tokens/sec (Completion Generation)</th>
      <th colspan=2 style="text-align:center;">Secs/step (Gradient Update)</th>
    </tr>
    <tr>
      <th style="text-align:center;">Standard</th>
      <th style="text-align:center;">w/ Model Compiled</th>
      <th style="text-align:center;">Standard</th>
      <th style="text-align:center;">w/ Model Compiled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">Qwen2.5-1.5B</td>
      <td style="text-align:center;">Full</td>
      <td style="text-align:center;">~16.5 GB</td>
      <td style="text-align:center;">24.4</td>
      <td style="text-align:center;">39.3</td>
      <td style="text-align:center;">69.2</td>
      <td style="text-align:center;">77.5</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-3B</td>
      <td style="text-align:center;">Full</td>
      <td style="text-align:center;">~19.6 GB</td>
      <td style="text-align:center;">17.7</td>
      <td style="text-align:center;">25.6</td>
      <td style="text-align:center;">63.5</td>
      <td style="text-align:center;">72.5</td>
    </tr>
  </tbody>
</table>

#### LoRA

<table>
  <thead>
    <tr>
      <th rowspan=2 style="text-align:center;">Model</th>
      <th rowspan=2 style="text-align:center;">GRPO Setting</th>
      <th rowspan=2 style="text-align:center;">Peak Memory per GPU</th>
      <th colspan=2 style="text-align:center;">Tokens/sec (Completion Generation)</th>
      <th colspan=2 style="text-align:center;">Secs/step (Gradient Update)</th>
    </tr>
    <tr>
      <th style="text-align:center;">Standard</th>
      <th style="text-align:center;">w/ Model Compiled</th>
      <th style="text-align:center;">Standard</th>
      <th style="text-align:center;">w/ Model Compiled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">Qwen2.5-1.5B</td>
      <td style="text-align:center;">LoRA</td>
      <td style="text-align:center;">~14.9 GB</td>
      <td style="text-align:center;">18.9</td>
      <td style="text-align:center;">28.4</td>
      <td style="text-align:center;">58.5</td>
      <td style="text-align:center;">49.7</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-3B</td>
      <td style="text-align:center;">LoRA</td>
      <td style="text-align:center;">~17.2 GB</td>
      <td style="text-align:center;">14.2</td>
      <td style="text-align:center;">21.3</td>
      <td style="text-align:center;">50.6</td>
      <td style="text-align:center;">48.8</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-7B</td>
      <td style="text-align:center;">LoRA</td>
      <td style="text-align:center;">~32.6 GB</td>
      <td style="text-align:center;">15.1</td>
      <td style="text-align:center;">20.5</td>
      <td style="text-align:center;">64.8</td>
      <td style="text-align:center;">68.0</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-1.5B</td>
      <td style="text-align:center;">QLoRA</td>
      <td style="text-align:center;">~12.3 GB</td>
      <td style="text-align:center;">7.9</td>
      <td style="text-align:center;">16.0</td>
      <td style="text-align:center;">142.7</td>
      <td style="text-align:center;">71.5</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-3B</td>
      <td style="text-align:center;">QLoRA</td>
      <td style="text-align:center;">~11.5 GB</td>
      <td style="text-align:center;">4.7</td>
      <td style="text-align:center;">17.7</td>
      <td style="text-align:center;">150.8</td>
      <td style="text-align:center;">87.6</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-7B</td>
      <td style="text-align:center;">QLoRA</td>
      <td style="text-align:center;">~19.1 GB</td>
      <td style="text-align:center;">2.6</td>
      <td style="text-align:center;">11.1</td>
      <td style="text-align:center;">410.0</td>
      <td style="text-align:center;">135.3</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-14B</td>
      <td style="text-align:center;">QLoRA</td>
      <td style="text-align:center;">~29.6 GB</td>
      <td style="text-align:center;">1.3</td>
      <td style="text-align:center;">6.6</td>
      <td style="text-align:center;">793.4</td>
      <td style="text-align:center;">189.7</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-32B</td>
      <td style="text-align:center;">QLoRA</td>
      <td style="text-align:center;">~45.5 GB</td>
      <td style="text-align:center;">0.6</td>
      <td style="text-align:center;">3.6</td>
      <td style="text-align:center;">1578.8</td>
      <td style="text-align:center;">312.6</td>
    </tr>
  </tbody>
</table>

#### DoRA w/ and w/o Inference Cache

DoRA w/ Inference Cache significantly speeds up the generation process by caching intermediate calculations, which is particularly effective for larger models where it can outperform `torch.compile` optimizations.

<table>
  <thead>
    <tr>
      <th rowspan=2 style="text-align:center;">Model</th>
      <th rowspan=2 style="text-align:center;">GRPO Setting</th>
      <th rowspan=2 style="text-align:center;">Peak Memory per GPU</th>
      <th colspan=2 style="text-align:center;">Tokens/sec (Completion Generation)</th>
      <th colspan=2 style="text-align:center;">Secs/step (Gradient Update)</th>
    </tr>
    <tr>
      <th style="text-align:center;">Standard</th>
      <th style="text-align:center;">w/ Model Compiled</th>
      <th style="text-align:center;">Standard</th>
      <th style="text-align:center;">w/ Model Compiled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center;">Qwen2.5-1.5B</td>
      <td style="text-align:center;">DoRA</td>
      <td style="text-align:center;">~14.9 GB</td>
      <td style="text-align:center;">9.1</td>
      <td style="text-align:center;">16.0</td>
      <td style="text-align:center;">190.0</td>
      <td style="text-align:center;">117.7</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-3B</td>
      <td style="text-align:center;">DoRA</td>
      <td style="text-align:center;">~17.2 GB</td>
      <td style="text-align:center;">6.1</td>
      <td style="text-align:center;">10.7</td>
      <td style="text-align:center;">101.3</td>
      <td style="text-align:center;">118.8</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-7B</td>
      <td style="text-align:center;">DoRA</td>
      <td style="text-align:center;">~32.5 GB</td>
      <td style="text-align:center;">3.5</td>
      <td style="text-align:center;">5.9</td>
      <td style="text-align:center;">328.1</td>
      <td style="text-align:center;">233.0</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-1.5B</td>
      <td style="text-align:center;">DoRA w/ Inference Cache</td>
      <td style="text-align:center;">~14.9 GB</td>
      <td colspan=2 style="text-align:center;">16.5</td>
      <td colspan=2 style="text-align:center;">93.2</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-3B</td>
      <td style="text-align:center;">DoRA w/ Inference Cache</td>
      <td style="text-align:center;">~17.3 GB</td>
      <td colspan=2 style="text-align:center;">12.5</td>
      <td colspan=2 style="text-align:center;">79.1</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-7B</td>
      <td style="text-align:center;">DoRA w/ Inference Cache</td>
      <td style="text-align:center;">~32.6 GB</td>
      <td colspan=2 style="text-align:center;">13.1</td>
      <td colspan=2 style="text-align:center;">101.2</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-1.5B</td>
      <td style="text-align:center;">QDoRA</td>
      <td style="text-align:center;">~12.3 GB</td>
      <td style="text-align:center;">4.0</td>
      <td style="text-align:center;">9.0</td>
      <td style="text-align:center;">486.5</td>
      <td style="text-align:center;">191.5</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-3B</td>
      <td style="text-align:center;">QDoRA</td>
      <td style="text-align:center;">~11.5 GB</td>
      <td style="text-align:center;">2.2</td>
      <td style="text-align:center;">6.0</td>
      <td style="text-align:center;">581.0</td>
      <td style="text-align:center;">219.8</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-7B</td>
      <td style="text-align:center;">QDoRA</td>
      <td style="text-align:center;">~19.1 GB</td>
      <td style="text-align:center;">1.1</td>
      <td style="text-align:center;">3.2</td>
      <td style="text-align:center;">1515.3</td>
      <td style="text-align:center;">488.3</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-14B</td>
      <td style="text-align:center;">QDoRA</td>
      <td style="text-align:center;">~29.6 GB</td>
      <td style="text-align:center;">0.6</td>
      <td style="text-align:center;">1.8</td>
      <td style="text-align:center;">2907.8</td>
      <td style="text-align:center;">911.6</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-32B</td>
      <td style="text-align:center;">QDoRA</td>
      <td style="text-align:center;">~45.5 GB</td>
      <td style="text-align:center;">0.2</td>
      <td style="text-align:center;">0.8</td>
      <td style="text-align:center;">4409.3</td>
      <td style="text-align:center;">1478.6</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-1.5B</td>
      <td style="text-align:center;">QDoRA w/ Inference Cache</td>
      <td style="text-align:center;">~12.3 GB</td>
      <td colspan=2 style="text-align:center;">7.2</td>
      <td colspan=2 style="text-align:center;">147.9</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-3B</td>
      <td style="text-align:center;">QDoRA w/ Inference Cache</td>
      <td style="text-align:center;">~11.5 GB</td>
      <td colspan=2 style="text-align:center;">3.3</td>
      <td colspan=2 style="text-align:center;">127.3</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-7B</td>
      <td style="text-align:center;">QDoRA w/ Inference Cache</td>
      <td style="text-align:center;">~19.1 GB</td>
      <td colspan=2 style="text-align:center;">2.3</td>
      <td colspan=2 style="text-align:center;">351.4</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-14B</td>
      <td style="text-align:center;">QDoRA w/ Inference Cache</td>
      <td style="text-align:center;">~29.6 GB</td>
      <td colspan=2 style="text-align:center;">1.3</td>
      <td colspan=2 style="text-align:center;">810.8</td>
    </tr>
    <tr>
      <td style="text-align:center;">Qwen2.5-32B</td>
      <td style="text-align:center;">QDoRA w/ Inference Cache</td>
      <td style="text-align:center;">~45.5 GB</td>
      <td colspan=2 style="text-align:center;">0.6</td>
      <td colspan=2 style="text-align:center;">1812.3</td>
    </tr>
  </tbody>
</table>

## Installation 🛠️

```bash
pip install torch torchvision torchao
pip install -e .
pip install wandb math_verify
```

## Get Started 🚀

### Downloading a model

First, download a model from the Hugging Face Hub.
```bash
MODEL_SIZE=1.5B  # 1.5B, 3B, 7B, 14B, or 32B
tune download "Qwen/Qwen2.5-${MODEL_SIZE}" \
--output-dir "/tmp/Qwen2.5-${MODEL_SIZE}" \
--hf-token <HF_TOKEN>
```

### Running finetuning recipes

Below are example commands for running distributed GRPO training on 2 GPUs. You can easily switch between LoRA methods by modifying the `lora_type` parameter in the config file or overriding it on the command line.

Full-Parameter Fine-Tuning:
```bash
tune run --nproc_per_node 2 full_grpo_distributed --config qwen2_5/1.5B_full_grpo
```

LoRA-Based Fine-Tuning:
```bash
# In the config file, set lora_type to "lora", "dora", or "dora_cache"
tune run --nproc_per_node 2 lora_grpo_distributed --config qwen2_5/1.5B_lora_grpo model.lora_type="lora"
```

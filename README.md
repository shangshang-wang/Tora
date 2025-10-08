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

## LoRA vs Full-Parameter Comparison

Unless specified otherwise, our experimental settings are as follows:
* We used Qwen2.5 base models in five sizes: 1.5B, 3B, 7B, 14B, and 32B parameters.
* All experiments were conducted on two NVIDIA RTX A40 GPUs using the GSM8K training dataset.
* We used a per-GPU batch size of 2 and a generation sequence length of 512.
* For all LoRA-based methods, LoRA was applied to all layers with a rank of 1, an alpha of 2, and zero dropout.
* In QLoRA and QDoRA, the base model was quantized to 4-bits.
* We enabled activation checkpointing and used Fully Sharded Data Parallelism (FSDP) across all experiments.
* The learning rate for LoRA-based methods was set to 20x that of full-parameter GRPO training.

### Training Dynamics

We show the reward dynamics during GRPO training of Qwen2.5-3B with different methods on GSM8K.
From the results, we can see that LoRA-based methods (with rank 1), even with base model quantization, achieve comparable performance with full-parameter GRPO training.

<p align="center">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/rewards_dark.png" width="630" >
      <source media="(prefers-color-scheme: light)" srcset="assets/rewards_light.png" width="630">
      <img alt="Alt text." style="border-radius: 8px;">
    </picture>
</p>

### Memory and Efficiency Benchmarking

In the tables below, we benchmark the peak memory usage per GPU, the number of generated tokens per second during GRPO completion generation, and the seconds per gradient step for different GRPO methods.

#### Full-Parameter GRPO

<table>
  <thead>
    <tr>
      <th rowspan=2 align="center">Model Size</th>
      <th rowspan=2 align="center">Setting</th>
      <th rowspan=2 align="center">Peak Memory/gpu</th>
      <th colspan=2 align="center">Generated Tokens/sec</th>
      <th colspan=2 align="center">Secs/grad step</th>
    </tr>
    <tr>
      <th align="center">Standard</th>
      <th align="center">Compiled</th>
      <th align="center">Standard</th>
      <th align="center">Compiled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">1.5B</td>
      <td align="center">Full</td>
      <td align="center">~16.5 GB</td>
      <td align="center">24.4</td>
      <td align="center">39.3</td>
      <td align="center">69.2</td>
      <td align="center">77.5</td>
    </tr>
    <tr>
      <td align="center">3B</td>
      <td align="center">Full</td>
      <td align="center">~19.6 GB</td>
      <td align="center">17.7</td>
      <td align="center">25.6</td>
      <td align="center">63.5</td>
      <td align="center">72.5</td>
    </tr>
  </tbody>
</table>

#### (Q)LoRA-based GRPO

<table>
  <thead>
    <tr>
      <th rowspan=2 align="center">Model Size</th>
      <th rowspan=2 align="center">Setting</th>
      <th rowspan=2 align="center">Peak Memory/gpu</th>
      <th colspan=2 align="center">Generated Tokens/sec</th>
      <th colspan=2 align="center">Secs/grad step</th>
    </tr>
    <tr>
      <th align="center">Standard</th>
      <th align="center">Compiled</th>
      <th align="center">Standard</th>
      <th align="center">Compiled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">1.5B</td>
      <td align="center">LoRA</td>
      <td align="center">~14.9 GB</td>
      <td align="center">18.9</td>
      <td align="center">28.4</td>
      <td align="center">58.5</td>
      <td align="center">49.7</td>
    </tr>
    <tr>
      <td align="center">3B</td>
      <td align="center">LoRA</td>
      <td align="center">~17.2 GB</td>
      <td align="center">14.2</td>
      <td align="center">21.3</td>
      <td align="center">50.6</td>
      <td align="center">48.8</td>
    </tr>
    <tr>
      <td align="center">7B</td>
      <td align="center">LoRA</td>
      <td align="center">~32.6 GB</td>
      <td align="center">15.1</td>
      <td align="center">20.5</td>
      <td align="center">64.8</td>
      <td align="center">68.0</td>
    </tr>
    <tr>
      <td align="center">1.5B</td>
      <td align="center">QLoRA</td>
      <td align="center">~12.3 GB</td>
      <td align="center">7.9</td>
      <td align="center">16.0</td>
      <td align="center">142.7</td>
      <td align="center">71.5</td>
    </tr>
    <tr>
      <td align="center">3B</td>
      <td align="center">QLoRA</td>
      <td align="center">~11.5 GB</td>
      <td align="center">4.7</td>
      <td align="center">17.7</td>
      <td align="center">150.8</td>
      <td align="center">87.6</td>
    </tr>
    <tr>
      <td align="center">7B</td>
      <td align="center">QLoRA</td>
      <td align="center">~19.1 GB</td>
      <td align="center">2.6</td>
      <td align="center">11.1</td>
      <td align="center">410.0</td>
      <td align="center">135.3</td>
    </tr>
    <tr>
      <td align="center">14B</td>
      <td align="center">QLoRA</td>
      <td align="center">~29.6 GB</td>
      <td align="center">1.3</td>
      <td align="center">6.6</td>
      <td align="center">793.4</td>
      <td align="center">189.7</td>
    </tr>
    <tr>
      <td align="center">32B</td>
      <td align="center">QLoRA</td>
      <td align="center">~45.5 GB</td>
      <td align="center">0.6</td>
      <td align="center">3.6</td>
      <td align="center">1578.8</td>
      <td align="center">312.6</td>
    </tr>
  </tbody>
</table>

#### (Q)DoRA-based GRPO

<table>
  <thead>
    <tr>
      <th rowspan=2 align="center">Model Size</th>
      <th rowspan=2 align="center">Setting</th>
      <th rowspan=2 align="center">Peak Memory/gpu</th>
      <th colspan=2 align="center">Generated Tokens/sec</th>
      <th colspan=2 align="center">Secs/grad step</th>
    </tr>
    <tr>
      <th align="center">Standard</th>
      <th align="center">Compiled</th>
      <th align="center">Standard</th>
      <th align="center">Compiled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">1.5B</td>
      <td align="center">DoRA</td>
      <td align="center">~14.9 GB</td>
      <td align="center">9.1</td>
      <td align="center">16.0</td>
      <td align="center">190.0</td>
      <td align="center">117.7</td>
    </tr>
    <tr>
      <td align="center">3B</td>
      <td align="center">DoRA</td>
      <td align="center">~17.2 GB</td>
      <td align="center">6.1</td>
      <td align="center">10.7</td>
      <td align="center">101.3</td>
      <td align="center">118.8</td>
    </tr>
    <tr>
      <td align="center">7B</td>
      <td align="center">DoRA</td>
      <td align="center">~32.5 GB</td>
      <td align="center">3.5</td>
      <td align="center">5.9</td>
      <td align="center">328.1</td>
      <td align="center">233.0</td>
    </tr>
    <tr>
      <td align="center">1.5B</td>
      <td align="center">QDoRA</td>
      <td align="center">~12.3 GB</td>
      <td align="center">4.0</td>
      <td align="center">9.0</td>
      <td align="center">486.5</td>
      <td align="center">191.5</td>
    </tr>
    <tr>
      <td align="center">3B</td>
      <td align="center">QDoRA</td>
      <td align="center">~11.5 GB</td>
      <td align="center">2.2</td>
      <td align="center">6.0</td>
      <td align="center">581.0</td>
      <td align="center">219.8</td>
    </tr>
    <tr>
      <td align="center">7B</td>
      <td align="center">QDoRA</td>
      <td align="center">~19.1 GB</td>
      <td align="center">1.1</td>
      <td align="center">3.2</td>
      <td align="center">1515.3</td>
      <td align="center">488.3</td>
    </tr>
    <tr>
      <td align="center">14B</td>
      <td align="center">QDoRA</td>
      <td align="center">~29.6 GB</td>
      <td align="center">0.6</td>
      <td align="center">1.8</td>
      <td align="center">2907.8</td>
      <td align="center">911.6</td>
    </tr>
    <tr>
      <td align="center">32B</td>
      <td align="center">QDoRA</td>
      <td align="center">~45.5 GB</td>
      <td align="center">0.2</td>
      <td align="center">0.8</td>
      <td align="center">4409.3</td>
      <td align="center">1478.6</td>
    </tr>
  </tbody>
</table>

#### (Q)DoRA-with-Cache-based GRPO

DoRA w/ Cache significantly speeds up the generation process by caching intermediate calculations, and it has comparable performance with `torch.compile` optimizations.

<table>
  <thead>
    <tr>
      <th align="center">Model Size</th>
      <th align="center">Setting</th>
      <th align="center">Peak Memory/gpu</th>
      <th align="center">Generated Tokens/sec</th>
      <th align="center">Secs/grad step</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">1.5B</td>
      <td align="center">DoRA w/ Cache</td>
      <td align="center">~14.9 GB</td>
      <td align="center">16.5</td>
      <td align="center">93.2</td>
    </tr>
    <tr>
      <td align="center">3B</td>
      <td align="center">DoRA w/ Cache</td>
      <td align="center">~17.3 GB</td>
      <td align="center">12.5</td>
      <td align="center">79.1</td>
    </tr>
    <tr>
      <td align="center">7B</td>
      <td align="center">DoRA w/ Cache</td>
      <td align="center">~32.6 GB</td>
      <td align="center">13.1</td>
      <td align="center">101.2</td>
    </tr>
    <tr>
      <td align="center">1.5B</td>
      <td align="center">QDoRA w/ Cache</td>
      <td align="center">~12.3 GB</td>
      <td align="center">7.2</td>
      <td align="center">147.9</td>
    </tr>
    <tr>
      <td align="center">3B</td>
      <td align="center">QDoRA w/ Cache</td>
      <td align="center">~11.5 GB</td>
      <td align="center">3.3</td>
      <td align="center">127.3</td>
    </tr>
    <tr>
      <td align="center">7B</td>
      <td align="center">QDoRA w/ Cache</td>
      <td align="center">~19.1 GB</td>
      <td align="center">2.3</td>
      <td align="center">351.4</td>
    </tr>
    <tr>
      <td align="center">14B</td>
      <td align="center">QDoRA w/ Cache</td>
      <td align="center">~29.6 GB</td>
      <td align="center">1.3</td>
      <td align="center">810.8</td>
    </tr>
    <tr>
      <td align="center">32B</td>
      <td align="center">QDoRA w/ Cache</td>
      <td align="center">~45.5 GB</td>
      <td align="center">0.6</td>
      <td align="center">1812.3</td>
    </tr>
  </tbody>
</table>

## Getting Started

Clone the repository and install the required packages.

```bash
git clone https://github.com/shangshang-wang/Tora.git && cd Tora
pip install torch torchvision torchao
pip install -e .
pip install wandb math_verify
```

Download a model from the Hugging Face Hub.
```bash
MODEL_SIZE=1.5B  # 1.5B, 3B, 7B, 14B, or 32B
tune download "Qwen/Qwen2.5-${MODEL_SIZE}" \
--output-dir "/tmp/Qwen2.5-${MODEL_SIZE}" \
--hf-token <HF_TOKEN>
```

Below are example commands for running distributed GRPO training on 2 GPUs.
You can easily switch between LoRA methods by modifying the `lora_type` parameter in the config file or overriding it on the command line.

Full-Parameter RL:
```bash
tune run --nproc_per_node 2 full_grpo_distributed --config qwen2_5/1.5B_full_grpo
```

LoRA-Based RL:
```bash
# In the config file, set lora_type to "lora", "dora", or "dora_cache"
tune run --nproc_per_node 2 lora_grpo_distributed --config qwen2_5/1.5B_lora_grpo model.lora_type="lora"
```

## Behind the Name

The name Tora (虎) means Tiger in Japanese. It's also a blend of **To**rchTune and Lo**RA**. 
The name is inspired by the film _Crouching Tiger, Hidden Dragon_, which refers to masters with hidden strengths.
This symbolism captures the role of LoRA in RL post-training: by updating only a tiny
fraction of a model's parameters, LoRA unleashes significant performance gains—a
"crouching tiger" of potential within the base model.

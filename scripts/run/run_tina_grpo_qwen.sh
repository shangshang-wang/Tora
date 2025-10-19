#!/bin/bash

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=2,3
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo -e "\nNumber of GPUs: ${GPU_COUNT}\n"

# Configuration
MODEL_SIZE="1.5B"           # Options: 1.5B
POST_TRAIN_TYPE="lora"      # Options: lora
USE_QLORA="false"           # Options: false
USE_COMPILE="false"         # Options: true, false
SEED=42                     # Set a seed for reproducibility

export BASE_MODEL_PATH="${CKPT_DIR}/qwen2_5/Qwen2.5-Math-${MODEL_SIZE}"
BASE_MODEL_NAME="Qwen/Qwen2.5-Math-${MODEL_SIZE}"

export DEEPSEEK_MODEL_PATH="${CKPT_DIR}/qwen2_5/DeepSeek-R1-Distill-Qwen-${MODEL_SIZE}"
DEEPSEEK_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-${MODEL_SIZE}"

# Override config in recipe
export CONFIG_NAME="${POST_TRAIN_TYPE}_grpo_qwen2_5_deepseek_distilled_${MODEL_SIZE}"
export BASE_MODEL_DIR="${CKPT_DIR}/qwen2_5"
export BASE_OUTPUT_DIR="${OUTPUT_DIR}/qwen2_5"
export MODEL_SIZE

# Download model if needed
if [ ! -d "${BASE_MODEL_PATH}" ]; then
    echo "Downloading base model to ${BASE_MODEL_PATH}..."
    tune download "${BASE_MODEL_NAME}" \
        --output-dir "${BASE_MODEL_PATH}" \
        --hf-token "${HF_TOKEN}"
else
    echo "Base model already exists at ${BASE_MODEL_PATH}. Skipping download."
fi

if [ ! -d "${DEEPSEEK_MODEL_PATH}" ]; then
    echo "Downloading deepseek model to ${DEEPSEEK_MODEL_PATH}..."
    tune download "${DEEPSEEK_MODEL_NAME}" \
        --output-dir "${DEEPSEEK_MODEL_PATH}" \
        --hf-token "${HF_TOKEN}"
else
    echo "Deepseek model already exists at ${DEEPSEEK_MODEL_PATH}. Skipping download."
fi

# Determine training configuration
CONFIG_FILE="qwen2_5_deepseek_distilled/${MODEL_SIZE}_lora_grpo"

# Map POST_TRAIN_TYPE to lora_type (default to "lora" if not specified)
LORA_TYPE="${POST_TRAIN_TYPE}"
[ "${POST_TRAIN_TYPE}" == "lora" ] && LORA_TYPE="lora"

echo "Using $( [ "${USE_QLORA}" == "true" ] && echo "QLoRA" || echo "LoRA" ) for post-training."
echo "Using ${LORA_TYPE}."
echo "Model Compile: ${USE_COMPILE}"
echo "Random Seed: ${SEED}"

tune run --nproc_per_node 2 lora_grpo_distributed \
    --config "${CONFIG_FILE}" \
    model.lora_type="${LORA_TYPE}" \
    seed="${SEED}" \
    compile="${USE_COMPILE}"

echo "TRAINING END TIME: $(date)"

echo ""
echo "========================================================================"
echo "SCRIPT COMPLETE"
echo "========================================================================"
echo "END TIME: $(date)"
echo "DONE"

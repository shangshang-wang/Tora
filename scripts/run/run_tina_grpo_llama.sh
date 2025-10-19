#!/bin/bash

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo -e "\nNumber of GPUs: ${GPU_COUNT}\n"

# Configuration
MODEL_SIZE="8B"             # Options: 8B
POST_TRAIN_TYPE="full"      # Options: full, lora
USE_QLORA="false"           # Options: false
USE_COMPILE="false"         # Options: true, false
SEED=42                     # Set a seed for reproducibility

export BASE_MODEL_PATH="${CKPT_DIR}/llama3_1/Llama-3.1-${MODEL_SIZE}"
BASE_MODEL_NAME="meta-llama/Llama-3.1-${MODEL_SIZE}"

# Override config in recipe
export CONFIG_NAME="${POST_TRAIN_TYPE}_grpo_llama3_1_${MODEL_SIZE}"
export BASE_MODEL_DIR="${CKPT_DIR}/llama3_1"
export BASE_OUTPUT_DIR="${OUTPUT_DIR}/llama3_1"
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

# Determine training configuration
if [ "${POST_TRAIN_TYPE}" == "full" ]; then
    echo "Using full model for post-training."
    echo "Random Seed: ${SEED}"
    tune run --nproc_per_node "${GPU_COUNT}" full_grpo_distributed \
        --config llama3_1/"${MODEL_SIZE}_full_grpo" \
        seed="${SEED}" \
        compile="${USE_COMPILE}"
else
    # Select config file based on QLoRA usage
    CONFIG_SUFFIX=$( [ "${USE_QLORA}" == "true" ] && echo "qlora_grpo" || echo "lora_grpo" )
    CONFIG_FILE="llama3_1/${MODEL_SIZE}_${CONFIG_SUFFIX}"

    # Map POST_TRAIN_TYPE to lora_type (default to "lora" if not specified)
    LORA_TYPE="${POST_TRAIN_TYPE}"
    [ "${POST_TRAIN_TYPE}" == "lora" ] && LORA_TYPE="lora"

    echo "Using $( [ "${USE_QLORA}" == "true" ] && echo "QLoRA" || echo "LoRA" ) for post-training."
    echo "Using ${LORA_TYPE}."
    echo "Model Compile: ${USE_COMPILE}"
    echo "Random Seed: ${SEED}"

    tune run --nproc_per_node "${GPU_COUNT}" lora_grpo_distributed \
        --config "${CONFIG_FILE}" \
        model.lora_type="${LORA_TYPE}" \
        seed="${SEED}" \
        compile="${USE_COMPILE}"
fi

echo "TRAINING END TIME: $(date)"

echo ""
echo "========================================================================"
echo "SCRIPT COMPLETE"
echo "========================================================================"
echo "END TIME: $(date)"
echo "DONE"

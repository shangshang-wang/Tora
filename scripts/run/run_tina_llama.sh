#!/bin/bash

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo -e "\nNumber of GPUs: ${GPU_COUNT}\n"

# Configuration
MODEL_SIZE="3B"           # Options: 3B
POST_TRAIN_TYPE="full"      # Options: full, lora
USE_QLORA="false"           # Options: false
USE_COMPILE="false"         # Options: true, false
SEED=42                     # Set a seed for reproducibility

export BASE_MODEL_PATH="${CKPT_DIR}/llama3_2/Llama-3.2-${MODEL_SIZE}"
BASE_MODEL_NAME="meta-llama/Llama-3.2-${MODEL_SIZE}"

export OCTO_THINKER_MODEL_PATH="${CKPT_DIR}/llama3_2/OctoThinker-${MODEL_SIZE}-Hybrid-Base"
OCTO_THINKER_MODEL_NAME="OctoThinker/OctoThinker-${MODEL_SIZE}-Hybrid-Base"

# Override config in recipe
export CONFIG_NAME="${POST_TRAIN_TYPE}_grpo_llama3_2_octo_thinker_${MODEL_SIZE}"
export BASE_MODEL_DIR="${CKPT_DIR}/llama3_2"
export BASE_OUTPUT_DIR="${OUTPUT_DIR}/llama3_2"
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

if [ ! -d "${OCTO_THINKER_MODEL_PATH}" ]; then
    echo "Downloading deepseek model to ${OCTO_THINKER_MODEL_PATH}..."
    tune download "${OCTO_THINKER_MODEL_NAME}" \
        --output-dir "${OCTO_THINKER_MODEL_PATH}" \
        --hf-token "${HF_TOKEN}"
else
    echo "Octothinker model already exists at ${OCTO_THINKER_MODEL_PATH}. Skipping download."
fi

# Determine training configuration
if [ "${POST_TRAIN_TYPE}" == "full" ]; then
    echo "Using full model for post-training."
    echo "Random Seed: ${SEED}"
    tune run --nproc_per_node 2 full_grpo_distributed \
        --config llama3_2_octo_thinker/"${MODEL_SIZE}_full_grpo" \
        seed="${SEED}" \
        compile="${USE_COMPILE}"
else
    # Select config file based on QLoRA usage
    CONFIG_SUFFIX=$( [ "${USE_QLORA}" == "true" ] && echo "qlora_grpo" || echo "lora_grpo" )
    CONFIG_FILE="llama3_2_octo_thinker/${MODEL_SIZE}_${CONFIG_SUFFIX}"

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
fi

echo "TRAINING END TIME: $(date)"

echo ""
echo "========================================================================"
echo "SCRIPT COMPLETE"
echo "========================================================================"
echo "END TIME: $(date)"
echo "DONE"

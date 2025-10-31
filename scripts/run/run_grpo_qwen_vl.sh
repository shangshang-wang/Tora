#!/bin/bash

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1,2,3
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo -e "\nNumber of GPUs: ${GPU_COUNT}\n"

# Configuration
MODEL_SIZE="8B"             # Options: 2B, 4B, 8B (no single device)
MODEL_FAMILY="qwen3_vl"
POST_TRAIN_TYPE="full"      # Options: full | lora, dora, dora_cache
USE_QLORA="false"           # LoRA Options: true, false
USE_COMPILE="false"         # Options: true, false
USE_SINGLE_DEVICE="false"   # Options: true, false
SEED=42                     # Set a seed for reproducibility

MODEL_SUBDIR="Qwen3-VL-${MODEL_SIZE}-Instruct"
export BASE_MODEL_PATH="${CKPT_DIR}/${MODEL_FAMILY}/${MODEL_SUBDIR}"
MODEL_PATH="Qwen/${MODEL_SUBDIR}"

# Override config in recipe
CONFIG_MODE_SUFFIX=$( [ "${USE_SINGLE_DEVICE}" == "true" ] && echo "_single_device" || echo "" )
export CONFIG_NAME="${POST_TRAIN_TYPE}_grpo_${MODEL_FAMILY}_${MODEL_SIZE}${CONFIG_MODE_SUFFIX}"
export BASE_MODEL_DIR="${CKPT_DIR}/${MODEL_FAMILY}"
export BASE_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_FAMILY}"
export MODEL_SIZE
export MODEL_SUBDIR
CONFIG_ROOT="recipes/configs"

# Download model if needed
if [ ! -d "${BASE_MODEL_PATH}" ]; then
    echo "Downloading base model to ${BASE_MODEL_PATH}..."
    tune download "${MODEL_PATH}" \
        --output-dir "${BASE_MODEL_PATH}" \
        --hf-token "${HF_TOKEN}"
else
    echo "Base model already exists at ${BASE_MODEL_PATH}. Skipping download."
fi

# Determine training configuration
if [ "${POST_TRAIN_TYPE}" == "full" ]; then
    echo "Using full model for post-training."
    echo "Random Seed: ${SEED}"
    if [ "${USE_SINGLE_DEVICE}" == "true" ]; then
        CONFIG_FILE="${MODEL_FAMILY}/${MODEL_SIZE}_full_grpo_single_device"
        CONFIG_PATH="${CONFIG_ROOT}/${CONFIG_FILE}.yaml"
        if [ ! -f "${CONFIG_PATH}" ]; then
            echo "Configuration ${CONFIG_PATH} not found. Please create it before running."
            exit 1
        fi
        tune run full_grpo_single_device \
            --config "${CONFIG_FILE}" \
            seed="${SEED}" \
            compile="${USE_COMPILE}"
    else
        CONFIG_FILE="${MODEL_FAMILY}/${MODEL_SIZE}_full_grpo"
        CONFIG_PATH="${CONFIG_ROOT}/${CONFIG_FILE}.yaml"
        if [ ! -f "${CONFIG_PATH}" ]; then
            echo "Configuration ${CONFIG_PATH} not found. Please create it before running."
            exit 1
        fi
        tune run --nproc_per_node "${GPU_COUNT}" full_grpo_distributed \
            --config "${CONFIG_FILE}" \
            seed="${SEED}" \
            compile="${USE_COMPILE}"
    fi
else
    # Select config file based on QLoRA usage
    CONFIG_SUFFIX=$( [ "${USE_QLORA}" == "true" ] && echo "qlora_grpo" || echo "lora_grpo" )
    CONFIG_FILE="${MODEL_FAMILY}/${MODEL_SIZE}_${CONFIG_SUFFIX}${CONFIG_MODE_SUFFIX}"
    CONFIG_PATH="${CONFIG_ROOT}/${CONFIG_FILE}.yaml"

    # Map POST_TRAIN_TYPE to lora_type (default to "lora" if not specified)
    LORA_TYPE="${POST_TRAIN_TYPE}"
    [ "${POST_TRAIN_TYPE}" == "lora" ] && LORA_TYPE="lora"

    echo "Using $( [ "${USE_QLORA}" == "true" ] && echo "QLoRA" || echo "LoRA" ) for post-training."
    echo "Using ${LORA_TYPE}."
    echo "Model Compile: ${USE_COMPILE}"
    echo "Random Seed: ${SEED}"

    if [ ! -f "${CONFIG_PATH}" ]; then
        echo "Configuration ${CONFIG_PATH} not found. Please create it before running."
        exit 1
    fi

    if [ "${USE_SINGLE_DEVICE}" == "true" ]; then
        tune run lora_grpo_single_device \
            --config "${CONFIG_FILE}" \
            model.lora_type="${LORA_TYPE}" \
            seed="${SEED}" \
            compile="${USE_COMPILE}"
    else
        tune run --nproc_per_node "${GPU_COUNT}" lora_grpo_distributed \
            --config "${CONFIG_FILE}" \
            model.lora_type="${LORA_TYPE}" \
            seed="${SEED}" \
            compile="${USE_COMPILE}"
    fi
fi

echo "TRAINING END TIME: $(date)"

echo ""
echo "========================================================================"
echo "SCRIPT COMPLETE"
echo "========================================================================"
echo "END TIME: $(date)"
echo "DONE"

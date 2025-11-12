#!/bin/bash

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo -e "\nNumber of GPUs: ${GPU_COUNT}\n"

# Configuration
MODEL_SIZE="3B"           # Qwen2.5 Options: 1.5B, 3B, 7B, 14B, Qwen3 Options: 0.6B, 1.7B, 4B, 8B, 14B
MODEL_FAMILY="qwen2_5"    # Options: qwen2_5, qwen3
POST_TRAIN_TYPE="full"    # Options: full | lora, dora, dora_cache
GRPO_LAUNCH_MODE="async"  # Options: sync | async (async currently supports Qwen2.5-3B full GRPO only)
USE_QLORA="false"         # Options: true, false
USE_COMPILE="false"       # Options: true, false
SEED=42                   # Set a seed for reproducibility

case "${MODEL_FAMILY}" in
  qwen2_5)
    MODEL_FAMILY_FOR_PATH="Qwen2.5"
    ;;
  qwen3)
    MODEL_FAMILY_FOR_PATH="Qwen3"
    ;;
  *)
    echo "Unsupported MODEL_FAMILY: ${MODEL_FAMILY}"
    exit 1
    ;;
esac

export BASE_MODEL_PATH="${CKPT_DIR}/${MODEL_FAMILY}/${MODEL_FAMILY_FOR_PATH}-${MODEL_SIZE}"
MODEL_PATH="Qwen/${MODEL_FAMILY_FOR_PATH}-${MODEL_SIZE}"

if [ "${MODEL_FAMILY}" == "qwen3" ] && [ "${USE_QLORA}" == "true" ]; then
    echo "QLoRA is not currently supported for ${MODEL_FAMILY}. Please set USE_QLORA=false."
    exit 1
fi

# Override config in recipe
export CONFIG_NAME="${POST_TRAIN_TYPE}_grpo_${MODEL_FAMILY}_${MODEL_SIZE}"
export BASE_MODEL_DIR="${CKPT_DIR}/${MODEL_FAMILY}"
export BASE_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_FAMILY}"
export MODEL_SIZE

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
    echo "Model Compile: ${USE_COMPILE}"
    if [ "${GRPO_LAUNCH_MODE}" == "async" ]; then
        CONFIG_FILE="${MODEL_FAMILY}/${MODEL_SIZE}_async_full_grpo"
        CONFIG_PATH="recipes/configs/${CONFIG_FILE}.yaml"
        if [ ! -f "${CONFIG_PATH}" ]; then
            echo "Could not find config file at ${CONFIG_PATH}."
            exit 1
        fi
        REF_CKPT_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG_NAME}/ref_model_ckpts"
        mkdir -p "${REF_CKPT_OUTPUT_DIR}"
        echo "Launching async orchestration."
        tune run async_full_grpo \
            --config "${CONFIG_FILE}" \
            training.seed="${SEED}" \
            training.compile="${USE_COMPILE}" \
            postprocessing.ref_checkpointer.output_dir="${REF_CKPT_OUTPUT_DIR}"
    else
        CONFIG_FILE="${MODEL_FAMILY}/${MODEL_SIZE}_full_grpo"
        CONFIG_PATH="recipes/configs/${CONFIG_FILE}.yaml"
        if [ ! -f "${CONFIG_PATH}" ]; then
            echo "Could not find config file at ${CONFIG_PATH}."
            exit 1
        fi
        tune run --nproc_per_node "${GPU_COUNT}" full_grpo_distributed \
            --config "${CONFIG_FILE}" \
            seed="${SEED}" \
            compile="${USE_COMPILE}"
    fi
else
    if [ "${GRPO_LAUNCH_MODE}" == "async" ]; then
        echo "Async GRPO launch mode is only supported when POST_TRAIN_TYPE=full."
        exit 1
    fi
    # Select config file based on QLoRA usage
    CONFIG_SUFFIX=$( [ "${USE_QLORA}" == "true" ] && echo "qlora_grpo" || echo "lora_grpo" )
    CONFIG_FILE="${MODEL_FAMILY}/${MODEL_SIZE}_${CONFIG_SUFFIX}"
    CONFIG_PATH="recipes/configs/${CONFIG_FILE}.yaml"
    if [ ! -f "${CONFIG_PATH}" ]; then
        echo "Could not find config file at ${CONFIG_PATH}."
        exit 1
    fi

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

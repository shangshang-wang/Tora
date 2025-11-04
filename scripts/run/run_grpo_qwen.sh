#!/bin/bash

echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=2,3
#export CUDA_VISIBLE_DEVICES=4,5
#export CUDA_VISIBLE_DEVICES=6,7
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo -e "\nNumber of GPUs: ${GPU_COUNT}\n"

# Configuration
MODEL_SIZE="3B"             # Options: 1.5B, 3B, 7B, 14B, 32B
POST_TRAIN_TYPE="full"      # Options: full, lora, dora, dora_cache
USE_QLORA="false"            # Options: true, false
USE_COMPILE="false"         # Options: true, false

# --- MODIFICATION: Define seeds as an array ---
# Set all the seeds you want to run here
SEEDS=(42)

export BASE_MODEL_PATH="${CKPT_DIR}/qwen2_5/Qwen2.5-${MODEL_SIZE}"
MODEL_PATH="Qwen/Qwen2.5-${MODEL_SIZE}"

# Override config in recipe
export CONFIG_NAME="${POST_TRAIN_TYPE}_grpo_qwen2_5_${MODEL_SIZE}"
export BASE_MODEL_DIR="${CKPT_DIR}/qwen2_5"
export BASE_OUTPUT_DIR="${OUTPUT_DIR}/qwen2_5"
export MODEL_SIZE

# Download model if needed (runs only once)
if [ ! -d "${BASE_MODEL_PATH}" ]; then
    echo "Downloading base model to ${BASE_MODEL_PATH}..."
    tune download "${MODEL_PATH}" \
        --output-dir "${BASE_MODEL_PATH}" \
        --hf-token "${HF_TOKEN}"
else
    echo "Base model already exists at ${BASE_MODEL_PATH}. Skipping download."
fi


for SEED in "${SEEDS[@]}"; do

    echo ""
    echo "========================================================================"
    echo "STARTING RUN FOR SEED: ${SEED}"
    echo "========================================================================"

    # Determine training configuration
    if [ "${POST_TRAIN_TYPE}" == "full" ]; then
        echo "Using full model for post-training."
        echo "Random Seed: ${SEED}"
        tune run --nproc_per_node "${GPU_COUNT}" full_grpo_distributed \
            --config qwen2_5/"${MODEL_SIZE}_full_grpo" \
            seed="${SEED}" \
            compile="${USE_COMPILE}"
    else
        # Select config file based on QLoRA usage
        CONFIG_SUFFIX=$( [ "${USE_QLORA}" == "true" ] && echo "qlora_grpo" || echo "lora_grpo" )
        CONFIG_FILE="qwen2_5/${MODEL_SIZE}_${CONFIG_SUFFIX}"

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

    echo "TRAINING END TIME FOR SEED ${SEED}: $(date)"

done


echo ""
echo "========================================================================"
echo "ALL SEED RUNS COMPLETE"
echo "========================================================================"
echo "END TIME: $(date)"
echo "DONE"
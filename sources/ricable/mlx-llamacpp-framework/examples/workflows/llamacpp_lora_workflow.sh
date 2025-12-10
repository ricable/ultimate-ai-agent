#!/bin/bash
# Example workflow for llama.cpp LoRA fine-tuning

set -e

# Configuration
LLAMA_CPP_DIR="/Users/cedric/dev/ran/flow2/llama.cpp-setup"
MODEL_PATH="${LLAMA_CPP_DIR}/models/llama-2-7b-q4_0.gguf"
DATASET_DIR="/Users/cedric/dev/ran/flow2/data/datasets"
OUTPUT_DIR="/Users/cedric/dev/ran/flow2/data/outputs/llamacpp_lora"

echo "Starting llama.cpp LoRA fine-tuning workflow"
echo "============================================="

# Create directories
mkdir -p "${DATASET_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Step 1: Prepare a sample dataset
echo "Step 1: Preparing sample dataset"
cat > "${DATASET_DIR}/sample_data.txt" << EOL
Q: What is Apple Silicon?
A: Apple Silicon refers to the custom ARM-based processors designed by Apple for their Mac computers and iPad tablets. These chips, like the M1, M2, and M3 series, offer high performance with excellent power efficiency.

Q: What are the advantages of Apple Silicon?
A: The advantages of Apple Silicon include superior performance-per-watt compared to Intel chips, integrated GPU and Neural Engine components, unified memory architecture, and native optimization for macOS applications.

Q: How does Apple Silicon compare to Intel processors?
A: Compared to Intel processors, Apple Silicon generally offers better power efficiency, improved thermal performance, and integration of more specialized components like Neural Engine. In many workloads, Apple Silicon chips offer better single-core performance and comparable multi-core performance while using significantly less power.

Q: What is the Neural Engine in Apple Silicon?
A: The Neural Engine is a specialized component in Apple Silicon chips designed to accelerate machine learning and AI workloads. It's a dedicated processor for neural network computations that can perform operations much faster and more efficiently than general-purpose CPU cores.

Q: Which applications benefit most from Apple Silicon?
A: Applications optimized for Apple Silicon benefit the most, particularly those that leverage the Neural Engine for ML tasks, those requiring high graphics performance utilizing the integrated GPU, and Apple's own applications like Final Cut Pro and Logic Pro. Native Apple Silicon apps generally show improved performance and battery life compared to running x86 apps through Rosetta 2 translation.
EOL

# Step 2: Convert text to JSONL format
echo "Step 2: Converting text to JSONL format"
python3 ../scripts/prepare_dataset.py --input "${DATASET_DIR}/sample_data.txt" --output "${DATASET_DIR}/sample_data.jsonl" convert --format qa

# Step 3: Split the dataset
echo "Step 3: Splitting the dataset"
python3 ../scripts/prepare_dataset.py --input "${DATASET_DIR}/sample_data.jsonl" --output "${DATASET_DIR}" split --shuffle --train-ratio 0.8 --val-ratio 0.2 --test-ratio 0.0

# Step 4: Verify llama-finetune binary exists
LLAMA_FINETUNE="${LLAMA_CPP_DIR}/build/bin/llama-finetune"
if [ ! -f "${LLAMA_FINETUNE}" ]; then
    echo "Error: llama-finetune binary not found at ${LLAMA_FINETUNE}"
    echo "Please build llama.cpp with fine-tuning support"
    exit 1
fi

# Step 5: Run LoRA fine-tuning
echo "Step 5: Running LoRA fine-tuning"
python3 ../scripts/finetune_llamacpp_lora.py \
    --llama-finetune-path "${LLAMA_FINETUNE}" \
    --model-path "${MODEL_PATH}" \
    --data-train-path "${DATASET_DIR}/train.jsonl" \
    --data-val-path "${DATASET_DIR}/val.jsonl" \
    --output-path "${OUTPUT_DIR}/apple_silicon_lora.bin" \
    --lora-rank 8 \
    --lora-alpha 16 \
    --epochs 3 \
    --learning-rate 3e-4 \
    --ctx-len 512 \
    --verbose

# Step 6: Run inference with fine-tuned model
echo "Step 6: Testing the fine-tuned model"
LLAMA_MAIN="${LLAMA_CPP_DIR}/build/bin/main"
if [ -f "${LLAMA_MAIN}" ]; then
    ${LLAMA_MAIN} \
        -m "${MODEL_PATH}" \
        --lora "${OUTPUT_DIR}/apple_silicon_lora.bin" \
        --temp 0.7 \
        -n 256 \
        --metal \
        -p "Tell me about the benefits of Apple Silicon for developers"
else
    echo "Warning: llama.cpp main binary not found at ${LLAMA_MAIN}"
    echo "Skipping inference test"
fi

echo "============================================="
echo "Workflow completed successfully"
echo "Fine-tuned LoRA adapter: ${OUTPUT_DIR}/apple_silicon_lora.bin"
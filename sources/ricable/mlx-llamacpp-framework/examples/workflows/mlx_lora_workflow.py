#!/usr/bin/env python3
"""
Example workflow for MLX LoRA fine-tuning.
"""

import os
import sys
import subprocess
import logging
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preparation.text_to_jsonl import convert_text_to_jsonl
from data_preparation.dataset_splitting import split_dataset
from mlx.lora_finetune import finetune_lora, apply_lora_adapter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Configuration
    model_name = "llama-2-7b"
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "mlx_lora")
    
    # Create directories
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting MLX LoRA fine-tuning workflow")
    logger.info("=======================================")
    
    # Step 1: Prepare a sample dataset
    logger.info("Step 1: Preparing sample dataset")
    sample_data_path = os.path.join(dataset_dir, "mlx_sample_data.txt")
    
    with open(sample_data_path, 'w') as f:
        f.write("""
Q: How can I optimize ML models for Apple Silicon?
A: To optimize ML models for Apple Silicon, leverage the Neural Engine through frameworks like MLX or Core ML, quantize models to INT8 or INT4 to reduce memory usage, use Metal for GPU acceleration, enable memory-efficient features like attention caching, and test with Apple's Instruments to identify bottlenecks. MLX is specifically designed for optimal performance on Apple Silicon.

Q: What is MLX framework?
A: MLX is Apple's open-source machine learning framework specifically designed for Apple Silicon. It features a Python-first API similar to PyTorch/JAX, unified memory model between CPU and GPU, built-in quantization support, Metal acceleration, and fine-tuning capabilities. MLX is optimized for research and Python ML workflows on Mac hardware.

Q: What advantages does MLX have over other frameworks on Mac?
A: MLX offers several advantages on Mac: native Apple Silicon optimization with significant performance gains, unified memory model eliminating CPU-GPU transfers, Python-first API familiar to PyTorch/JAX users, first-party support from Apple, built-in Metal acceleration, and seamless quantization. It's specifically designed for the Mac's architecture rather than being adapted from other platforms.

Q: How does MLX handle memory management?
A: MLX uses a unified memory model where CPU and GPU share the same memory space, eliminating the need for costly data transfers between them. This approach leverages Apple Silicon's architecture where both CPU and GPU access the same physical memory. The framework uses lazy computation and just-in-time compilation to optimize memory usage and performance.

Q: What quantization options does MLX support?
A: MLX supports several quantization options including FP16 (half-precision floating point), INT8 (8-bit integer), and INT4 (4-bit integer) quantization. These options provide different trade-offs between model size, inference speed, and accuracy. INT4 quantization offers the most significant size reduction (approximately 75%) with moderate quality impact.
""")
    
    # Step 2: Convert text to JSONL format
    logger.info("Step 2: Converting text to JSONL format")
    jsonl_path = os.path.join(dataset_dir, "mlx_sample_data.jsonl")
    
    success, message = convert_text_to_jsonl(
        input_file=sample_data_path,
        output_file=jsonl_path,
        format_type="qa"
    )
    
    if not success:
        logger.error(f"Failed to convert text to JSONL: {message}")
        return 1
    
    logger.info(message)
    
    # Step 3: Split the dataset
    logger.info("Step 3: Splitting the dataset")
    
    success, message = split_dataset(
        input_file=jsonl_path,
        output_dir=dataset_dir,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        shuffle=True,
        seed=42
    )
    
    if not success:
        logger.error(f"Failed to split dataset: {message}")
        return 1
    
    logger.info(message)
    
    # Step 4: Run LoRA fine-tuning
    logger.info("Step 4: Running LoRA fine-tuning")
    
    success, message = finetune_lora(
        model_name=model_name,
        train_data_path=os.path.join(dataset_dir, "train.jsonl"),
        output_dir=output_dir,
        val_data_path=os.path.join(dataset_dir, "val.jsonl"),
        lora_rank=8,
        lora_alpha=16,
        epochs=3,
        batch_size=1,
        learning_rate=3e-4,
        quantization="int4"  # Use quantization for memory efficiency
    )
    
    if not success:
        logger.error(f"Failed to run LoRA fine-tuning: {message}")
        return 1
    
    logger.info(message)
    
    # Step 5: Test the fine-tuned model
    logger.info("Step 5: Testing the fine-tuned model")
    
    prompt = "What are the best practices for deploying MLX models on Mac?"
    
    success, output = apply_lora_adapter(
        model_name=model_name,
        lora_path=os.path.join(output_dir, "final"),
        prompt=prompt,
        max_tokens=256,
        temperature=0.7,
        quantization="int4"
    )
    
    if not success:
        logger.error(f"Failed to run inference: {output}")
        return 1
    
    logger.info(f"Generated response:\n{output}")
    
    logger.info("=======================================")
    logger.info("Workflow completed successfully")
    logger.info(f"Fine-tuned LoRA adapter: {os.path.join(output_dir, 'final')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Example workflow for MLX QLoRA fine-tuning on a low-resource device.
"""

import os
import sys
import subprocess
import logging
import json

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preparation.text_to_jsonl import convert_text_to_jsonl
from data_preparation.format_dataset import format_instruction_dataset
from data_preparation.dataset_splitting import split_dataset
from mlx.qlora_finetune import finetune_qlora, apply_qlora_adapter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Configuration
    model_name = "llama-2-7b"
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "mlx_qlora")
    
    # Create directories
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting MLX QLoRA fine-tuning workflow for low-resource devices")
    logger.info("==============================================================")
    
    # Step 1: Prepare a sample dataset for Mac hardware support
    logger.info("Step 1: Preparing sample dataset")
    sample_data_path = os.path.join(dataset_dir, "mac_support_data.txt")
    
    with open(sample_data_path, 'w') as f:
        f.write("""
Q: What Mac hardware is best for running LLMs locally?
A: For running LLMs locally on Mac, M1/M2/M3 MacBook Pro or Mac Studio with at least 16GB RAM is recommended. The MacBook Pro M2 Pro/Max with 32GB RAM can run 7B-13B models at good speeds. Mac Studio with M1/M2 Ultra and 64GB+ RAM is ideal for larger models (33B-70B). More RAM allows using less aggressive quantization for better quality. For professional workloads, look for M2 Max/Ultra with 64-128GB RAM.

Q: How much RAM do I need to run a 7B parameter model on Mac?
A: For a 7B parameter model on Mac, you need at minimum 8GB RAM using INT4 quantization, though performance will be limited. 16GB RAM is recommended for comfortable usage with INT8 quantization. With 32GB RAM, you can run 7B models with minimal quantization (FP16) for best quality or run multiple models simultaneously. Memory requirements scale with context length, so longer contexts require more RAM.

Q: What quantization settings work best on M1 Macs with 16GB RAM?
A: On M1 Macs with 16GB RAM, INT8/Q8_0 quantization works well for 7B models, offering a good balance between quality and memory usage (approximately 50% size reduction). For 13B models, use INT4/Q4_K quantization (approximately 75% size reduction). Avoid running larger models (33B+) on 16GB systems. For maximum quality on 7B models with 16GB RAM, use Q6_K quantization with a context length of 2048 tokens or less.

Q: How can I optimize performance when running LLMs on MacBook Air?
A: To optimize LLM performance on MacBook Air: 1) Use INT4/Q4_K quantization for 7B models, 2) Keep context length under 2048 tokens, 3) Enable Metal acceleration, 4) Close other applications to free up memory, 5) Use the laptop while plugged in to prevent thermal throttling, 6) Set temperature parameters lower (0.1-0.5) for faster generation, 7) Consider batch processing rather than real-time chat for better efficiency, and 8) Use a cooling pad for longer sessions.

Q: What's the largest model I can run on a Mac Mini M2 with 8GB RAM?
A: On a Mac Mini M2 with 8GB RAM, the largest model you can practically run is a 7B parameter model (like Llama 2 7B) with INT4/Q4_K quantization. You'll need to keep context lengths short (1024-2048 tokens) and use Metal acceleration. While technically you might be able to load a heavily quantized 13B model (with Q2_K), performance would be poor due to excessive memory swapping. For optimal experience with 8GB RAM, stick with smaller 7B models and consider using smaller specialized models like Phi-2 (2.7B) or TinyLlama (1.1B).
""")
    
    # Step 2: Convert text to JSONL format
    logger.info("Step 2: Converting text to JSONL format")
    jsonl_path = os.path.join(dataset_dir, "mac_support_data.jsonl")
    
    success, message = convert_text_to_jsonl(
        input_file=sample_data_path,
        output_file=jsonl_path,
        format_type="qa"
    )
    
    if not success:
        logger.error(f"Failed to convert text to JSONL: {message}")
        return 1
    
    logger.info(message)
    
    # Step 3: Format data into instruction format
    logger.info("Step 3: Formatting data into instruction format")
    instruction_path = os.path.join(dataset_dir, "mac_support_instructions.jsonl")
    
    success, message = format_instruction_dataset(
        input_file=jsonl_path,
        output_file=instruction_path,
        instruction_template="### Instruction:\n{instruction}\n\n### Response:\n{response}"
    )
    
    if not success:
        logger.error(f"Failed to format data: {message}")
        return 1
    
    logger.info(message)
    
    # Step 4: Split the dataset
    logger.info("Step 4: Splitting the dataset")
    
    success, message = split_dataset(
        input_file=instruction_path,
        output_dir=dataset_dir,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        shuffle=True,
        seed=42,
        train_filename="qlora_train.jsonl",
        val_filename="qlora_val.jsonl"
    )
    
    if not success:
        logger.error(f"Failed to split dataset: {message}")
        return 1
    
    logger.info(message)
    
    # Step 5: Run QLoRA fine-tuning with INT4 quantization
    logger.info("Step 5: Running QLoRA fine-tuning with INT4 quantization")
    
    success, message = finetune_qlora(
        model_name=model_name,
        train_data_path=os.path.join(dataset_dir, "qlora_train.jsonl"),
        output_dir=output_dir,
        val_data_path=os.path.join(dataset_dir, "qlora_val.jsonl"),
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        quantization="int4",  # INT4 quantization for maximum memory efficiency
        epochs=3,
        batch_size=1,
        learning_rate=5e-4,
        max_length=512,
        prompt_key="instruction",
        response_key="output"
    )
    
    if not success:
        logger.error(f"Failed to run QLoRA fine-tuning: {message}")
        return 1
    
    logger.info(message)
    
    # Step 6: Test the fine-tuned model
    logger.info("Step 6: Testing the fine-tuned model")
    
    prompt = "I have a MacBook Air M1 with 8GB RAM. What's the best way to run Mixtral 8x7B on it?"
    
    success, output = apply_qlora_adapter(
        model_name=model_name,
        qlora_path=os.path.join(output_dir, "final"),
        prompt=prompt,
        max_tokens=512,
        temperature=0.7
    )
    
    if not success:
        logger.error(f"Failed to run inference: {output}")
        return 1
    
    logger.info(f"Generated response:\n{output}")
    
    # Step 7: Show memory usage comparison
    logger.info("Step 7: Memory usage comparison")
    
    memory_comparison = """
Memory Usage Comparison:
------------------------
Full Fine-tuning (7B model):      ~28GB RAM
LoRA Fine-tuning (7B model):      ~14GB RAM
QLoRA Fine-tuning (INT8, 7B):     ~10GB RAM
QLoRA Fine-tuning (INT4, 7B):     ~6GB RAM

The QLoRA approach with INT4 quantization makes fine-tuning possible
even on lower-end Mac hardware with 8GB RAM, while preserving most
of the quality benefits of fine-tuning.
"""
    logger.info(memory_comparison)
    
    logger.info("==============================================================")
    logger.info("Workflow completed successfully")
    logger.info(f"Fine-tuned QLoRA adapter: {os.path.join(output_dir, 'final')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
LoRA fine-tuning implementation for llama.cpp.
"""

import os
import argparse
import subprocess
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_finetune_args(args: Dict[str, Any]) -> List[str]:
    """
    Prepare command-line arguments for llama-finetune.
    
    Args:
        args: Dictionary of fine-tuning arguments
        
    Returns:
        List of command-line arguments for the llama-finetune binary
    """
    cmd_args = []
    
    # Required arguments
    cmd_args.extend(["--model-base", args["model_path"]])
    cmd_args.extend(["--lora-out", args["output_path"]])
    
    # Data arguments
    if "data_train" in args:
        cmd_args.extend(["--data-train", args["data_train"]])
    if "data_val" in args:
        cmd_args.extend(["--data-val", args["data_val"]])
    
    # LoRA parameters
    if "lora_rank" in args:
        cmd_args.extend(["--lora-rank", str(args["lora_rank"])])
    if "lora_alpha" in args:
        cmd_args.extend(["--lora-alpha", str(args["lora_alpha"])])
    if "lora_dropout" in args:
        cmd_args.extend(["--lora-dropout", str(args["lora_dropout"])])
    if "lora_layers" in args:
        cmd_args.extend(["--lora-layers", args["lora_layers"]])
    
    # Training parameters
    if "batch_size" in args:
        cmd_args.extend(["--batch-size", str(args["batch_size"])])
    if "epochs" in args:
        cmd_args.extend(["--epochs", str(args["epochs"])])
    if "learning_rate" in args:
        cmd_args.extend(["--learning-rate", str(args["learning_rate"])])
    if "warmup_steps" in args:
        cmd_args.extend(["--warmup-steps", str(args["warmup_steps"])])
    if "weight_decay" in args:
        cmd_args.extend(["--weight-decay", str(args["weight_decay"])])
    
    # Context and thread settings
    if "ctx_len" in args:
        cmd_args.extend(["--ctx-len", str(args["ctx_len"])])
    if "threads" in args:
        cmd_args.extend(["--threads", str(args["threads"])])
    
    # Metal acceleration
    if args.get("use_metal", True):
        cmd_args.append("--metal")
    
    return cmd_args

def finetune_lora(
    llama_finetune_path: str,
    model_path: str,
    data_train_path: str,
    output_path: str,
    data_val_path: Optional[str] = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_layers: str = "all",
    batch_size: int = 1,
    epochs: int = 3,
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    ctx_len: int = 512,
    threads: int = 4,
    use_metal: bool = True,
    seed: int = 42,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Fine-tune a model using LoRA with llama.cpp.
    
    Args:
        llama_finetune_path: Path to the llama-finetune binary
        model_path: Path to the base model (GGUF format)
        data_train_path: Path to the training data (JSONL format)
        output_path: Path to save the LoRA adapter weights
        data_val_path: Path to the validation data (JSONL format)
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        lora_layers: Which layers to apply LoRA to ("all" or comma-separated layer indices)
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay value
        ctx_len: Context length for training
        threads: Number of threads to use
        use_metal: Whether to use Metal acceleration (Apple Silicon)
        seed: Random seed for reproducibility
        verbose: Whether to show verbose output
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Verify the model path exists
    if not os.path.exists(model_path):
        return False, f"Model path does not exist: {model_path}"
    
    # Verify the training data exists
    if not os.path.exists(data_train_path):
        return False, f"Training data path does not exist: {data_train_path}"
    
    # Verify the validation data if provided
    if data_val_path and not os.path.exists(data_val_path):
        return False, f"Validation data path does not exist: {data_val_path}"
    
    # Verify the llama-finetune binary exists
    if not os.path.exists(llama_finetune_path):
        return False, f"llama-finetune binary not found at: {llama_finetune_path}"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare the command arguments
    args = {
        "model_path": model_path,
        "output_path": output_path,
        "data_train": data_train_path,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_layers": lora_layers,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "ctx_len": ctx_len,
        "threads": threads,
        "use_metal": use_metal
    }
    
    if data_val_path:
        args["data_val"] = data_val_path
    
    cmd_args = prepare_finetune_args(args)
    
    # Add seed if provided
    if seed is not None:
        cmd_args.extend(["--seed", str(seed)])
    
    # Run the fine-tuning command
    cmd = [llama_finetune_path] + cmd_args
    
    try:
        logger.info(f"Running fine-tuning command: {' '.join(cmd)}")
        
        # Set verbose output if requested
        stdout = None if verbose else subprocess.PIPE
        stderr = None if verbose else subprocess.PIPE
        
        process = subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True)
        
        if process.returncode != 0:
            error_msg = process.stderr if process.stderr else "Unknown error"
            return False, f"Fine-tuning failed with return code {process.returncode}: {error_msg}"
        
        return True, f"Fine-tuning completed successfully. LoRA adapter saved to {output_path}"
        
    except Exception as e:
        return False, f"Fine-tuning failed with exception: {str(e)}"

def apply_lora_adapter(
    llama_main_path: str,
    model_path: str,
    lora_path: str,
    prompt: str,
    output_tokens: int = 512,
    ctx_len: int = 2048,
    use_metal: bool = True,
    temp: float = 0.7,
    top_p: float = 0.9,
    seed: int = 42,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Apply a trained LoRA adapter to a model and run inference.
    
    Args:
        llama_main_path: Path to the llama main binary
        model_path: Path to the base model (GGUF format)
        lora_path: Path to the LoRA adapter weights
        prompt: Text prompt for generation
        output_tokens: Number of tokens to generate
        ctx_len: Context length for inference
        use_metal: Whether to use Metal acceleration (Apple Silicon)
        temp: Temperature for sampling
        top_p: Top-p sampling value
        seed: Random seed for reproducibility
        verbose: Whether to show verbose output
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Verify the model path exists
    if not os.path.exists(model_path):
        return False, f"Model path does not exist: {model_path}"
    
    # Verify the LoRA adapter exists
    if not os.path.exists(lora_path):
        return False, f"LoRA adapter path does not exist: {lora_path}"
    
    # Verify the llama binary exists
    if not os.path.exists(llama_main_path):
        return False, f"llama binary not found at: {llama_main_path}"
    
    # Prepare the command
    cmd = [
        llama_main_path,
        "-m", model_path,
        "--lora", lora_path,
        "-n", str(output_tokens),
        "-c", str(ctx_len),
        "--temp", str(temp),
        "--top_p", str(top_p),
        "--seed", str(seed),
        "-p", prompt
    ]
    
    if use_metal:
        cmd.append("--metal")
    
    try:
        logger.info(f"Running inference command: {' '.join(cmd)}")
        
        # Set verbose output if requested
        stdout = None if verbose else subprocess.PIPE
        stderr = None if verbose else subprocess.PIPE
        
        process = subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True)
        
        if process.returncode != 0:
            error_msg = process.stderr if process.stderr else "Unknown error"
            return False, f"Inference failed with return code {process.returncode}: {error_msg}"
        
        output = process.stdout if process.stdout else "No output"
        return True, output
        
    except Exception as e:
        return False, f"Inference failed with exception: {str(e)}"
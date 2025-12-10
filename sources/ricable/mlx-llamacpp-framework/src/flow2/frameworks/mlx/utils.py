"""
Utility functions for MLX fine-tuning.
"""

import os
import json
import random
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import mlx.core as mx

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)

def load_jsonl_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    Load a dataset from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the dataset examples
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    example = json.loads(line)
                    data.append(example)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line}")
    
    return data

def prepare_dataset(
    examples: List[Dict[str, str]],
    tokenizer: Any,
    max_length: int = 512,
    prompt_key: str = "prompt",
    response_key: str = "response"
) -> Tuple[mx.array, mx.array]:
    """
    Prepare a dataset for training.
    
    Args:
        examples: List of dictionaries containing prompts and responses
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        prompt_key: Key for prompt in the examples dictionary
        response_key: Key for response in the examples dictionary
        
    Returns:
        Tuple of (input_ids, labels)
    """
    input_ids = []
    labels = []
    
    for example in examples:
        if prompt_key not in example or response_key not in example:
            logger.warning(f"Skipping example missing keys: {example}")
            continue
        
        prompt = example[prompt_key]
        response = example[response_key]
        
        # Tokenize prompt and response
        prompt_ids = tokenizer.encode(prompt)
        response_ids = tokenizer.encode(response)
        
        # Create full sequence
        full_ids = prompt_ids + response_ids
        
        # Truncate if needed
        if len(full_ids) > max_length:
            logger.warning(f"Truncating sequence from {len(full_ids)} to {max_length} tokens")
            full_ids = full_ids[:max_length]
        
        # Create labels: -100 for prompt tokens (don't compute loss), actual ids for response tokens
        example_labels = [-100] * len(prompt_ids) + response_ids
        
        # Truncate labels if needed
        if len(example_labels) > max_length:
            example_labels = example_labels[:max_length]
        
        input_ids.append(full_ids)
        labels.append(example_labels)
    
    # Pad sequences to max length
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_labels = []
    
    for ids, lbs in zip(input_ids, labels):
        # Pad with tokenizer.pad_id
        padded_ids = ids + [tokenizer.pad_id] * (max_len - len(ids))
        padded_lbs = lbs + [-100] * (max_len - len(lbs))
        
        padded_input_ids.append(padded_ids)
        padded_labels.append(padded_lbs)
    
    return mx.array(padded_input_ids), mx.array(padded_labels)

def create_optimizer(
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
):
    """
    Create an optimizer for fine-tuning.
    
    Args:
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer_type: Optimizer type (adamw, sgd)
        beta1: Beta1 for AdamW
        beta2: Beta2 for AdamW
        eps: Epsilon for AdamW
        
    Returns:
        Optimizer
    """
    import mlx.optimizers as optim
    
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            eps=eps
        )
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def create_scheduler(
    optimizer: Any,
    scheduler_type: str = "linear",
    warmup_steps: int = 100,
    total_steps: int = 1000,
    min_lr_ratio: float = 0.1
):
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Scheduler type (linear, cosine)
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate ratio
        
    Returns:
        Scheduler
    """
    import mlx.optimizers as optim
    
    if scheduler_type.lower() == "linear":
        return optim.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=total_steps,
            warmup_iters=warmup_steps
        )
    elif scheduler_type.lower() == "cosine":
        return optim.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=optimizer.learning_rate * min_lr_ratio
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def compute_loss(logits, targets):
    """
    Compute cross entropy loss.
    
    Args:
        logits: Model logits of shape (batch_size, seq_len, vocab_size)
        targets: Target labels of shape (batch_size, seq_len)
        
    Returns:
        Loss value
    """
    import mlx.nn as nn
    
    # Reshape logits to (batch_size * seq_len, vocab_size)
    logits = logits.reshape(-1, logits.shape[-1])
    
    # Reshape targets to (batch_size * seq_len)
    targets = targets.reshape(-1)
    
    # Create a mask for non-padding and non-ignored tokens
    mask = targets != -100
    
    # Apply mask to logits and targets
    logits = logits[mask]
    targets = targets[mask]
    
    # Compute cross entropy loss
    return nn.losses.cross_entropy(logits, targets)

def create_data_loader(
    input_ids: mx.array,
    labels: mx.array,
    batch_size: int = 1,
    shuffle: bool = True
):
    """
    Create a data loader for training.
    
    Args:
        input_ids: Input token IDs
        labels: Target labels
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        Generator yielding batches
    """
    num_examples = input_ids.shape[0]
    indices = np.arange(num_examples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield input_ids[batch_indices], labels[batch_indices]
"""
Full Fine-tuning Implementation for HuggingFace
===============================================

Complete parameter fine-tuning with Accelerate integration.
"""

import os
import json
import torch
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback
    )
    from datasets import Dataset, load_dataset
    from accelerate import Accelerator
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("Required libraries not available for full fine-tuning")

from ..utils import setup_mps_device, get_optimal_dtype, cleanup_memory

@dataclass
class TrainingConfig:
    """Full fine-tuning configuration."""
    output_dir: str
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 1000
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    remove_unused_columns: bool = False
    dataloader_pin_memory: bool = False
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    report_to: Optional[str] = None
    max_grad_norm: float = 1.0
    
    def to_training_args(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(**asdict(self))

def create_training_config(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    **kwargs
) -> TrainingConfig:
    """
    Create training configuration for full fine-tuning.
    
    Args:
        output_dir: Output directory
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        **kwargs: Additional training arguments
        
    Returns:
        Training configuration
    """
    return TrainingConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs
    )

def prepare_dataset(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    train_split: str = "train",
    eval_split: Optional[str] = "validation"
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Prepare dataset for full fine-tuning.
    
    Args:
        dataset_path: Path to dataset
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        train_split: Training split name
        eval_split: Evaluation split name
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    if not HF_AVAILABLE:
        raise ImportError("Required libraries not available")
    
    # Load dataset
    if os.path.exists(dataset_path):
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            dataset = load_dataset(dataset_path)
    else:
        dataset = load_dataset(dataset_path)
    
    # Split dataset
    if isinstance(dataset, dict):
        train_dataset = dataset[train_split]
        eval_dataset = dataset.get(eval_split) if eval_split else None
    else:
        if eval_split and hasattr(dataset, train_split):
            train_dataset = dataset[train_split]
            eval_dataset = dataset[eval_split] if eval_split in dataset else None
        else:
            split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
    
    def tokenize_function(examples):
        """Tokenize text examples."""
        if 'text' in examples:
            texts = examples['text']
        elif 'input' in examples and 'output' in examples:
            texts = [f"### Input:\n{inp}\n\n### Output:\n{out}" 
                    for inp, out in zip(examples['input'], examples['output'])]
        elif 'prompt' in examples and 'completion' in examples:
            texts = [f"{prompt}{completion}" 
                    for prompt, completion in zip(examples['prompt'], examples['completion'])]
        else:
            raise ValueError("Dataset must contain appropriate text fields")
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_overflowing_tokens=False,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    
    eval_dataset_tokenized = None
    if eval_dataset:
        eval_dataset_tokenized = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval dataset"
        )
    
    return train_dataset, eval_dataset_tokenized

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    import numpy as np
    
    predictions, labels = eval_pred
    
    # For causal LM, we typically just use perplexity
    # This is a simplified metric computation
    
    # Shift predictions and labels for causal LM
    shift_preds = predictions[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate perplexity (simplified)
    loss = torch.nn.functional.cross_entropy(
        shift_preds.view(-1, shift_preds.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
    
    perplexity = torch.exp(loss)
    
    return {
        "perplexity": perplexity.item(),
        "loss": loss.item()
    }

def finetune_full(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    training_config: Optional[TrainingConfig] = None,
    max_length: int = 512,
    device: Optional[torch.device] = None,
    use_accelerate: bool = True
) -> str:
    """
    Perform full parameter fine-tuning.
    
    Args:
        model_name: Model name or path
        dataset_path: Dataset path
        output_dir: Output directory
        training_config: Training configuration
        max_length: Maximum sequence length
        device: Target device
        use_accelerate: Use Accelerate for training
        
    Returns:
        Path to saved model
    """
    if not HF_AVAILABLE:
        raise ImportError("Required libraries not available")
    
    print(f"Starting full fine-tuning of {model_name}")
    
    # Setup device
    if device is None:
        device = setup_mps_device()
    
    # Create configurations
    if training_config is None:
        training_config = create_training_config(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=get_optimal_dtype(device),
        device_map="auto" if device.type != "cpu" else None
    )
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Prepare dataset
    print("Preparing dataset...")
    train_dataset, eval_dataset = prepare_dataset(
        dataset_path, tokenizer, max_length
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if device.type != "mps" else None
    )
    
    # Setup training arguments
    training_args = training_config.to_training_args()
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_dataset else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else None
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Save training config
    config_path = os.path.join(model_path, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "training_config": asdict(training_config),
            "model_name": model_name,
            "dataset_path": dataset_path,
            "max_length": max_length,
            "num_parameters": num_params,
            "trainable_parameters": trainable_params
        }, f, indent=2)
    
    print(f"Fine-tuned model saved to: {model_path}")
    cleanup_memory()
    
    return model_path
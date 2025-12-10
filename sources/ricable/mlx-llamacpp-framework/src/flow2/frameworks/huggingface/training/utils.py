"""
HuggingFace Training Utilities
=============================

Utility functions for HuggingFace training with Accelerate integration.
"""

import os
import torch
import warnings
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

try:
    from accelerate import Accelerator
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup
    )
    from torch.optim import AdamW, Adam
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("Required libraries not available for training utilities")

def setup_accelerator(
    mixed_precision: str = "fp16",
    gradient_accumulation_steps: int = 1,
    device_placement: bool = True,
    split_batches: bool = False
) -> Optional[Accelerator]:
    """
    Setup Accelerate for distributed training.
    
    Args:
        mixed_precision: Mixed precision mode ("no", "fp16", "bf16")
        gradient_accumulation_steps: Gradient accumulation steps
        device_placement: Enable automatic device placement
        split_batches: Split batches across devices
        
    Returns:
        Accelerator instance or None if not available
    """
    if not HF_AVAILABLE:
        warnings.warn("Accelerate not available")
        return None
    
    try:
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device_placement=device_placement,
            split_batches=split_batches
        )
        
        print(f"Accelerator setup:")
        print(f"  Device: {accelerator.device}")
        print(f"  Num processes: {accelerator.num_processes}")
        print(f"  Mixed precision: {mixed_precision}")
        
        return accelerator
        
    except Exception as e:
        warnings.warn(f"Failed to setup Accelerator: {e}")
        return None

def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    optimizer_type: str = "adamw",
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer for training.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer_type: Optimizer type ("adamw", "adam")
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    # Get parameters that require gradients
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    
    # Create parameter groups with different weight decay for different layers
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    
    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            **kwargs
        )
    elif optimizer_type.lower() == "adam":
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    print(f"Created {optimizer_type} optimizer with lr={learning_rate}, wd={weight_decay}")
    return optimizer

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    scheduler_type: str = "linear",
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps
        scheduler_type: Scheduler type ("linear", "cosine")
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type.lower() == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    elif scheduler_type.lower() == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    print(f"Created {scheduler_type} scheduler with {num_warmup_steps} warmup steps")
    return scheduler

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    loss: float,
    checkpoint_dir: str,
    accelerator: Optional[Accelerator] = None
) -> str:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current training step
        loss: Current loss
        checkpoint_dir: Checkpoint directory
        accelerator: Accelerator instance
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    if accelerator is not None:
        # Use accelerator to save
        accelerator.save_state(checkpoint_path)
        
        # Save additional metadata
        if accelerator.is_main_process:
            metadata = {
                "step": step,
                "loss": loss,
                "learning_rate": scheduler.get_last_lr()[0] if scheduler else None
            }
            
            import json
            with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
    else:
        # Manual save
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'step': step,
            'loss': loss,
        }, os.path.join(checkpoint_path, "pytorch_model.bin"))
    
    print(f"Checkpoint saved to: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str,
    accelerator: Optional[Accelerator] = None
) -> Tuple[int, float]:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        scheduler: Scheduler to load into
        checkpoint_path: Checkpoint path
        accelerator: Accelerator instance
        
    Returns:
        Tuple of (step, loss)
    """
    if accelerator is not None:
        # Use accelerator to load
        accelerator.load_state(checkpoint_path)
        
        # Load metadata
        import json
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            step = metadata.get("step", 0)
            loss = metadata.get("loss", 0.0)
        else:
            step, loss = 0, 0.0
    else:
        # Manual load
        checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        step = checkpoint.get('step', 0)
        loss = checkpoint.get('loss', 0.0)
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"Resuming from step {step} with loss {loss:.4f}")
    
    return step, loss

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_percent": (trainable_params / total_params) * 100 if total_params > 0 else 0
    }

def print_training_info(
    model: torch.nn.Module,
    dataset_size: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    gradient_accumulation_steps: int = 1
):
    """Print comprehensive training information."""
    param_info = count_parameters(model)
    
    steps_per_epoch = dataset_size // (batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    
    print("\n" + "="*50)
    print("🎯 TRAINING CONFIGURATION")
    print("="*50)
    print(f"Model Parameters:")
    print(f"  Total: {param_info['total']:,}")
    print(f"  Trainable: {param_info['trainable']:,} ({param_info['trainable_percent']:.1f}%)")
    print(f"  Frozen: {param_info['frozen']:,}")
    
    print(f"\nTraining Setup:")
    print(f"  Dataset size: {dataset_size:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print("="*50)
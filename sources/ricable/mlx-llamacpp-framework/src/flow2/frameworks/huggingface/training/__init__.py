"""
HuggingFace Training Module
==========================

Comprehensive training utilities with Accelerate integration:
- LoRA (Low-Rank Adaptation) fine-tuning
- QLoRA (Quantized LoRA) fine-tuning  
- Full parameter fine-tuning
- Distributed training support
- MPS acceleration for Apple Silicon
- Advanced optimization strategies
"""

from .lora_finetune import (
    finetune_lora,
    finetune_qlora,
    merge_lora_adapter,
    create_lora_config
)

from .full_finetune import (
    finetune_full,
    prepare_dataset,
    compute_metrics,
    create_training_config
)

from .utils import (
    setup_accelerator,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    load_checkpoint
)

__all__ = [
    # LoRA training
    "finetune_lora",
    "finetune_qlora", 
    "merge_lora_adapter",
    "create_lora_config",
    # Full fine-tuning
    "finetune_full",
    "prepare_dataset",
    "compute_metrics", 
    "create_training_config",
    # Training utilities
    "setup_accelerator",
    "create_optimizer",
    "create_scheduler",
    "save_checkpoint",
    "load_checkpoint"
]
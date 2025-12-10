"""
MLX Framework Integration
=========================

Complete MLX framework support including training, inference, and quantization
with Flash Attention optimization.

Modules:
    training: LoRA, QLoRA, and full fine-tuning implementations
    inference: Model inference and generation utilities
    quantization: Model quantization utilities
    utils: MLX-specific utility functions
"""

# Check MLX availability
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

if MLX_AVAILABLE:
    from .training.lora_finetune import finetune_lora, apply_lora_adapter
    from .training.qlora_finetune import finetune_qlora, apply_qlora_adapter  
    from .training.full_finetune import finetune_full
    from .inference.inference import load_mlx_model, generate_completion, chat_completion
    from .quantization.quantize import quantize_model, batch_quantize_models

__all__ = [
    "MLX_AVAILABLE",
    "finetune_lora",
    "apply_lora_adapter", 
    "finetune_qlora",
    "apply_qlora_adapter",
    "finetune_full",
    "load_mlx_model",
    "generate_completion", 
    "chat_completion",
    "quantize_model",
    "batch_quantize_models",
] if MLX_AVAILABLE else ["MLX_AVAILABLE"]
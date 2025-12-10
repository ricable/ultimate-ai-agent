"""
LlamaCpp Framework Integration
=============================

Complete LlamaCpp framework support including training, inference, and quantization.

Modules:
    training: LoRA and fine-tuning implementations
    inference: Model inference and generation utilities  
    quantization: Model quantization utilities
"""

# Check LlamaCpp availability
try:
    import llama_cpp
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

if LLAMACPP_AVAILABLE:
    from .training.lora import finetune_lora, apply_lora_adapter
    from .inference.inference import create_llama_model, generate_completion, chat_completion
    from .quantization.quantize import quantize_model, batch_quantize_models

__all__ = [
    "LLAMACPP_AVAILABLE",
    "finetune_lora",
    "apply_lora_adapter",
    "create_llama_model", 
    "generate_completion",
    "chat_completion",
    "quantize_model",
    "batch_quantize_models",
] if LLAMACPP_AVAILABLE else ["LLAMACPP_AVAILABLE"]
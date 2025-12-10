"""
HuggingFace Transformers Framework Implementation
================================================

This module provides comprehensive HuggingFace transformers support with:
- MPS (Metal Performance Shaders) acceleration for Apple Silicon
- Text Generation Inference (TGI) integration
- Accelerate library optimization
- Advanced quantization (4-bit, 8-bit, GPTQ, AWQ)
- LoRA/QLoRA fine-tuning with accelerate
- Multi-GPU and distributed training support

Features:
- MPS backend optimization for M1/M2/M3/M4 Macs
- TGI server integration for production inference
- BitsAndBytes quantization support
- PEFT (Parameter Efficient Fine-Tuning) integration
- Streaming text generation
- Chat template support
"""

import os
import sys
import warnings
from typing import Optional, Dict, Any, List, Union

# Framework availability detection
HF_AVAILABLE = False
ACCELERATE_AVAILABLE = False
TGI_AVAILABLE = False
MPS_AVAILABLE = False
QUANTIZATION_AVAILABLE = False

try:
    import torch
    import transformers
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        AutoConfig,
        TextStreamer,
        pipeline
    )
    HF_AVAILABLE = True
    
    # Check MPS availability
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        MPS_AVAILABLE = True
    
except ImportError as e:
    warnings.warn(f"HuggingFace transformers not available: {e}")

try:
    import accelerate
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    warnings.warn("Accelerate library not available")

try:
    from text_generation import Client
    TGI_AVAILABLE = True
except ImportError:
    warnings.warn("Text Generation Inference client not available")

try:
    import bitsandbytes
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    warnings.warn("BitsAndBytes quantization not available")

# Conditional imports based on availability
if HF_AVAILABLE:
    from .inference.inference import (
        load_hf_model,
        generate_completion,
        chat_completion,
        streaming_completion,
        batch_generate
    )
    from .training.lora_finetune import (
        finetune_lora,
        finetune_qlora,
        merge_lora_adapter,
        create_lora_config,
        TrainingConfig
    )
    from .training.full_finetune import (
        finetune_full,
        prepare_dataset,
        compute_metrics,
        create_training_config
    )
    from .quantization.quantize import (
        quantize_model,
        load_quantized_model,
        benchmark_quantization,
        batch_quantize_models,
        QuantizationMethod,
        create_quantization_config
    )
    from .utils import (
        get_model_info,
        estimate_memory_usage,
        optimize_for_inference,
        setup_mps_device,
        cleanup_memory
    )
    from .inference.inference import GenerationParams

# Export framework status
__all__ = [
    # Framework status
    "HF_AVAILABLE",
    "ACCELERATE_AVAILABLE", 
    "TGI_AVAILABLE",
    "MPS_AVAILABLE",
    "QUANTIZATION_AVAILABLE",
]

# Add conditional exports
if HF_AVAILABLE:
    __all__.extend([
        # Inference
        "load_hf_model",
        "generate_completion", 
        "chat_completion",
        "streaming_completion",
        "batch_generate",
        # Training
        "finetune_lora",
        "finetune_qlora", 
        "merge_lora_adapter",
        "create_lora_config",
        "TrainingConfig",
        "finetune_full",
        "prepare_dataset",
        "compute_metrics",
        "create_training_config",
        # Quantization
        "quantize_model",
        "load_quantized_model",
        "benchmark_quantization",
        "batch_quantize_models",
        "QuantizationMethod",
        "create_quantization_config",
        # Utils
        "get_model_info",
        "estimate_memory_usage",
        "optimize_for_inference",
        "setup_mps_device",
        "cleanup_memory",
        "GenerationParams",
    ])

def get_framework_info() -> Dict[str, Any]:
    """Get comprehensive framework information."""
    info = {
        "framework": "huggingface",
        "available": HF_AVAILABLE,
        "accelerate_available": ACCELERATE_AVAILABLE,
        "tgi_available": TGI_AVAILABLE,
        "mps_available": MPS_AVAILABLE,
        "quantization_available": QUANTIZATION_AVAILABLE,
        "device_info": {},
        "memory_info": {}
    }
    
    if HF_AVAILABLE:
        import torch
        info["torch_version"] = torch.__version__
        info["transformers_version"] = transformers.__version__
        
        # Device information
        if torch.cuda.is_available():
            info["device_info"]["cuda"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name()
            }
        
        if MPS_AVAILABLE:
            info["device_info"]["mps"] = {
                "available": True,
                "device": "mps"
            }
        
        # Memory information
        if torch.cuda.is_available():
            info["memory_info"]["cuda"] = {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved()
            }
    
    if ACCELERATE_AVAILABLE:
        info["accelerate_version"] = accelerate.__version__
    
    return info

def print_framework_status():
    """Print comprehensive framework status."""
    info = get_framework_info()
    
    print("🤗 HuggingFace Framework Status")
    print("=" * 40)
    print(f"HuggingFace Available: {info['available']}")
    print(f"Accelerate Available: {info['accelerate_available']}")
    print(f"TGI Available: {info['tgi_available']}")
    print(f"MPS Available: {info['mps_available']}")
    print(f"Quantization Available: {info['quantization_available']}")
    
    if info['available']:
        print(f"PyTorch Version: {info.get('torch_version', 'Unknown')}")
        print(f"Transformers Version: {info.get('transformers_version', 'Unknown')}")
        
        if info['accelerate_available']:
            print(f"Accelerate Version: {info.get('accelerate_version', 'Unknown')}")
        
        if info['device_info'].get('mps', {}).get('available'):
            print("🚀 MPS (Metal) acceleration enabled!")
        
        if info['device_info'].get('cuda', {}).get('available'):
            cuda_info = info['device_info']['cuda']
            print(f"🚀 CUDA available: {cuda_info['device_count']} device(s)")
            print(f"   Device: {cuda_info['device_name']}")
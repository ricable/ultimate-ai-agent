"""
HuggingFace Framework Utilities
==============================

Utility functions for HuggingFace transformers framework including:
- MPS device setup and optimization
- Memory estimation and optimization
- Model information and analysis
- Performance monitoring
"""

import os
import gc
import torch
import psutil
import warnings
from typing import Dict, Any, Optional, List, Tuple, Union
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

def setup_mps_device() -> torch.device:
    """
    Setup and optimize MPS device for Apple Silicon.
    
    Returns:
        torch.device: The optimal device (mps, cuda, or cpu)
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("🚀 MPS (Metal Performance Shaders) available - using Apple Silicon GPU")
        device = torch.device("mps")
        
        # Set MPS optimization flags
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        return device
    elif torch.cuda.is_available():
        print("🚀 CUDA available - using NVIDIA GPU")
        return torch.device("cuda")
    else:
        print("⚠️  Using CPU - consider using a GPU for better performance")
        return torch.device("cpu")

def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """
    Get optimal dtype based on device capabilities.
    
    Args:
        device: Target device
        
    Returns:
        torch.dtype: Optimal data type
    """
    if device.type == "mps":
        # MPS works best with float16 for memory efficiency
        return torch.float16
    elif device.type == "cuda":
        # CUDA supports bfloat16 on newer cards
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    else:
        # CPU typically uses float32
        return torch.float32

def estimate_memory_usage(
    model_name: str,
    batch_size: int = 1,
    sequence_length: int = 512,
    dtype: Optional[torch.dtype] = None
) -> Dict[str, float]:
    """
    Estimate memory usage for a model.
    
    Args:
        model_name: HuggingFace model name
        batch_size: Batch size for inference
        sequence_length: Input sequence length
        dtype: Data type (auto-detected if None)
        
    Returns:
        Dict with memory estimates in GB
    """
    try:
        config = AutoConfig.from_pretrained(model_name)
        
        # Get model parameters
        num_params = getattr(config, 'num_parameters', None)
        if num_params is None:
            # Estimate based on hidden size and layers
            hidden_size = config.hidden_size
            num_layers = config.num_hidden_layers
            vocab_size = config.vocab_size
            
            # Rough estimation formula
            num_params = (
                vocab_size * hidden_size +  # Embedding
                num_layers * (4 * hidden_size * hidden_size) +  # Transformer layers
                hidden_size * vocab_size  # Output projection
            )
        
        # Determine dtype size
        if dtype is None:
            dtype = torch.float16
        
        dtype_bytes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.int8: 1,
            torch.int4: 0.5
        }.get(dtype, 2)
        
        # Memory calculations (in GB)
        model_memory = (num_params * dtype_bytes) / (1024**3)
        
        # KV cache memory (approximate)
        kv_cache_memory = (
            2 * batch_size * sequence_length * 
            config.num_hidden_layers * config.hidden_size * dtype_bytes
        ) / (1024**3)
        
        # Activation memory (approximate)
        activation_memory = (
            batch_size * sequence_length * config.hidden_size * 
            config.num_hidden_layers * dtype_bytes * 4  # Factor for activations
        ) / (1024**3)
        
        total_memory = model_memory + kv_cache_memory + activation_memory
        
        return {
            "model_memory_gb": model_memory,
            "kv_cache_memory_gb": kv_cache_memory,
            "activation_memory_gb": activation_memory,
            "total_memory_gb": total_memory,
            "num_parameters": num_params,
            "dtype_bytes": dtype_bytes,
            "batch_size": batch_size,
            "sequence_length": sequence_length
        }
        
    except Exception as e:
        warnings.warn(f"Could not estimate memory for {model_name}: {e}")
        return {"error": str(e)}

def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get comprehensive model information.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Dict with model information
    """
    try:
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        info = {
            "model_name": model_name,
            "model_type": config.model_type,
            "architectures": getattr(config, 'architectures', []),
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
            "tokenizer_type": type(tokenizer).__name__,
            "pad_token": tokenizer.pad_token,
            "eos_token": tokenizer.eos_token,
            "bos_token": tokenizer.bos_token,
            "special_tokens": len(tokenizer.special_tokens_map),
        }
        
        # Add memory estimates
        memory_info = estimate_memory_usage(model_name)
        info.update(memory_info)
        
        return info
        
    except Exception as e:
        return {"error": str(e), "model_name": model_name}

def optimize_for_inference(
    model: torch.nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None
) -> torch.nn.Module:
    """
    Optimize model for inference.
    
    Args:
        model: PyTorch model
        device: Target device
        dtype: Target dtype
        
    Returns:
        Optimized model
    """
    # Move to device
    model = model.to(device)
    
    # Convert dtype if specified
    if dtype is not None:
        model = model.to(dtype)
    
    # Set to eval mode
    model.eval()
    
    # Enable inference optimizations
    try:
        if hasattr(torch.jit, 'optimize_for_inference'):
            # Only try if model is scriptable
            if hasattr(model, '_c'):  # Check if already scripted
                model = torch.jit.optimize_for_inference(model)
    except Exception:
        # Skip optimization if it fails
        pass
    
    # MPS specific optimizations
    if device.type == "mps":
        # Enable MPS optimizations
        torch.mps.empty_cache()
        
        # Set memory format for better performance
        if hasattr(model, 'to_memory_format'):
            model = model.to(memory_format=torch.channels_last)
    
    # CUDA specific optimizations  
    elif device.type == "cuda":
        # Enable CUDA optimizations
        torch.cuda.empty_cache()
        
        # Enable cudnn optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    return model

def cleanup_memory():
    """Clean up GPU and system memory."""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def get_system_info() -> Dict[str, Any]:
    """Get system resource information."""
    memory_info = psutil.virtual_memory()
    
    info = {
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": memory_info.total / (1024**3),
        "memory_available_gb": memory_info.available / (1024**3),
        "memory_percent": memory_info.percent,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        })
    
    return info

def format_model_size(num_params: int) -> str:
    """Format parameter count in human readable format."""
    if num_params >= 1e9:
        return f"{num_params/1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.1f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.1f}K"
    else:
        return str(num_params)

def check_model_compatibility(model_name: str, device: torch.device) -> Dict[str, Any]:
    """
    Check if model is compatible with target device.
    
    Args:
        model_name: HuggingFace model name
        device: Target device
        
    Returns:
        Compatibility report
    """
    try:
        model_info = get_model_info(model_name)
        system_info = get_system_info()
        
        # Memory check
        required_memory = model_info.get("total_memory_gb", 0)
        available_memory = system_info["memory_available_gb"]
        
        memory_ok = required_memory < (available_memory * 0.8)  # Leave 20% buffer
        
        # Device compatibility
        device_ok = True
        warnings_list = []
        
        if device.type == "mps" and not system_info["mps_available"]:
            device_ok = False
            warnings_list.append("MPS requested but not available")
        
        if device.type == "cuda" and not system_info["cuda_available"]:
            device_ok = False
            warnings_list.append("CUDA requested but not available")
        
        # Model architecture checks
        model_type = model_info.get("model_type", "")
        
        # Some models have known issues with MPS
        if device.type == "mps" and model_type in ["falcon", "mpt"]:
            warnings_list.append(f"{model_type} models may have compatibility issues with MPS")
        
        return {
            "compatible": memory_ok and device_ok,
            "memory_ok": memory_ok,
            "device_ok": device_ok,
            "required_memory_gb": required_memory,
            "available_memory_gb": available_memory,
            "warnings": warnings_list,
            "recommendations": get_recommendations(model_info, system_info, device)
        }
        
    except Exception as e:
        return {"error": str(e), "compatible": False}

def get_recommendations(
    model_info: Dict[str, Any], 
    system_info: Dict[str, Any],
    device: torch.device
) -> List[str]:
    """Get optimization recommendations."""
    recommendations = []
    
    required_memory = model_info.get("total_memory_gb", 0)
    available_memory = system_info["memory_available_gb"]
    
    if required_memory > available_memory:
        recommendations.append("Consider using quantization (4-bit or 8-bit) to reduce memory usage")
        recommendations.append("Reduce batch size or sequence length")
    
    if device.type == "cpu" and (system_info["cuda_available"] or system_info["mps_available"]):
        if system_info["mps_available"]:
            recommendations.append("Consider using MPS device for better performance on Apple Silicon")
        elif system_info["cuda_available"]:
            recommendations.append("Consider using CUDA device for better performance")
    
    if device.type == "mps":
        recommendations.append("Use float16 dtype for optimal MPS performance")
        recommendations.append("Enable MPS fallback for unsupported operations")
    
    num_params = model_info.get("num_parameters", 0)
    if num_params > 7e9:  # > 7B parameters
        recommendations.append("Large model detected - consider using gradient checkpointing during training")
        recommendations.append("Use streaming generation for better memory efficiency")
    
    return recommendations
"""
HuggingFace Quantization Implementation
======================================

Comprehensive quantization support for HuggingFace models.
"""

import os
import time
import torch
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    from optimum.gptq import GPTQQuantizer, load_quantized_model
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("HuggingFace quantization libraries not available")

from ..utils import setup_mps_device, get_optimal_dtype, cleanup_memory

class QuantizationMethod(Enum):
    """Supported quantization methods."""
    BITSANDBYTES_4BIT = "bnb_4bit"
    BITSANDBYTES_8BIT = "bnb_8bit"
    GPTQ = "gptq"
    AWQ = "awq"
    DYNAMIC = "dynamic"

@dataclass
class QuantizationConfig:
    """Quantization configuration."""
    method: QuantizationMethod
    bits: int = 4
    group_size: int = 128
    desc_act: bool = False
    static_groups: bool = False
    sym: bool = True
    true_sequential: bool = True
    model_seqlen: int = 2048
    dataset: str = "c4"
    compute_dtype: torch.dtype = torch.float16
    quant_type: str = "nf4"
    use_double_quant: bool = True
    
    def to_bnb_config(self) -> BitsAndBytesConfig:
        """Convert to BitsAndBytes configuration."""
        if self.method == QuantizationMethod.BITSANDBYTES_4BIT:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=self.use_double_quant,
                bnb_4bit_quant_type=self.quant_type
            )
        elif self.method == QuantizationMethod.BITSANDBYTES_8BIT:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            raise ValueError(f"Method {self.method} not supported for BitsAndBytes")

def create_quantization_config(
    method: QuantizationMethod,
    bits: int = 4,
    group_size: int = 128,
    **kwargs
) -> QuantizationConfig:
    """
    Create quantization configuration.
    
    Args:
        method: Quantization method
        bits: Number of bits
        group_size: Group size for quantization
        **kwargs: Additional parameters
        
    Returns:
        Quantization configuration
    """
    return QuantizationConfig(
        method=method,
        bits=bits,
        group_size=group_size,
        **kwargs
    )

def quantize_model(
    model_name: str,
    output_dir: str,
    quantization_config: QuantizationConfig,
    save_model: bool = True,
    device: Optional[torch.device] = None
) -> Union[str, Tuple[Any, Any]]:
    """
    Quantize a HuggingFace model.
    
    Args:
        model_name: Model name or path
        output_dir: Output directory
        quantization_config: Quantization configuration
        save_model: Whether to save the quantized model
        device: Target device
        
    Returns:
        Path to quantized model or (model, tokenizer) tuple
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace quantization libraries not available")
    
    print(f"Quantizing model: {model_name}")
    print(f"Method: {quantization_config.method.value}")
    print(f"Bits: {quantization_config.bits}")
    
    if device is None:
        device = setup_mps_device()
    
    # Create output directory
    if save_model:
        os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    if quantization_config.method in [QuantizationMethod.BITSANDBYTES_4BIT, QuantizationMethod.BITSANDBYTES_8BIT]:
        # BitsAndBytes quantization
        return _quantize_bitsandbytes(
            model_name, output_dir, quantization_config, save_model, device
        )
    
    elif quantization_config.method == QuantizationMethod.GPTQ:
        # GPTQ quantization
        return _quantize_gptq(
            model_name, output_dir, quantization_config, save_model, device
        )
    
    elif quantization_config.method == QuantizationMethod.AWQ:
        # AWQ quantization
        return _quantize_awq(
            model_name, output_dir, quantization_config, save_model, device
        )
    
    elif quantization_config.method == QuantizationMethod.DYNAMIC:
        # Dynamic quantization
        return _quantize_dynamic(
            model_name, output_dir, quantization_config, save_model, device
        )
    
    else:
        raise ValueError(f"Unsupported quantization method: {quantization_config.method}")

def _quantize_bitsandbytes(
    model_name: str,
    output_dir: str,
    config: QuantizationConfig,
    save_model: bool,
    device: torch.device
) -> Union[str, Tuple[Any, Any]]:
    """Quantize using BitsAndBytes."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load quantized model
    bnb_config = config.to_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=config.compute_dtype
    )
    
    if save_model:
        # Save quantized model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save config
        import json
        config_path = os.path.join(output_dir, "quantization_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "method": config.method.value,
                "bits": config.bits,
                "compute_dtype": str(config.compute_dtype),
                "quant_type": config.quant_type,
                "use_double_quant": config.use_double_quant,
                "original_model": model_name
            }, f, indent=2)
        
        print(f"BitsAndBytes quantized model saved to: {output_dir}")
        return output_dir
    else:
        return model, tokenizer

def _quantize_gptq(
    model_name: str,
    output_dir: str,
    config: QuantizationConfig,
    save_model: bool,
    device: torch.device
) -> Union[str, Tuple[Any, Any]]:
    """Quantize using GPTQ."""
    
    try:
        from optimum.gptq import GPTQQuantizer
        from datasets import load_dataset
    except ImportError:
        raise ImportError("GPTQ quantization requires optimum[gptq] and datasets")
    
    # Prepare calibration dataset
    if config.dataset == "c4":
        dataset = load_dataset("allenai/c4", "allenai--c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"})
        dataset = dataset["train"]
    else:
        dataset = load_dataset(config.dataset, split="train")
    
    # Create quantizer
    quantizer = GPTQQuantizer(
        bits=config.bits,
        group_size=config.group_size,
        desc_act=config.desc_act,
        static_groups=config.static_groups,
        sym=config.sym,
        true_sequential=config.true_sequential,
        model_seqlen=config.model_seqlen,
        dataset=dataset
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=config.compute_dtype,
        device_map="auto"
    )
    
    # Quantize
    print("Starting GPTQ quantization...")
    quantized_model = quantizer.quantize_model(model, tokenizer)
    
    if save_model:
        # Save quantized model
        quantized_model.save_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        
        print(f"GPTQ quantized model saved to: {output_dir}")
        return output_dir
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return quantized_model, tokenizer

def _quantize_awq(
    model_name: str,
    output_dir: str,
    config: QuantizationConfig,
    save_model: bool,
    device: torch.device
) -> Union[str, Tuple[Any, Any]]:
    """Quantize using AWQ."""
    
    try:
        from awq import AutoAWQForCausalLM
        from awq.quantize.quantizer import AwqQuantizer
    except ImportError:
        raise ImportError("AWQ quantization requires autoawq library")
    
    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare calibration data (simplified)
    quant_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
    ] * 32  # Repeat to get more samples
    
    # Quantize
    print("Starting AWQ quantization...")
    model.quantize(tokenizer, quant_data=quant_data)
    
    if save_model:
        # Save quantized model
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"AWQ quantized model saved to: {output_dir}")
        return output_dir
    else:
        return model, tokenizer

def _quantize_dynamic(
    model_name: str,
    output_dir: str,
    config: QuantizationConfig,
    save_model: bool,
    device: torch.device
) -> Union[str, Tuple[Any, Any]]:
    """Apply dynamic quantization."""
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=config.compute_dtype,
        device_map=device if device.type != "cpu" else None
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Apply dynamic quantization
    if config.bits == 8:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    else:
        warnings.warn("Dynamic quantization currently only supports 8-bit")
    
    if save_model:
        # Save quantized model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"Dynamic quantized model saved to: {output_dir}")
        return output_dir
    else:
        return model, tokenizer

def load_quantized_model(
    model_path: str,
    device: Optional[torch.device] = None
) -> Tuple[Any, Any]:
    """
    Load a quantized model.
    
    Args:
        model_path: Path to quantized model
        device: Target device
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = setup_mps_device()
    
    # Check for quantization config
    config_path = os.path.join(model_path, "quantization_config.json")
    
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            quant_config = json.load(f)
        
        method = quant_config.get("method", "unknown")
        print(f"Loading {method} quantized model from: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model (auto-detects quantization)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if device.type != "cpu" else None,
        torch_dtype=torch.float16
    )
    
    return model, tokenizer

def benchmark_quantization(
    model_name: str,
    methods: List[QuantizationMethod],
    test_prompts: List[str],
    output_dir: str = "quantization_benchmark"
) -> Dict[str, Any]:
    """
    Benchmark different quantization methods.
    
    Args:
        model_name: Model to benchmark
        methods: Quantization methods to test
        test_prompts: Test prompts for evaluation
        output_dir: Output directory for results
        
    Returns:
        Benchmark results
    """
    results = {
        "model_name": model_name,
        "methods": {},
        "test_prompts": test_prompts
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for method in methods:
        print(f"\nBenchmarking {method.value}...")
        
        try:
            # Create quantization config
            config = create_quantization_config(method)
            
            # Quantize model
            start_time = time.time()
            model, tokenizer = quantize_model(
                model_name, 
                os.path.join(output_dir, method.value),
                config,
                save_model=False
            )
            quantize_time = time.time() - start_time
            
            # Test inference
            inference_times = []
            for prompt in test_prompts:
                start_time = time.time()
                
                inputs = tokenizer(prompt, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            # Calculate model size
            model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)  # MB
            
            results["methods"][method.value] = {
                "quantize_time": quantize_time,
                "avg_inference_time": sum(inference_times) / len(inference_times),
                "model_size_mb": model_size,
                "successful": True
            }
            
            # Cleanup
            del model, tokenizer
            cleanup_memory()
            
        except Exception as e:
            print(f"Error with {method.value}: {e}")
            results["methods"][method.value] = {
                "error": str(e),
                "successful": False
            }
    
    # Save results
    import json
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nBenchmark results saved to: {results_path}")
    return results

def batch_quantize_models(
    model_names: List[str],
    output_base_dir: str,
    quantization_config: QuantizationConfig
) -> Dict[str, str]:
    """
    Quantize multiple models in batch.
    
    Args:
        model_names: List of model names
        output_base_dir: Base output directory
        quantization_config: Quantization configuration
        
    Returns:
        Dictionary mapping model names to output paths
    """
    results = {}
    
    for model_name in model_names:
        print(f"\nQuantizing {model_name}...")
        
        # Create model-specific output directory
        model_dir = model_name.replace("/", "_")
        output_dir = os.path.join(output_base_dir, model_dir)
        
        try:
            result_path = quantize_model(
                model_name,
                output_dir, 
                quantization_config,
                save_model=True
            )
            results[model_name] = result_path
            print(f"✅ {model_name} quantized successfully")
            
        except Exception as e:
            print(f"❌ Failed to quantize {model_name}: {e}")
            results[model_name] = f"Error: {e}"
        
        # Cleanup between models
        cleanup_memory()
    
    return results
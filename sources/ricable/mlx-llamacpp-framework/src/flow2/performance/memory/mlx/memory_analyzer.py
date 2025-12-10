#!/usr/bin/env python3
"""
Memory Usage Analyzer for MLX models on Apple Silicon.

This script analyzes memory usage patterns of MLX models with different configurations,
helping users optimize memory usage for their specific hardware.

Usage:
    python memory_analyzer.py --model <model_name> [--quant <quantization>] [--ctx <context_length>]

Example:
    python memory_analyzer.py --model llama-2-7b --quant int4 --ctx 2048
"""

import argparse
import os
import time
import json
import platform
import psutil
import traceback
import gc
import numpy as np
from datetime import datetime
import threading
import tempfile

# Third-party imports conditionally handled
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. This script requires MLX and MLX-LM to be installed.")

# Constants
QUANTIZATION_TYPES = ["none", "int8", "int4"]
DEFAULT_PROMPT = "Explain the theory of relativity in detail, covering both special and general relativity."

def get_system_info():
    """Get system information"""
    mem = psutil.virtual_memory()
    
    return {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "ram_total": mem.total / (1024 ** 3),  # Convert to GB
        "ram_available": mem.available / (1024 ** 3),  # Convert to GB
        "mac_model": subprocess.getoutput("sysctl hw.model").replace("hw.model: ", ""),
        "timestamp": datetime.now().isoformat()
    }

class MemoryMonitor(threading.Thread):
    """Thread for monitoring memory usage"""
    def __init__(self, pid):
        super().__init__()
        self.pid = pid
        self.running = True
        self.memory_usage = []
        self.peak_memory = 0
    
    def run(self):
        while self.running:
            try:
                proc = psutil.Process(self.pid)
                mem_info = proc.memory_info()
                current_mem = mem_info.rss / (1024 ** 3)  # Convert to GB
                
                self.memory_usage.append(current_mem)
                self.peak_memory = max(self.peak_memory, current_mem)
                
                time.sleep(0.1)
            except:
                # Process might have ended
                break
    
    def stop(self):
        self.running = False

def measure_memory_usage(model_name, prompt=DEFAULT_PROMPT, context_length=2048, quantization=None):
    """
    Measure memory usage during MLX model inference
    """
    if not MLX_AVAILABLE:
        return {
            "error": "MLX not available. Please install MLX and MLX-LM."
        }
    
    # Start memory monitoring for current process
    monitor = MemoryMonitor(os.getpid())
    monitor.start()
    
    start_time = time.time()
    load_time = None
    inference_time = None
    tokens_per_second = None
    
    try:
        # Record base memory before loading
        gc.collect()
        base_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        
        # Load the model
        load_start = time.time()
        model, tokenizer = load(model_name, quantization=quantization)
        load_end = time.time()
        load_time = load_end - load_start
        
        # Force compile the model with a small input to avoid measuring compilation time during inference
        input_tokens = tokenizer.encode("Test")
        _ = model(mx.array([input_tokens]))
        
        # Measure inference
        inference_start = time.time()
        tokens = generate(model, tokenizer, prompt, max_tokens=256)
        inference_end = time.time()
        inference_time = inference_end - inference_start
        
        # Calculate tokens per second
        output_tokens = len(tokens)
        tokens_per_second = output_tokens / inference_time
        
        # Clean up to measure base model size
        del tokens
        gc.collect()
        
    except Exception as e:
        print(f"Error during MLX inference: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
    finally:
        # Stop monitoring
        monitor.stop()
        monitor.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        "peak_memory_gb": monitor.peak_memory,
        "avg_memory_gb": np.mean(monitor.memory_usage) if monitor.memory_usage else 0,
        "memory_timeline": monitor.memory_usage,
        "base_memory_gb": base_memory,
        "model_memory_gb": monitor.peak_memory - base_memory,
        "load_time_seconds": load_time,
        "inference_time_seconds": inference_time,
        "tokens_per_second": tokens_per_second,
        "total_time_seconds": total_time
    }

def estimate_model_size(model_name):
    """Estimate model size based on name"""
    # Extract model size (e.g., 7B, 13B)
    if "7b" in model_name.lower():
        return "7B"
    elif "13b" in model_name.lower():
        return "13B"
    elif "34b" in model_name.lower() or "33b" in model_name.lower():
        return "33B"
    elif "70b" in model_name.lower():
        return "70B"
    else:
        return "Unknown"

def estimate_model_memory_requirements(model_size, quantization):
    """Estimate memory requirements based on model size and quantization"""
    # Base memory sizes for FP16 models in GB
    base_sizes = {
        "7B": 14,
        "13B": 26,
        "33B": 65,
        "70B": 140
    }
    
    # Memory reduction factors for quantization
    quant_factors = {
        "none": 1.0,  # No reduction (FP16)
        "int8": 0.5,  # 50% reduction
        "int4": 0.25  # 75% reduction
    }
    
    if model_size not in base_sizes:
        return None
    
    quant = quantization if quantization in quant_factors else "none"
    base_memory = base_sizes.get(model_size, 0)
    
    return base_memory * quant_factors[quant]

def estimate_optimal_settings(system_info, model_size, quantization):
    """Estimate optimal settings based on system and model information"""
    available_ram = system_info["ram_available"]
    estimated_model_memory = estimate_model_memory_requirements(model_size, quantization)
    
    if estimated_model_memory is None:
        return {
            "error": "Unknown model size"
        }
    
    # Estimate memory needed for inference (approximation)
    estimated_inference_memory = estimated_model_memory * 1.2  # 20% overhead for inference
    
    # Calculate maximum safe context length
    if available_ram < estimated_inference_memory:
        recommendation = "Model too large for available memory. Consider using a smaller model or higher quantization."
        max_context = 0
    else:
        remaining_ram = available_ram - estimated_inference_memory
        # Rough estimate: each 1K tokens in context uses X GB additional RAM depending on model size
        tokens_per_gb = 2048 / (0.5 if model_size == "7B" else 
                               1.0 if model_size == "13B" else 
                               2.0)  # Very rough approximation
        
        max_context = int((remaining_ram * tokens_per_gb) / 1024) * 1024  # Round to nearest 1K
        max_context = min(32768, max(1024, max_context))  # Keep between 1K and 32K
        
        if max_context < 2048:
            recommendation = "Limited context length available. Consider higher quantization."
        elif max_context < 8192:
            recommendation = "Moderate context length available. Settings should be adequate for most use cases."
        else:
            recommendation = "Large context length available. Model can run efficiently with extensive context."
    
    return {
        "estimated_model_memory_gb": estimated_model_memory,
        "estimated_inference_memory_gb": estimated_inference_memory,
        "recommended_max_context_length": max_context,
        "recommendation": recommendation
    }

def main():
    if not MLX_AVAILABLE:
        print("Error: MLX not available. Please install MLX and MLX-LM.")
        return
    
    parser = argparse.ArgumentParser(description="Analyze memory usage of MLX models")
    parser.add_argument("--model", required=True, help="Model name (e.g., llama-2-7b)")
    parser.add_argument("--quant", choices=QUANTIZATION_TYPES, default=None, help="Quantization type")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length")
    parser.add_argument("--output", help="Output file for results (JSON)")
    args = parser.parse_args()
    
    # Get system information
    import subprocess  # Import here for cleaner handling
    system_info = get_system_info()
    print(f"System: {system_info['mac_model']}, RAM: {system_info['ram_total']:.2f} GB")
    
    # Extract model size
    model_size = estimate_model_size(args.model)
    print(f"Model size: {model_size}, Quantization: {args.quant or 'none (FP16)'}")
    
    # Estimate optimal settings
    optimal_settings = estimate_optimal_settings(system_info, model_size, args.quant)
    if "error" in optimal_settings:
        print(f"Error: {optimal_settings['error']}")
        return
    
    print(f"Estimated model memory: {optimal_settings['estimated_model_memory_gb']:.2f} GB")
    print(f"Estimated inference memory: {optimal_settings['estimated_inference_memory_gb']:.2f} GB")
    print(f"Recommended max context: {optimal_settings['recommended_max_context_length']}")
    print(f"Recommendation: {optimal_settings['recommendation']}")
    
    # Measure actual memory usage
    print("\nMeasuring memory usage during inference...")
    memory_usage = measure_memory_usage(
        args.model,
        context_length=args.ctx,
        quantization=args.quant
    )
    
    if "error" in memory_usage:
        print(f"Error during measurement: {memory_usage['error']}")
        return
    
    print(f"Peak memory usage: {memory_usage['peak_memory_gb']:.2f} GB")
    print(f"Model memory usage: {memory_usage['model_memory_gb']:.2f} GB")
    print(f"Average memory usage: {memory_usage['avg_memory_gb']:.2f} GB")
    print(f"Load time: {memory_usage['load_time_seconds']:.2f} seconds")
    print(f"Inference time: {memory_usage['inference_time_seconds']:.2f} seconds")
    print(f"Performance: {memory_usage['tokens_per_second']:.2f} tokens/second")
    
    # Compile results
    results = {
        "system_info": system_info,
        "model_info": {
            "name": args.model,
            "size": model_size,
            "quantization": args.quant or "none (FP16)"
        },
        "optimal_settings": optimal_settings,
        "memory_usage": memory_usage,
        "test_parameters": {
            "context_length": args.ctx,
            "quantization": args.quant
        }
    }
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
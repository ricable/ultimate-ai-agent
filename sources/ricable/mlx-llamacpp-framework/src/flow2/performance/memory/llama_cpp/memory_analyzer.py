#!/usr/bin/env python3
"""
Memory Usage Analyzer for llama.cpp models on Apple Silicon.

This script analyzes memory usage patterns of llama.cpp models with different configurations,
helping users optimize memory usage for their specific hardware.

Usage:
    python memory_analyzer.py --model <model_path> [--quant <quantization>] [--ctx <context_length>]

Example:
    python memory_analyzer.py --model models/llama-2-7b.gguf --quant q4_k --ctx 2048
"""

import argparse
import subprocess
import re
import os
import time
import json
import platform
import psutil
from datetime import datetime

# Constants
QUANTIZATION_TYPES = ["f16", "q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "q2_k"]
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

def measure_memory_usage(llama_cpp_path, model_path, prompt=DEFAULT_PROMPT, context_length=2048, threads=4, use_metal=True):
    """
    Measure memory usage during llama.cpp model inference
    """
    # Construct command
    cmd = [
        os.path.join(llama_cpp_path, "main"),
        "-m", model_path,
        "-p", prompt,
        "-c", str(context_length),
        "-n", "256",
        "-t", str(threads)
    ]
    
    if use_metal:
        cmd.append("--metal")
    
    # Start memory monitoring
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Track memory usage over time
    memory_usage = []
    peak_memory = 0
    
    try:
        while process.poll() is None:
            # Get memory usage of the process
            proc = psutil.Process(process.pid)
            mem_info = proc.memory_info()
            current_mem = mem_info.rss / (1024 ** 3)  # Convert to GB
            
            memory_usage.append(current_mem)
            peak_memory = max(peak_memory, current_mem)
            
            time.sleep(0.1)
    
    except Exception as e:
        print(f"Error during memory monitoring: {e}")
    finally:
        # Ensure process is terminated
        if process.poll() is None:
            process.terminate()
            process.wait()
    
    # Get output
    stdout, stderr = process.communicate()
    
    # Parse timing information from output
    load_time = None
    inference_time = None
    
    output = stdout.decode()
    
    # Example parsing - adjust based on llama.cpp output format
    if "llama_model_load: loaded" in output:
        load_time_match = re.search(r"llama_model_load: loaded in (\d+\.\d+) ms", output)
        if load_time_match:
            load_time = float(load_time_match.group(1)) / 1000  # convert to seconds
    
    tokens_per_second_match = re.search(r"(\d+\.\d+) tokens/s", output)
    if tokens_per_second_match:
        tokens_per_second = float(tokens_per_second_match.group(1))
        inference_time = 256 / tokens_per_second  # Approximate inference time based on tokens generated
    
    return {
        "peak_memory_gb": peak_memory,
        "avg_memory_gb": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
        "memory_timeline": memory_usage,
        "load_time_seconds": load_time,
        "inference_time_seconds": inference_time
    }

def analyze_model_file_size(model_path):
    """Analyze the model file size"""
    file_size_bytes = os.path.getsize(model_path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    
    return {
        "file_path": model_path,
        "file_size_bytes": file_size_bytes,
        "file_size_gb": file_size_gb
    }

def extract_model_info(model_path):
    """Extract model information from filename"""
    filename = os.path.basename(model_path)
    
    # Try to extract model size (e.g., 7B, 13B)
    model_size_match = re.search(r"(\d+)[bB]", filename)
    model_size = model_size_match.group(1) + "B" if model_size_match else "Unknown"
    
    # Try to extract quantization (e.g., q4_0, q8_0)
    quant_match = re.search(r"[qQ](\d+)_[kK\d]", filename)
    quantization = quant_match.group(0) if quant_match else "f16"
    
    return {
        "model_size": model_size,
        "quantization": quantization
    }

def estimate_optimal_settings(system_info, model_info):
    """Estimate optimal settings based on system and model information"""
    available_ram = system_info["ram_available"]
    model_size = model_info["file_size_gb"]
    
    # Estimate memory needed for inference (approximation)
    estimated_inference_memory = model_size * 1.3  # 30% overhead for inference
    
    # Calculate maximum safe context length
    if available_ram < estimated_inference_memory:
        recommendation = "Model too large for available memory. Consider using a smaller model or higher quantization."
        max_context = 0
    else:
        remaining_ram = available_ram - estimated_inference_memory
        # Rough estimate: each 1K tokens in context uses X GB additional RAM depending on model size
        tokens_per_gb = 2048 / (0.5 if "7B" in model_info["model_size"] else 
                               1.0 if "13B" in model_info["model_size"] else 
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
        "estimated_inference_memory_gb": estimated_inference_memory,
        "recommended_max_context_length": max_context,
        "recommendation": recommendation
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze memory usage of llama.cpp models")
    parser.add_argument("--model", required=True, help="Path to the model file (.gguf)")
    parser.add_argument("--llama_cpp", default="./llama.cpp", help="Path to llama.cpp directory")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--no-metal", action="store_true", help="Disable Metal acceleration")
    parser.add_argument("--output", help="Output file for results (JSON)")
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Get system information
    system_info = get_system_info()
    print(f"System: {system_info['mac_model']}, RAM: {system_info['ram_total']:.2f} GB")
    
    # Analyze model file
    model_file_info = analyze_model_file_size(args.model)
    print(f"Model file size: {model_file_info['file_size_gb']:.2f} GB")
    
    # Extract model information
    model_info = extract_model_info(args.model)
    print(f"Model size: {model_info['model_size']}, Quantization: {model_info['quantization']}")
    
    # Estimate optimal settings
    optimal_settings = estimate_optimal_settings(system_info, model_file_info)
    print(f"Estimated inference memory: {optimal_settings['estimated_inference_memory_gb']:.2f} GB")
    print(f"Recommended max context: {optimal_settings['recommended_max_context_length']}")
    print(f"Recommendation: {optimal_settings['recommendation']}")
    
    # Measure actual memory usage
    print("\nMeasuring memory usage during inference...")
    memory_usage = measure_memory_usage(
        args.llama_cpp, 
        args.model, 
        context_length=args.ctx, 
        threads=args.threads, 
        use_metal=not args.no_metal
    )
    
    print(f"Peak memory usage: {memory_usage['peak_memory_gb']:.2f} GB")
    print(f"Average memory usage: {memory_usage['avg_memory_gb']:.2f} GB")
    print(f"Load time: {memory_usage['load_time_seconds']:.2f} seconds")
    print(f"Inference time: {memory_usage['inference_time_seconds']:.2f} seconds")
    
    # Compile results
    results = {
        "system_info": system_info,
        "model_file_info": model_file_info,
        "model_info": model_info,
        "optimal_settings": optimal_settings,
        "memory_usage": memory_usage,
        "test_parameters": {
            "context_length": args.ctx,
            "threads": args.threads,
            "metal_enabled": not args.no_metal
        }
    }
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Metal GPU Optimization Utility for MLX on Apple Silicon.

This script analyzes and optimizes Metal GPU performance for MLX models,
providing configuration recommendations to maximize performance.

Usage:
    python metal_optimizer.py --model <model_name> [--options]

Example:
    python metal_optimizer.py --model llama-2-7b --quant int4 --output results/mlx_optimized.json
"""

import argparse
import os
import time
import json
import platform
import psutil
import numpy as np
import tempfile
import subprocess
from datetime import datetime
import traceback
import gc
import threading

# Third-party imports conditionally handled
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. This script requires MLX and MLX-LM to be installed.")

# Constants
DEFAULT_PROMPT = "Explain the theory of relativity in detail, covering both special and general relativity."

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

def get_gpu_info():
    """Get information about the GPU(s) in the system"""
    gpu_info = {}
    
    try:
        # Check if we're on a Mac
        if platform.system() != "Darwin":
            return {"error": "Not running on macOS"}
        
        # Get GPU info from system_profiler
        gpu_data = subprocess.getoutput("system_profiler SPDisplaysDataType")
        
        # Parse the output
        # This is a simple parsing approach - a more thorough check would use the Metal framework
        gpu_name = None
        metal_support = False
        
        for line in gpu_data.split('\n'):
            line = line.strip()
            
            if "Chipset Model:" in line:
                gpu_name = line.replace("Chipset Model:", "").strip()
            
            if "Metal:" in line and "supported" in line.lower():
                metal_support = True
        
        gpu_info["name"] = gpu_name
        gpu_info["metal_support"] = metal_support
        
        # Try to get more specific information for Apple Silicon
        if "Apple" in gpu_name:
            # Determine the chip type (M1, M2, M3)
            if "M1" in gpu_name:
                chip_type = "M1"
            elif "M2" in gpu_name:
                chip_type = "M2"
            elif "M3" in gpu_name:
                chip_type = "M3"
            else:
                chip_type = "Unknown Apple Silicon"
            
            # Determine the variant (Base, Pro, Max, Ultra)
            if "Max" in gpu_name:
                variant = "Max"
            elif "Ultra" in gpu_name:
                variant = "Ultra"
            elif "Pro" in gpu_name:
                variant = "Pro"
            else:
                variant = "Base"
            
            gpu_info["chip_type"] = chip_type
            gpu_info["variant"] = variant
        
    except Exception as e:
        gpu_info["error"] = str(e)
    
    return gpu_info

def run_mlx_benchmark(model_name, prompt=DEFAULT_PROMPT, max_tokens=256, 
                     quantization=None, device="gpu", stream=True, seed=None,
                     temperature=0.7, top_p=0.95, repetition_penalty=1.1):
    """
    Run MLX benchmark with specified device settings
    """
    if not MLX_AVAILABLE:
        return {
            "error": "MLX not available. Please install MLX and MLX-LM."
        }
    
    # Store original device
    original_device = mx.default_device()
    
    try:
        # Set device
        if device == "gpu":
            mx.set_default_device(mx.gpu)
        elif device == "cpu":
            mx.set_default_device(mx.cpu)
        else:
            return {"error": f"Unknown device: {device}"}
        
        # Set seed if provided
        if seed is not None:
            mx.random.seed(seed)
        
        # Start memory monitoring for current process
        monitor = MemoryMonitor(os.getpid())
        monitor.start()
        
        # Clear cache and collect garbage
        gc.collect()
        mx.clear_cache()
        
        # Record base memory before loading
        base_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
        
        # Load model - measure time
        load_start = time.time()
        model, tokenizer = load(model_name, quantization=quantization)
        load_end = time.time()
        load_time = load_end - load_start
        
        # Force compile the model with a small input to avoid measuring compilation time during inference
        input_tokens = tokenizer.encode("Test")
        _ = model(mx.array([input_tokens]))
        
        # Measure inference - first encode the prompt
        encode_start = time.time()
        prompt_tokens = tokenizer.encode(prompt)
        encode_end = time.time()
        encode_time = encode_end - encode_start
        
        # Then generate the response
        generate_start = time.time()
        
        # Track tokens per second during generation
        tokens_generated = 0
        generation_times = []
        
        if stream:
            # Stream mode to calculate tokens per second more accurately
            for token in generate(
                model, 
                tokenizer, 
                prompt, 
                max_tokens=max_tokens, 
                temp=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stream=True
            ):
                tokens_generated += 1
                generation_times.append(time.time())
        else:
            # Non-streaming mode
            generation = generate(
                model, 
                tokenizer, 
                prompt, 
                max_tokens=max_tokens, 
                temp=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            tokens_generated = len(generation)
        
        generate_end = time.time()
        generate_time = generate_end - generate_start
        
        # Calculate tokens per second
        tokens_per_second = tokens_generated / generate_time if generate_time > 0 else 0
        
        # Calculate incremental tokens per second if streaming
        incremental_tokens_per_second = None
        if stream and len(generation_times) > 10:
            # Calculate after the first few tokens to skip compilation time
            start_idx = 5
            token_intervals = [generation_times[i] - generation_times[i-1] for i in range(start_idx + 1, len(generation_times))]
            if token_intervals:
                avg_token_time = np.mean(token_intervals)
                incremental_tokens_per_second = 1.0 / avg_token_time if avg_token_time > 0 else 0
        
        # Clean up
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
        
        # Stop memory monitoring
        monitor.stop()
        monitor.join()
        
        result = {
            "device": device,
            "model": model_name,
            "quantization": quantization,
            "load_time_seconds": load_time,
            "encode_time_seconds": encode_time,
            "generate_time_seconds": generate_time,
            "total_time_seconds": load_time + encode_time + generate_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second,
            "incremental_tokens_per_second": incremental_tokens_per_second,
            "base_memory_gb": base_memory,
            "peak_memory_gb": monitor.peak_memory,
            "model_memory_gb": monitor.peak_memory - base_memory,
            "success": True
        }
        
        return result
        
    except Exception as e:
        return {
            "device": device,
            "model": model_name,
            "quantization": quantization,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False
        }
    finally:
        # Restore original device
        mx.set_default_device(original_device)

def analyze_device_performance(model_name, quantization=None):
    """Analyze performance between GPU and CPU"""
    print(f"Analyzing device performance for {model_name} (quantization: {quantization or 'none'})...")
    
    # Run GPU benchmark
    print("Testing GPU (Metal) performance...")
    gpu_result = run_mlx_benchmark(
        model_name=model_name,
        quantization=quantization,
        device="gpu"
    )
    
    if gpu_result["success"]:
        print(f"  GPU Performance: {gpu_result['tokens_per_second']:.2f} tokens/second")
        print(f"  GPU Load time: {gpu_result['load_time_seconds']:.2f} seconds")
        print(f"  GPU Peak memory: {gpu_result['peak_memory_gb']:.2f} GB")
    else:
        print(f"  GPU Error: {gpu_result.get('error', 'Unknown error')}")
    
    # Run CPU benchmark
    print("\nTesting CPU performance...")
    cpu_result = run_mlx_benchmark(
        model_name=model_name,
        quantization=quantization,
        device="cpu"
    )
    
    if cpu_result["success"]:
        print(f"  CPU Performance: {cpu_result['tokens_per_second']:.2f} tokens/second")
        print(f"  CPU Load time: {cpu_result['load_time_seconds']:.2f} seconds")
        print(f"  CPU Peak memory: {cpu_result['peak_memory_gb']:.2f} GB")
    else:
        print(f"  CPU Error: {cpu_result.get('error', 'Unknown error')}")
    
    # Compare results
    if gpu_result["success"] and cpu_result["success"]:
        speedup = gpu_result["tokens_per_second"] / cpu_result["tokens_per_second"] if cpu_result["tokens_per_second"] > 0 else float('inf')
        
        print("\nGPU vs CPU Performance:")
        print(f"  GPU: {gpu_result['tokens_per_second']:.2f} tokens/second")
        print(f"  CPU: {cpu_result['tokens_per_second']:.2f} tokens/second")
        print(f"  Speedup: {speedup:.2f}x")
        
        recommended_device = "gpu" if speedup > 1.0 else "cpu"
        print(f"  Recommended device: {recommended_device}")
    else:
        recommended_device = "gpu" if gpu_result["success"] else "cpu"
        print(f"\nRecommended device: {recommended_device} (based on successful tests)")
    
    return {
        "gpu_result": gpu_result,
        "cpu_result": cpu_result,
        "recommended_device": recommended_device
    }

def optimize_quantization(model_name):
    """Test different quantization methods to find the best performance-memory tradeoff"""
    print(f"\nAnalyzing quantization performance for {model_name}...")
    
    quantization_types = [None, "int8", "int4"]  # None means FP16
    results = []
    
    for quant in quantization_types:
        quant_name = quant or "none (FP16)"
        print(f"Testing {quant_name} quantization...")
        
        result = run_mlx_benchmark(
            model_name=model_name,
            quantization=quant,
            device="gpu"  # Always use GPU for quantization tests
        )
        
        if result["success"]:
            print(f"  Performance: {result['tokens_per_second']:.2f} tokens/second")
            print(f"  Peak memory: {result['peak_memory_gb']:.2f} GB")
            print(f"  Model memory: {result['model_memory_gb']:.2f} GB")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
        
        results.append(result)
    
    # Find optimal quantization
    valid_results = [r for r in results if r["success"]]
    
    if valid_results:
        # Calculate a balanced score: performance / memory usage
        for r in valid_results:
            r["perf_memory_ratio"] = r["tokens_per_second"] / r["model_memory_gb"] if r["model_memory_gb"] > 0 else 0
        
        # Sort by performance/memory ratio (higher is better)
        valid_results.sort(key=lambda r: r["perf_memory_ratio"], reverse=True)
        
        best_result = valid_results[0]
        best_quant = best_result["quantization"] or "none (FP16)"
        
        print("\nQuantization Performance Comparison:")
        for r in valid_results:
            quant_name = r["quantization"] or "none (FP16)"
            print(f"  {quant_name}: {r['tokens_per_second']:.2f} tokens/s, {r['model_memory_gb']:.2f} GB, ratio: {r['perf_memory_ratio']:.2f}")
        
        print(f"\nRecommended quantization: {best_quant}")
        print(f"  Best perf/memory ratio: {best_result['perf_memory_ratio']:.2f}")
    else:
        print("\nNo valid quantization results obtained.")
        best_quant = None
    
    return {
        "quantization_results": results,
        "recommended_quantization": best_quant if best_quant != "none (FP16)" else None
    }

def optimize_generation_parameters(model_name, quantization=None):
    """Optimize generation parameters for best performance"""
    print(f"\nOptimizing generation parameters for {model_name} (quantization: {quantization or 'none'})...")
    
    # Test different temperature values
    temperatures = [0.0, 0.7, 1.0]
    # Test different repetition penalties
    rep_penalties = [1.0, 1.1, 1.2]
    
    results = []
    
    for temp in temperatures:
        for rep_penalty in rep_penalties:
            print(f"Testing temperature={temp}, repetition_penalty={rep_penalty}...")
            
            result = run_mlx_benchmark(
                model_name=model_name,
                quantization=quantization,
                device="gpu",
                temperature=temp,
                repetition_penalty=rep_penalty
            )
            
            if result["success"]:
                print(f"  Performance: {result['tokens_per_second']:.2f} tokens/second")
                # Add parameters to result
                result["temperature"] = temp
                result["repetition_penalty"] = rep_penalty
                results.append(result)
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Find optimal parameters
    valid_results = [r for r in results if r["success"]]
    
    if valid_results:
        # Sort by tokens per second (higher is better)
        valid_results.sort(key=lambda r: r["tokens_per_second"], reverse=True)
        
        best_result = valid_results[0]
        
        print("\nGeneration Parameter Comparison:")
        for r in valid_results:
            print(f"  temp={r['temperature']}, rep_penalty={r['repetition_penalty']}: {r['tokens_per_second']:.2f} tokens/s")
        
        print(f"\nRecommended parameters:")
        print(f"  Temperature: {best_result['temperature']}")
        print(f"  Repetition Penalty: {best_result['repetition_penalty']}")
    else:
        print("\nNo valid parameter optimization results obtained.")
        best_result = {
            "temperature": 0.7,  # Default
            "repetition_penalty": 1.1  # Default
        }
    
    return {
        "parameter_results": results,
        "recommended_parameters": {
            "temperature": best_result["temperature"],
            "repetition_penalty": best_result["repetition_penalty"]
        }
    }

def create_optimized_script(output_path, model_name, quantization=None, 
                          temperature=0.7, repetition_penalty=1.1):
    """Create an optimized script for running the model with the best settings"""
    script_content = f"""#!/usr/bin/env python3
# Optimized MLX settings for {model_name}
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# 
# This script uses the optimal settings determined by the MLX Metal optimizer

import mlx.core as mx
from mlx_lm import load, generate

# Set device to GPU for optimal performance
mx.set_default_device(mx.gpu)

# Load model with optimal settings
model, tokenizer = load(
    "{model_name}",
    quantization="{quantization}" if "{quantization}" != "None" else None
)

def chat():
    \"\"\"Interactive chat function with optimized settings\"\"\"
    history = "You are a helpful assistant.\\n\\n"
    print("Assistant: Hello! How can I help you today?")
    
    while True:
        user_input = input("\\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        prompt = history + f"User: {{user_input}}\\nAssistant: "
        history = prompt
        
        print("\\nAssistant: ", end="", flush=True)
        
        # Generate with optimal settings
        for token in generate(
            model,
            tokenizer,
            prompt,
            max_tokens=512,
            temp={temperature},
            top_p=0.95,
            repetition_penalty={repetition_penalty},
            stream=True
        ):
            print(token, end="", flush=True)
            history += token
        
        history += "\\n"

if __name__ == "__main__":
    print("Starting optimized MLX chat with the following settings:")
    print(f"  Model: {model_name}")
    print(f"  Quantization: {quantization or 'none (FP16)'}")
    print(f"  Temperature: {temperature}")
    print(f"  Repetition Penalty: {repetition_penalty}")
    print(f"  Device: GPU (Metal)")
    print("\\nType 'exit' or 'quit' to end the conversation.\\n")
    
    chat()
"""

    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(output_path, 0o755)
    
    return output_path

def main():
    if not MLX_AVAILABLE:
        print("Error: MLX not available. Please install MLX and MLX-LM.")
        return
    
    parser = argparse.ArgumentParser(description="Optimize Metal GPU settings for MLX")
    parser.add_argument("--model", required=True, help="Model name (e.g., llama-2-7b)")
    parser.add_argument("--quant", choices=["none", "int8", "int4"], help="Initial quantization to test")
    parser.add_argument("--optimize-quant", action="store_true", help="Optimize quantization")
    parser.add_argument("--optimize-params", action="store_true", help="Optimize generation parameters")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--script", help="Output path for optimized script")
    args = parser.parse_args()
    
    # Convert "none" to None for quantization
    quantization = None if args.quant == "none" else args.quant
    
    # Get GPU information
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info.get('name', 'Unknown')}")
    print(f"Metal Support: {gpu_info.get('metal_support', False)}")
    
    if gpu_info.get('metal_support', False) == False:
        print("Warning: Metal support not detected. This script is designed for Apple Silicon with Metal support.")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return
    
    # Run benchmarks and optimizations
    all_results = {}
    
    # Analyze device performance (GPU vs CPU)
    device_analysis = analyze_device_performance(model_name=args.model, quantization=quantization)
    all_results["device_analysis"] = device_analysis
    
    # Optimize quantization if requested
    quantization_results = None
    if args.optimize_quant:
        quantization_results = optimize_quantization(model_name=args.model)
        all_results["quantization_optimization"] = quantization_results
        
        # Update quantization for parameter optimization if better quantization found
        if quantization_results["recommended_quantization"] is not None:
            quantization = quantization_results["recommended_quantization"]
            print(f"Using recommended quantization ({quantization}) for parameter optimization")
    
    # Optimize generation parameters if requested
    parameter_results = None
    if args.optimize_params:
        parameter_results = optimize_generation_parameters(model_name=args.model, quantization=quantization)
        all_results["parameter_optimization"] = parameter_results
    
    # Compile final recommendations
    recommendations = {
        "system_info": {
            "os": platform.platform(),
            "gpu": gpu_info,
            "metal_support": gpu_info.get('metal_support', False)
        },
        "model_name": args.model,
        "recommendations": {
            "device": device_analysis["recommended_device"],
            "quantization": quantization_results["recommended_quantization"] if quantization_results else quantization,
            "temperature": parameter_results["recommended_parameters"]["temperature"] if parameter_results else 0.7,
            "repetition_penalty": parameter_results["recommended_parameters"]["repetition_penalty"] if parameter_results else 1.1,
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Print final recommendations
    print("\n==== FINAL RECOMMENDATIONS ====")
    print(f"Model: {args.model}")
    print(f"Device: {recommendations['recommendations']['device']}")
    print(f"Quantization: {recommendations['recommendations']['quantization'] or 'none (FP16)'}")
    print(f"Temperature: {recommendations['recommendations']['temperature']}")
    print(f"Repetition Penalty: {recommendations['recommendations']['repetition_penalty']}")
    
    # Example code
    print("\nOptimal Python Code:")
    print("```python")
    print("import mlx.core as mx")
    print("from mlx_lm import load, generate")
    print("")
    print("# Set device")
    print(f"mx.set_default_device(mx.{recommendations['recommendations']['device']})")
    print("")
    print("# Load model with optimal settings")
    if recommendations['recommendations']['quantization']:
        print(f"model, tokenizer = load(\"{args.model}\", quantization=\"{recommendations['recommendations']['quantization']}\")")
    else:
        print(f"model, tokenizer = load(\"{args.model}\")")
    print("")
    print("# Generate with optimal settings")
    print("tokens = generate(")
    print("    model,")
    print("    tokenizer,")
    print("    \"Your prompt here\",")
    print("    max_tokens=512,")
    print(f"    temp={recommendations['recommendations']['temperature']},")
    print("    top_p=0.95,")
    print(f"    repetition_penalty={recommendations['recommendations']['repetition_penalty']}")
    print(")")
    print("```")
    
    # Save results if output file specified
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output, 'w') as f:
            json.dump({**recommendations, "detailed_results": all_results}, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Create optimized script if requested
    if args.script:
        script_path = create_optimized_script(
            output_path=args.script,
            model_name=args.model,
            quantization=recommendations['recommendations']['quantization'],
            temperature=recommendations['recommendations']['temperature'],
            repetition_penalty=recommendations['recommendations']['repetition_penalty']
        )
        print(f"\nOptimized script created at {script_path}")
        print(f"Run with: python {script_path}")

if __name__ == "__main__":
    main()
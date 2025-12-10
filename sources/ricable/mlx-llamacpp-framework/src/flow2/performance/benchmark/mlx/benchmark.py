#!/usr/bin/env python3
"""
Benchmark Utility for MLX models on Apple Silicon.

This script benchmarks MLX models with various configurations to measure
inference speed, memory usage, and other performance metrics.

Usage:
    python benchmark.py --model <model_name> [--options]

Example:
    python benchmark.py --model llama-2-7b --quant int4 --ctx 2048,4096,8192
"""

import argparse
import os
import time
import json
import platform
import psutil
import numpy as np
from datetime import datetime
import csv
import tempfile
import threading
import gc
import traceback
import subprocess

# Third-party imports conditionally handled
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. This script requires MLX and MLX-LM to be installed.")

# Flash Attention Integration
try:
    from flash_attention_mlx import OptimizedMLXMultiHeadAttention, FlashAttentionBenchmark
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

# Constants
DEFAULT_PROMPT_SHORT = "Explain quantum computing briefly."
DEFAULT_PROMPT_MEDIUM = "Write a short essay on the impact of artificial intelligence on society, discussing both potential benefits and concerns."
DEFAULT_PROMPT_LONG = "Explain the history, theory, and practical applications of deep learning, including major milestones, key algorithms, and its impact on various fields such as computer vision, natural language processing, and reinforcement learning."

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

def get_system_info():
    """Get detailed system information"""
    mem = psutil.virtual_memory()
    
    # Get additional Mac-specific information
    mac_info = {}
    try:
        # Get Mac model
        mac_model = subprocess.getoutput("sysctl hw.model").replace("hw.model: ", "")
        mac_info["model"] = mac_model
        
        # Get CPU details
        cpu_brand = subprocess.getoutput("sysctl machdep.cpu.brand_string").replace("machdep.cpu.brand_string: ", "")
        cpu_cores = int(subprocess.getoutput("sysctl hw.ncpu").replace("hw.ncpu: ", ""))
        mac_info["cpu_brand"] = cpu_brand
        mac_info["cpu_cores"] = cpu_cores
        
        # Check for Apple Silicon
        if "Apple" in platform.processor():
            apple_silicon = True
            # Try to determine the specific chip
            if "M1" in cpu_brand:
                chip = "M1"
            elif "M2" in cpu_brand:
                chip = "M2"
            elif "M3" in cpu_brand:
                chip = "M3"
            else:
                chip = "Apple Silicon (unknown)"
            
            # Try to determine variant (Base/Pro/Max/Ultra)
            if "Max" in cpu_brand:
                variant = "Max"
            elif "Ultra" in cpu_brand:
                variant = "Ultra"
            elif "Pro" in cpu_brand:
                variant = "Pro"
            else:
                variant = "Base"
            
            mac_info["chip"] = chip
            mac_info["variant"] = variant
        else:
            apple_silicon = False
            mac_info["chip"] = "Intel"
            mac_info["variant"] = "N/A"
        
        mac_info["apple_silicon"] = apple_silicon
        
    except Exception as e:
        print(f"Error getting Mac details: {e}")
        mac_info = {"error": str(e)}
    
    # Get Metal support info
    metal_info = {}
    try:
        # This is a simple check - a more thorough check would use the Metal framework
        metal_devices = subprocess.getoutput("system_profiler SPDisplaysDataType | grep Metal")
        metal_info["supported"] = "Metal" in metal_devices
    except:
        metal_info["supported"] = False
    
    return {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "ram_total_gb": mem.total / (1024 ** 3),  # Convert to GB
        "ram_available_gb": mem.available / (1024 ** 3),  # Convert to GB
        "mac_info": mac_info,
        "metal_info": metal_info,
        "timestamp": datetime.now().isoformat()
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

def extract_model_info(model_name):
    """Extract model information from model name"""
    # Extract model size
    model_size = estimate_model_size(model_name)
    
    # Try to extract model family/name
    model_families = ["llama", "mistral", "falcon", "mpt", "stablelm", "phi"]
    model_family = "unknown"
    for family in model_families:
        if family in model_name.lower():
            model_family = family
            break
    
    return {
        "model_name": model_name,
        "model_size": model_size,
        "model_family": model_family
    }


def apply_flash_attention_to_model(model, use_flash_attention=True, block_size=None):
    """
    Apply Flash Attention optimizations to model attention layers
    """
    if not use_flash_attention or not FLASH_ATTENTION_AVAILABLE:
        return model, 0
    
    attention_replacements = 0
    
    def replace_attention_recursive(module, name_prefix=""):
        nonlocal attention_replacements
        
        # Handle MLX models which may have different attribute access patterns
        try:
            for name in dir(module):
                if name.startswith('_') or name in ['training', 'parameters', 'modules']:
                    continue
                    
                try:
                    child = getattr(module, name)
                    if not hasattr(child, '__class__'):
                        continue
                        
                    full_name = f"{name_prefix}.{name}" if name_prefix else name
                    
                    # Check if this is an attention layer we should replace
                    if hasattr(child, '__class__') and 'MultiHeadAttention' in str(child.__class__):
                        # Create optimized replacement
                        try:
                            flash_attention = OptimizedMLXMultiHeadAttention(
                                child.dims,
                                child.num_heads,
                                bias=hasattr(child, 'bias'),
                                use_flash_attention=True,
                                block_size=block_size
                            )
                            
                            # Copy weights from original layer
                            if hasattr(child, 'q_proj') and hasattr(child.q_proj, 'weight'):
                                flash_attention.q_proj.weight = child.q_proj.weight
                                flash_attention.k_proj.weight = child.k_proj.weight  
                                flash_attention.v_proj.weight = child.v_proj.weight
                                flash_attention.out_proj.weight = child.out_proj.weight
                                
                                if hasattr(child.q_proj, 'bias') and child.q_proj.bias is not None:
                                    flash_attention.q_proj.bias = child.q_proj.bias
                                    flash_attention.k_proj.bias = child.k_proj.bias
                                    flash_attention.v_proj.bias = child.v_proj.bias
                                    flash_attention.out_proj.bias = child.out_proj.bias
                            
                            # Replace the layer
                            setattr(module, name, flash_attention)
                            attention_replacements += 1
                        except Exception:
                            pass  # Skip failed replacements
                    else:
                        # Recursively process child modules
                        replace_attention_recursive(child, full_name)
                        
                except (AttributeError, TypeError):
                    continue
                    
        except (AttributeError, TypeError):
            pass
    
    try:
        replace_attention_recursive(model)
    except Exception:
        pass  # Continue with standard attention if Flash Attention fails
    
    return model, attention_replacements

def run_benchmark(model_name, prompt, max_tokens=256, quantization=None, 
                  temperature=0.7, top_p=0.95, top_k=40, repetition_penalty=1.1,
                  n_runs=3, use_flash_attention=True, flash_block_size=None):
    """
    Run MLX benchmark with specified parameters
    """
    if not MLX_AVAILABLE:
        return {
            "error": "MLX not available. Please install MLX and MLX-LM."
        }
    
    results = []
    
    for i in range(n_runs):
        try:
            # Start memory monitoring for current process
            monitor = MemoryMonitor(os.getpid())
            monitor.start()
            
            # Clear previous models and cache
            gc.collect()
            mx.clear_cache()
            
            # Record base memory before loading
            base_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
            
            # Load model - measure time
            load_start = time.time()
            model, tokenizer = load(model_name, quantization=quantization)
            
            # Apply Flash Attention optimizations
            flash_replacements = 0
            if use_flash_attention and FLASH_ATTENTION_AVAILABLE:
                model, flash_replacements = apply_flash_attention_to_model(
                    model, 
                    use_flash_attention=use_flash_attention,
                    block_size=flash_block_size
                )
            
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
            generation = generate(
                model, 
                tokenizer, 
                prompt, 
                max_tokens=max_tokens, 
                temp=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            generate_end = time.time()
            generate_time = generate_end - generate_start
            
            # Calculate tokens per second
            output_tokens = len(generation)
            tokens_per_second = output_tokens / generate_time if generate_time > 0 else 0
            
            # Clean up
            del model, tokenizer, generation
            gc.collect()
            mx.clear_cache()
            
            # Stop memory monitoring
            monitor.stop()
            monitor.join()
            
            results.append({
                "run": i + 1,
                "load_time_seconds": load_time,
                "encode_time_seconds": encode_time,
                "generate_time_seconds": generate_time,
                "total_time_seconds": load_time + encode_time + generate_time,
                "tokens_per_second": tokens_per_second,
                "output_tokens": output_tokens,
                "base_memory_gb": base_memory,
                "peak_memory_gb": monitor.peak_memory,
                "model_memory_gb": monitor.peak_memory - base_memory,
                "memory_samples": monitor.memory_usage,
                "flash_attention_layers": flash_replacements
            })
            
        except Exception as e:
            print(f"Error during benchmark run {i+1}: {e}")
            print(traceback.format_exc())
            results.append({
                "run": i + 1,
                "error": str(e)
            })
    
    # Calculate averages and statistics
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        return {
            "error": "All benchmark runs failed",
            "runs": results
        }
    
    avg_load_time = np.mean([r["load_time_seconds"] for r in valid_results])
    avg_generate_time = np.mean([r["generate_time_seconds"] for r in valid_results])
    avg_tokens_per_second = np.mean([r["tokens_per_second"] for r in valid_results])
    avg_peak_memory = np.mean([r["peak_memory_gb"] for r in valid_results])
    avg_model_memory = np.mean([r["model_memory_gb"] for r in valid_results])
    
    return {
        "avg_load_time_seconds": avg_load_time,
        "avg_generate_time_seconds": avg_generate_time,
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_peak_memory_gb": avg_peak_memory,
        "avg_model_memory_gb": avg_model_memory,
        "runs": results
    }

def run_comprehensive_benchmark(model_name, quantization_types, max_tokens_list, 
                               temperature_list, repetition_penalty_list, output_file=None,
                               use_flash_attention=True, flash_block_size=None):
    """
    Run comprehensive benchmarks across different configurations
    """
    if not MLX_AVAILABLE:
        print("Error: MLX not available. Please install MLX and MLX-LM.")
        return {
            "error": "MLX not available. Please install MLX and MLX-LM."
        }
    
    system_info = get_system_info()
    model_info = extract_model_info(model_name)
    
    print(f"System: {system_info['mac_info'].get('model', 'Unknown Mac')}")
    print(f"Chip: {system_info['mac_info'].get('chip', 'Unknown')} {system_info['mac_info'].get('variant', '')}")
    print(f"RAM: {system_info['ram_total_gb']:.2f} GB")
    print(f"Model: {model_info['model_name']}")
    print(f"Model Size: {model_info['model_size']}, Family: {model_info['model_family']}")
    print(f"Metal Support: {system_info['metal_info'].get('supported', False)}")
    print("\nRunning comprehensive benchmarks. This may take some time...\n")
    
    # Generate prompt variants (short, medium, long)
    prompts = {
        "short": DEFAULT_PROMPT_SHORT,
        "medium": DEFAULT_PROMPT_MEDIUM,
        "long": DEFAULT_PROMPT_LONG
    }
    
    all_results = []
    total_configs = len(quantization_types) * len(max_tokens_list) * len(temperature_list) * len(repetition_penalty_list) * len(prompts)
    completed = 0
    
    for quant in quantization_types:
        for max_tokens in max_tokens_list:
            for temp in temperature_list:
                for rep_penalty in repetition_penalty_list:
                    for prompt_type, prompt in prompts.items():
                        config_desc = f"Quant: {quant or 'none'}, Max Tokens: {max_tokens}, Temp: {temp}, Rep Penalty: {rep_penalty}, Prompt: {prompt_type}"
                        print(f"Running benchmark {completed+1}/{total_configs}: {config_desc}")
                        
                        # Run the benchmark for this configuration
                        benchmark_result = run_benchmark(
                            model_name=model_name,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            quantization=quant,
                            temperature=temp,
                            repetition_penalty=rep_penalty,
                            n_runs=1,  # Use 1 run per config to save time in comprehensive benchmark
                            use_flash_attention=use_flash_attention,
                            flash_block_size=flash_block_size
                        )
                        
                        # Add configuration details to result
                        result_entry = {
                            "quantization": quant,
                            "max_tokens": max_tokens,
                            "temperature": temp,
                            "repetition_penalty": rep_penalty,
                            "prompt_type": prompt_type,
                            "benchmark_result": benchmark_result
                        }
                        
                        all_results.append(result_entry)
                        
                        # Print summary of this configuration
                        if "error" in benchmark_result:
                            print(f"  Error: {benchmark_result['error']}")
                        else:
                            print(f"  Tokens/second: {benchmark_result['avg_tokens_per_second']:.2f}")
                            print(f"  Peak Memory: {benchmark_result['avg_peak_memory_gb']:.2f} GB")
                        
                        completed += 1
                        print("")
    
    # Find optimal configurations
    valid_results = [r for r in all_results if "error" not in r["benchmark_result"]]
    
    if valid_results:
        # Optimal for speed
        speed_optimal = max(valid_results, key=lambda r: r["benchmark_result"]["avg_tokens_per_second"])
        
        # Optimal for memory efficiency
        memory_optimal = min(valid_results, key=lambda r: r["benchmark_result"]["avg_peak_memory_gb"])
        
        # Best balance (simplified - you could use a weighted score)
        def balance_score(result):
            speed = result["benchmark_result"]["avg_tokens_per_second"]
            memory = result["benchmark_result"]["avg_peak_memory_gb"]
            # Normalize and combine (higher is better)
            max_speed = max(r["benchmark_result"]["avg_tokens_per_second"] for r in valid_results)
            max_memory = max(r["benchmark_result"]["avg_peak_memory_gb"] for r in valid_results)
            speed_norm = speed / max_speed
            memory_norm = 1 - (memory / max_memory)  # Lower memory is better
            return 0.7 * speed_norm + 0.3 * memory_norm  # Weight speed higher
        
        balance_optimal = max(valid_results, key=balance_score)
    else:
        speed_optimal = {"error": "No valid results"}
        memory_optimal = {"error": "No valid results"}
        balance_optimal = {"error": "No valid results"}
    
    # Compile comprehensive results
    comprehensive_results = {
        "system_info": system_info,
        "model_info": model_info,
        "benchmark_configs": {
            "quantization_types": quantization_types,
            "max_tokens_list": max_tokens_list,
            "temperature_list": temperature_list,
            "repetition_penalty_list": repetition_penalty_list,
        },
        "all_results": all_results,
        "optimal_configs": {
            "speed": speed_optimal,
            "memory": memory_optimal,
            "balance": balance_optimal
        }
    }
    
    # Print optimal configurations
    print("\n==== OPTIMAL CONFIGURATIONS ====")
    
    if "error" not in speed_optimal:
        print("\nBest for Speed:")
        print(f"  Quantization: {speed_optimal['quantization'] or 'none (FP16)'}, Max Tokens: {speed_optimal['max_tokens']}")
        print(f"  Temperature: {speed_optimal['temperature']}, Rep Penalty: {speed_optimal['repetition_penalty']}")
        print(f"  Tokens/second: {speed_optimal['benchmark_result']['avg_tokens_per_second']:.2f}")
    
    if "error" not in memory_optimal:
        print("\nBest for Memory Efficiency:")
        print(f"  Quantization: {memory_optimal['quantization'] or 'none (FP16)'}, Max Tokens: {memory_optimal['max_tokens']}")
        print(f"  Peak Memory: {memory_optimal['benchmark_result']['avg_peak_memory_gb']:.2f} GB")
    
    if "error" not in balance_optimal:
        print("\nBest Balance of Speed and Memory:")
        print(f"  Quantization: {balance_optimal['quantization'] or 'none (FP16)'}, Max Tokens: {balance_optimal['max_tokens']}")
        print(f"  Temperature: {balance_optimal['temperature']}, Rep Penalty: {balance_optimal['repetition_penalty']}")
        print(f"  Tokens/second: {balance_optimal['benchmark_result']['avg_tokens_per_second']:.2f}")
        print(f"  Peak Memory: {balance_optimal['benchmark_result']['avg_peak_memory_gb']:.2f} GB")
    
    # Save results if output file specified
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        # Also save as CSV for easier analysis
        csv_file = os.path.splitext(output_file)[0] + ".csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow([
                "Quantization", "Max Tokens", "Temperature", "Rep Penalty", 
                "Prompt Type", "Tokens/second", "Peak Memory (GB)"
            ])
            
            # Write data
            for result in all_results:
                if "error" not in result["benchmark_result"]:
                    writer.writerow([
                        result["quantization"] or "none (FP16)",
                        result["max_tokens"],
                        result["temperature"],
                        result["repetition_penalty"],
                        result["prompt_type"],
                        result["benchmark_result"]["avg_tokens_per_second"],
                        result["benchmark_result"]["avg_peak_memory_gb"]
                    ])
        print(f"CSV summary saved to {csv_file}")
    
    return comprehensive_results

def parse_comma_separated_ints(value):
    """Parse comma-separated integers from command-line arguments"""
    return [int(x) for x in value.split(',')]

def parse_comma_separated_floats(value):
    """Parse comma-separated floats from command-line arguments"""
    return [float(x) for x in value.split(',')]

def parse_comma_separated_quants(value):
    """Parse comma-separated quantization values"""
    result = []
    for x in value.split(','):
        if x.lower() == "none" or x.lower() == "fp16":
            result.append(None)  # None means no quantization (FP16)
        else:
            result.append(x)
    return result

def main():
    if not MLX_AVAILABLE:
        print("Error: MLX not available. Please install MLX and MLX-LM.")
        return
    
    parser = argparse.ArgumentParser(description="Benchmark MLX models across different configurations")
    parser.add_argument("--model", required=True, help="Model name (e.g., llama-2-7b)")
    parser.add_argument("--quant", default="none,int8,int4", help="Comma-separated list of quantization types (none, int8, int4)")
    parser.add_argument("--max-tokens", default="256", help="Comma-separated list of max token counts")
    parser.add_argument("--temp", default="0.7", help="Comma-separated list of temperature values")
    parser.add_argument("--rep-penalty", default="1.1", help="Comma-separated list of repetition penalty values")
    parser.add_argument("--quick", action="store_true", help="Run a quick benchmark with fewer configurations")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--disable-flash-attention", action="store_true", help="Disable Flash Attention optimization")
    parser.add_argument("--flash-block-size", type=int, default=None, help="Flash Attention block size (auto if None)")
    args = parser.parse_args()
    
    # Parse lists of configurations to test
    try:
        quantization_types = parse_comma_separated_quants(args.quant)
        max_tokens_list = parse_comma_separated_ints(args.max_tokens)
        temperature_list = parse_comma_separated_floats(args.temp)
        repetition_penalty_list = parse_comma_separated_floats(args.rep_penalty)
    except ValueError:
        print("Error: Invalid format for one of the parameters.")
        print("Use comma-separated values, e.g., --quant none,int8,int4")
        return
    
    # If quick mode, use smaller sets of configurations
    if args.quick:
        if len(quantization_types) > 1:
            quantization_types = [quantization_types[0]]
        if len(max_tokens_list) > 1:
            max_tokens_list = [max_tokens_list[0]]
        if len(temperature_list) > 1:
            temperature_list = [temperature_list[0]]
        if len(repetition_penalty_list) > 1:
            repetition_penalty_list = [repetition_penalty_list[0]]
    
    # Run the comprehensive benchmark
    run_comprehensive_benchmark(
        model_name=args.model,
        quantization_types=quantization_types,
        max_tokens_list=max_tokens_list,
        temperature_list=temperature_list,
        repetition_penalty_list=repetition_penalty_list,
        output_file=args.output,
        use_flash_attention=not args.disable_flash_attention,
        flash_block_size=args.flash_block_size
    )

if __name__ == "__main__":
    main()
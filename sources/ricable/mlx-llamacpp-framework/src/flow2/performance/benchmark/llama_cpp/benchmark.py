#!/usr/bin/env python3
"""
Benchmark Utility for llama.cpp models on Apple Silicon.

This script benchmarks llama.cpp models with various configurations to measure
inference speed, memory usage, and other performance metrics.

Usage:
    python benchmark.py --model <model_path> [--options]

Example:
    python benchmark.py --model models/llama-2-7b-q4_k.gguf --ctx 2048,4096,8192 --threads 4,8
"""

import argparse
import subprocess
import re
import os
import time
import json
import platform
import psutil
import numpy as np
from datetime import datetime
import csv
import multiprocessing
import tempfile

# Constants
DEFAULT_PROMPT_SHORT = "Explain quantum computing briefly."
DEFAULT_PROMPT_MEDIUM = "Write a short essay on the impact of artificial intelligence on society, discussing both potential benefits and concerns."
DEFAULT_PROMPT_LONG = "Explain the history, theory, and practical applications of deep learning, including major milestones, key algorithms, and its impact on various fields such as computer vision, natural language processing, and reinforcement learning."

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

def run_benchmark(llama_cpp_path, model_path, prompt, n_tokens=128, n_predict=256, 
                  context_length=2048, threads=4, use_metal=True, metal_mmq=True, 
                  batch_size=512, n_runs=3):
    """
    Run llama.cpp benchmark with specified parameters
    """
    results = []
    
    for i in range(n_runs):
        # Create a temporary file for the prompt
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(prompt)
            prompt_file = f.name
        
        try:
            # Construct command
            cmd = [
                os.path.join(llama_cpp_path, "main"),
                "-m", model_path,
                "-f", prompt_file,
                "-n", str(n_predict),
                "-c", str(context_length),
                "-t", str(threads),
                "-b", str(batch_size),
                "--log-disable"  # Disable most logs for cleaner output
            ]
            
            if use_metal:
                cmd.append("--metal")
                if metal_mmq:
                    cmd.append("--metal-mmq")
            
            # Measure memory before
            mem_before = psutil.virtual_memory().available / (1024 ** 3)
            
            # Start timing
            start_time = time.time()
            
            # Run the process
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Track memory during execution
            memory_samples = []
            peak_memory = 0
            pid = process.pid
            
            while process.poll() is None:
                try:
                    proc = psutil.Process(pid)
                    mem_info = proc.memory_info()
                    current_mem = mem_info.rss / (1024 ** 3)  # Convert to GB
                    memory_samples.append(current_mem)
                    peak_memory = max(peak_memory, current_mem)
                    time.sleep(0.1)
                except:
                    # Process might have ended
                    break
            
            # Get output
            stdout, stderr = process.communicate()
            
            # End timing
            end_time = time.time()
            total_time = end_time - start_time
            
            # Measure memory after
            mem_after = psutil.virtual_memory().available / (1024 ** 3)
            mem_diff = mem_before - mem_after
            
            # Process output
            output = stdout.decode()
            error_output = stderr.decode()
            
            # Parse timing and tokens per second
            tokens_per_second = None
            tokens_per_second_match = re.search(r"(\d+\.\d+) tokens/s", output)
            if tokens_per_second_match:
                tokens_per_second = float(tokens_per_second_match.group(1))
            
            # Parse load time
            load_time = None
            load_time_match = re.search(r"llama_model_load: loaded in (\d+\.\d+) ms", output)
            if load_time_match:
                load_time = float(load_time_match.group(1)) / 1000  # convert to seconds
            
            # Extract any other relevant metrics
            # ...
            
            results.append({
                "run": i + 1,
                "total_time_seconds": total_time,
                "tokens_per_second": tokens_per_second,
                "load_time_seconds": load_time,
                "peak_memory_gb": peak_memory,
                "avg_memory_gb": np.mean(memory_samples) if memory_samples else None,
                "memory_diff_gb": mem_diff,
                "memory_samples": memory_samples
            })
            
        except Exception as e:
            print(f"Error during benchmark run {i+1}: {e}")
            results.append({
                "run": i + 1,
                "error": str(e)
            })
        finally:
            # Clean up temporary file
            try:
                os.unlink(prompt_file)
            except:
                pass
    
    # Calculate averages and statistics
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        return {
            "error": "All benchmark runs failed",
            "runs": results
        }
    
    avg_total_time = np.mean([r["total_time_seconds"] for r in valid_results])
    avg_tokens_per_second = np.mean([r["tokens_per_second"] for r in valid_results if r["tokens_per_second"] is not None])
    avg_peak_memory = np.mean([r["peak_memory_gb"] for r in valid_results if r["peak_memory_gb"] is not None])
    
    return {
        "avg_total_time_seconds": avg_total_time,
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_peak_memory_gb": avg_peak_memory,
        "runs": results
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
    
    # Try to extract model family/name
    model_families = ["llama", "mistral", "falcon", "mpt", "stablelm", "phi"]
    model_family = "unknown"
    for family in model_families:
        if family in filename.lower():
            model_family = family
            break
    
    return {
        "filename": filename,
        "model_size": model_size,
        "quantization": quantization,
        "model_family": model_family
    }

def run_comprehensive_benchmark(llama_cpp_path, model_path, context_lengths, thread_counts, 
                                batch_sizes, use_metal=True, metal_mmq=True, output_file=None):
    """
    Run comprehensive benchmarks across different configurations
    """
    system_info = get_system_info()
    model_info = extract_model_info(model_path)
    
    print(f"System: {system_info['mac_info'].get('model', 'Unknown Mac')}")
    print(f"Chip: {system_info['mac_info'].get('chip', 'Unknown')} {system_info['mac_info'].get('variant', '')}")
    print(f"RAM: {system_info['ram_total_gb']:.2f} GB")
    print(f"Model: {model_info['filename']}")
    print(f"Model Size: {model_info['model_size']}, Quantization: {model_info['quantization']}")
    print(f"Metal Support: {system_info['metal_info'].get('supported', False)}")
    print("\nRunning comprehensive benchmarks. This may take some time...\n")
    
    # Generate prompt variants (short, medium, long)
    prompts = {
        "short": DEFAULT_PROMPT_SHORT,
        "medium": DEFAULT_PROMPT_MEDIUM,
        "long": DEFAULT_PROMPT_LONG
    }
    
    all_results = []
    total_configs = len(context_lengths) * len(thread_counts) * len(batch_sizes) * len(prompts)
    completed = 0
    
    for ctx in context_lengths:
        for threads in thread_counts:
            for batch in batch_sizes:
                for prompt_type, prompt in prompts.items():
                    config_desc = f"Context: {ctx}, Threads: {threads}, Batch: {batch}, Prompt: {prompt_type}"
                    print(f"Running benchmark {completed+1}/{total_configs}: {config_desc}")
                    
                    # Run the benchmark for this configuration
                    benchmark_result = run_benchmark(
                        llama_cpp_path=llama_cpp_path,
                        model_path=model_path,
                        prompt=prompt,
                        context_length=ctx,
                        threads=threads,
                        batch_size=batch,
                        use_metal=use_metal,
                        metal_mmq=metal_mmq,
                        n_runs=1  # Use 1 run per config to save time in comprehensive benchmark
                    )
                    
                    # Add configuration details to result
                    result_entry = {
                        "context_length": ctx,
                        "threads": threads,
                        "batch_size": batch,
                        "prompt_type": prompt_type,
                        "metal": use_metal,
                        "metal_mmq": metal_mmq,
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
            "context_lengths": context_lengths,
            "thread_counts": thread_counts,
            "batch_sizes": batch_sizes,
            "metal": use_metal,
            "metal_mmq": metal_mmq
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
        print(f"  Context: {speed_optimal['context_length']}, Threads: {speed_optimal['threads']}, Batch: {speed_optimal['batch_size']}")
        print(f"  Tokens/second: {speed_optimal['benchmark_result']['avg_tokens_per_second']:.2f}")
    
    if "error" not in memory_optimal:
        print("\nBest for Memory Efficiency:")
        print(f"  Context: {memory_optimal['context_length']}, Threads: {memory_optimal['threads']}, Batch: {memory_optimal['batch_size']}")
        print(f"  Peak Memory: {memory_optimal['benchmark_result']['avg_peak_memory_gb']:.2f} GB")
    
    if "error" not in balance_optimal:
        print("\nBest Balance of Speed and Memory:")
        print(f"  Context: {balance_optimal['context_length']}, Threads: {balance_optimal['threads']}, Batch: {balance_optimal['batch_size']}")
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
                "Context Length", "Threads", "Batch Size", "Prompt Type", 
                "Metal", "Metal MMQ", "Tokens/second", "Peak Memory (GB)"
            ])
            
            # Write data
            for result in all_results:
                if "error" not in result["benchmark_result"]:
                    writer.writerow([
                        result["context_length"],
                        result["threads"],
                        result["batch_size"],
                        result["prompt_type"],
                        result["metal"],
                        result["metal_mmq"],
                        result["benchmark_result"]["avg_tokens_per_second"],
                        result["benchmark_result"]["avg_peak_memory_gb"]
                    ])
        print(f"CSV summary saved to {csv_file}")
    
    return comprehensive_results

def parse_comma_separated_ints(value):
    """Parse comma-separated integers from command-line arguments"""
    return [int(x) for x in value.split(',')]

def main():
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp models across different configurations")
    parser.add_argument("--model", required=True, help="Path to the model file (.gguf)")
    parser.add_argument("--llama_cpp", default="./llama.cpp", help="Path to llama.cpp directory")
    parser.add_argument("--ctx", default="2048", help="Comma-separated list of context lengths to test")
    parser.add_argument("--threads", default="4", help="Comma-separated list of thread counts to test")
    parser.add_argument("--batch", default="512", help="Comma-separated list of batch sizes to test")
    parser.add_argument("--no-metal", action="store_true", help="Disable Metal acceleration")
    parser.add_argument("--no-metal-mmq", action="store_true", help="Disable Metal matrix multiplication")
    parser.add_argument("--quick", action="store_true", help="Run a quick benchmark with fewer configurations")
    parser.add_argument("--output", help="Output file for results (JSON)")
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Parse lists of configurations to test
    try:
        context_lengths = parse_comma_separated_ints(args.ctx)
        thread_counts = parse_comma_separated_ints(args.threads)
        batch_sizes = parse_comma_separated_ints(args.batch)
    except ValueError:
        print("Error: Invalid format for context lengths, thread counts, or batch sizes.")
        print("Use comma-separated integers, e.g., --ctx 2048,4096,8192")
        return
    
    # If quick mode, use smaller sets of configurations
    if args.quick:
        if len(context_lengths) > 1:
            context_lengths = [context_lengths[0]]
        if len(thread_counts) > 1:
            thread_counts = [thread_counts[0]]
        if len(batch_sizes) > 1:
            batch_sizes = [batch_sizes[0]]
    
    # Run the comprehensive benchmark
    run_comprehensive_benchmark(
        llama_cpp_path=args.llama_cpp,
        model_path=args.model,
        context_lengths=context_lengths,
        thread_counts=thread_counts,
        batch_sizes=batch_sizes,
        use_metal=not args.no_metal,
        metal_mmq=not args.no_metal_mmq,
        output_file=args.output
    )

if __name__ == "__main__":
    main()
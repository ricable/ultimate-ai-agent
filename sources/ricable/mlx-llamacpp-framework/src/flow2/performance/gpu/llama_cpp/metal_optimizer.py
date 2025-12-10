#!/usr/bin/env python3
"""
Metal GPU Optimization Utility for llama.cpp on Apple Silicon.

This script analyzes and optimizes Metal GPU acceleration settings for llama.cpp
to maximize performance on Apple Silicon hardware.

Usage:
    python metal_optimizer.py --model <model_path> [--options]

Example:
    python metal_optimizer.py --model models/llama-2-7b-q4_k.gguf --output results/metal_optimized.json
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
import tempfile
import sys

# Constants
DEFAULT_PROMPT = "Explain the theory of relativity in detail, covering both special and general relativity."
DEFAULT_METAL_SETTINGS = [
    {"metal": True, "metal_mmq": False, "batch_size": 512, "group_name": "Metal Basic"},
    {"metal": True, "metal_mmq": True, "batch_size": 512, "group_name": "Metal with MMQ"},
    {"metal": False, "metal_mmq": False, "batch_size": 512, "group_name": "CPU Only"}
]

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
        # This is a simple parsing approach - a more robust one would use regex
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

def estimate_optimal_batch_size(gpu_info):
    """Estimate optimal batch size based on GPU info"""
    if "error" in gpu_info:
        return 512  # Default fallback
    
    # Base batch size on chip type and variant
    if "chip_type" in gpu_info and "variant" in gpu_info:
        chip_type = gpu_info["chip_type"]
        variant = gpu_info["variant"]
        
        # These are estimates and would need to be validated with actual benchmarks
        if chip_type == "M3":
            if variant == "Ultra":
                return 2048
            elif variant == "Max":
                return 1536
            elif variant == "Pro":
                return 1024
            else:  # Base
                return 768
        elif chip_type == "M2":
            if variant == "Ultra":
                return 1536
            elif variant == "Max":
                return 1024
            elif variant == "Pro":
                return 768
            else:  # Base
                return 512
        elif chip_type == "M1":
            if variant == "Ultra":
                return 1024
            elif variant == "Max":
                return 768
            elif variant == "Pro":
                return 512
            else:  # Base
                return 384
    
    # Default fallback
    return 512

def run_metal_benchmark(llama_cpp_path, model_path, prompt=DEFAULT_PROMPT, 
                        n_predict=256, context_length=2048, threads=4, 
                        use_metal=True, metal_mmq=True, batch_size=512):
    """
    Run llama.cpp benchmark with specified Metal settings
    """
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
        
        # Check for Metal-specific errors or warnings
        metal_issues = []
        if "Metal" in error_output:
            for line in error_output.split("\n"):
                if "Metal" in line and ("error" in line.lower() or "warning" in line.lower()):
                    metal_issues.append(line.strip())
        
        return {
            "metal": use_metal,
            "metal_mmq": metal_mmq,
            "batch_size": batch_size,
            "total_time_seconds": total_time,
            "tokens_per_second": tokens_per_second,
            "load_time_seconds": load_time,
            "peak_memory_gb": peak_memory,
            "avg_memory_gb": np.mean(memory_samples) if memory_samples else None,
            "metal_issues": metal_issues,
            "success": tokens_per_second is not None
        }
        
    except Exception as e:
        return {
            "metal": use_metal,
            "metal_mmq": metal_mmq,
            "batch_size": batch_size,
            "error": str(e),
            "success": False
        }
    finally:
        # Clean up temporary file
        try:
            os.unlink(prompt_file)
        except:
            pass

def analyze_metal_performance(llama_cpp_path, model_path, metal_settings, context_length=2048, threads=4):
    """Analyze performance across different Metal settings"""
    results = []
    
    for setting in metal_settings:
        print(f"Testing {setting['group_name']}...")
        
        # Use smaller batch size for initial tests
        result = run_metal_benchmark(
            llama_cpp_path=llama_cpp_path,
            model_path=model_path,
            use_metal=setting["metal"],
            metal_mmq=setting["metal_mmq"],
            batch_size=setting["batch_size"],
            context_length=context_length,
            threads=threads
        )
        
        # Add group name for easier identification
        result["group_name"] = setting["group_name"]
        results.append(result)
        
        # Print immediate result
        if result["success"]:
            print(f"  Performance: {result['tokens_per_second']:.2f} tokens/second")
            print(f"  Load time: {result['load_time_seconds']:.2f} seconds")
            print(f"  Peak memory: {result['peak_memory_gb']:.2f} GB")
        else:
            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print("  Failed to get valid results")
        
        if result["metal_issues"] if "metal_issues" in result else []:
            print("  Metal issues detected:")
            for issue in result["metal_issues"]:
                print(f"    - {issue}")
        
        print("")
    
    return results

def optimize_batch_size(llama_cpp_path, model_path, use_metal=True, metal_mmq=True, 
                       context_length=2048, threads=4, min_batch=128, max_batch=2048):
    """Find the optimal batch size for Metal GPU acceleration"""
    print("Optimizing batch size...")
    batch_sizes = []
    
    # Use exponential search to find a good range
    batch = min_batch
    while batch <= max_batch:
        batch_sizes.append(batch)
        batch *= 2
    
    # Add the maximum as well if not already included
    if max_batch not in batch_sizes:
        batch_sizes.append(max_batch)
    
    results = []
    best_batch = min_batch
    best_performance = 0
    
    for batch in batch_sizes:
        print(f"  Testing batch size {batch}...")
        
        result = run_metal_benchmark(
            llama_cpp_path=llama_cpp_path,
            model_path=model_path,
            use_metal=use_metal,
            metal_mmq=metal_mmq,
            batch_size=batch,
            context_length=context_length,
            threads=threads
        )
        
        results.append(result)
        
        if result["success"] and result["tokens_per_second"] > best_performance:
            best_performance = result["tokens_per_second"]
            best_batch = batch
        
        print(f"    Performance: {result['tokens_per_second']:.2f} tokens/second")
    
    # Refine search around the best batch size using binary search
    if best_batch > min_batch:
        lower_bound = best_batch // 2
        upper_bound = min(best_batch * 2, max_batch)
        
        # Skip if the best batch size is already at the extremes
        if lower_bound < best_batch < upper_bound:
            print("\nRefining batch size search...")
            
            # Try a few intermediate batch sizes
            refinement_batches = [
                lower_bound + (best_batch - lower_bound) // 2,
                best_batch + (upper_bound - best_batch) // 4,
                best_batch + (upper_bound - best_batch) // 2
            ]
            
            for batch in refinement_batches:
                if batch not in batch_sizes:  # Skip if already tested
                    print(f"  Testing batch size {batch}...")
                    
                    result = run_metal_benchmark(
                        llama_cpp_path=llama_cpp_path,
                        model_path=model_path,
                        use_metal=use_metal,
                        metal_mmq=metal_mmq,
                        batch_size=batch,
                        context_length=context_length,
                        threads=threads
                    )
                    
                    results.append(result)
                    
                    if result["success"] and result["tokens_per_second"] > best_performance:
                        best_performance = result["tokens_per_second"]
                        best_batch = batch
                    
                    print(f"    Performance: {result['tokens_per_second']:.2f} tokens/second")
    
    print(f"\nOptimal batch size: {best_batch}")
    print(f"Best performance: {best_performance:.2f} tokens/second")
    
    return {
        "optimal_batch_size": best_batch,
        "best_performance": best_performance,
        "all_results": results
    }

def create_optimized_script(output_path, model_path, context_length, batch_size, 
                          use_metal=True, metal_mmq=True, threads=4):
    """Create an optimized script for running the model with the best settings"""
    script_content = f"""#!/bin/bash
# Optimized Metal GPU settings for llama.cpp
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Path to the model
MODEL="{model_path}"

# Optimal settings determined by metal_optimizer.py
CONTEXT_LENGTH={context_length}
BATCH_SIZE={batch_size}
THREADS={threads}
METAL={"1" if use_metal else "0"}
METAL_MMQ={"1" if metal_mmq else "0"}

# Run llama.cpp with optimized settings
./main \\
    -m "$MODEL" \\
    -c $CONTEXT_LENGTH \\
    -b $BATCH_SIZE \\
    -t $THREADS \\
    {"--metal \\" if use_metal else ""} \\
    {"--metal-mmq \\" if use_metal and metal_mmq else ""} \\
    --color \\
    --interactive

echo "Running with optimal Metal settings:"
echo "  Model: $MODEL"
echo "  Context Length: $CONTEXT_LENGTH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Threads: $THREADS"
echo "  Metal: {"Enabled" if use_metal else "Disabled"}"
echo "  Metal MMQ: {"Enabled" if use_metal and metal_mmq else "Disabled"}"
"""

    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(output_path, 0o755)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Optimize Metal GPU settings for llama.cpp")
    parser.add_argument("--model", required=True, help="Path to the model file (.gguf)")
    parser.add_argument("--llama_cpp", default="./llama.cpp", help="Path to llama.cpp directory")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--script", help="Output path for optimized script")
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Get GPU information
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info.get('name', 'Unknown')}")
    print(f"Metal Support: {gpu_info.get('metal_support', False)}")
    
    if gpu_info.get('metal_support', False) == False:
        print("Warning: Metal support not detected. This script is designed for Apple Silicon with Metal support.")
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return
    
    # Analyze Metal performance
    print("\nAnalyzing Metal performance...")
    metal_analysis = analyze_metal_performance(
        llama_cpp_path=args.llama_cpp,
        model_path=args.model,
        metal_settings=DEFAULT_METAL_SETTINGS,
        context_length=args.ctx,
        threads=args.threads
    )
    
    # Determine if Metal is beneficial
    metal_results = [r for r in metal_analysis if r["metal"] and r["success"]]
    cpu_results = [r for r in metal_analysis if not r["metal"] and r["success"]]
    
    use_metal = True
    use_metal_mmq = True
    
    if metal_results and cpu_results:
        best_metal = max(metal_results, key=lambda r: r["tokens_per_second"])
        best_cpu = max(cpu_results, key=lambda r: r["tokens_per_second"])
        
        metal_speedup = best_metal["tokens_per_second"] / best_cpu["tokens_per_second"] if best_cpu["tokens_per_second"] > 0 else float('inf')
        
        print(f"\nMetal vs CPU Performance:")
        print(f"  Best Metal: {best_metal['tokens_per_second']:.2f} tokens/second")
        print(f"  Best CPU: {best_cpu['tokens_per_second']:.2f} tokens/second")
        print(f"  Speedup: {metal_speedup:.2f}x")
        
        if metal_speedup < 1.1:
            print("\nWarning: Metal acceleration provides minimal benefit (<10% speedup).")
            use_metal = input("Continue with Metal optimization? (y/n): ").lower() == 'y'
    elif not metal_results:
        print("\nWarning: Metal acceleration failed or provided no measurable results.")
        use_metal = False
    
    # Determine if Metal MMQ is beneficial
    if use_metal:
        metal_mmq_results = [r for r in metal_analysis if r["metal"] and r["metal_mmq"] and r["success"]]
        metal_no_mmq_results = [r for r in metal_analysis if r["metal"] and not r["metal_mmq"] and r["success"]]
        
        if metal_mmq_results and metal_no_mmq_results:
            best_mmq = max(metal_mmq_results, key=lambda r: r["tokens_per_second"])
            best_no_mmq = max(metal_no_mmq_results, key=lambda r: r["tokens_per_second"])
            
            mmq_speedup = best_mmq["tokens_per_second"] / best_no_mmq["tokens_per_second"] if best_no_mmq["tokens_per_second"] > 0 else float('inf')
            
            print(f"\nMetal MMQ Impact:")
            print(f"  With MMQ: {best_mmq['tokens_per_second']:.2f} tokens/second")
            print(f"  Without MMQ: {best_no_mmq['tokens_per_second']:.2f} tokens/second")
            print(f"  Speedup: {mmq_speedup:.2f}x")
            
            use_metal_mmq = mmq_speedup >= 1.0
        elif not metal_mmq_results:
            print("\nMetal MMQ failed or provided no measurable results.")
            use_metal_mmq = False
    
    # Optimize batch size if using Metal
    batch_size_results = None
    optimal_batch_size = estimate_optimal_batch_size(gpu_info)
    
    if use_metal:
        print("\nOptimizing batch size for Metal acceleration...")
        batch_size_results = optimize_batch_size(
            llama_cpp_path=args.llama_cpp,
            model_path=args.model,
            use_metal=use_metal,
            metal_mmq=use_metal_mmq,
            context_length=args.ctx,
            threads=args.threads
        )
        optimal_batch_size = batch_size_results["optimal_batch_size"]
    
    # Generate final recommendations
    recommendations = {
        "system_info": {
            "os": platform.platform(),
            "gpu": gpu_info,
            "metal_support": gpu_info.get('metal_support', False)
        },
        "model_path": args.model,
        "context_length": args.ctx,
        "threads": args.threads,
        "recommendations": {
            "use_metal": use_metal,
            "use_metal_mmq": use_metal_mmq,
            "optimal_batch_size": optimal_batch_size,
            "estimated_performance": batch_size_results["best_performance"] if batch_size_results else None
        },
        "metal_analysis": metal_analysis,
        "batch_size_optimization": batch_size_results,
        "timestamp": datetime.now().isoformat()
    }
    
    # Print final recommendations
    print("\n==== FINAL RECOMMENDATIONS ====")
    print(f"Metal Acceleration: {'Enabled' if use_metal else 'Disabled'}")
    if use_metal:
        print(f"Metal MMQ: {'Enabled' if use_metal_mmq else 'Disabled'}")
        print(f"Optimal Batch Size: {optimal_batch_size}")
        print(f"Estimated Performance: {batch_size_results['best_performance']:.2f} tokens/second")
    
    # Command line example
    print("\nOptimal Command Line:")
    cmd = f"./main -m {args.model} -c {args.ctx} -t {args.threads} -b {optimal_batch_size}"
    if use_metal:
        cmd += " --metal"
        if use_metal_mmq:
            cmd += " --metal-mmq"
    print(cmd)
    
    # Save results if output file specified
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Create optimized script if requested
    if args.script:
        script_path = create_optimized_script(
            output_path=args.script,
            model_path=args.model,
            context_length=args.ctx,
            batch_size=optimal_batch_size,
            use_metal=use_metal,
            metal_mmq=use_metal_mmq,
            threads=args.threads
        )
        print(f"\nOptimized script created at {script_path}")
        print(f"Run with: bash {script_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Comparative Benchmark Tool for llama.cpp vs MLX on Apple Silicon.

This script provides a direct comparison of performance between llama.cpp and MLX frameworks,
measuring inference speed, memory usage, and quality metrics across various configurations.

Usage:
    python framework_comparison.py --llama-model <path> --mlx-model <name> [--options]

Example:
    python framework_comparison.py --llama-model models/llama-2-7b-q4_k.gguf --mlx-model llama-2-7b --quant int4,q4_k
"""

import argparse
import os
import time
import json
import platform
import psutil
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import subprocess
import tempfile
import gc
import threading
import sys
import re
from pathlib import Path
import importlib.util

# Add parent directory to path to ensure imports work
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import from the benchmark modules
llama_cpp_benchmark_available = False
mlx_benchmark_available = False

try:
    from benchmark.llama_cpp.benchmark import run_benchmark as llama_cpp_run_benchmark
    from benchmark.llama_cpp.benchmark import get_system_info, extract_model_info
    llama_cpp_benchmark_available = True
except ImportError:
    print("Warning: llama.cpp benchmark module not available.")

try:
    from benchmark.mlx.benchmark import run_benchmark as mlx_run_benchmark
    mlx_benchmark_available = True
except ImportError:
    print("Warning: MLX benchmark module not available.")

# Check for MLX
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. MLX benchmarks will be skipped.")

# Constants
DEFAULT_PROMPT_SHORT = "Explain quantum computing briefly."
DEFAULT_PROMPT_MEDIUM = "Write a short essay on the impact of artificial intelligence on society, discussing both potential benefits and concerns."
DEFAULT_PROMPT_LONG = "Explain the history, theory, and practical applications of deep learning, including major milestones, key algorithms, and its impact on various fields such as computer vision, natural language processing, and reinforcement learning."

# Quality benchmark questions (simplified)
QUALITY_BENCHMARK_QUESTIONS = [
    "What is the capital of France?",
    "Explain how a transformer neural network works.",
    "Write a function in Python to find prime numbers.",
    "What are the main provisions of the GDPR?",
    "Compare and contrast renewable and non-renewable energy sources."
]

def parse_comma_separated(value, parser_func=str):
    """Parse comma-separated values with a parser function"""
    if not value:
        return []
    return [parser_func(x.strip()) for x in value.split(',')]

def map_quantization(quant, framework):
    """Map quantization between frameworks for fair comparison"""
    # Mapping from MLX to llama.cpp and vice versa
    mlx_to_llama = {
        "int4": ["q4_k", "q4_0"],
        "int8": ["q8_0"],
        None: ["f16"] 
    }
    
    llama_to_mlx = {
        "q4_k": "int4",
        "q4_0": "int4",
        "q5_k": "int4",  # Approximate mapping
        "q8_0": "int8",
        "f16": None  # No quantization (FP16)
    }
    
    if framework == "mlx":
        for mlx_quant, llama_quants in mlx_to_llama.items():
            if quant in llama_quants:
                return mlx_quant
        return None  # Default to no quantization
    else:  # llama.cpp
        return llama_to_mlx.get(quant, "q4_k")  # Default to q4_k if unknown

def run_comparative_benchmark(llama_cpp_path, llama_model_path, mlx_model_name,
                             prompt, n_predict=256, context_length=2048, threads=4,
                             use_metal=True, metal_mmq=True, batch_size=512,
                             mlx_quant=None, llama_quant=None, n_runs=3):
    """Run benchmarks on both frameworks and collect results"""
    
    llama_results = None
    mlx_results = None
    
    print(f"Running comparison with prompt length: {len(prompt)} characters")
    
    # Run llama.cpp benchmark if available
    if llama_cpp_benchmark_available:
        print("\nRunning llama.cpp benchmark...")
        try:
            llama_results = llama_cpp_run_benchmark(
                llama_cpp_path=llama_cpp_path,
                model_path=llama_model_path,
                prompt=prompt,
                n_predict=n_predict,
                context_length=context_length,
                threads=threads,
                use_metal=use_metal,
                metal_mmq=metal_mmq,
                batch_size=batch_size,
                n_runs=n_runs
            )
            print(f"  llama.cpp tokens/second: {llama_results['avg_tokens_per_second']:.2f}")
            print(f"  llama.cpp peak memory: {llama_results['avg_peak_memory_gb']:.2f} GB")
        except Exception as e:
            print(f"Error in llama.cpp benchmark: {e}")
            llama_results = {"error": str(e)}
    
    # Run MLX benchmark if available
    if mlx_benchmark_available and MLX_AVAILABLE:
        print("\nRunning MLX benchmark...")
        try:
            mlx_results = mlx_run_benchmark(
                model_name=mlx_model_name,
                prompt=prompt,
                max_tokens=n_predict,
                quantization=mlx_quant,
                n_runs=n_runs
            )
            print(f"  MLX tokens/second: {mlx_results['avg_tokens_per_second']:.2f}")
            print(f"  MLX peak memory: {mlx_results['avg_peak_memory_gb']:.2f} GB")
        except Exception as e:
            print(f"Error in MLX benchmark: {e}")
            mlx_results = {"error": str(e)}
    
    return {
        "llama_cpp": llama_results,
        "mlx": mlx_results,
        "config": {
            "prompt_length": len(prompt),
            "n_predict": n_predict,
            "context_length": context_length,
            "threads": threads,
            "use_metal": use_metal,
            "metal_mmq": metal_mmq,
            "batch_size": batch_size,
            "mlx_quant": mlx_quant,
            "llama_quant": llama_quant
        }
    }

def compare_quality(llama_cpp_path, llama_model_path, mlx_model_name,
                   questions=QUALITY_BENCHMARK_QUESTIONS, max_tokens=256,
                   use_metal=True, threads=4, mlx_quant=None):
    """
    Compare the quality of responses between llama.cpp and MLX models
    """
    results = {
        "questions": questions,
        "responses": {
            "llama_cpp": [],
            "mlx": []
        },
        "timing": {
            "llama_cpp": [],
            "mlx": []
        }
    }
    
    # Generate responses with llama.cpp
    print("\nRunning llama.cpp quality test...")
    for i, question in enumerate(questions):
        print(f"  Question {i+1}/{len(questions)}")
        try:
            # Create a temporary file for the prompt
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(question)
                prompt_file = f.name
            
            # Construct command
            cmd = [
                os.path.join(llama_cpp_path, "main"),
                "-m", llama_model_path,
                "-f", prompt_file,
                "-n", str(max_tokens),
                "-t", str(threads),
                "--log-disable"  # Disable most logs for cleaner output
            ]
            
            if use_metal:
                cmd.append("--metal")
            
            # Start timing
            start_time = time.time()
            
            # Run the process
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            # End timing
            end_time = time.time()
            total_time = end_time - start_time
            
            # Process output
            output = stdout.decode()
            # Try to extract just the generated text (not prompt)
            try:
                # Simple extraction (may need adjustment based on llama.cpp output format)
                response = output.split(question)[1].strip()
            except:
                response = output
            
            results["responses"]["llama_cpp"].append(response)
            results["timing"]["llama_cpp"].append(total_time)
            
            # Clean up temporary file
            os.unlink(prompt_file)
            
        except Exception as e:
            print(f"  Error in llama.cpp quality test: {e}")
            results["responses"]["llama_cpp"].append(f"ERROR: {str(e)}")
            results["timing"]["llama_cpp"].append(None)
    
    # Generate responses with MLX
    if MLX_AVAILABLE:
        print("\nRunning MLX quality test...")
        try:
            # Load model
            model, tokenizer = load(mlx_model_name, quantization=mlx_quant)
            
            for i, question in enumerate(questions):
                print(f"  Question {i+1}/{len(questions)}")
                try:
                    # Start timing
                    start_time = time.time()
                    
                    # Generate response
                    tokens = generate(model, tokenizer, question, max_tokens=max_tokens)
                    response = tokenizer.decode(tokens)
                    
                    # End timing
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    results["responses"]["mlx"].append(response)
                    results["timing"]["mlx"].append(total_time)
                    
                except Exception as e:
                    print(f"  Error in MLX quality test question {i+1}: {e}")
                    results["responses"]["mlx"].append(f"ERROR: {str(e)}")
                    results["timing"]["mlx"].append(None)
                    
            # Clean up
            del model, tokenizer
            gc.collect()
            
        except Exception as e:
            print(f"Error loading MLX model: {e}")
            results["responses"]["mlx"] = ["ERROR: Model loading failed"] * len(questions)
            results["timing"]["mlx"] = [None] * len(questions)
    else:
        print("MLX not available, skipping quality test.")
        results["responses"]["mlx"] = ["MLX NOT AVAILABLE"] * len(questions)
        results["timing"]["mlx"] = [None] * len(questions)
    
    return results

def generate_comparison_report(results, output_dir):
    """Generate comprehensive report with visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results for valid runs
    valid_results = [r for r in results if "error" not in r.get("llama_cpp", {}) and "error" not in r.get("mlx", {})]
    
    if not valid_results:
        print("No valid results for reporting")
        return
    
    # Prepare data for charts
    prompts = [r["config"]["prompt_length"] for r in valid_results]
    llama_tokens_per_second = [r["llama_cpp"]["avg_tokens_per_second"] for r in valid_results]
    mlx_tokens_per_second = [r["mlx"]["avg_tokens_per_second"] for r in valid_results]
    
    llama_memory = [r["llama_cpp"]["avg_peak_memory_gb"] for r in valid_results]
    mlx_memory = [r["mlx"]["avg_peak_memory_gb"] for r in valid_results]
    
    # Generate tokens/second comparison chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(valid_results))
    
    plt.bar(index, llama_tokens_per_second, bar_width, label='llama.cpp')
    plt.bar(index + bar_width, mlx_tokens_per_second, bar_width, label='MLX')
    
    plt.xlabel('Prompt Size')
    plt.ylabel('Tokens per Second')
    plt.title('Inference Speed Comparison')
    plt.xticks(index + bar_width / 2, [f"Prompt {i+1}" for i in range(len(valid_results))])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_comparison.png'))
    
    # Generate memory usage comparison chart
    plt.figure(figsize=(10, 6))
    plt.bar(index, llama_memory, bar_width, label='llama.cpp')
    plt.bar(index + bar_width, mlx_memory, bar_width, label='MLX')
    
    plt.xlabel('Prompt Size')
    plt.ylabel('Peak Memory (GB)')
    plt.title('Memory Usage Comparison')
    plt.xticks(index + bar_width / 2, [f"Prompt {i+1}" for i in range(len(valid_results))])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_comparison.png'))
    
    # Save results as JSON
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary CSV
    with open(os.path.join(output_dir, 'summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Prompt', 'Framework', 'Tokens/Second', 'Peak Memory (GB)', 'Load Time (s)'])
        
        for i, result in enumerate(valid_results):
            writer.writerow([
                f"Prompt {i+1}",
                'llama.cpp',
                result["llama_cpp"]["avg_tokens_per_second"],
                result["llama_cpp"]["avg_peak_memory_gb"],
                result["llama_cpp"].get("avg_load_time_seconds", "N/A")
            ])
            writer.writerow([
                f"Prompt {i+1}",
                'MLX',
                result["mlx"]["avg_tokens_per_second"],
                result["mlx"]["avg_peak_memory_gb"],
                result["mlx"]["avg_load_time_seconds"]
            ])
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>llama.cpp vs MLX Benchmark Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart {{ margin: 20px 0; max-width: 800px; }}
            .winner {{ font-weight: bold; color: green; }}
            .section {{ margin: 30px 0; }}
        </style>
    </head>
    <body>
        <h1>llama.cpp vs MLX Benchmark Results</h1>
        
        <div class="section">
            <h2>Performance Summary</h2>
            <p>Average across all tests:</p>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>llama.cpp</th>
                    <th>MLX</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td>Tokens per Second</td>
                    <td>{np.mean(llama_tokens_per_second):.2f}</td>
                    <td>{np.mean(mlx_tokens_per_second):.2f}</td>
                    <td>{(np.mean(mlx_tokens_per_second) / np.mean(llama_tokens_per_second) - 1) * 100:.1f}% 
                        {'(MLX faster)' if np.mean(mlx_tokens_per_second) > np.mean(llama_tokens_per_second) else '(llama.cpp faster)'}</td>
                </tr>
                <tr>
                    <td>Peak Memory (GB)</td>
                    <td>{np.mean(llama_memory):.2f}</td>
                    <td>{np.mean(mlx_memory):.2f}</td>
                    <td>{(np.mean(mlx_memory) / np.mean(llama_memory) - 1) * 100:.1f}% 
                        {'(MLX uses more)' if np.mean(mlx_memory) > np.mean(llama_memory) else '(llama.cpp uses more)'}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Inference Speed Comparison</h2>
            <img src="speed_comparison.png" class="chart" alt="Speed Comparison Chart">
        </div>
        
        <div class="section">
            <h2>Memory Usage Comparison</h2>
            <img src="memory_comparison.png" class="chart" alt="Memory Comparison Chart">
        </div>
        
        <div class="section">
            <h2>Detailed Results</h2>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Framework</th>
                    <th>Tokens/Second</th>
                    <th>Peak Memory (GB)</th>
                    <th>Load Time (s)</th>
                </tr>
    """
    
    for i, result in enumerate(valid_results):
        llama_speed = result["llama_cpp"]["avg_tokens_per_second"]
        mlx_speed = result["mlx"]["avg_tokens_per_second"]
        
        llama_memory = result["llama_cpp"]["avg_peak_memory_gb"]
        mlx_memory = result["mlx"]["avg_peak_memory_gb"]
        
        html_report += f"""
                <tr>
                    <td rowspan="2">Prompt {i+1}</td>
                    <td>llama.cpp</td>
                    <td class="{'winner' if llama_speed > mlx_speed else ''}">{llama_speed:.2f}</td>
                    <td class="{'winner' if llama_memory < mlx_memory else ''}">{llama_memory:.2f}</td>
                    <td>{result["llama_cpp"].get("avg_load_time_seconds", "N/A")}</td>
                </tr>
                <tr>
                    <td>MLX</td>
                    <td class="{'winner' if mlx_speed > llama_speed else ''}">{mlx_speed:.2f}</td>
                    <td class="{'winner' if mlx_memory < llama_memory else ''}">{mlx_memory:.2f}</td>
                    <td>{result["mlx"]["avg_load_time_seconds"]:.2f}</td>
                </tr>
        """
    
    html_report += """
            </table>
        </div>
        
        <div class="section">
            <h2>Configuration Details</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
    """
    
    # Include configuration details from the first valid result
    for key, value in valid_results[0]["config"].items():
        html_report += f"""
                <tr>
                    <td>{key}</td>
                    <td>{value}</td>
                </tr>
        """
    
    html_report += """
            </table>
        </div>
        
        <div class="section">
            <h2>System Information</h2>
            <pre id="system-info">
            </pre>
        </div>
        
        <footer>
            <p>Generated on {timestamp}</p>
        </footer>
        
        <script>
            // You could add JavaScript here to enhance the report
            document.getElementById('system-info').textContent = JSON.stringify({system_info}, null, 2);
        </script>
    </body>
    </html>
    """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
               system_info=get_system_info())
    
    with open(os.path.join(output_dir, 'report.html'), 'w') as f:
        f.write(html_report)
    
    print(f"Benchmark report generated in {output_dir}")

def run_framework_comparison(llama_cpp_path, llama_model_path, mlx_model_name,
                            prompts=None, n_predict=256, context_lengths=[2048],
                            thread_counts=[4], use_metal=True, metal_mmq=True,
                            batch_sizes=[512], mlx_quant=None, llama_quant=None,
                            n_runs=3, output_dir=None, run_quality_test=False):
    """
    Run comprehensive comparison between llama.cpp and MLX frameworks
    """
    if not prompts:
        prompts = {
            "short": DEFAULT_PROMPT_SHORT,
            "medium": DEFAULT_PROMPT_MEDIUM,
            "long": DEFAULT_PROMPT_LONG
        }
    
    system_info = get_system_info()
    llama_model_info = extract_model_info(llama_model_path) if llama_model_path else {"error": "No llama.cpp model provided"}
    
    print(f"System: {system_info['mac_info'].get('model', 'Unknown Mac')}")
    print(f"Chip: {system_info['mac_info'].get('chip', 'Unknown')} {system_info['mac_info'].get('variant', '')}")
    print(f"RAM: {system_info['ram_total_gb']:.2f} GB")
    print(f"llama.cpp Model: {llama_model_info.get('filename', 'N/A')}")
    print(f"MLX Model: {mlx_model_name}")
    print(f"Metal Support: {system_info['metal_info'].get('supported', False)}")
    
    if not llama_cpp_benchmark_available:
        print("Warning: llama.cpp benchmark module not available. Some tests will be skipped.")
    
    if not mlx_benchmark_available or not MLX_AVAILABLE:
        print("Warning: MLX benchmark not available. Some tests will be skipped.")
    
    print("\nRunning comparative benchmarks. This may take some time...\n")
    
    all_results = []
    
    # Run performance benchmarks with different prompts
    for prompt_name, prompt in prompts.items():
        for ctx in context_lengths:
            for threads in thread_counts:
                for batch in batch_sizes:
                    config_desc = f"Prompt: {prompt_name}, Context: {ctx}, Threads: {threads}, Batch: {batch}"
                    print(f"Running benchmark: {config_desc}")
                    
                    result = run_comparative_benchmark(
                        llama_cpp_path=llama_cpp_path,
                        llama_model_path=llama_model_path,
                        mlx_model_name=mlx_model_name,
                        prompt=prompt,
                        n_predict=n_predict,
                        context_length=ctx,
                        threads=threads,
                        use_metal=use_metal,
                        metal_mmq=metal_mmq,
                        batch_size=batch,
                        mlx_quant=mlx_quant,
                        llama_quant=llama_quant,
                        n_runs=n_runs
                    )
                    
                    result["prompt_name"] = prompt_name
                    all_results.append(result)
    
    # Run quality test if requested
    quality_results = None
    if run_quality_test:
        print("\nRunning quality comparison test...")
        quality_results = compare_quality(
            llama_cpp_path=llama_cpp_path,
            llama_model_path=llama_model_path,
            mlx_model_name=mlx_model_name,
            use_metal=use_metal,
            threads=thread_counts[0],
            mlx_quant=mlx_quant
        )
    
    # Compile comprehensive results
    comprehensive_results = {
        "system_info": system_info,
        "llama_model_info": llama_model_info,
        "mlx_model_name": mlx_model_name,
        "benchmark_configs": {
            "n_predict": n_predict,
            "context_lengths": context_lengths,
            "thread_counts": thread_counts,
            "batch_sizes": batch_sizes,
            "use_metal": use_metal,
            "metal_mmq": metal_mmq,
            "mlx_quant": mlx_quant,
            "llama_quant": llama_quant,
            "n_runs": n_runs
        },
        "performance_results": all_results,
        "quality_results": quality_results
    }
    
    # Generate report if output directory specified
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save comprehensive results as JSON
        with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Generate visual report
        generate_comparison_report(all_results, output_dir)
    
    return comprehensive_results

def main():
    parser = argparse.ArgumentParser(description="Compare llama.cpp and MLX performance on Apple Silicon")
    
    # Required arguments for the models
    parser.add_argument("--llama-model", help="Path to the llama.cpp model file (.gguf)")
    parser.add_argument("--mlx-model", help="Name of the MLX model (e.g., llama-2-7b)")
    
    # Optional arguments
    parser.add_argument("--llama-cpp", default="./llama.cpp", help="Path to llama.cpp directory")
    parser.add_argument("--ctx", default="2048", help="Comma-separated list of context lengths to test")
    parser.add_argument("--threads", default="4", help="Comma-separated list of thread counts to test")
    parser.add_argument("--batch", default="512", help="Comma-separated list of batch sizes to test")
    parser.add_argument("--n-predict", type=int, default=256, help="Number of tokens to predict")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for each configuration")
    parser.add_argument("--no-metal", action="store_true", help="Disable Metal acceleration")
    parser.add_argument("--no-metal-mmq", action="store_true", help="Disable Metal matrix multiplication")
    parser.add_argument("--quant", help="Quantization to use (will be mapped between frameworks)")
    parser.add_argument("--output", help="Output directory for results and report")
    parser.add_argument("--quality-test", action="store_true", help="Run quality comparison test")
    parser.add_argument("--quick", action="store_true", help="Run a quick comparison with minimal configurations")
    
    args = parser.parse_args()
    
    # Check that at least one model is provided
    if not args.llama_model and not args.mlx_model:
        print("Error: You must provide at least one model (--llama-model or --mlx-model)")
        return
    
    # Parse lists of configurations to test
    try:
        context_lengths = parse_comma_separated(args.ctx, int)
        thread_counts = parse_comma_separated(args.threads, int)
        batch_sizes = parse_comma_separated(args.batch, int)
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
        
        # Use only short prompt for quick test
        prompts = {"short": DEFAULT_PROMPT_SHORT}
    else:
        prompts = {
            "short": DEFAULT_PROMPT_SHORT,
            "medium": DEFAULT_PROMPT_MEDIUM,
            "long": DEFAULT_PROMPT_LONG
        }
    
    # Handle quantization mapping
    mlx_quant = None
    llama_quant = None
    
    if args.quant:
        if args.mlx_model:
            mlx_quant = map_quantization(args.quant, "mlx")
        if args.llama_model:
            llama_quant = args.quant
    
    # Run the comparison
    run_framework_comparison(
        llama_cpp_path=args.llama_cpp,
        llama_model_path=args.llama_model,
        mlx_model_name=args.mlx_model,
        prompts=prompts,
        n_predict=args.n_predict,
        context_lengths=context_lengths,
        thread_counts=thread_counts,
        use_metal=not args.no_metal,
        metal_mmq=not args.no_metal_mmq,
        batch_sizes=batch_sizes,
        mlx_quant=mlx_quant,
        llama_quant=llama_quant,
        n_runs=args.runs,
        output_dir=args.output,
        run_quality_test=args.quality_test
    )

if __name__ == "__main__":
    main()
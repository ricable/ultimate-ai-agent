#!/usr/bin/env python3
"""
Quantization Comparison Tool for LLMs on Apple Silicon.

This script evaluates and compares model quality and performance across different 
quantization levels for both llama.cpp and MLX frameworks.

Usage:
    python quantization_comparison.py --llama-base <base_model> --mlx-model <name> [--options]

Example:
    python quantization_comparison.py --llama-base models/llama-2-7b.gguf --mlx-model llama-2-7b --quants q4_k,q8_0,f16,int4,int8
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
import shutil

# Add parent directory to path to ensure imports work
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import from the benchmark modules
try:
    from benchmark.llama_cpp.benchmark import run_benchmark as llama_cpp_run_benchmark
    from benchmark.llama_cpp.benchmark import get_system_info, extract_model_info
    llama_cpp_benchmark_available = True
except ImportError:
    print("Warning: llama.cpp benchmark module not available.")
    llama_cpp_benchmark_available = False

try:
    from benchmark.mlx.benchmark import run_benchmark as mlx_run_benchmark
    mlx_benchmark_available = True
except ImportError:
    print("Warning: MLX benchmark module not available.")
    mlx_benchmark_available = False

# Check for MLX
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Warning: MLX not available. MLX benchmarks will be skipped.")

# Constants
DEFAULT_PROMPT = "Explain how transformers work in deep learning, including the self-attention mechanism and the advantages over recurrent neural networks."

# Quality benchmark questions
QUALITY_BENCHMARK_QUESTIONS = [
    "What is the square root of 144?",
    "Write a recursive function to calculate Fibonacci numbers in Python.",
    "Explain the concept of quantum entanglement.",
    "Write a short poem about artificial intelligence.",
    "Translate 'Hello, how are you?' to French, Spanish, and German."
]

# Example MMLU questions for quality assessment
MMLU_QUESTIONS = [
    "Which of the following is a noble gas? A) Nitrogen B) Oxygen C) Helium D) Hydrogen",
    "What is the capital of Canada? A) Toronto B) Vancouver C) Ottawa D) Montreal",
    "Which hormone is primarily responsible for regulating blood glucose levels? A) Insulin B) Estrogen C) Testosterone D) Adrenaline",
    "Who wrote 'Pride and Prejudice'? A) Charles Dickens B) Jane Austen C) Mark Twain D) Virginia Woolf",
    "Which of these is NOT a primary color in the RGB color model? A) Red B) Green C) Yellow D) Blue"
]

def parse_comma_separated(value, parser_func=str):
    """Parse comma-separated values with a parser function"""
    if not value:
        return []
    return [parser_func(x.strip()) for x in value.split(',')]

def classify_quantization(quant):
    """Classify quantization type as llama.cpp or MLX"""
    if quant in ["int4", "int8", "int4_gptq", "int8_kv"]:
        return "mlx"
    elif quant in ["q4_0", "q4_k", "q5_k", "q5_0", "q6_k", "q8_0", "f16"]:
        return "llama"
    elif quant is None or quant.lower() in ["none", "fp16"]:
        return "both"  # No quantization (FP16) available in both
    else:
        return "unknown"

def get_corresponding_model_path(base_model_path, quantization):
    """Get the corresponding model path for a given quantization"""
    if quantization == "f16" or quantization is None:
        return base_model_path  # Use base model for FP16
    
    # Extract path components
    base_dir = os.path.dirname(base_model_path)
    filename = os.path.basename(base_model_path)
    
    # Check if base model already contains quantization info
    if "q2_" in filename or "q3_" in filename or "q4_" in filename or "q5_" in filename or "q6_" in filename or "q8_" in filename:
        # Replace existing quantization with new one
        for q in ["q2_k", "q2_0", "q3_k", "q3_0", "q4_k", "q4_0", "q5_k", "q5_0", "q6_k", "q8_0", "f16"]:
            if q in filename:
                return os.path.join(base_dir, filename.replace(q, quantization))
    
    # If no quantization in filename, insert before file extension
    name, ext = os.path.splitext(filename)
    return os.path.join(base_dir, f"{name}-{quantization}{ext}")

def evaluate_model_quality(llama_cpp_path, llama_model_path, mlx_model_name, 
                         quantization, questions=QUALITY_BENCHMARK_QUESTIONS,
                         max_tokens=256, use_metal=True, threads=4):
    """Evaluate model quality for a specific quantization level"""
    results = {
        "quantization": quantization,
        "framework": classify_quantization(quantization),
        "questions": questions,
        "responses": [],
        "timing": [],
        "token_counts": []
    }
    
    try:
        # Determine which framework to use
        framework = classify_quantization(quantization)
        
        if framework == "llama" or framework == "both":
            # Use llama.cpp
            model_path = get_corresponding_model_path(llama_model_path, quantization)
            
            if not os.path.exists(model_path) and quantization != "f16":
                print(f"Model not found: {model_path}")
                results["error"] = f"Model not found: {model_path}"
                return results
            
            print(f"Evaluating llama.cpp with quantization {quantization}...")
            
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
                        "-m", model_path,
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
                    
                    # Count tokens (rough estimate)
                    token_count = len(response.split())
                    
                    results["responses"].append(response)
                    results["timing"].append(total_time)
                    results["token_counts"].append(token_count)
                    
                    # Clean up temporary file
                    os.unlink(prompt_file)
                    
                except Exception as e:
                    print(f"  Error in llama.cpp quality test: {e}")
                    results["responses"].append(f"ERROR: {str(e)}")
                    results["timing"].append(None)
                    results["token_counts"].append(0)
                
        elif framework == "mlx" and MLX_AVAILABLE:
            # Use MLX
            print(f"Evaluating MLX with quantization {quantization}...")
            
            try:
                # Load model
                model, tokenizer = load(mlx_model_name, quantization=quantization)
                
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
                        
                        # Count tokens
                        token_count = len(tokens)
                        
                        results["responses"].append(response)
                        results["timing"].append(total_time)
                        results["token_counts"].append(token_count)
                        
                    except Exception as e:
                        print(f"  Error in MLX quality test question {i+1}: {e}")
                        results["responses"].append(f"ERROR: {str(e)}")
                        results["timing"].append(None)
                        results["token_counts"].append(0)
                
                # Clean up
                del model, tokenizer
                gc.collect()
                
            except Exception as e:
                print(f"Error loading MLX model: {e}")
                results["error"] = f"Error loading MLX model: {str(e)}"
        else:
            print(f"Unknown framework or MLX not available for quantization {quantization}")
            results["error"] = "Unknown framework or MLX not available"
            
    except Exception as e:
        print(f"Error in quality evaluation: {e}")
        results["error"] = str(e)
    
    return results

def benchmark_quantization_performance(llama_cpp_path, llama_model_path, mlx_model_name,
                                     quantization, prompt=DEFAULT_PROMPT, n_predict=256,
                                     context_length=2048, threads=4, use_metal=True,
                                     metal_mmq=True, batch_size=512, n_runs=3):
    """Benchmark performance for a specific quantization level"""
    results = {
        "quantization": quantization,
        "framework": classify_quantization(quantization),
        "config": {
            "prompt_length": len(prompt),
            "n_predict": n_predict,
            "context_length": context_length,
            "threads": threads,
            "use_metal": use_metal,
            "metal_mmq": metal_mmq,
            "batch_size": batch_size
        }
    }
    
    try:
        # Determine which framework to use
        framework = classify_quantization(quantization)
        
        if framework == "llama" or framework == "both":
            # Use llama.cpp
            if llama_cpp_benchmark_available:
                model_path = get_corresponding_model_path(llama_model_path, quantization)
                
                if not os.path.exists(model_path) and quantization != "f16":
                    print(f"Model not found: {model_path}")
                    results["llama_cpp"] = {"error": f"Model not found: {model_path}"}
                else:
                    print(f"Benchmarking llama.cpp with quantization {quantization}...")
                    
                    llama_results = llama_cpp_run_benchmark(
                        llama_cpp_path=llama_cpp_path,
                        model_path=model_path,
                        prompt=prompt,
                        n_predict=n_predict,
                        context_length=context_length,
                        threads=threads,
                        use_metal=use_metal,
                        metal_mmq=metal_mmq,
                        batch_size=batch_size,
                        n_runs=n_runs
                    )
                    
                    results["llama_cpp"] = llama_results
                    print(f"  llama.cpp tokens/second: {llama_results['avg_tokens_per_second']:.2f}")
                    print(f"  llama.cpp peak memory: {llama_results['avg_peak_memory_gb']:.2f} GB")
            else:
                results["llama_cpp"] = {"error": "llama.cpp benchmark not available"}
        
        if framework == "mlx" or framework == "both":
            # Use MLX
            if mlx_benchmark_available and MLX_AVAILABLE:
                print(f"Benchmarking MLX with quantization {quantization}...")
                
                # Map quantization string to MLX format
                mlx_quant = quantization if framework == "mlx" else None
                
                mlx_results = mlx_run_benchmark(
                    model_name=mlx_model_name,
                    prompt=prompt,
                    max_tokens=n_predict,
                    quantization=mlx_quant,
                    n_runs=n_runs
                )
                
                results["mlx"] = mlx_results
                print(f"  MLX tokens/second: {mlx_results['avg_tokens_per_second']:.2f}")
                print(f"  MLX peak memory: {mlx_results['avg_peak_memory_gb']:.2f} GB")
            else:
                results["mlx"] = {"error": "MLX benchmark not available"}
    
    except Exception as e:
        print(f"Error in quantization benchmark: {e}")
        results["error"] = str(e)
    
    return results

def generate_quantization_report(perf_results, quality_results, output_dir):
    """Generate comprehensive report on quantization comparisons"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract framework-specific results
    llama_perf_results = [r for r in perf_results if "llama_cpp" in r and "error" not in r["llama_cpp"]]
    mlx_perf_results = [r for r in perf_results if "mlx" in r and "error" not in r["mlx"]]
    
    llama_quality_results = [r for r in quality_results if r["framework"] in ["llama", "both"] and "error" not in r]
    mlx_quality_results = [r for r in quality_results if r["framework"] in ["mlx", "both"] and "error" not in r]
    
    # Generate performance comparison charts
    if llama_perf_results:
        plt.figure(figsize=(10, 6))
        llama_quants = [r["quantization"] for r in llama_perf_results]
        llama_speeds = [r["llama_cpp"]["avg_tokens_per_second"] for r in llama_perf_results]
        llama_memory = [r["llama_cpp"]["avg_peak_memory_gb"] for r in llama_perf_results]
        
        x = np.arange(len(llama_quants))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot tokens/second
        color = 'tab:blue'
        ax1.set_xlabel('Quantization')
        ax1.set_ylabel('Tokens per Second', color=color)
        bars1 = ax1.bar(x - width/2, llama_speeds, width, label='Tokens/s', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for memory
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Memory (GB)', color=color)
        bars2 = ax2.bar(x + width/2, llama_memory, width, label='Memory', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('llama.cpp Performance by Quantization Level')
        plt.xticks(x, llama_quants)
        
        # Add legend with custom handles
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='tab:blue', lw=4),
                        Line2D([0], [0], color='tab:red', lw=4)]
        ax1.legend(custom_lines, ['Tokens/s', 'Memory (GB)'], loc='upper right')
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, 'llama_quantization_performance.png'))
    
    if mlx_perf_results:
        plt.figure(figsize=(10, 6))
        mlx_quants = [r["quantization"] for r in mlx_perf_results]
        mlx_speeds = [r["mlx"]["avg_tokens_per_second"] for r in mlx_perf_results]
        mlx_memory = [r["mlx"]["avg_peak_memory_gb"] for r in mlx_perf_results]
        
        x = np.arange(len(mlx_quants))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot tokens/second
        color = 'tab:green'
        ax1.set_xlabel('Quantization')
        ax1.set_ylabel('Tokens per Second', color=color)
        bars1 = ax1.bar(x - width/2, mlx_speeds, width, label='Tokens/s', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for memory
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Memory (GB)', color=color)
        bars2 = ax2.bar(x + width/2, mlx_memory, width, label='Memory', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('MLX Performance by Quantization Level')
        plt.xticks(x, mlx_quants)
        
        # Add legend with custom handles
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='tab:green', lw=4),
                        Line2D([0], [0], color='tab:orange', lw=4)]
        ax1.legend(custom_lines, ['Tokens/s', 'Memory (GB)'], loc='upper right')
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mlx_quantization_performance.png'))
    
    # Generate quality comparison charts (using response time as a proxy for quality)
    if llama_quality_results:
        plt.figure(figsize=(10, 6))
        llama_quants = [r["quantization"] for r in llama_quality_results]
        llama_times = [np.mean([t for t in r["timing"] if t is not None]) for r in llama_quality_results]
        
        plt.bar(llama_quants, llama_times)
        plt.xlabel('Quantization')
        plt.ylabel('Average Response Time (s)')
        plt.title('llama.cpp Response Time by Quantization Level')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'llama_quantization_response_time.png'))
    
    if mlx_quality_results:
        plt.figure(figsize=(10, 6))
        mlx_quants = [r["quantization"] for r in mlx_quality_results]
        mlx_times = [np.mean([t for t in r["timing"] if t is not None]) for r in mlx_quality_results]
        
        plt.bar(mlx_quants, mlx_times)
        plt.xlabel('Quantization')
        plt.ylabel('Average Response Time (s)')
        plt.title('MLX Response Time by Quantization Level')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mlx_quantization_response_time.png'))
    
    # Save results as JSON
    with open(os.path.join(output_dir, 'quantization_performance_results.json'), 'w') as f:
        json.dump(perf_results, f, indent=2)
    
    with open(os.path.join(output_dir, 'quantization_quality_results.json'), 'w') as f:
        json.dump(quality_results, f, indent=2)
    
    # Generate summary CSV
    with open(os.path.join(output_dir, 'quantization_summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Framework', 'Quantization', 'Tokens/Second', 'Memory (GB)', 'Avg Response Time (s)'])
        
        # Combine performance and quality data
        for perf_result in llama_perf_results:
            quant = perf_result["quantization"]
            
            # Find corresponding quality result
            quality_result = next((r for r in llama_quality_results if r["quantization"] == quant), None)
            
            avg_response_time = None
            if quality_result:
                valid_times = [t for t in quality_result["timing"] if t is not None]
                if valid_times:
                    avg_response_time = np.mean(valid_times)
            
            writer.writerow([
                'llama.cpp',
                quant,
                perf_result["llama_cpp"]["avg_tokens_per_second"],
                perf_result["llama_cpp"]["avg_peak_memory_gb"],
                avg_response_time
            ])
        
        for perf_result in mlx_perf_results:
            quant = perf_result["quantization"]
            
            # Find corresponding quality result
            quality_result = next((r for r in mlx_quality_results if r["quantization"] == quant), None)
            
            avg_response_time = None
            if quality_result:
                valid_times = [t for t in quality_result["timing"] if t is not None]
                if valid_times:
                    avg_response_time = np.mean(valid_times)
            
            writer.writerow([
                'MLX',
                quant,
                perf_result["mlx"]["avg_tokens_per_second"],
                perf_result["mlx"]["avg_peak_memory_gb"],
                avg_response_time
            ])
    
    # Generate HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quantization Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart {{ margin: 20px 0; max-width: 800px; }}
            .winner {{ font-weight: bold; color: green; }}
            .section {{ margin: 30px 0; }}
            .response {{ font-family: monospace; white-space: pre-wrap; margin: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; max-height: 200px; overflow-y: auto; }}
        </style>
    </head>
    <body>
        <h1>Quantization Comparison Report</h1>
        
        <div class="section">
            <h2>Performance Summary</h2>
            
            <h3>llama.cpp Performance by Quantization Level</h3>
            <img src="llama_quantization_performance.png" class="chart" alt="llama.cpp Performance Chart">
            
            <h3>MLX Performance by Quantization Level</h3>
            <img src="mlx_quantization_performance.png" class="chart" alt="MLX Performance Chart">
            
            <h3>Detailed Performance Metrics</h3>
            <table>
                <tr>
                    <th>Framework</th>
                    <th>Quantization</th>
                    <th>Tokens/Second</th>
                    <th>Memory (GB)</th>
                    <th>Performance/Memory Ratio</th>
                </tr>
    """
    
    # Add llama.cpp performance data
    for result in llama_perf_results:
        quant = result["quantization"]
        tokens_per_second = result["llama_cpp"]["avg_tokens_per_second"]
        memory_gb = result["llama_cpp"]["avg_peak_memory_gb"]
        ratio = tokens_per_second / memory_gb if memory_gb > 0 else 0
        
        html_report += f"""
                <tr>
                    <td>llama.cpp</td>
                    <td>{quant}</td>
                    <td>{tokens_per_second:.2f}</td>
                    <td>{memory_gb:.2f}</td>
                    <td>{ratio:.2f}</td>
                </tr>
        """
    
    # Add MLX performance data
    for result in mlx_perf_results:
        quant = result["quantization"]
        tokens_per_second = result["mlx"]["avg_tokens_per_second"]
        memory_gb = result["mlx"]["avg_peak_memory_gb"]
        ratio = tokens_per_second / memory_gb if memory_gb > 0 else 0
        
        html_report += f"""
                <tr>
                    <td>MLX</td>
                    <td>{quant}</td>
                    <td>{tokens_per_second:.2f}</td>
                    <td>{memory_gb:.2f}</td>
                    <td>{ratio:.2f}</td>
                </tr>
        """
    
    html_report += """
            </table>
        </div>
        
        <div class="section">
            <h2>Quality Analysis</h2>
            
            <h3>Response Time Comparison</h3>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
                <div>
                    <h4>llama.cpp Response Time by Quantization</h4>
                    <img src="llama_quantization_response_time.png" class="chart" alt="llama.cpp Response Time Chart">
                </div>
                <div>
                    <h4>MLX Response Time by Quantization</h4>
                    <img src="mlx_quantization_response_time.png" class="chart" alt="MLX Response Time Chart">
                </div>
            </div>
            
            <h3>Sample Responses by Quantization Level</h3>
    """
    
    # Add sample responses for quality comparison
    for framework, results in [("llama.cpp", llama_quality_results), ("MLX", mlx_quality_results)]:
        if results:
            html_report += f"""
            <h4>{framework} Responses</h4>
            <table>
                <tr>
                    <th>Quantization</th>
                    <th>Question</th>
                    <th>Response</th>
                    <th>Time (s)</th>
                </tr>
            """
            
            # Show first question response for each quantization
            for result in results:
                quant = result["quantization"]
                if result["responses"] and len(result["responses"]) > 0:
                    question = result["questions"][0]
                    response = result["responses"][0]
                    time_taken = result["timing"][0] if result["timing"] and len(result["timing"]) > 0 else None
                    
                    html_report += f"""
                    <tr>
                        <td>{quant}</td>
                        <td>{question[:50]}...</td>
                        <td><div class="response">{response[:200]}...</div></td>
                        <td>{time_taken:.2f if time_taken else 'N/A'}</td>
                    </tr>
                    """
            
            html_report += """
            </table>
            """
    
    html_report += """
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <h3>Best Performance Settings</h3>
    """
    
    # Find best performance settings
    if llama_perf_results:
        best_llama_speed = max(llama_perf_results, key=lambda r: r["llama_cpp"]["avg_tokens_per_second"])
        best_llama_memory = min(llama_perf_results, key=lambda r: r["llama_cpp"]["avg_peak_memory_gb"])
        
        # Calculate a balanced score (higher is better)
        def balance_score(result):
            speed = result["llama_cpp"]["avg_tokens_per_second"]
            memory = result["llama_cpp"]["avg_peak_memory_gb"]
            return speed / (memory ** 0.5)  # Favor speed slightly
        
        best_llama_balanced = max(llama_perf_results, key=balance_score)
        
        html_report += f"""
            <h4>llama.cpp Recommendations</h4>
            <ul>
                <li><strong>Best Speed:</strong> {best_llama_speed["quantization"]} ({best_llama_speed["llama_cpp"]["avg_tokens_per_second"]:.2f} tokens/s)</li>
                <li><strong>Best Memory Efficiency:</strong> {best_llama_memory["quantization"]} ({best_llama_memory["llama_cpp"]["avg_peak_memory_gb"]:.2f} GB)</li>
                <li><strong>Best Balance:</strong> {best_llama_balanced["quantization"]} ({best_llama_balanced["llama_cpp"]["avg_tokens_per_second"]:.2f} tokens/s, {best_llama_balanced["llama_cpp"]["avg_peak_memory_gb"]:.2f} GB)</li>
            </ul>
        """
    
    if mlx_perf_results:
        best_mlx_speed = max(mlx_perf_results, key=lambda r: r["mlx"]["avg_tokens_per_second"])
        best_mlx_memory = min(mlx_perf_results, key=lambda r: r["mlx"]["avg_peak_memory_gb"])
        
        # Calculate a balanced score
        def balance_score(result):
            speed = result["mlx"]["avg_tokens_per_second"]
            memory = result["mlx"]["avg_peak_memory_gb"]
            return speed / (memory ** 0.5)  # Favor speed slightly
        
        best_mlx_balanced = max(mlx_perf_results, key=balance_score)
        
        html_report += f"""
            <h4>MLX Recommendations</h4>
            <ul>
                <li><strong>Best Speed:</strong> {best_mlx_speed["quantization"]} ({best_mlx_speed["mlx"]["avg_tokens_per_second"]:.2f} tokens/s)</li>
                <li><strong>Best Memory Efficiency:</strong> {best_mlx_memory["quantization"]} ({best_mlx_memory["mlx"]["avg_peak_memory_gb"]:.2f} GB)</li>
                <li><strong>Best Balance:</strong> {best_mlx_balanced["quantization"]} ({best_mlx_balanced["mlx"]["avg_tokens_per_second"]:.2f} tokens/s, {best_mlx_balanced["mlx"]["avg_peak_memory_gb"]:.2f} GB)</li>
            </ul>
        """
    
    html_report += """
        </div>
        
        <footer>
            <p>Generated on {timestamp}</p>
        </footer>
    </body>
    </html>
    """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with open(os.path.join(output_dir, 'quantization_report.html'), 'w') as f:
        f.write(html_report)
    
    print(f"Quantization report generated in {output_dir}")

def run_quantization_comparison(llama_cpp_path, llama_base_model, mlx_model_name,
                              quantizations, prompt=DEFAULT_PROMPT, n_predict=256,
                              context_length=2048, threads=4, use_metal=True,
                              metal_mmq=True, batch_size=512, n_runs=3,
                              output_dir=None, run_quality_test=True):
    """
    Run comprehensive comparison of different quantization levels
    """
    system_info = get_system_info()
    
    print(f"System: {system_info['mac_info'].get('model', 'Unknown Mac')}")
    print(f"Chip: {system_info['mac_info'].get('chip', 'Unknown')} {system_info['mac_info'].get('variant', '')}")
    print(f"RAM: {system_info['ram_total_gb']:.2f} GB")
    print(f"Base llama.cpp Model: {llama_base_model}")
    print(f"MLX Model: {mlx_model_name}")
    print(f"Testing Quantizations: {', '.join(quantizations)}")
    
    # Benchmark performance for each quantization
    perf_results = []
    for quant in quantizations:
        print(f"\nBenchmarking quantization: {quant}")
        result = benchmark_quantization_performance(
            llama_cpp_path=llama_cpp_path,
            llama_model_path=llama_base_model,
            mlx_model_name=mlx_model_name,
            quantization=quant,
            prompt=prompt,
            n_predict=n_predict,
            context_length=context_length,
            threads=threads,
            use_metal=use_metal,
            metal_mmq=metal_mmq,
            batch_size=batch_size,
            n_runs=n_runs
        )
        perf_results.append(result)
    
    # Evaluate quality for each quantization if requested
    quality_results = []
    if run_quality_test:
        for quant in quantizations:
            print(f"\nEvaluating quality for quantization: {quant}")
            result = evaluate_model_quality(
                llama_cpp_path=llama_cpp_path,
                llama_model_path=llama_base_model,
                mlx_model_name=mlx_model_name,
                quantization=quant,
                questions=QUALITY_BENCHMARK_QUESTIONS,
                max_tokens=n_predict,
                use_metal=use_metal,
                threads=threads
            )
            quality_results.append(result)
    
    # Compile comprehensive results
    comprehensive_results = {
        "system_info": system_info,
        "benchmark_configs": {
            "llama_base_model": llama_base_model,
            "mlx_model_name": mlx_model_name,
            "quantizations": quantizations,
            "n_predict": n_predict,
            "context_length": context_length,
            "threads": threads,
            "use_metal": use_metal,
            "metal_mmq": metal_mmq,
            "batch_size": batch_size,
            "n_runs": n_runs
        },
        "performance_results": perf_results,
        "quality_results": quality_results
    }
    
    # Generate report if output directory specified
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save comprehensive results as JSON
        with open(os.path.join(output_dir, 'quantization_comparison_results.json'), 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Generate visual report
        generate_quantization_report(perf_results, quality_results, output_dir)
    
    return comprehensive_results

def main():
    parser = argparse.ArgumentParser(description="Compare quantization levels for llama.cpp and MLX")
    
    # Required arguments for the models
    parser.add_argument("--llama-base", help="Path to the base llama.cpp model file (.gguf)")
    parser.add_argument("--mlx-model", help="Name of the MLX model (e.g., llama-2-7b)")
    
    # Optional arguments
    parser.add_argument("--llama-cpp", default="./llama.cpp", help="Path to llama.cpp directory")
    parser.add_argument("--quants", default="q4_k,q8_0,f16,int4,int8", help="Comma-separated list of quantizations to test")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length to test")
    parser.add_argument("--threads", type=int, default=4, help="Thread count to test")
    parser.add_argument("--n-predict", type=int, default=256, help="Number of tokens to predict")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for each configuration")
    parser.add_argument("--no-metal", action="store_true", help="Disable Metal acceleration")
    parser.add_argument("--no-metal-mmq", action="store_true", help="Disable Metal matrix multiplication")
    parser.add_argument("--output", help="Output directory for results and report")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality comparison test")
    
    args = parser.parse_args()
    
    # Check that at least one model is provided
    if not args.llama_base and not args.mlx_model:
        print("Error: You must provide at least one model (--llama-base or --mlx-model)")
        return
    
    # Parse quantizations to test
    quantizations = parse_comma_separated(args.quants)
    
    # Run the comparison
    run_quantization_comparison(
        llama_cpp_path=args.llama_cpp,
        llama_base_model=args.llama_base,
        mlx_model_name=args.mlx_model,
        quantizations=quantizations,
        context_length=args.ctx,
        threads=args.threads,
        use_metal=not args.no_metal,
        metal_mmq=not args.no_metal_mmq,
        n_predict=args.n_predict,
        n_runs=args.runs,
        output_dir=args.output,
        run_quality_test=not args.skip_quality
    )

if __name__ == "__main__":
    main()
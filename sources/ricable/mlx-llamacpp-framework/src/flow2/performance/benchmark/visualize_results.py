#!/usr/bin/env python3
"""
Benchmark Results Visualization Tool

This script generates visualizations and reports from benchmark results.
It can process JSON results from framework_comparison.py, quantization_comparison.py,
or the individual benchmark scripts.

Usage:
    python visualize_results.py --input <results_file.json> [--output <output_dir>] [--type <visualization_type>]

Example:
    python visualize_results.py --input results/comparison_results.json --output reports --type all
"""

import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
from pathlib import Path
import sys

def load_benchmark_results(file_path):
    """Load benchmark results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading benchmark results: {e}")
        return None

def detect_result_type(data):
    """Detect the type of benchmark results"""
    if "performance_results" in data and "quality_results" in data:
        return "quantization_comparison"
    elif "all_results" in data and isinstance(data["all_results"], list):
        # Check if this is a framework comparison
        first_result = data["all_results"][0] if data["all_results"] else {}
        if "llama_cpp" in first_result and "mlx" in first_result:
            return "framework_comparison"
    elif "runs" in data and isinstance(data["runs"], list):
        return "single_benchmark"
    elif "all_results" in data and "optimal_configs" in data:
        return "comprehensive_benchmark"
    
    return "unknown"

def create_framework_comparison_charts(data, output_dir):
    """Create charts for framework comparison results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results
    all_results = data.get("all_results", []) or data.get("performance_results", [])
    
    if not all_results:
        print("No valid results found for visualization")
        return
    
    # Filter valid results
    valid_results = [r for r in all_results if "llama_cpp" in r and "mlx" in r 
                    and "error" not in r.get("llama_cpp", {}) and "error" not in r.get("mlx", {})]
    
    if not valid_results:
        print("No valid comparison results found")
        return
    
    # Extract data for plotting
    labels = [r.get("prompt_name", f"Test {i+1}") for i, r in enumerate(valid_results)]
    llama_tokens_per_second = [r["llama_cpp"]["avg_tokens_per_second"] for r in valid_results]
    mlx_tokens_per_second = [r["mlx"]["avg_tokens_per_second"] for r in valid_results]
    
    llama_memory = [r["llama_cpp"]["avg_peak_memory_gb"] for r in valid_results]
    mlx_memory = [r["mlx"]["avg_peak_memory_gb"] for r in valid_results]
    
    # 1. Tokens per second comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, llama_tokens_per_second, width, label='llama.cpp')
    plt.bar(x + width/2, mlx_tokens_per_second, width, label='MLX')
    
    plt.xlabel('Test Case')
    plt.ylabel('Tokens per Second')
    plt.title('Inference Speed Comparison: llama.cpp vs MLX')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_comparison.png'), dpi=300)
    
    # 2. Memory usage comparison
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, llama_memory, width, label='llama.cpp')
    plt.bar(x + width/2, mlx_memory, width, label='MLX')
    
    plt.xlabel('Test Case')
    plt.ylabel('Peak Memory Usage (GB)')
    plt.title('Memory Usage Comparison: llama.cpp vs MLX')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_comparison.png'), dpi=300)
    
    # 3. Performance/Memory ratio (higher is better)
    plt.figure(figsize=(12, 6))
    llama_ratio = [s/m if m > 0 else 0 for s, m in zip(llama_tokens_per_second, llama_memory)]
    mlx_ratio = [s/m if m > 0 else 0 for s, m in zip(mlx_tokens_per_second, mlx_memory)]
    
    plt.bar(x - width/2, llama_ratio, width, label='llama.cpp')
    plt.bar(x + width/2, mlx_ratio, width, label='MLX')
    
    plt.xlabel('Test Case')
    plt.ylabel('Performance/Memory Ratio (tokens/s/GB)')
    plt.title('Efficiency Comparison: llama.cpp vs MLX')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'), dpi=300)
    
    # 4. Relative performance chart (MLX vs llama.cpp)
    plt.figure(figsize=(12, 6))
    speed_diff_percent = [(m/l - 1) * 100 if l > 0 else 0 for m, l in zip(mlx_tokens_per_second, llama_tokens_per_second)]
    memory_diff_percent = [(m/l - 1) * 100 if l > 0 else 0 for m, l in zip(mlx_memory, llama_memory)]
    
    plt.bar(x - width/2, speed_diff_percent, width, label='Speed Difference %')
    plt.bar(x + width/2, memory_diff_percent, width, label='Memory Difference %')
    
    plt.xlabel('Test Case')
    plt.ylabel('Relative Difference (% vs llama.cpp)')
    plt.title('MLX Performance Relative to llama.cpp')
    plt.xticks(x, labels)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'relative_performance.png'), dpi=300)
    
    # 5. Create summary table
    summary_data = []
    for i, result in enumerate(valid_results):
        llama_tps = result["llama_cpp"]["avg_tokens_per_second"]
        mlx_tps = result["mlx"]["avg_tokens_per_second"]
        llama_mem = result["llama_cpp"]["avg_peak_memory_gb"]
        mlx_mem = result["mlx"]["avg_peak_memory_gb"]
        
        speed_diff = (mlx_tps / llama_tps - 1) * 100 if llama_tps > 0 else 0
        memory_diff = (mlx_mem / llama_mem - 1) * 100 if llama_mem > 0 else 0
        
        summary_data.append({
            "Test": labels[i],
            "llama.cpp Speed (tok/s)": f"{llama_tps:.2f}",
            "MLX Speed (tok/s)": f"{mlx_tps:.2f}",
            "Speed Diff": f"{speed_diff:.1f}%",
            "llama.cpp Memory (GB)": f"{llama_mem:.2f}",
            "MLX Memory (GB)": f"{mlx_mem:.2f}",
            "Memory Diff": f"{memory_diff:.1f}%",
            "llama.cpp Efficiency (tok/s/GB)": f"{(llama_tps/llama_mem):.2f}" if llama_mem > 0 else "N/A",
            "MLX Efficiency (tok/s/GB)": f"{(mlx_tps/mlx_mem):.2f}" if mlx_mem > 0 else "N/A"
        })
    
    # Save as CSV
    pd.DataFrame(summary_data).to_csv(os.path.join(output_dir, 'framework_comparison_summary.csv'), index=False)
    
    # Also save as HTML
    pd.DataFrame(summary_data).to_html(os.path.join(output_dir, 'framework_comparison_summary.html'), index=False)
    
    print(f"Framework comparison charts created in {output_dir}")

def create_quantization_comparison_charts(data, output_dir):
    """Create charts for quantization comparison results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract performance results
    perf_results = data.get("performance_results", [])
    quality_results = data.get("quality_results", [])
    
    if not perf_results:
        print("No valid performance results found for visualization")
        return
    
    # Separate by framework
    llama_results = [r for r in perf_results if "llama_cpp" in r and "error" not in r.get("llama_cpp", {})]
    mlx_results = [r for r in perf_results if "mlx" in r and "error" not in r.get("mlx", {})]
    
    # Create performance charts for llama.cpp
    if llama_results:
        plt.figure(figsize=(12, 6))
        quants = [r["quantization"] for r in llama_results]
        speeds = [r["llama_cpp"]["avg_tokens_per_second"] for r in llama_results]
        memory = [r["llama_cpp"]["avg_peak_memory_gb"] for r in llama_results]
        
        x = np.arange(len(quants))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot tokens/second
        color = 'tab:blue'
        ax1.set_xlabel('Quantization')
        ax1.set_ylabel('Tokens per Second', color=color)
        bars1 = ax1.bar(x - width/2, speeds, width, label='Tokens/s', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for memory
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Memory (GB)', color=color)
        bars2 = ax2.bar(x + width/2, memory, width, label='Memory', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('llama.cpp Performance by Quantization Level')
        plt.xticks(x, quants, rotation=45)
        
        # Add legend with custom handles
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='tab:blue', lw=4),
                        Line2D([0], [0], color='tab:red', lw=4)]
        ax1.legend(custom_lines, ['Tokens/s', 'Memory (GB)'], loc='upper right')
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, 'llama_quantization_performance.png'), dpi=300)
        
        # Efficiency chart (tokens/s/GB)
        plt.figure(figsize=(12, 6))
        efficiency = [s/m if m > 0 else 0 for s, m in zip(speeds, memory)]
        
        plt.bar(quants, efficiency, color='tab:purple')
        plt.xlabel('Quantization')
        plt.ylabel('Efficiency (tokens/s/GB)')
        plt.title('llama.cpp Efficiency by Quantization Level')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'llama_quantization_efficiency.png'), dpi=300)
    
    # Create performance charts for MLX
    if mlx_results:
        plt.figure(figsize=(12, 6))
        quants = [r["quantization"] for r in mlx_results]
        speeds = [r["mlx"]["avg_tokens_per_second"] for r in mlx_results]
        memory = [r["mlx"]["avg_peak_memory_gb"] for r in mlx_results]
        
        x = np.arange(len(quants))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot tokens/second
        color = 'tab:green'
        ax1.set_xlabel('Quantization')
        ax1.set_ylabel('Tokens per Second', color=color)
        bars1 = ax1.bar(x - width/2, speeds, width, label='Tokens/s', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for memory
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Memory (GB)', color=color)
        bars2 = ax2.bar(x + width/2, memory, width, label='Memory', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('MLX Performance by Quantization Level')
        plt.xticks(x, quants, rotation=45)
        
        # Add legend with custom handles
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='tab:green', lw=4),
                        Line2D([0], [0], color='tab:orange', lw=4)]
        ax1.legend(custom_lines, ['Tokens/s', 'Memory (GB)'], loc='upper right')
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mlx_quantization_performance.png'), dpi=300)
        
        # Efficiency chart (tokens/s/GB)
        plt.figure(figsize=(12, 6))
        efficiency = [s/m if m > 0 else 0 for s, m in zip(speeds, memory)]
        
        plt.bar(quants, efficiency, color='tab:cyan')
        plt.xlabel('Quantization')
        plt.ylabel('Efficiency (tokens/s/GB)')
        plt.title('MLX Efficiency by Quantization Level')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mlx_quantization_efficiency.png'), dpi=300)
    
    # Create quality comparison charts if available
    if quality_results:
        llama_quality = [r for r in quality_results if r["framework"] in ["llama", "both"] and "error" not in r]
        mlx_quality = [r for r in quality_results if r["framework"] in ["mlx", "both"] and "error" not in r]
        
        if llama_quality:
            plt.figure(figsize=(12, 6))
            quants = [r["quantization"] for r in llama_quality]
            avg_times = [np.mean([t for t in r["timing"] if t is not None]) for r in llama_quality]
            
            plt.bar(quants, avg_times, color='tab:blue')
            plt.xlabel('Quantization')
            plt.ylabel('Average Response Time (s)')
            plt.title('llama.cpp Response Time by Quantization Level')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'llama_quantization_response_time.png'), dpi=300)
        
        if mlx_quality:
            plt.figure(figsize=(12, 6))
            quants = [r["quantization"] for r in mlx_quality]
            avg_times = [np.mean([t for t in r["timing"] if t is not None]) for r in mlx_quality]
            
            plt.bar(quants, avg_times, color='tab:green')
            plt.xlabel('Quantization')
            plt.ylabel('Average Response Time (s)')
            plt.title('MLX Response Time by Quantization Level')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mlx_quantization_response_time.png'), dpi=300)
    
    # Create summary tables
    llama_summary = []
    for result in llama_results:
        quant = result["quantization"]
        llama_data = result["llama_cpp"]
        
        # Find corresponding quality result if available
        quality_data = next((r for r in llama_quality if r["quantization"] == quant), None) if quality_results else None
        
        avg_response_time = None
        if quality_data:
            valid_times = [t for t in quality_data["timing"] if t is not None]
            if valid_times:
                avg_response_time = np.mean(valid_times)
        
        llama_summary.append({
            "Quantization": quant,
            "Speed (tok/s)": f"{llama_data['avg_tokens_per_second']:.2f}",
            "Memory (GB)": f"{llama_data['avg_peak_memory_gb']:.2f}",
            "Efficiency (tok/s/GB)": f"{(llama_data['avg_tokens_per_second']/llama_data['avg_peak_memory_gb']):.2f}",
            "Load Time (s)": f"{llama_data.get('avg_load_time_seconds', 'N/A')}",
            "Response Time (s)": f"{avg_response_time:.2f}" if avg_response_time else "N/A",
        })
    
    mlx_summary = []
    for result in mlx_results:
        quant = result["quantization"]
        mlx_data = result["mlx"]
        
        # Find corresponding quality result if available
        quality_data = next((r for r in mlx_quality if r["quantization"] == quant), None) if quality_results else None
        
        avg_response_time = None
        if quality_data:
            valid_times = [t for t in quality_data["timing"] if t is not None]
            if valid_times:
                avg_response_time = np.mean(valid_times)
        
        mlx_summary.append({
            "Quantization": quant,
            "Speed (tok/s)": f"{mlx_data['avg_tokens_per_second']:.2f}",
            "Memory (GB)": f"{mlx_data['avg_peak_memory_gb']:.2f}",
            "Efficiency (tok/s/GB)": f"{(mlx_data['avg_tokens_per_second']/mlx_data['avg_peak_memory_gb']):.2f}",
            "Load Time (s)": f"{mlx_data.get('avg_load_time_seconds', 'N/A')}",
            "Response Time (s)": f"{avg_response_time:.2f}" if avg_response_time else "N/A",
        })
    
    # Save as CSV
    if llama_summary:
        pd.DataFrame(llama_summary).to_csv(os.path.join(output_dir, 'llama_quantization_summary.csv'), index=False)
        pd.DataFrame(llama_summary).to_html(os.path.join(output_dir, 'llama_quantization_summary.html'), index=False)
    
    if mlx_summary:
        pd.DataFrame(mlx_summary).to_csv(os.path.join(output_dir, 'mlx_quantization_summary.csv'), index=False)
        pd.DataFrame(mlx_summary).to_html(os.path.join(output_dir, 'mlx_quantization_summary.html'), index=False)
    
    print(f"Quantization comparison charts created in {output_dir}")

def create_comprehensive_benchmark_charts(data, output_dir):
    """Create charts for comprehensive benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results
    all_results = data.get("all_results", [])
    system_info = data.get("system_info", {})
    model_info = data.get("model_info", {})
    optimal_configs = data.get("optimal_configs", {})
    
    if not all_results:
        print("No valid results found for visualization")
        return
    
    # Filter valid results
    valid_results = [r for r in all_results if "benchmark_result" in r and "error" not in r["benchmark_result"]]
    
    if not valid_results:
        print("No valid benchmark results found")
        return
    
    # Organize data for different parameter types
    contexts = sorted(list(set([r["context_length"] for r in valid_results if "context_length" in r])))
    threads = sorted(list(set([r["threads"] for r in valid_results if "threads" in r])))
    batch_sizes = sorted(list(set([r["batch_size"] for r in valid_results if "batch_size" in r])))
    prompt_types = sorted(list(set([r["prompt_type"] for r in valid_results if "prompt_type" in r])))
    
    # 1. Context Length vs Performance
    if len(contexts) > 1:
        plt.figure(figsize=(12, 6))
        
        # Group by context length
        ctx_data = {}
        for ctx in contexts:
            ctx_results = [r for r in valid_results if r.get("context_length") == ctx]
            if ctx_results:
                ctx_data[ctx] = np.mean([r["benchmark_result"]["avg_tokens_per_second"] for r in ctx_results])
        
        plt.bar(list(ctx_data.keys()), list(ctx_data.values()), color='tab:blue')
        plt.xlabel('Context Length')
        plt.ylabel('Tokens per Second')
        plt.title('Performance by Context Length')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_by_context.png'), dpi=300)
    
    # 2. Thread Count vs Performance
    if len(threads) > 1:
        plt.figure(figsize=(12, 6))
        
        # Group by thread count
        thread_data = {}
        for t in threads:
            thread_results = [r for r in valid_results if r.get("threads") == t]
            if thread_results:
                thread_data[t] = np.mean([r["benchmark_result"]["avg_tokens_per_second"] for r in thread_results])
        
        plt.bar(list(thread_data.keys()), list(thread_data.values()), color='tab:green')
        plt.xlabel('Thread Count')
        plt.ylabel('Tokens per Second')
        plt.title('Performance by Thread Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_by_threads.png'), dpi=300)
    
    # 3. Batch Size vs Performance
    if len(batch_sizes) > 1:
        plt.figure(figsize=(12, 6))
        
        # Group by batch size
        batch_data = {}
        for b in batch_sizes:
            batch_results = [r for r in valid_results if r.get("batch_size") == b]
            if batch_results:
                batch_data[b] = np.mean([r["benchmark_result"]["avg_tokens_per_second"] for r in batch_results])
        
        plt.bar(list(batch_data.keys()), list(batch_data.values()), color='tab:orange')
        plt.xlabel('Batch Size')
        plt.ylabel('Tokens per Second')
        plt.title('Performance by Batch Size')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_by_batch.png'), dpi=300)
    
    # 4. Prompt Type vs Performance
    if len(prompt_types) > 1:
        plt.figure(figsize=(12, 6))
        
        # Group by prompt type
        prompt_data = {}
        for p in prompt_types:
            prompt_results = [r for r in valid_results if r.get("prompt_type") == p]
            if prompt_results:
                prompt_data[p] = np.mean([r["benchmark_result"]["avg_tokens_per_second"] for r in prompt_results])
        
        plt.bar(list(prompt_data.keys()), list(prompt_data.values()), color='tab:purple')
        plt.xlabel('Prompt Type')
        plt.ylabel('Tokens per Second')
        plt.title('Performance by Prompt Type')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_by_prompt.png'), dpi=300)
    
    # 5. Heat map of parameters (if enough data points)
    if len(contexts) > 1 and len(threads) > 1:
        # Create matrix of context x threads
        perf_matrix = np.zeros((len(contexts), len(threads)))
        
        for i, ctx in enumerate(contexts):
            for j, t in enumerate(threads):
                results = [r for r in valid_results if r.get("context_length") == ctx and r.get("threads") == t]
                if results:
                    perf_matrix[i, j] = np.mean([r["benchmark_result"]["avg_tokens_per_second"] for r in results])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(perf_matrix, annot=True, fmt=".1f", xticklabels=threads, yticklabels=contexts)
        plt.xlabel('Thread Count')
        plt.ylabel('Context Length')
        plt.title('Performance Heat Map (Tokens/Second)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_heatmap.png'), dpi=300)
    
    # 6. Optimal Configuration Chart
    if optimal_configs:
        plt.figure(figsize=(12, 6))
        
        optimal_types = []
        speeds = []
        memories = []
        
        if "speed" in optimal_configs and "error" not in optimal_configs["speed"]:
            optimal_types.append("Speed")
            speeds.append(optimal_configs["speed"]["benchmark_result"]["avg_tokens_per_second"])
            memories.append(optimal_configs["speed"]["benchmark_result"]["avg_peak_memory_gb"])
        
        if "memory" in optimal_configs and "error" not in optimal_configs["memory"]:
            optimal_types.append("Memory")
            speeds.append(optimal_configs["memory"]["benchmark_result"]["avg_tokens_per_second"])
            memories.append(optimal_configs["memory"]["benchmark_result"]["avg_peak_memory_gb"])
        
        if "balance" in optimal_configs and "error" not in optimal_configs["balance"]:
            optimal_types.append("Balance")
            speeds.append(optimal_configs["balance"]["benchmark_result"]["avg_tokens_per_second"])
            memories.append(optimal_configs["balance"]["benchmark_result"]["avg_peak_memory_gb"])
        
        if optimal_types:
            x = np.arange(len(optimal_types))
            width = 0.35
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot tokens/second
            color = 'tab:blue'
            ax1.set_xlabel('Optimization Target')
            ax1.set_ylabel('Tokens per Second', color=color)
            bars1 = ax1.bar(x - width/2, speeds, width, label='Tokens/s', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Create second y-axis for memory
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Memory (GB)', color=color)
            bars2 = ax2.bar(x + width/2, memories, width, label='Memory', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Optimal Configurations')
            plt.xticks(x, optimal_types)
            
            # Add legend with custom handles
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='tab:blue', lw=4),
                            Line2D([0], [0], color='tab:red', lw=4)]
            ax1.legend(custom_lines, ['Tokens/s', 'Memory (GB)'], loc='upper right')
            
            fig.tight_layout()
            plt.savefig(os.path.join(output_dir, 'optimal_configurations.png'), dpi=300)
    
    # 7. Create summary table of all results
    summary_data = []
    for result in valid_results:
        summary_data.append({
            "Context": result.get("context_length", "N/A"),
            "Threads": result.get("threads", "N/A"),
            "Batch": result.get("batch_size", "N/A"),
            "Prompt": result.get("prompt_type", "N/A"),
            "Metal": "Yes" if result.get("metal", False) else "No",
            "Metal MMQ": "Yes" if result.get("metal_mmq", False) else "No",
            "Tokens/Second": f"{result['benchmark_result']['avg_tokens_per_second']:.2f}",
            "Peak Memory (GB)": f"{result['benchmark_result']['avg_peak_memory_gb']:.2f}",
            "Load Time (s)": f"{result['benchmark_result'].get('avg_load_time_seconds', 'N/A')}"
        })
    
    # Save as CSV
    pd.DataFrame(summary_data).to_csv(os.path.join(output_dir, 'benchmark_summary.csv'), index=False)
    
    # Also save as HTML
    pd.DataFrame(summary_data).to_html(os.path.join(output_dir, 'benchmark_summary.html'), index=False)
    
    # 8. Create optimal configuration summary
    if optimal_configs:
        optimal_summary = []
        
        if "speed" in optimal_configs and "error" not in optimal_configs["speed"]:
            speed_opt = optimal_configs["speed"]
            optimal_summary.append({
                "Optimization": "Speed",
                "Context": speed_opt.get("context_length", "N/A"),
                "Threads": speed_opt.get("threads", "N/A"),
                "Batch": speed_opt.get("batch_size", "N/A"),
                "Prompt": speed_opt.get("prompt_type", "N/A"),
                "Tokens/Second": f"{speed_opt['benchmark_result']['avg_tokens_per_second']:.2f}",
                "Memory (GB)": f"{speed_opt['benchmark_result']['avg_peak_memory_gb']:.2f}"
            })
        
        if "memory" in optimal_configs and "error" not in optimal_configs["memory"]:
            mem_opt = optimal_configs["memory"]
            optimal_summary.append({
                "Optimization": "Memory",
                "Context": mem_opt.get("context_length", "N/A"),
                "Threads": mem_opt.get("threads", "N/A"),
                "Batch": mem_opt.get("batch_size", "N/A"),
                "Prompt": mem_opt.get("prompt_type", "N/A"),
                "Tokens/Second": f"{mem_opt['benchmark_result']['avg_tokens_per_second']:.2f}",
                "Memory (GB)": f"{mem_opt['benchmark_result']['avg_peak_memory_gb']:.2f}"
            })
        
        if "balance" in optimal_configs and "error" not in optimal_configs["balance"]:
            bal_opt = optimal_configs["balance"]
            optimal_summary.append({
                "Optimization": "Balance",
                "Context": bal_opt.get("context_length", "N/A"),
                "Threads": bal_opt.get("threads", "N/A"),
                "Batch": bal_opt.get("batch_size", "N/A"),
                "Prompt": bal_opt.get("prompt_type", "N/A"),
                "Tokens/Second": f"{bal_opt['benchmark_result']['avg_tokens_per_second']:.2f}",
                "Memory (GB)": f"{bal_opt['benchmark_result']['avg_peak_memory_gb']:.2f}"
            })
        
        # Save as CSV
        pd.DataFrame(optimal_summary).to_csv(os.path.join(output_dir, 'optimal_configurations.csv'), index=False)
        
        # Also save as HTML
        pd.DataFrame(optimal_summary).to_html(os.path.join(output_dir, 'optimal_configurations.html'), index=False)
    
    print(f"Comprehensive benchmark charts created in {output_dir}")

def create_single_benchmark_charts(data, output_dir):
    """Create charts for single benchmark run results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the runs data
    runs = data.get("runs", [])
    
    if not runs:
        print("No run data found in benchmark results")
        return
    
    # Filter valid runs
    valid_runs = [r for r in runs if "error" not in r]
    
    if not valid_runs:
        print("No valid run data found")
        return
    
    # 1. Memory usage over time
    plt.figure(figsize=(12, 6))
    
    for i, run in enumerate(valid_runs):
        if "memory_samples" in run and run["memory_samples"]:
            plt.plot(run["memory_samples"], label=f"Run {i+1}")
    
    plt.xlabel('Sample Number')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_usage_over_time.png'), dpi=300)
    
    # 2. Tokens per second comparison between runs
    plt.figure(figsize=(12, 6))
    
    run_numbers = [f"Run {r['run']}" for r in valid_runs]
    tokens_per_second = [r.get("tokens_per_second", 0) for r in valid_runs]
    
    plt.bar(run_numbers, tokens_per_second, color='tab:blue')
    plt.xlabel('Run Number')
    plt.ylabel('Tokens per Second')
    plt.title('Inference Speed Comparison Between Runs')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tokens_per_second_comparison.png'), dpi=300)
    
    # 3. Peak memory comparison between runs
    plt.figure(figsize=(12, 6))
    
    peak_memory = [r.get("peak_memory_gb", 0) for r in valid_runs]
    
    plt.bar(run_numbers, peak_memory, color='tab:red')
    plt.xlabel('Run Number')
    plt.ylabel('Peak Memory (GB)')
    plt.title('Peak Memory Comparison Between Runs')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'peak_memory_comparison.png'), dpi=300)
    
    # 4. Create summary table
    summary_data = []
    for i, run in enumerate(valid_runs):
        summary_data.append({
            "Run": run.get("run", i+1),
            "Tokens/Second": f"{run.get('tokens_per_second', 0):.2f}",
            "Total Time (s)": f"{run.get('total_time_seconds', 0):.2f}",
            "Load Time (s)": f"{run.get('load_time_seconds', 'N/A')}",
            "Peak Memory (GB)": f"{run.get('peak_memory_gb', 0):.2f}",
            "Avg Memory (GB)": f"{run.get('avg_memory_gb', 0):.2f}"
        })
    
    # Save as CSV
    pd.DataFrame(summary_data).to_csv(os.path.join(output_dir, 'run_summary.csv'), index=False)
    
    # Also save as HTML
    pd.DataFrame(summary_data).to_html(os.path.join(output_dir, 'run_summary.html'), index=False)
    
    print(f"Single benchmark charts created in {output_dir}")

def create_interactive_html_report(data, output_dir, result_type):
    """Create an interactive HTML report with all visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'interactive_report.html')
    
    # Create report title based on result type
    if result_type == "framework_comparison":
        report_title = "Framework Comparison: llama.cpp vs MLX"
    elif result_type == "quantization_comparison":
        report_title = "Quantization Comparison Report"
    elif result_type == "comprehensive_benchmark":
        report_title = "Comprehensive Benchmark Report"
    elif result_type == "single_benchmark":
        report_title = "Benchmark Run Analysis"
    else:
        report_title = "Benchmark Results Report"
    
    # Get system info if available
    system_info = data.get("system_info", {})
    mac_info = system_info.get("mac_info", {})
    
    # Build HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report_title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart-container {{ display: flex; flex-wrap: wrap; justify-content: space-around; margin: 20px 0; }}
            .chart {{ margin: 10px; max-width: 90%; }}
            .nav {{ position: sticky; top: 0; background-color: #333; padding: 10px; color: white; z-index: 100; }}
            .nav a {{ color: white; text-decoration: none; margin: 0 10px; }}
            .nav a:hover {{ text-decoration: underline; }}
            .system-info {{ font-family: monospace; white-space: pre-wrap; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }}
            .footer {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #ddd; color: #777; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="nav">
            <a href="#overview">Overview</a>
            <a href="#visualizations">Visualizations</a>
            <a href="#data">Data Tables</a>
            <a href="#system">System Info</a>
        </div>
        
        <h1 id="overview">{report_title}</h1>
        
        <div class="section">
            <h2>Overview</h2>
    """
    
    # Add overview content based on result type
    if result_type == "framework_comparison":
        html_content += """
            <p>This report compares the performance of llama.cpp and MLX frameworks on Apple Silicon hardware, measuring:</p>
            <ul>
                <li>Inference speed (tokens per second)</li>
                <li>Memory usage (peak GB)</li>
                <li>Efficiency (tokens per second per GB)</li>
            </ul>
        """
    elif result_type == "quantization_comparison":
        html_content += """
            <p>This report analyzes the impact of different quantization levels on model performance and quality, comparing:</p>
            <ul>
                <li>Performance differences between quantization levels</li>
                <li>Memory usage differences</li>
                <li>Quality implications of quantization</li>
                <li>Optimal quantization levels for different use cases</li>
            </ul>
        """
    elif result_type == "comprehensive_benchmark":
        html_content += """
            <p>This report provides a comprehensive analysis of model performance across different parameters:</p>
            <ul>
                <li>Impact of context length on performance</li>
                <li>Impact of thread count on performance</li>
                <li>Impact of batch size on performance</li>
                <li>Optimal configurations for different priorities</li>
            </ul>
        """
    elif result_type == "single_benchmark":
        html_content += """
            <p>This report analyzes the results of individual benchmark runs, showing:</p>
            <ul>
                <li>Performance consistency between runs</li>
                <li>Memory usage patterns</li>
                <li>Detailed timing information</li>
            </ul>
        """
    
    # System information section
    mac_model = mac_info.get("model", "Unknown Mac")
    chip = mac_info.get("chip", "Unknown")
    variant = mac_info.get("variant", "")
    ram = system_info.get("ram_total_gb", 0)
    
    html_content += f"""
        </div>
        
        <div class="section">
            <h3>Test Environment</h3>
            <p><strong>Hardware:</strong> {mac_model}</p>
            <p><strong>Chip:</strong> {chip} {variant}</p>
            <p><strong>RAM:</strong> {ram:.2f} GB</p>
            <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d")}</p>
        </div>
        
        <div class="section" id="visualizations">
            <h2>Visualizations</h2>
            
            <div class="chart-container">
    """
    
    # Add chart images based on result type
    if result_type == "framework_comparison":
        charts = [
            "speed_comparison.png", 
            "memory_comparison.png",
            "efficiency_comparison.png",
            "relative_performance.png"
        ]
    elif result_type == "quantization_comparison":
        charts = [
            "llama_quantization_performance.png",
            "mlx_quantization_performance.png",
            "llama_quantization_efficiency.png",
            "mlx_quantization_efficiency.png",
            "llama_quantization_response_time.png",
            "mlx_quantization_response_time.png"
        ]
    elif result_type == "comprehensive_benchmark":
        charts = [
            "performance_by_context.png",
            "performance_by_threads.png",
            "performance_by_batch.png",
            "performance_by_prompt.png",
            "performance_heatmap.png",
            "optimal_configurations.png"
        ]
    elif result_type == "single_benchmark":
        charts = [
            "memory_usage_over_time.png",
            "tokens_per_second_comparison.png",
            "peak_memory_comparison.png"
        ]
    else:
        charts = []
    
    # Add available charts
    for chart in charts:
        chart_path = os.path.join(output_dir, chart)
        if os.path.exists(chart_path):
            chart_title = chart.replace(".png", "").replace("_", " ").title()
            html_content += f"""
                <div class="chart">
                    <h3>{chart_title}</h3>
                    <img src="{chart}" alt="{chart_title}" style="max-width: 100%;">
                </div>
            """
    
    html_content += """
            </div>
        </div>
        
        <div class="section" id="data">
            <h2>Data Tables</h2>
    """
    
    # Add data tables based on result type
    if result_type == "framework_comparison":
        table_path = os.path.join(output_dir, "framework_comparison_summary.html")
        if os.path.exists(table_path):
            with open(table_path, 'r') as f:
                table_content = f.read()
            html_content += f"""
                <h3>Framework Comparison Summary</h3>
                {table_content}
            """
    
    elif result_type == "quantization_comparison":
        llama_table = os.path.join(output_dir, "llama_quantization_summary.html")
        mlx_table = os.path.join(output_dir, "mlx_quantization_summary.html")
        
        if os.path.exists(llama_table):
            with open(llama_table, 'r') as f:
                table_content = f.read()
            html_content += f"""
                <h3>llama.cpp Quantization Summary</h3>
                {table_content}
            """
        
        if os.path.exists(mlx_table):
            with open(mlx_table, 'r') as f:
                table_content = f.read()
            html_content += f"""
                <h3>MLX Quantization Summary</h3>
                {table_content}
            """
    
    elif result_type == "comprehensive_benchmark":
        summary_table = os.path.join(output_dir, "benchmark_summary.html")
        optimal_table = os.path.join(output_dir, "optimal_configurations.html")
        
        if os.path.exists(summary_table):
            with open(summary_table, 'r') as f:
                table_content = f.read()
            html_content += f"""
                <h3>Benchmark Summary</h3>
                {table_content}
            """
        
        if os.path.exists(optimal_table):
            with open(optimal_table, 'r') as f:
                table_content = f.read()
            html_content += f"""
                <h3>Optimal Configurations</h3>
                {table_content}
            """
    
    elif result_type == "single_benchmark":
        run_table = os.path.join(output_dir, "run_summary.html")
        
        if os.path.exists(run_table):
            with open(run_table, 'r') as f:
                table_content = f.read()
            html_content += f"""
                <h3>Run Summary</h3>
                {table_content}
            """
    
    # Add system info section
    html_content += f"""
        </div>
        
        <div class="section" id="system">
            <h2>System Information</h2>
            <div class="system-info">
                {json.dumps(system_info, indent=2)}
            </div>
        </div>
        
        <div class="footer">
            <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Generated with Performance Benchmarking Tools for LLMs on Apple Silicon</p>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Interactive HTML report generated at {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--input", required=True, help="Path to benchmark results JSON file")
    parser.add_argument("--output", default="benchmark_report", help="Output directory for visualizations")
    parser.add_argument("--type", choices=["auto", "framework", "quantization", "comprehensive", "single", "all"], 
                        default="auto", help="Type of visualization to generate")
    args = parser.parse_args()
    
    # Load benchmark results
    data = load_benchmark_results(args.input)
    
    if not data:
        print("Failed to load benchmark results")
        return
    
    # Detect result type if auto
    result_type = detect_result_type(data)
    if args.type != "auto" and args.type != "all":
        if args.type == "framework":
            result_type = "framework_comparison"
        elif args.type == "quantization":
            result_type = "quantization_comparison"
        elif args.type == "comprehensive":
            result_type = "comprehensive_benchmark"
        elif args.type == "single":
            result_type = "single_benchmark"
    
    print(f"Detected result type: {result_type}")
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations based on type
    if result_type == "framework_comparison" or args.type == "all":
        create_framework_comparison_charts(data, output_dir)
    
    if result_type == "quantization_comparison" or args.type == "all":
        create_quantization_comparison_charts(data, output_dir)
    
    if result_type == "comprehensive_benchmark" or args.type == "all":
        create_comprehensive_benchmark_charts(data, output_dir)
    
    if result_type == "single_benchmark" or args.type == "all":
        create_single_benchmark_charts(data, output_dir)
    
    # Create interactive HTML report
    create_interactive_html_report(data, output_dir, result_type)

if __name__ == "__main__":
    main()
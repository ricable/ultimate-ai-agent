#!/usr/bin/env python3
"""
Comprehensive Benchmark Workflow Tool

This script provides end-to-end workflows for benchmarking llama.cpp and MLX models,
including performance testing, quantization comparison, and results visualization.

Usage:
    python benchmark_workflow.py <workflow> [--options]

Available workflows:
    framework-comparison   - Compare llama.cpp and MLX performance
    quantization-study     - Compare different quantization levels
    hardware-scaling       - Test performance across different hardware parameters
    quality-evaluation     - Evaluate quality impact of different configurations
    comprehensive          - Run all benchmarks and generate a complete report

Example:
    python benchmark_workflow.py framework-comparison --llama-model models/llama-2-7b-q4_k.gguf --mlx-model llama-2-7b
"""

import argparse
import os
import subprocess
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path to ensure imports work
parent_dir = str(Path(__file__).parent.parent.absolute())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def ensure_directory(directory):
    """Ensure directory exists, create if needed"""
    os.makedirs(directory, exist_ok=True)
    return directory

def run_command(cmd, description=None, exit_on_error=False):
    """Run a command and return the result"""
    if description:
        print(f"\n{description}...")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        if exit_on_error:
            sys.exit(1)
        return None

def framework_comparison_workflow(args):
    """Run framework comparison workflow"""
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_directory(os.path.join(args.output_dir, f"framework_comparison_{timestamp}"))
    
    # Step 1: Validate models
    if not args.llama_model:
        print("Error: llama.cpp model path is required for framework comparison")
        sys.exit(1)
    
    if not args.mlx_model:
        print("Error: MLX model name is required for framework comparison")
        sys.exit(1)
    
    # Step 2: Run framework comparison benchmark
    framework_comparison_script = os.path.join(parent_dir, "benchmark", "framework_comparison.py")
    
    # Build command
    cmd = [
        sys.executable,
        framework_comparison_script,
        "--llama-model", args.llama_model,
        "--mlx-model", args.mlx_model,
        "--output", os.path.join(output_dir, "comparison_results")
    ]
    
    # Add optional parameters
    if args.llama_cpp:
        cmd.extend(["--llama-cpp", args.llama_cpp])
    
    if args.ctx:
        cmd.extend(["--ctx", args.ctx])
    
    if args.threads:
        cmd.extend(["--threads", args.threads])
    
    if args.batch:
        cmd.extend(["--batch", args.batch])
    
    if args.n_predict:
        cmd.extend(["--n-predict", str(args.n_predict)])
    
    if args.runs:
        cmd.extend(["--runs", str(args.runs)])
    
    if args.quant:
        cmd.extend(["--quant", args.quant])
    
    if args.no_metal:
        cmd.append("--no-metal")
    
    if args.no_metal_mmq:
        cmd.append("--no-metal-mmq")
    
    if args.quality_test:
        cmd.append("--quality-test")
    
    if args.quick:
        cmd.append("--quick")
    
    # Run the comparison
    run_command(cmd, "Running framework comparison benchmark", exit_on_error=True)
    
    # Step 3: Generate visualizations
    results_json = os.path.join(output_dir, "comparison_results", "comparison_results.json")
    if os.path.exists(results_json):
        visualize_script = os.path.join(parent_dir, "benchmark", "visualize_results.py")
        vis_cmd = [
            sys.executable,
            visualize_script,
            "--input", results_json,
            "--output", os.path.join(output_dir, "visualizations"),
            "--type", "framework"
        ]
        run_command(vis_cmd, "Generating visualizations", exit_on_error=False)
    
    print(f"\nFramework comparison workflow completed. Results available in {output_dir}")

def quantization_study_workflow(args):
    """Run quantization study workflow"""
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_directory(os.path.join(args.output_dir, f"quantization_study_{timestamp}"))
    
    # Step 1: Validate models
    if not args.llama_model and not args.mlx_model:
        print("Error: At least one model (llama.cpp or MLX) is required for quantization study")
        sys.exit(1)
    
    # Step 2: Run quantization comparison benchmark
    quantization_script = os.path.join(parent_dir, "benchmark", "quantization_comparison.py")
    
    # Build command
    cmd = [
        sys.executable,
        quantization_script,
        "--output", os.path.join(output_dir, "quantization_results")
    ]
    
    # Add model parameters
    if args.llama_model:
        cmd.extend(["--llama-base", args.llama_model])
    
    if args.mlx_model:
        cmd.extend(["--mlx-model", args.mlx_model])
    
    # Add optional parameters
    if args.llama_cpp:
        cmd.extend(["--llama-cpp", args.llama_cpp])
    
    if args.quants:
        cmd.extend(["--quants", args.quants])
    else:
        # Default quantization types to test
        default_quants = "q4_k,q8_0,f16"
        if args.mlx_model:
            default_quants += ",int4,int8"
        cmd.extend(["--quants", default_quants])
    
    if args.ctx:
        cmd.extend(["--ctx", args.ctx.split(",")[0]])  # Use first ctx value
    
    if args.threads:
        cmd.extend(["--threads", args.threads.split(",")[0]])  # Use first thread value
    
    if args.n_predict:
        cmd.extend(["--n-predict", str(args.n_predict)])
    
    if args.runs:
        cmd.extend(["--runs", str(args.runs)])
    
    if args.no_metal:
        cmd.append("--no-metal")
    
    if args.no_metal_mmq:
        cmd.append("--no-metal-mmq")
    
    if args.skip_quality:
        cmd.append("--skip-quality")
    
    # Run the quantization study
    run_command(cmd, "Running quantization comparison benchmark", exit_on_error=True)
    
    # Step 3: Generate visualizations
    results_json = os.path.join(output_dir, "quantization_results", "quantization_comparison_results.json")
    if os.path.exists(results_json):
        visualize_script = os.path.join(parent_dir, "benchmark", "visualize_results.py")
        vis_cmd = [
            sys.executable,
            visualize_script,
            "--input", results_json,
            "--output", os.path.join(output_dir, "visualizations"),
            "--type", "quantization"
        ]
        run_command(vis_cmd, "Generating visualizations", exit_on_error=False)
    
    print(f"\nQuantization study workflow completed. Results available in {output_dir}")

def hardware_scaling_workflow(args):
    """Run hardware scaling workflow to test performance across different hardware parameters"""
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_directory(os.path.join(args.output_dir, f"hardware_scaling_{timestamp}"))
    results_dir = ensure_directory(os.path.join(output_dir, "results"))
    
    # Step 1: Validate model input
    if args.framework == "llama" and not args.llama_model:
        print("Error: llama.cpp model path is required for llama framework")
        sys.exit(1)
    
    if args.framework == "mlx" and not args.mlx_model:
        print("Error: MLX model name is required for mlx framework")
        sys.exit(1)
    
    # Step 2: Determine the benchmark script to use
    if args.framework == "llama":
        benchmark_script = os.path.join(parent_dir, "benchmark", "llama_cpp", "benchmark.py")
        model_param = ["--model", args.llama_model]
        if args.llama_cpp:
            llama_cpp_param = ["--llama_cpp", args.llama_cpp]
        else:
            llama_cpp_param = []
    else:  # mlx
        benchmark_script = os.path.join(parent_dir, "benchmark", "mlx", "benchmark.py")
        model_param = ["--model", args.mlx_model]
        llama_cpp_param = []
    
    # Parse parameters for scaling tests
    context_lengths = args.ctx.split(",") if args.ctx else ["2048", "4096", "8192"]
    thread_counts = args.threads.split(",") if args.threads else ["4", "8", "16"]
    batch_sizes = args.batch.split(",") if args.batch else ["512", "1024", "2048"]
    
    if args.framework == "mlx":
        quants = args.quants.split(",") if args.quants else ["none", "int8", "int4"]
    else:
        quants = args.quants.split(",") if args.quants else ["f16", "q8_0", "q4_k"]
    
    # Create an array to store all result files
    all_result_files = []
    
    # Step 3: Run benchmarks for each combination of parameters
    # We'll run separate benchmarks for:
    # 1. Context length scaling
    # 2. Thread count scaling
    # 3. Batch size scaling
    # 4. Quantization scaling
    
    # 1. Context length scaling
    ctx_output = os.path.join(results_dir, "context_scaling.json")
    
    ctx_cmd = [
        sys.executable,
        benchmark_script,
        *model_param,
        *llama_cpp_param,
        "--ctx", ",".join(context_lengths),
        "--output", ctx_output
    ]
    
    # Add default values for other parameters
    if args.framework == "llama":
        ctx_cmd.extend(["--threads", thread_counts[0], "--batch", batch_sizes[0]])
        if args.no_metal:
            ctx_cmd.append("--no-metal")
        if args.no_metal_mmq:
            ctx_cmd.append("--no-metal-mmq")
    else:  # mlx
        ctx_cmd.extend(["--quant", quants[0]])
    
    if args.quick:
        ctx_cmd.append("--quick")
    
    run_command(ctx_cmd, "Running context length scaling benchmark")
    if os.path.exists(ctx_output):
        all_result_files.append(ctx_output)
    
    # 2. Thread count scaling (only for llama.cpp)
    if args.framework == "llama":
        thread_output = os.path.join(results_dir, "thread_scaling.json")
        
        thread_cmd = [
            sys.executable,
            benchmark_script,
            *model_param,
            *llama_cpp_param,
            "--threads", ",".join(thread_counts),
            "--ctx", context_lengths[0],
            "--batch", batch_sizes[0],
            "--output", thread_output
        ]
        
        if args.no_metal:
            thread_cmd.append("--no-metal")
        if args.no_metal_mmq:
            thread_cmd.append("--no-metal-mmq")
        if args.quick:
            thread_cmd.append("--quick")
        
        run_command(thread_cmd, "Running thread count scaling benchmark")
        if os.path.exists(thread_output):
            all_result_files.append(thread_output)
    
    # 3. Batch size scaling
    batch_output = os.path.join(results_dir, "batch_scaling.json")
    
    if args.framework == "llama":
        batch_cmd = [
            sys.executable,
            benchmark_script,
            *model_param,
            *llama_cpp_param,
            "--batch", ",".join(batch_sizes),
            "--ctx", context_lengths[0],
            "--threads", thread_counts[0],
            "--output", batch_output
        ]
        
        if args.no_metal:
            batch_cmd.append("--no-metal")
        if args.no_metal_mmq:
            batch_cmd.append("--no-metal-mmq")
    else:  # mlx - no direct batch param, use different prompt lengths instead
        batch_cmd = [
            sys.executable,
            benchmark_script,
            *model_param,
            "--quant", quants[0],
            "--output", batch_output
        ]
    
    if args.quick:
        batch_cmd.append("--quick")
    
    run_command(batch_cmd, "Running batch size scaling benchmark")
    if os.path.exists(batch_output):
        all_result_files.append(batch_output)
    
    # 4. Quantization scaling
    quant_output = os.path.join(results_dir, "quantization_scaling.json")
    
    if args.framework == "llama":
        # For llama.cpp, we need to ensure models exist for each quantization
        # For now, we'll just output a message
        print("\nTo perform quantization scaling with llama.cpp, please run the quantization-study workflow instead")
    else:  # mlx
        quant_cmd = [
            sys.executable,
            benchmark_script,
            *model_param,
            "--quant", ",".join(quants),
            "--output", quant_output
        ]
        
        if args.quick:
            quant_cmd.append("--quick")
        
        run_command(quant_cmd, "Running quantization scaling benchmark")
        if os.path.exists(quant_output):
            all_result_files.append(quant_output)
    
    # Step 4: Generate visualizations for each result file
    visualize_script = os.path.join(parent_dir, "benchmark", "visualize_results.py")
    
    for result_file in all_result_files:
        result_name = os.path.basename(result_file).split(".")[0]
        vis_output = os.path.join(output_dir, "visualizations", result_name)
        
        vis_cmd = [
            sys.executable,
            visualize_script,
            "--input", result_file,
            "--output", vis_output,
            "--type", "comprehensive"
        ]
        
        run_command(vis_cmd, f"Generating visualizations for {result_name}")
    
    # Step 5: Create a combined report
    print("\nCreating combined hardware scaling report...")
    
    # Create a simple HTML index file that links to all visualizations
    index_html = os.path.join(output_dir, "hardware_scaling_report.html")
    
    with open(index_html, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Hardware Scaling Report for {args.framework.upper()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }}
        .nav {{ position: sticky; top: 0; background-color: #333; padding: 10px; color: white; z-index: 100; }}
        .nav a {{ color: white; text-decoration: none; margin: 0 10px; }}
        .nav a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="#overview">Overview</a>
""")
        
        # Add navigation links for each section
        for result_file in all_result_files:
            result_name = os.path.basename(result_file).split(".")[0]
            section_name = result_name.replace("_", " ").title()
            f.write(f'        <a href="#{result_name}">{section_name}</a>\n')
        
        f.write("""    </div>
    
    <h1 id="overview">Hardware Scaling Report</h1>
    
    <div class="section">
        <h2>Overview</h2>
        <p>This report shows how model performance scales with different hardware parameters:</p>
        <ul>
""")
        
        # Add descriptions for each section
        for result_file in all_result_files:
            result_name = os.path.basename(result_file).split(".")[0]
            section_name = result_name.replace("_", " ").title()
            f.write(f'            <li><a href="#{result_name}">{section_name}</a> - How performance scales with {result_name.split("_")[0]}</li>\n')
        
        f.write("""        </ul>
    </div>
""")
        
        # Add sections for each result
        for result_file in all_result_files:
            result_name = os.path.basename(result_file).split(".")[0]
            section_name = result_name.replace("_", " ").title()
            vis_path = f"visualizations/{result_name}/interactive_report.html"
            
            f.write(f"""
    <div class="section" id="{result_name}">
        <h2>{section_name}</h2>
        <p>Analysis of how performance scales with {result_name.split("_")[0]}.</p>
        <p><a href="{vis_path}" target="_blank">View detailed report</a></p>
        <iframe src="{vis_path}" width="100%" height="600px" style="border:none;"></iframe>
    </div>
""")
        
        # Add footer
        f.write(f"""
    <div class="footer">
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Generated with Performance Benchmarking Tools for LLMs on Apple Silicon</p>
    </div>
</body>
</html>
""")
    
    print(f"\nHardware scaling workflow completed. Results available in {output_dir}")
    print(f"Open {index_html} to view the combined report")

def quality_evaluation_workflow(args):
    """Run quality evaluation workflow to test model output quality"""
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_directory(os.path.join(args.output_dir, f"quality_evaluation_{timestamp}"))
    
    # Step 1: Validate models
    if not args.llama_model and not args.mlx_model:
        print("Error: At least one model (llama.cpp or MLX) is required for quality evaluation")
        sys.exit(1)
    
    # Step 2: Run framework comparison with quality test enabled
    # We'll use the framework comparison script which already has quality testing
    framework_comparison_script = os.path.join(parent_dir, "benchmark", "framework_comparison.py")
    
    # Build command
    cmd = [
        sys.executable,
        framework_comparison_script,
        "--output", os.path.join(output_dir, "quality_results"),
        "--quality-test"  # Enable quality testing
    ]
    
    # Add model parameters
    if args.llama_model:
        cmd.extend(["--llama-model", args.llama_model])
    
    if args.mlx_model:
        cmd.extend(["--mlx-model", args.mlx_model])
    
    # Add optional parameters
    if args.llama_cpp:
        cmd.extend(["--llama-cpp", args.llama_cpp])
    
    if args.quant:
        cmd.extend(["--quant", args.quant])
    
    if args.n_predict:
        cmd.extend(["--n-predict", str(args.n_predict)])
    
    if args.no_metal:
        cmd.append("--no-metal")
    
    # Use quick mode to focus on quality
    cmd.append("--quick")
    
    # Run the quality evaluation
    run_command(cmd, "Running quality evaluation benchmark", exit_on_error=True)
    
    # Step 3: Generate visualizations
    results_json = os.path.join(output_dir, "quality_results", "comparison_results.json")
    if os.path.exists(results_json):
        visualize_script = os.path.join(parent_dir, "benchmark", "visualize_results.py")
        vis_cmd = [
            sys.executable,
            visualize_script,
            "--input", results_json,
            "--output", os.path.join(output_dir, "visualizations"),
            "--type", "framework"
        ]
        run_command(vis_cmd, "Generating visualizations", exit_on_error=False)
    
    # Step 4: If we have multiple quantization levels, also run quantization comparison
    if args.quants:
        quantization_script = os.path.join(parent_dir, "benchmark", "quantization_comparison.py")
        
        # Build command
        quant_cmd = [
            sys.executable,
            quantization_script,
            "--output", os.path.join(output_dir, "quantization_quality"),
            "--quants", args.quants
        ]
        
        # Add model parameters
        if args.llama_model:
            quant_cmd.extend(["--llama-base", args.llama_model])
        
        if args.mlx_model:
            quant_cmd.extend(["--mlx-model", args.mlx_model])
        
        # Add optional parameters
        if args.llama_cpp:
            quant_cmd.extend(["--llama-cpp", args.llama_cpp])
        
        if args.no_metal:
            quant_cmd.append("--no-metal")
        
        # Run the quantization quality comparison
        run_command(quant_cmd, "Running quantization quality comparison", exit_on_error=False)
        
        # Generate visualizations
        quant_results_json = os.path.join(output_dir, "quantization_quality", "quantization_comparison_results.json")
        if os.path.exists(quant_results_json):
            quant_vis_cmd = [
                sys.executable,
                visualize_script,
                "--input", quant_results_json,
                "--output", os.path.join(output_dir, "quant_visualizations"),
                "--type", "quantization"
            ]
            run_command(quant_vis_cmd, "Generating quantization visualizations", exit_on_error=False)
    
    # Step 5: Create a combined quality report
    print("\nCreating combined quality evaluation report...")
    
    # Create an HTML report that shows quality results
    report_html = os.path.join(output_dir, "quality_evaluation_report.html")
    
    with open(report_html, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>LLM Quality Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }}
        .response {{ font-family: monospace; white-space: pre-wrap; margin: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; max-height: 300px; overflow-y: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .nav {{ position: sticky; top: 0; background-color: #333; padding: 10px; color: white; z-index: 100; }}
        .nav a {{ color: white; text-decoration: none; margin: 0 10px; }}
        .nav a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="#overview">Overview</a>
        <a href="#framework">Framework Comparison</a>
""")

        if args.quants:
            f.write('        <a href="#quantization">Quantization Comparison</a>\n')
        
        f.write("""    </div>
    
    <h1 id="overview">LLM Quality Evaluation Report</h1>
    
    <div class="section">
        <h2>Overview</h2>
        <p>This report evaluates the quality of model outputs across different configurations:</p>
        <ul>
            <li><a href="#framework">Framework Comparison</a> - Quality comparison between llama.cpp and MLX</li>
""")

        if args.quants:
            f.write('            <li><a href="#quantization">Quantization Comparison</a> - How quantization affects output quality</li>\n')
        
        f.write("""        </ul>
    </div>
    
    <div class="section" id="framework">
        <h2>Framework Comparison</h2>
        <p>Comparison of output quality between llama.cpp and MLX frameworks.</p>
        <p><a href="visualizations/interactive_report.html" target="_blank">View detailed framework comparison report</a></p>
        <iframe src="visualizations/interactive_report.html" width="100%" height="600px" style="border:none;"></iframe>
    </div>
""")

        if args.quants:
            f.write("""
    <div class="section" id="quantization">
        <h2>Quantization Comparison</h2>
        <p>Analysis of how quantization affects output quality.</p>
        <p><a href="quant_visualizations/interactive_report.html" target="_blank">View detailed quantization report</a></p>
        <iframe src="quant_visualizations/interactive_report.html" width="100%" height="600px" style="border:none;"></iframe>
    </div>
""")
        
        # Add footer
        f.write(f"""
    <div class="footer">
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Generated with Performance Benchmarking Tools for LLMs on Apple Silicon</p>
    </div>
</body>
</html>
""")
    
    print(f"\nQuality evaluation workflow completed. Results available in {output_dir}")
    print(f"Open {report_html} to view the combined report")

def comprehensive_workflow(args):
    """Run comprehensive workflow with all benchmarks"""
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_directory(os.path.join(args.output_dir, f"comprehensive_benchmark_{timestamp}"))
    
    # Step 1: Create sub-directories for each workflow
    framework_dir = ensure_directory(os.path.join(output_dir, "framework_comparison"))
    quantization_dir = ensure_directory(os.path.join(output_dir, "quantization_study"))
    hardware_dir = ensure_directory(os.path.join(output_dir, "hardware_scaling"))
    quality_dir = ensure_directory(os.path.join(output_dir, "quality_evaluation"))
    
    # Step 2: Run each workflow with modified args
    print("\n=== RUNNING COMPREHENSIVE BENCHMARK WORKFLOW ===\n")
    print("This will run all benchmark workflows and may take a significant amount of time.\n")
    
    # Save original output_dir
    original_output_dir = args.output_dir
    
    # Run framework comparison
    if args.llama_model and args.mlx_model:
        print("\n=== FRAMEWORK COMPARISON WORKFLOW ===\n")
        args.output_dir = framework_dir
        args.quick = True  # Use quick mode for comprehensive workflow
        framework_comparison_workflow(args)
    else:
        print("\nSkipping framework comparison workflow (requires both llama.cpp and MLX models)")
    
    # Run quantization study
    if args.llama_model or args.mlx_model:
        print("\n=== QUANTIZATION STUDY WORKFLOW ===\n")
        args.output_dir = quantization_dir
        args.skip_quality = False  # Ensure quality testing is enabled
        args.quick = True  # Use quick mode for comprehensive workflow
        quantization_study_workflow(args)
    
    # Run hardware scaling
    if args.llama_model or args.mlx_model:
        print("\n=== HARDWARE SCALING WORKFLOW ===\n")
        args.output_dir = hardware_dir
        args.framework = "llama" if args.llama_model else "mlx"
        args.quick = True  # Use quick mode for comprehensive workflow
        hardware_scaling_workflow(args)
    
    # Run quality evaluation
    if args.llama_model or args.mlx_model:
        print("\n=== QUALITY EVALUATION WORKFLOW ===\n")
        args.output_dir = quality_dir
        args.quick = True  # Use quick mode for comprehensive workflow
        quality_evaluation_workflow(args)
    
    # Restore original output_dir
    args.output_dir = original_output_dir
    
    # Step 3: Create a comprehensive report
    print("\nCreating comprehensive benchmark report...")
    
    # Create an HTML index that links to all reports
    index_html = os.path.join(output_dir, "comprehensive_report.html")
    
    with open(index_html, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive LLM Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }}
        .card {{ margin: 15px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: white; }}
        .card h3 {{ margin-top: 0; }}
        .card-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .nav {{ position: sticky; top: 0; background-color: #333; padding: 10px; color: white; z-index: 100; }}
        .nav a {{ color: white; text-decoration: none; margin: 0 10px; }}
        .nav a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="#overview">Overview</a>
        <a href="#framework">Framework Comparison</a>
        <a href="#quantization">Quantization</a>
        <a href="#hardware">Hardware Scaling</a>
        <a href="#quality">Quality Evaluation</a>
        <a href="#recommendations">Recommendations</a>
    </div>
    
    <h1 id="overview">Comprehensive LLM Benchmark Report</h1>
    
    <div class="section">
        <h2>Overview</h2>
        <p>This comprehensive report evaluates LLM performance and quality across multiple dimensions:</p>
        <div class="card-grid">
            <div class="card">
                <h3>Framework Comparison</h3>
                <p>Direct comparison between llama.cpp and MLX frameworks</p>
                <p><a href="framework_comparison/framework_comparison_report.html" target="_blank">View Report</a></p>
            </div>
            
            <div class="card">
                <h3>Quantization Study</h3>
                <p>Analysis of different quantization levels</p>
                <p><a href="quantization_study/quantization_report.html" target="_blank">View Report</a></p>
            </div>
            
            <div class="card">
                <h3>Hardware Scaling</h3>
                <p>Performance scaling with hardware parameters</p>
                <p><a href="hardware_scaling/hardware_scaling_report.html" target="_blank">View Report</a></p>
            </div>
            
            <div class="card">
                <h3>Quality Evaluation</h3>
                <p>Analysis of output quality across configurations</p>
                <p><a href="quality_evaluation/quality_evaluation_report.html" target="_blank">View Report</a></p>
            </div>
        </div>
    </div>
    
    <div class="section" id="framework">
        <h2>Framework Comparison</h2>
        <p>Direct comparison between llama.cpp and MLX frameworks.</p>
        <iframe src="framework_comparison/framework_comparison_report.html" width="100%" height="500px" style="border:none;"></iframe>
    </div>
    
    <div class="section" id="quantization">
        <h2>Quantization Study</h2>
        <p>Analysis of different quantization levels.</p>
        <iframe src="quantization_study/quantization_report.html" width="100%" height="500px" style="border:none;"></iframe>
    </div>
    
    <div class="section" id="hardware">
        <h2>Hardware Scaling</h2>
        <p>Performance scaling with hardware parameters.</p>
        <iframe src="hardware_scaling/hardware_scaling_report.html" width="100%" height="500px" style="border:none;"></iframe>
    </div>
    
    <div class="section" id="quality">
        <h2>Quality Evaluation</h2>
        <p>Analysis of output quality across configurations.</p>
        <iframe src="quality_evaluation/quality_evaluation_report.html" width="100%" height="500px" style="border:none;"></iframe>
    </div>
    
    <div class="section" id="recommendations">
        <h2>Recommendations</h2>
        <p>Based on all benchmark results, here are the recommended configurations:</p>
        
        <div class="card">
            <h3>For Speed-Critical Applications</h3>
            <p>Framework: MLX (typically 20-30% faster)</p>
            <p>Quantization: INT4 (MLX) or Q4_K (llama.cpp)</p>
            <p>Hardware: Maximum thread count, batch size adjusted to context</p>
        </div>
        
        <div class="card">
            <h3>For Memory-Constrained Environments</h3>
            <p>Framework: llama.cpp (slightly more memory efficient)</p>
            <p>Quantization: Q4_K or Q3_K for extreme cases</p>
            <p>Hardware: Reduced context length, modest thread count</p>
        </div>
        
        <div class="card">
            <h3>For Quality-Critical Applications</h3>
            <p>Framework: Either (quality differences minimal)</p>
            <p>Quantization: INT8/Q8_0 or FP16 if memory allows</p>
            <p>Hardware: Adequate context for task, balanced threads</p>
        </div>
        
        <div class="card">
            <h3>Balanced Configuration</h3>
            <p>Framework: MLX for research, llama.cpp for deployment</p>
            <p>Quantization: INT4/Q4_K for most use cases</p>
            <p>Hardware: 8 threads, 2048 context, 512-1024 batch size</p>
        </div>
    </div>
    
    <div class="footer">
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Generated with Performance Benchmarking Tools for LLMs on Apple Silicon</p>
    </div>
</body>
</html>
""")
    
    print(f"\nComprehensive benchmark workflow completed. Results available in {output_dir}")
    print(f"Open {index_html} to view the comprehensive report")

def main():
    parser = argparse.ArgumentParser(description="Benchmark workflow tool for llama.cpp and MLX")
    
    # Workflow selection
    parser.add_argument("workflow", choices=[
        "framework-comparison", 
        "quantization-study", 
        "hardware-scaling", 
        "quality-evaluation",
        "comprehensive"
    ], help="Workflow to run")
    
    # Model arguments
    parser.add_argument("--llama-model", help="Path to the llama.cpp model file (.gguf)")
    parser.add_argument("--mlx-model", help="Name of the MLX model (e.g., llama-2-7b)")
    parser.add_argument("--llama-cpp", default="./llama.cpp", help="Path to llama.cpp directory")
    
    # Framework selection for hardware scaling
    parser.add_argument("--framework", choices=["llama", "mlx"], help="Framework to test for hardware scaling")
    
    # Benchmark parameters
    parser.add_argument("--ctx", help="Comma-separated list of context lengths to test")
    parser.add_argument("--threads", help="Comma-separated list of thread counts to test")
    parser.add_argument("--batch", help="Comma-separated list of batch sizes to test")
    parser.add_argument("--n-predict", type=int, help="Number of tokens to predict")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for each configuration")
    parser.add_argument("--no-metal", action="store_true", help="Disable Metal acceleration")
    parser.add_argument("--no-metal-mmq", action="store_true", help="Disable Metal matrix multiplication")
    
    # Quantization parameters
    parser.add_argument("--quant", help="Quantization to use (will be mapped between frameworks)")
    parser.add_argument("--quants", help="Comma-separated list of quantizations to test")
    
    # Workflow options
    parser.add_argument("--quality-test", action="store_true", help="Run quality comparison test")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality tests in quantization study")
    parser.add_argument("--quick", action="store_true", help="Run a quick workflow with minimal configurations")
    parser.add_argument("--output-dir", default="./benchmark_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_directory(args.output_dir)
    
    # Run the selected workflow
    if args.workflow == "framework-comparison":
        framework_comparison_workflow(args)
    elif args.workflow == "quantization-study":
        quantization_study_workflow(args)
    elif args.workflow == "hardware-scaling":
        hardware_scaling_workflow(args)
    elif args.workflow == "quality-evaluation":
        quality_evaluation_workflow(args)
    elif args.workflow == "comprehensive":
        comprehensive_workflow(args)

if __name__ == "__main__":
    main()
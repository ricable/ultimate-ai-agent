#!/usr/bin/env python3
"""
Optimized Batch Inference Script for llama.cpp on Apple Silicon

This script processes a batch of prompts using llama.cpp with optimized settings
for Apple Silicon, leveraging Metal GPU acceleration.

Usage:
  python batch_inference.py --model <model_path> --input <input_file> --output <output_file> [options]

Example:
  python batch_inference.py --model models/llama-2-7b-q4_k.gguf --input prompts.txt --output results.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time
import platform
import tempfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def detect_apple_silicon():
    """Detect if running on Apple Silicon"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def get_optimal_batch_size(model_path):
    """Estimate optimal batch size based on model path"""
    if "7b" in model_path.lower():
        return 512
    elif "13b" in model_path.lower():
        return 512
    elif "70b" in model_path.lower():
        return 256
    else:
        return 512  # Default

def get_optimal_threads():
    """Estimate optimal thread count based on system"""
    if platform.system() == "Darwin":
        # Get CPU core count
        try:
            cpu_count = os.cpu_count()
            if "M1" in platform.processor() or "M2" in platform.processor() or "M3" in platform.processor():
                # For Apple Silicon, a good rule is about half of all cores
                return max(4, cpu_count // 2)
            else:
                return max(4, cpu_count - 2)
        except:
            return 4  # Default fallback
    else:
        return os.cpu_count() or 4

def process_prompt(llama_cpp_path, model_path, prompt, context_length=2048, 
                  max_tokens=512, batch_size=512, threads=4, use_metal=True, 
                  temperature=0.7, top_p=0.9, repeat_penalty=1.1):
    """Process a single prompt using llama.cpp"""
    # Create a temporary file for the prompt
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name
    
    try:
        # Build command
        cmd = [
            os.path.join(llama_cpp_path, "main"),
            "-m", model_path,
            "-f", prompt_file,
            "-n", str(max_tokens),
            "-c", str(context_length),
            "-b", str(batch_size),
            "-t", str(threads),
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--repeat-penalty", str(repeat_penalty),
            "--log-disable"  # Disable most logs for cleaner output
        ]
        
        if use_metal and detect_apple_silicon():
            cmd.append("--metal")
            cmd.append("--metal-mmq")
        
        # Record start time
        start_time = time.time()
        
        # Run the process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # Record end time
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Parse output
        output = stdout.decode('utf-8', errors='replace')
        
        # Extract completion text (after the prompt)
        # This is a simple extraction that assumes the model output follows the prompt
        completion = output.split(prompt)[-1].strip()
        
        return {
            "prompt": prompt,
            "completion": completion,
            "inference_time": inference_time,
            "success": True
        }
        
    except Exception as e:
        return {
            "prompt": prompt,
            "error": str(e),
            "success": False
        }
    finally:
        # Clean up the temporary file
        try:
            os.unlink(prompt_file)
        except:
            pass

def process_batch(llama_cpp_path, model_path, prompts, context_length=2048, 
                 max_tokens=512, batch_size=512, threads=4, use_metal=True,
                 temperature=0.7, top_p=0.9, repeat_penalty=1.1, max_workers=4):
    """Process a batch of prompts in parallel"""
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        futures = [
            executor.submit(
                process_prompt, 
                llama_cpp_path, 
                model_path, 
                prompt, 
                context_length, 
                max_tokens, 
                batch_size, 
                threads, 
                use_metal,
                temperature,
                top_p,
                repeat_penalty
            )
            for prompt in prompts
        ]
        
        # Process as they complete with a progress bar
        for future in tqdm(futures, total=len(futures), desc="Processing prompts"):
            results.append(future.result())
    
    return results

def read_prompts(input_file):
    """Read prompts from input file (text or JSONL)"""
    prompts = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.jsonl'):
                # JSONL format - one JSON object per line with a "prompt" field
                for line in f:
                    try:
                        data = json.loads(line)
                        if "prompt" in data:
                            prompts.append(data["prompt"])
                    except:
                        pass
            else:
                # Text format - one prompt per line
                prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading prompts: {e}")
        sys.exit(1)
    
    return prompts

def write_results(results, output_file):
    """Write results to output file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            if output_file.endswith('.json'):
                # Single JSON array
                json.dump(results, f, indent=2, ensure_ascii=False)
            elif output_file.endswith('.jsonl'):
                # JSONL format - one JSON object per line
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            else:
                # Plain text format
                for result in results:
                    f.write(f"Prompt: {result['prompt']}\n")
                    f.write(f"Completion: {result['completion']}\n")
                    f.write(f"Time: {result['inference_time']:.2f} seconds\n")
                    f.write("-" * 80 + "\n\n")
    except Exception as e:
        print(f"Error writing results: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Optimized Batch Inference for llama.cpp on Apple Silicon")
    parser.add_argument("--model", required=True, help="Path to the model file (.gguf)")
    parser.add_argument("--input", required=True, help="Input file with prompts (one per line, or JSONL)")
    parser.add_argument("--output", required=True, help="Output file for results (.json, .jsonl, or .txt)")
    parser.add_argument("--llama_cpp", default="../../llama.cpp", help="Path to llama.cpp directory")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate per prompt")
    parser.add_argument("--batch", type=int, help="Batch size (default: auto-detected)")
    parser.add_argument("--threads", type=int, help="Number of threads (default: auto-detected)")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--no-metal", action="store_true", help="Disable Metal acceleration")
    parser.add_argument("--parallel", type=int, help="Number of parallel processes (default: auto-detected)")
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Validate llama.cpp path
    llama_cpp_path = args.llama_cpp
    if not os.path.exists(os.path.join(llama_cpp_path, "main")):
        print(f"Error: llama.cpp binary not found at {os.path.join(llama_cpp_path, 'main')}")
        print("Please provide the correct path to llama.cpp directory with the --llama_cpp argument.")
        sys.exit(1)
    
    # Auto-detect optimal settings if not specified
    batch_size = args.batch or get_optimal_batch_size(args.model)
    threads = args.threads or get_optimal_threads()
    
    # Determine parallel processes
    # Default is CPU count divided by thread count (minimum 1, maximum 8)
    if args.parallel:
        parallel_processes = args.parallel
    else:
        cpu_count = os.cpu_count() or 4
        parallel_processes = max(1, min(8, cpu_count // threads))
    
    # Check if running on Apple Silicon
    is_apple_silicon = detect_apple_silicon()
    if is_apple_silicon:
        print("✅ Apple Silicon detected. Metal acceleration will be enabled.")
    else:
        print("⚠️ Not running on Apple Silicon. Metal acceleration will be disabled.")
        if not args.no_metal:
            args.no_metal = True
    
    # Read prompts
    prompts = read_prompts(args.input)
    print(f"📥 Loaded {len(prompts)} prompts from {args.input}")
    
    # Print settings
    print("\n🚀 Running batch inference with the following settings:")
    print(f"📦 Model: {args.model}")
    print(f"📏 Context Length: {args.ctx} tokens")
    print(f"📊 Batch Size: {batch_size}")
    print(f"🧵 Threads per Process: {threads}")
    print(f"⚙️ Parallel Processes: {parallel_processes}")
    print(f"🌡️ Temperature: {args.temp}")
    print(f"🎯 Top-p: {args.top_p}")
    print(f"🔄 Repetition Penalty: {args.repeat_penalty}")
    print(f"🔥 Metal Acceleration: {'Disabled' if args.no_metal else 'Enabled'}")
    
    # Process prompts
    print("\n⏳ Processing prompts...")
    start_time = time.time()
    
    results = process_batch(
        llama_cpp_path=llama_cpp_path,
        model_path=args.model,
        prompts=prompts,
        context_length=args.ctx,
        max_tokens=args.max_tokens,
        batch_size=batch_size,
        threads=threads,
        use_metal=not args.no_metal,
        temperature=args.temp,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        max_workers=parallel_processes
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    avg_time = sum(r["inference_time"] for r in successful_results) / len(successful_results) if successful_results else 0
    total_tokens = sum(len(r["completion"].split()) for r in successful_results) if successful_results else 0
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    # Print summary
    print("\n✅ Batch processing complete!")
    print(f"⏱️ Total Time: {total_time:.2f} seconds")
    print(f"⚡ Average Inference Time: {avg_time:.2f} seconds per prompt")
    print(f"🔣 Estimated Total Tokens: {total_tokens}")
    print(f"⚡ Throughput: {tokens_per_second:.2f} tokens/second")
    
    # Write results
    write_results(results, args.output)
    print(f"📤 Results written to {args.output}")

if __name__ == "__main__":
    main()
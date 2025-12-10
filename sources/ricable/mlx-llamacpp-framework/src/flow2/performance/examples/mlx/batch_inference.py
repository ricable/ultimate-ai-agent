#!/usr/bin/env python3
"""
Optimized Batch Inference Script for MLX on Apple Silicon

This script processes a batch of prompts using MLX with optimized settings
for Apple Silicon, leveraging Metal GPU acceleration.

Usage:
  python batch_inference.py --model <model_name> --input <input_file> --output <output_file> [options]

Example:
  python batch_inference.py --model llama-2-7b --quant int4 --input prompts.txt --output results.jsonl
"""

import argparse
import json
import os
import sys
import time
import platform
import gc
import threading
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing

# Check for MLX and required packages
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Error: MLX not available. Please install MLX and MLX-LM.")
    print("pip install mlx mlx-lm")
    sys.exit(1)

# Flash Attention Integration
try:
    from flash_attention_mlx import OptimizedMLXMultiHeadAttention, FlashAttentionBenchmark
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

def detect_apple_silicon():
    """Detect if running on Apple Silicon"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def get_optimal_processes():
    """Estimate optimal process count based on system"""
    if platform.system() == "Darwin":
        # Get CPU core count
        try:
            cpu_count = os.cpu_count()
            if "M1" in platform.processor() or "M2" in platform.processor() or "M3" in platform.processor():
                # For Apple Silicon, use about half the cores
                return max(2, min(4, cpu_count // 2))
            else:
                return max(2, min(4, cpu_count // 2))
        except:
            return 2  # Default fallback
    else:
        return max(2, min(4, (os.cpu_count() or 4) // 2))


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

def process_prompt(prompt, model_name, quantization=None, max_tokens=512,
                  temperature=0.7, top_p=0.95, top_k=40, repetition_penalty=1.1,
                  use_gpu=True, use_flash_attention=True, flash_block_size=None):
    """Process a single prompt using MLX"""
    if not MLX_AVAILABLE:
        return {
            "prompt": prompt,
            "error": "MLX not available",
            "success": False
        }
    
    # Set device based on arguments and availability
    if use_gpu and detect_apple_silicon():
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)
    
    try:
        # Clear cache and collect garbage
        gc.collect()
        mx.clear_cache()
        
        # Load the model
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
        
        # Generate completion
        generate_start = time.time()
        completion_tokens = generate(
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
        
        # Convert tokens to text
        completion = tokenizer.decode(completion_tokens)
        
        # Clean up
        del model, tokenizer, completion_tokens
        gc.collect()
        mx.clear_cache()
        
        return {
            "prompt": prompt,
            "completion": completion,
            "tokens": len(completion_tokens),
            "load_time": load_time,
            "generate_time": generate_time,
            "total_time": load_time + generate_time,
            "flash_attention_layers": flash_replacements,
            "success": True
        }
        
    except Exception as e:
        return {
            "prompt": prompt,
            "error": str(e),
            "success": False
        }

def process_prompt_wrapper(args):
    """Wrapper for process_prompt to use with ProcessPoolExecutor"""
    return process_prompt(**args)

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
                    if 'generate_time' in result:
                        f.write(f"Time: {result['generate_time']:.2f} seconds\n")
                    f.write("-" * 80 + "\n\n")
    except Exception as e:
        print(f"Error writing results: {e}")
        sys.exit(1)

def main():
    if not MLX_AVAILABLE:
        print("Error: MLX not available. Please install MLX and MLX-LM.")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Optimized Batch Inference for MLX on Apple Silicon")
    parser.add_argument("--model", required=True, help="Model name (e.g., llama-2-7b)")
    parser.add_argument("--input", required=True, help="Input file with prompts (one per line, or JSONL)")
    parser.add_argument("--output", required=True, help="Output file for results (.json, .jsonl, or .txt)")
    parser.add_argument("--quant", choices=["int4", "int8"], help="Quantization type")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate per prompt")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--processes", type=int, help="Number of parallel processes (default: auto-detected)")
    parser.add_argument("--disable-flash-attention", action="store_true", help="Disable Flash Attention optimization")
    parser.add_argument("--flash-block-size", type=int, default=None, help="Flash Attention block size (auto if None)")
    args = parser.parse_args()
    
    # Auto-detect optimal settings if not specified
    processes = args.processes or get_optimal_processes()
    
    # Check if running on Apple Silicon
    is_apple_silicon = detect_apple_silicon()
    if is_apple_silicon and not args.no_gpu:
        print("✅ Apple Silicon detected. Metal acceleration will be enabled.")
    else:
        if is_apple_silicon and args.no_gpu:
            print("⚠️ Apple Silicon detected but GPU is disabled. Using CPU.")
        else:
            print("⚠️ Not running on Apple Silicon. Using CPU.")
            if not args.no_gpu:
                args.no_gpu = True
    
    # Read prompts
    prompts = read_prompts(args.input)
    print(f"📥 Loaded {len(prompts)} prompts from {args.input}")
    
    # Print settings
    print("\n🚀 Running batch inference with the following settings:")
    print(f"📦 Model: {args.model}")
    print(f"📊 Quantization: {args.quant or 'none (FP16)'}")
    print(f"📏 Max Tokens: {args.max_tokens}")
    print(f"⚙️ Parallel Processes: {processes}")
    print(f"🌡️ Temperature: {args.temp}")
    print(f"🎯 Top-p: {args.top_p}")
    print(f"🔝 Top-k: {args.top_k}")
    print(f"🔄 Repetition Penalty: {args.repeat_penalty}")
    print(f"🔥 GPU Acceleration: {'Disabled' if args.no_gpu else 'Enabled'}")
    print(f"⚡ Flash Attention: {'Disabled' if args.disable_flash_attention else 'Enabled'}")
    
    # Prepare arguments for each prompt
    prompt_args = [
        {
            "prompt": prompt,
            "model_name": args.model,
            "quantization": args.quant,
            "max_tokens": args.max_tokens,
            "temperature": args.temp,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repeat_penalty,
            "use_gpu": not args.no_gpu,
            "use_flash_attention": not args.disable_flash_attention,
            "flash_block_size": args.flash_block_size
        }
        for prompt in prompts
    ]
    
    # Process prompts
    print("\n⏳ Processing prompts...")
    start_time = time.time()
    
    results = []
    
    # Use ProcessPoolExecutor for parallel processing
    # Note: Using processes instead of threads because MLX operations need independent contexts
    if processes > 1 and len(prompts) > 1:
        # Process in parallel
        with ProcessPoolExecutor(max_workers=processes) as executor:
            # Process with progress bar
            for result in tqdm(executor.map(process_prompt_wrapper, prompt_args), 
                              total=len(prompt_args), desc="Processing"):
                results.append(result)
    else:
        # Process sequentially
        for args in tqdm(prompt_args, desc="Processing"):
            results.append(process_prompt_wrapper(args))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    successful_results = [r for r in results if r["success"]]
    avg_generate_time = sum(r.get("generate_time", 0) for r in successful_results) / len(successful_results) if successful_results else 0
    total_tokens = sum(r.get("tokens", 0) for r in successful_results) if successful_results else 0
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    # Print summary
    print("\n✅ Batch processing complete!")
    print(f"⏱️ Total Time: {total_time:.2f} seconds")
    print(f"⚡ Average Generation Time: {avg_generate_time:.2f} seconds per prompt")
    print(f"🔣 Total Tokens Generated: {total_tokens}")
    print(f"⚡ Throughput: {tokens_per_second:.2f} tokens/second")
    
    # Write results
    write_results(results, args.output)
    print(f"📤 Results written to {args.output}")

if __name__ == "__main__":
    # Set start method for multiprocessing to 'spawn' for better compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main()
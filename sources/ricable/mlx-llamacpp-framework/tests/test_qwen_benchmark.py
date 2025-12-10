#!/usr/bin/env python3
"""
Custom benchmark script for Qwen2.5-1.5B-Instruct
"""

from mlx_lm import load, generate
import time
import json
import psutil
import platform
import os

def get_system_info():
    """Get system information"""
    return {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
        "ram_available_gb": psutil.virtual_memory().available / (1024**3),
        "cpu_count": psutil.cpu_count(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

def get_model_size(model_path):
    """Calculate model size in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024**3)

def benchmark_qwen():
    """Run comprehensive benchmarks on Qwen model"""
    
    model_path = "./models/mlx/qwen2.5-1.5b-instruct"
    
    # System info
    system_info = get_system_info()
    model_size_gb = get_model_size(model_path)
    
    print("="*60)
    print("QWEN 2.5-1.5B-INSTRUCT BENCHMARK")
    print("="*60)
    print(f"Model path: {model_path}")
    print(f"Model size: {model_size_gb:.2f} GB")
    print(f"System: {system_info['os']}")
    print(f"RAM: {system_info['ram_total_gb']:.1f} GB")
    print(f"CPU: {system_info['cpu_count']} cores")
    
    # Load model and measure loading time
    print("\nLoading model...")
    start_load = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Test prompts of different lengths
    test_prompts = [
        {
            "name": "Short prompt",
            "prompt": "What is AI?",
            "max_tokens": 50
        },
        {
            "name": "Medium prompt", 
            "prompt": "Explain how machine learning works and provide examples of its applications in daily life",
            "max_tokens": 150
        },
        {
            "name": "Long prompt",
            "prompt": "Write a detailed technical explanation of neural networks, including how they process information, learn from data, and make predictions. Include examples of different types of neural networks and their specific use cases.",
            "max_tokens": 300
        },
        {
            "name": "Code generation",
            "prompt": "Write a Python function that implements a binary search algorithm with proper error handling and documentation",
            "max_tokens": 200
        },
        {
            "name": "Creative writing",
            "prompt": "Write a creative short story about a time traveler who discovers that changing small things in the past has unexpected consequences",
            "max_tokens": 400
        }
    ]
    
    benchmark_results = []
    
    for i, test in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Testing: {test['name']}")
        print(f"Prompt: {test['prompt'][:100]}...")
        
        # Run multiple iterations for stable results
        iterations = 3
        times = []
        token_counts = []
        memory_usage = []
        
        for iteration in range(iterations):
            # Measure memory before generation
            memory_before = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            # Generate response
            start_time = time.time()
            response = generate(
                model, 
                tokenizer, 
                test['prompt'], 
                max_tokens=test['max_tokens']
            )
            generation_time = time.time() - start_time
            
            # Measure memory after generation
            memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            # Count tokens (approximate)
            response_tokens = len(response.split())
            
            times.append(generation_time)
            token_counts.append(response_tokens)
            memory_usage.append(memory_after - memory_before)
            
            print(f"  Iteration {iteration + 1}: {generation_time:.2f}s, {response_tokens} tokens, {response_tokens/generation_time:.1f} tok/s")
        
        # Calculate averages
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        avg_tokens_per_sec = avg_tokens / avg_time
        avg_memory_delta = sum(memory_usage) / len(memory_usage)
        
        # Show sample response
        print(f"  Average: {avg_time:.2f}s, {avg_tokens:.0f} tokens, {avg_tokens_per_sec:.1f} tok/s")
        print(f"  Memory delta: {avg_memory_delta:.1f} MB")
        print(f"  Sample response: {response[:150]}...")
        
        # Store results
        benchmark_results.append({
            "test_name": test['name'],
            "prompt": test['prompt'],
            "max_tokens": test['max_tokens'],
            "avg_generation_time": avg_time,
            "avg_tokens_generated": avg_tokens,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "avg_memory_delta_mb": avg_memory_delta,
            "sample_response": response,
            "individual_times": times,
            "individual_token_counts": token_counts
        })
    
    # Calculate overall statistics
    overall_stats = {
        "total_tests": len(benchmark_results),
        "avg_generation_time": sum(r['avg_generation_time'] for r in benchmark_results) / len(benchmark_results),
        "avg_tokens_per_sec": sum(r['avg_tokens_per_sec'] for r in benchmark_results) / len(benchmark_results),
        "total_tokens_generated": sum(r['avg_tokens_generated'] for r in benchmark_results),
        "peak_tokens_per_sec": max(r['avg_tokens_per_sec'] for r in benchmark_results),
        "min_tokens_per_sec": min(r['avg_tokens_per_sec'] for r in benchmark_results)
    }
    
    # Create comprehensive report
    full_report = {
        "model_info": {
            "name": "Qwen2.5-1.5B-Instruct",
            "path": model_path,
            "size_gb": model_size_gb,
            "load_time": load_time
        },
        "system_info": system_info,
        "benchmark_results": benchmark_results,
        "overall_stats": overall_stats
    }
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: Qwen2.5-1.5B-Instruct")
    print(f"Model size: {model_size_gb:.2f} GB")
    print(f"Load time: {load_time:.2f} seconds")
    print(f"Tests completed: {overall_stats['total_tests']}")
    print(f"Average generation time: {overall_stats['avg_generation_time']:.2f} seconds")
    print(f"Average tokens/sec: {overall_stats['avg_tokens_per_sec']:.1f}")
    print(f"Peak tokens/sec: {overall_stats['peak_tokens_per_sec']:.1f}")
    print(f"Min tokens/sec: {overall_stats['min_tokens_per_sec']:.1f}")
    print(f"Total tokens generated: {overall_stats['total_tokens_generated']:.0f}")
    
    # Save results
    with open('qwen_comprehensive_benchmark.json', 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"\nDetailed results saved to qwen_comprehensive_benchmark.json")
    return full_report

if __name__ == "__main__":
    benchmark_qwen()
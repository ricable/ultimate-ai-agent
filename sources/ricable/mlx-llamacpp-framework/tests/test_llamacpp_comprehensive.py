#!/usr/bin/env python3
"""
Comprehensive llama.cpp testing for both TinyLlama and Qwen2.5 models
"""

from llama_cpp import Llama
import time
import json
import psutil
import platform
import os
from typing import Dict, List, Any

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
    """Get model file size in GB"""
    return os.path.getsize(model_path) / (1024**3)

def test_model_llamacpp(model_path: str, model_name: str) -> Dict[str, Any]:
    """Test a single model with llama.cpp"""
    
    print(f"\n{'='*60}")
    print(f"TESTING {model_name.upper()} WITH LLAMA.CPP")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    
    model_size_gb = get_model_size(model_path)
    print(f"Model size: {model_size_gb:.2f} GB")
    
    # Load model and measure loading time
    print("\nLoading model...")
    start_load = time.time()
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=1,  # Use Metal on Apple Silicon
            verbose=False
        )
        load_time = time.time() - start_load
        print(f"Model loaded in {load_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"error": str(e)}
    
    # Test prompts
    test_prompts = [
        {
            "name": "AI Explanation",
            "prompt": "Explain artificial intelligence in simple terms",
            "max_tokens": 128
        },
        {
            "name": "Programming Help", 
            "prompt": "Write a simple Python function to calculate fibonacci numbers",
            "max_tokens": 200
        },
        {
            "name": "Creative Writing",
            "prompt": "Write a short story about a robot learning to paint",
            "max_tokens": 300
        },
        {
            "name": "Math Problem",
            "prompt": "Solve this math problem step by step: What is 15% of 240?",
            "max_tokens": 150
        },
        {
            "name": "Conversational",
            "prompt": "Hi! How are you doing today? What's your favorite hobby?",
            "max_tokens": 128
        }
    ]
    
    results = []
    
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
            try:
                response = llm(
                    test['prompt'],
                    max_tokens=test['max_tokens'],
                    temperature=0.7,
                    top_p=0.95,
                    repeat_penalty=1.1,
                    stop=["Human:", "User:", "\n\n"],
                    echo=False
                )
                generation_time = time.time() - start_time
                
                # Measure memory after generation
                memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
                
                # Extract response text
                response_text = response['choices'][0]['text']
                response_tokens = len(response_text.split())
                
                times.append(generation_time)
                token_counts.append(response_tokens)
                memory_usage.append(memory_after - memory_before)
                
                print(f"  Iteration {iteration + 1}: {generation_time:.2f}s, {response_tokens} tokens, {response_tokens/generation_time:.1f} tok/s")
                
            except Exception as e:
                print(f"  Iteration {iteration + 1}: Error - {e}")
                times.append(float('inf'))
                token_counts.append(0)
                memory_usage.append(0)
        
        if len([t for t in times if t != float('inf')]) > 0:
            # Calculate averages (excluding failed runs)
            valid_times = [t for t in times if t != float('inf')]
            valid_tokens = [token_counts[i] for i, t in enumerate(times) if t != float('inf')]
            valid_memory = [memory_usage[i] for i, t in enumerate(times) if t != float('inf')]
            
            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                avg_tokens = sum(valid_tokens) / len(valid_tokens)
                avg_tokens_per_sec = avg_tokens / avg_time
                avg_memory_delta = sum(valid_memory) / len(valid_memory)
                
                # Show sample response from last successful iteration
                try:
                    sample_response = llm(test['prompt'], max_tokens=test['max_tokens'], temperature=0.7)
                    sample_text = sample_response['choices'][0]['text']
                except:
                    sample_text = "Error generating sample"
                
                print(f"  Average: {avg_time:.2f}s, {avg_tokens:.0f} tokens, {avg_tokens_per_sec:.1f} tok/s")
                print(f"  Memory delta: {avg_memory_delta:.1f} MB")
                print(f"  Sample response: {sample_text[:150]}...")
                
                # Store results
                results.append({
                    "test_name": test['name'],
                    "prompt": test['prompt'],
                    "max_tokens": test['max_tokens'],
                    "avg_generation_time": avg_time,
                    "avg_tokens_generated": avg_tokens,
                    "avg_tokens_per_sec": avg_tokens_per_sec,
                    "avg_memory_delta_mb": avg_memory_delta,
                    "sample_response": sample_text,
                    "successful_runs": len(valid_times),
                    "total_runs": iterations
                })
            else:
                print(f"  All iterations failed for {test['name']}")
                results.append({
                    "test_name": test['name'],
                    "error": "All iterations failed",
                    "successful_runs": 0,
                    "total_runs": iterations
                })
        else:
            print(f"  No valid results for {test['name']}")
    
    # Calculate overall statistics
    successful_results = [r for r in results if 'avg_tokens_per_sec' in r]
    
    if successful_results:
        overall_stats = {
            "total_tests": len(results),
            "successful_tests": len(successful_results),
            "avg_generation_time": sum(r['avg_generation_time'] for r in successful_results) / len(successful_results),
            "avg_tokens_per_sec": sum(r['avg_tokens_per_sec'] for r in successful_results) / len(successful_results),
            "total_tokens_generated": sum(r['avg_tokens_generated'] for r in successful_results),
            "peak_tokens_per_sec": max(r['avg_tokens_per_sec'] for r in successful_results),
            "min_tokens_per_sec": min(r['avg_tokens_per_sec'] for r in successful_results)
        }
    else:
        overall_stats = {
            "total_tests": len(results),
            "successful_tests": 0,
            "error": "No successful tests"
        }
    
    # Create report for this model
    model_report = {
        "model_info": {
            "name": model_name,
            "path": model_path,
            "size_gb": model_size_gb,
            "load_time": load_time,
            "framework": "llama.cpp"
        },
        "test_results": results,
        "overall_stats": overall_stats
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY FOR {model_name.upper()}")
    print(f"{'='*60}")
    if 'error' not in overall_stats:
        print(f"Model: {model_name}")
        print(f"Framework: llama.cpp")
        print(f"Model size: {model_size_gb:.2f} GB")
        print(f"Load time: {load_time:.2f} seconds")
        print(f"Successful tests: {overall_stats['successful_tests']}/{overall_stats['total_tests']}")
        print(f"Average generation time: {overall_stats['avg_generation_time']:.2f} seconds")
        print(f"Average tokens/sec: {overall_stats['avg_tokens_per_sec']:.1f}")
        print(f"Peak tokens/sec: {overall_stats['peak_tokens_per_sec']:.1f}")
        print(f"Total tokens generated: {overall_stats['total_tokens_generated']:.0f}")
    else:
        print(f"Testing failed: {overall_stats['error']}")
    
    return model_report

def run_comprehensive_llamacpp_tests():
    """Run comprehensive tests on both models with llama.cpp"""
    
    print("="*80)
    print("COMPREHENSIVE LLAMA.CPP TESTING")
    print("="*80)
    
    system_info = get_system_info()
    print(f"System: {system_info['os']}")
    print(f"RAM: {system_info['ram_total_gb']:.1f} GB")
    print(f"CPU: {system_info['cpu_count']} cores")
    
    # Model paths
    models = {
        "TinyLlama-1.1B-Chat": "./models/llamacpp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "Qwen2.5-1.5B-Instruct": "./models/llamacpp/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    }
    
    all_results = {}
    
    # Test each model
    for model_name, model_path in models.items():
        if os.path.exists(model_path):
            print(f"\nTesting {model_name}...")
            result = test_model_llamacpp(model_path, model_name)
            all_results[model_name] = result
        else:
            print(f"\nModel not found: {model_path}")
            all_results[model_name] = {"error": f"Model file not found: {model_path}"}
    
    # Create comprehensive report
    comprehensive_report = {
        "system_info": system_info,
        "framework": "llama.cpp",
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_tested": all_results
    }
    
    # Save results
    output_file = 'llamacpp_comprehensive_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(comprehensive_report, f, indent=2)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    
    return comprehensive_report

if __name__ == "__main__":
    results = run_comprehensive_llamacpp_tests()
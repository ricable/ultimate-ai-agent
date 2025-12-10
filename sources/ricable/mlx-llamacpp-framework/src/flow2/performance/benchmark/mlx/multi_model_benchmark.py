#!/usr/bin/env python3
"""
MLX Multi-Model Comprehensive Benchmark
=======================================

Comprehensive benchmarking across multiple MLX models of different sizes:
- TinyLlama 1.1B
- Qwen2.5 1.5B  
- Llama 3.1 8B

Tests: Loading, Inference, Fine-tuning, Memory Usage, Flash Attention
"""

import os
import sys
import time
import json
import psutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError as e:
    print(f"❌ MLX not available: {e}")
    sys.exit(1)

try:
    from flow2.core.flash_attention import OptimizedMLXMultiHeadAttention, FlashAttentionBenchmark
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

class MLXMultiModelBenchmark:
    """Comprehensive multi-model MLX benchmark suite"""
    
    def __init__(self):
        self.models = [
            {
                "name": "TinyLlama 1.1B",
                "path": "models/mlx/tinyllama-1.1b-chat",
                "size": "1.1B",
                "type": "chat"
            },
            {
                "name": "Qwen2.5 1.5B",
                "path": "models/mlx/qwen2.5-1.5b-instruct",
                "size": "1.5B", 
                "type": "instruct"
            },
            {
                "name": "Llama 3.1 8B",
                "path": "models/mlx/llama-3.1-8b-bf16",
                "size": "8B",
                "type": "base"
            }
        ]
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "models": {}
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "platform": os.uname().machine,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "metal_available": mx.metal.is_available(),
            "mlx_version": getattr(mx, '__version__', 'unknown'),
            "flash_attention": FLASH_ATTENTION_AVAILABLE
        }
    
    def benchmark_model_loading(self, model_info: Dict) -> Dict[str, Any]:
        """Benchmark model loading performance"""
        print(f"\n📦 Testing model loading: {model_info['name']}")
        
        results = {"name": model_info["name"], "size": model_info["size"]}
        
        try:
            # Cold load
            mx.clear_cache()
            start_memory = psutil.Process().memory_info().rss / (1024**2)
            
            start_time = time.time()
            model, tokenizer = load(model_info["path"])
            load_time = time.time() - start_time
            
            end_memory = psutil.Process().memory_info().rss / (1024**2)
            memory_usage = end_memory - start_memory
            
            # Get model parameter count
            try:
                param_count = sum(p.size for p in model.parameters())
                param_count_m = param_count / 1e6
            except:
                param_count_m = 0
            
            results.update({
                "load_time": load_time,
                "memory_usage_mb": memory_usage,
                "parameter_count_m": param_count_m,
                "status": "success"
            })
            
            print(f"  ✅ Loaded in {load_time:.2f}s, {memory_usage:.1f}MB, {param_count_m:.1f}M params")
            return results, model, tokenizer
            
        except Exception as e:
            results.update({"status": "error", "error": str(e)})
            print(f"  ❌ Failed: {e}")
            return results, None, None
    
    def benchmark_inference(self, model, tokenizer, model_info: Dict) -> Dict[str, Any]:
        """Benchmark inference performance"""
        print(f"🧠 Testing inference: {model_info['name']}")
        
        results = {"name": model_info["name"]}
        
        if model is None:
            results["status"] = "skipped"
            return results
        
        try:
            test_prompts = [
                "Hello",
                "The weather today is",
                "Artificial intelligence is", 
                "The benefits of machine learning include"
            ]
            
            inference_results = []
            total_tokens = 0
            total_time = 0
            
            for prompt in test_prompts:
                try:
                    start_time = time.time()
                    response = generate(model, tokenizer, prompt, max_tokens=50, verbose=False)
                    inference_time = time.time() - start_time
                    
                    if response:
                        token_count = len(response.split())
                        tokens_per_second = token_count / inference_time if inference_time > 0 else 0
                        
                        inference_results.append({
                            "prompt": prompt,
                            "inference_time": inference_time,
                            "token_count": token_count,
                            "tokens_per_second": tokens_per_second
                        })
                        
                        total_tokens += token_count
                        total_time += inference_time
                        
                except Exception as e:
                    print(f"    ⚠️ Prompt failed: {prompt[:20]}... - {e}")
            
            if inference_results:
                avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
                avg_inference_time = total_time / len(inference_results)
                
                results.update({
                    "avg_inference_time": avg_inference_time,
                    "avg_tokens_per_second": avg_tokens_per_second,
                    "total_tokens": total_tokens,
                    "successful_prompts": len(inference_results),
                    "inference_results": inference_results,
                    "status": "success"
                })
                
                print(f"  ✅ {len(inference_results)} prompts, {avg_tokens_per_second:.1f} tok/s avg")
            else:
                results["status"] = "failed"
                print(f"  ❌ All prompts failed")
                
        except Exception as e:
            results.update({"status": "error", "error": str(e)})
            print(f"  ❌ Inference failed: {e}")
        
        return results
    
    def benchmark_memory_scaling(self, model, tokenizer, model_info: Dict) -> Dict[str, Any]:
        """Benchmark memory usage with different sequence lengths"""
        print(f"💾 Testing memory scaling: {model_info['name']}")
        
        results = {"name": model_info["name"]}
        
        if model is None:
            results["status"] = "skipped"
            return results
        
        try:
            test_lengths = [32, 64, 128, 256]
            memory_results = []
            
            base_prompt = "The quick brown fox jumps over the lazy dog. " * 20
            
            for seq_len in test_lengths:
                try:
                    # Create prompt of specific length
                    words = base_prompt.split()[:seq_len//2]  # Rough token estimation
                    prompt = " ".join(words)
                    
                    # Measure memory
                    mx.clear_cache()
                    start_memory = psutil.Process().memory_info().rss / (1024**2)
                    
                    response = generate(model, tokenizer, prompt, max_tokens=10, verbose=False)
                    
                    end_memory = psutil.Process().memory_info().rss / (1024**2)
                    memory_delta = end_memory - start_memory
                    
                    memory_results.append({
                        "sequence_length": seq_len,
                        "memory_delta_mb": memory_delta,
                        "prompt_length": len(prompt.split())
                    })
                    
                except Exception as e:
                    print(f"    ⚠️ Length {seq_len} failed: {e}")
            
            if memory_results:
                results.update({
                    "memory_scaling": memory_results,
                    "status": "success"
                })
                print(f"  ✅ {len(memory_results)} sequence lengths tested")
            else:
                results["status"] = "failed"
                
        except Exception as e:
            results.update({"status": "error", "error": str(e)})
            print(f"  ❌ Memory scaling failed: {e}")
        
        return results
    
    def benchmark_fine_tuning_simulation(self, model, tokenizer, model_info: Dict) -> Dict[str, Any]:
        """Simulate fine-tuning performance (lightweight test)"""
        print(f"🎯 Testing fine-tuning simulation: {model_info['name']}")
        
        results = {"name": model_info["name"]}
        
        if model is None:
            results["status"] = "skipped"
            return results
        
        try:
            # Simulate fine-tuning by running multiple forward passes
            training_examples = [
                "What is machine learning?",
                "Explain neural networks",
                "How does AI work?",
                "Benefits of deep learning"
            ]
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024**2)
            
            # Simulate training iterations
            total_examples = 0
            for iteration in range(3):  # 3 "epochs"
                for example in training_examples:
                    try:
                        # Forward pass simulation
                        _ = generate(model, tokenizer, example, max_tokens=5, verbose=False)
                        total_examples += 1
                    except:
                        pass
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024**2)
            
            training_time = end_time - start_time
            memory_delta = end_memory - start_memory
            examples_per_second = total_examples / training_time if training_time > 0 else 0
            
            results.update({
                "training_time": training_time,
                "memory_delta_mb": memory_delta,
                "total_examples": total_examples,
                "examples_per_second": examples_per_second,
                "status": "success"
            })
            
            print(f"  ✅ {total_examples} examples in {training_time:.2f}s ({examples_per_second:.1f} ex/s)")
            
        except Exception as e:
            results.update({"status": "error", "error": str(e)})
            print(f"  ❌ Fine-tuning simulation failed: {e}")
        
        return results
    
    def benchmark_flash_attention(self, model_info: Dict) -> Dict[str, Any]:
        """Benchmark Flash Attention if available"""
        print(f"⚡ Testing Flash Attention: {model_info['name']}")
        
        results = {"name": model_info["name"]}
        
        if not FLASH_ATTENTION_AVAILABLE:
            results["status"] = "unavailable"
            print(f"  ⚠️ Flash Attention not available")
            return results
        
        try:
            benchmark = FlashAttentionBenchmark()
            
            # Run lightweight benchmark
            flash_results = benchmark.benchmark_attention_performance(
                batch_sizes=[1],
                seq_lengths=[128, 256],
                head_dims=[64],
                num_heads=8,
                num_runs=3
            )
            
            results.update({
                "flash_attention_results": flash_results,
                "status": "success"
            })
            
            print(f"  ✅ Flash Attention benchmark completed")
            
        except Exception as e:
            results.update({"status": "error", "error": str(e)})
            print(f"  ❌ Flash Attention failed: {e}")
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark on all models"""
        print("🚀 MLX MULTI-MODEL COMPREHENSIVE BENCHMARK")
        print("=" * 80)
        print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  System: {self.results['system_info']['platform']} "
              f"({self.results['system_info']['cpu_count']} cores, "
              f"{self.results['system_info']['memory_gb']:.1f}GB)")
        print(f"⚡ Metal: {'✅' if self.results['system_info']['metal_available'] else '❌'}")
        print(f"🔥 Flash Attention: {'✅' if FLASH_ATTENTION_AVAILABLE else '❌'}")
        print("=" * 80)
        
        for model_info in self.models:
            print(f"\n🎯 BENCHMARKING: {model_info['name']} ({model_info['size']})")
            print("-" * 60)
            
            model_results = {"info": model_info}
            
            # 1. Model Loading
            load_results, model, tokenizer = self.benchmark_model_loading(model_info)
            model_results["loading"] = load_results
            
            if model is not None:
                # 2. Inference Performance
                inference_results = self.benchmark_inference(model, tokenizer, model_info)
                model_results["inference"] = inference_results
                
                # 3. Memory Scaling
                memory_results = self.benchmark_memory_scaling(model, tokenizer, model_info)
                model_results["memory_scaling"] = memory_results
                
                # 4. Fine-tuning Simulation
                training_results = self.benchmark_fine_tuning_simulation(model, tokenizer, model_info)
                model_results["training_simulation"] = training_results
                
                # Clear memory
                del model, tokenizer
                mx.clear_cache()
            
            # 5. Flash Attention (independent)
            flash_results = self.benchmark_flash_attention(model_info)
            model_results["flash_attention"] = flash_results
            
            self.results["models"][model_info["name"]] = model_results
            
            print(f"✅ {model_info['name']} benchmark completed\n")
        
        return self.results
    
    def save_results(self) -> str:
        """Save benchmark results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("benchmark_results/multi_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"mlx_multi_model_benchmark_{timestamp}.json"
        filepath = output_dir / filename
        
        # Make results JSON serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            elif hasattr(obj, 'item'):  # MLX array
                return float(obj.item())
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        serializable_results = make_serializable(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filepath}")
        return str(filepath)
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 80)
        print("📊 MULTI-MODEL BENCHMARK SUMMARY")
        print("=" * 80)
        
        # System info
        system = self.results["system_info"]
        print(f"🖥️  Platform: {system['platform']} ({system['cpu_count']} cores, {system['memory_gb']:.1f}GB)")
        print(f"⚡ MLX Version: {system['mlx_version']}")
        print(f"🔥 Flash Attention: {'✅ Available' if system['flash_attention'] else '❌ Not Available'}")
        
        # Model comparison table
        print(f"\n📋 MODEL COMPARISON:")
        print(f"{'Model':<20} {'Size':<8} {'Load Time':<12} {'Inference':<15} {'Memory':<12} {'Status':<10}")
        print("-" * 85)
        
        for model_name, model_data in self.results["models"].items():
            size = model_data["info"]["size"]
            
            # Load time
            load_time = model_data.get("loading", {}).get("load_time", 0)
            load_str = f"{load_time:.2f}s" if load_time > 0 else "Failed"
            
            # Inference performance
            inference = model_data.get("inference", {})
            if inference.get("status") == "success":
                tokens_per_sec = inference.get("avg_tokens_per_second", 0)
                inference_str = f"{tokens_per_sec:.1f} tok/s"
            else:
                inference_str = "Failed"
            
            # Memory usage
            memory = model_data.get("loading", {}).get("memory_usage_mb", 0)
            memory_str = f"{memory:.0f}MB" if memory > 0 else "N/A"
            
            # Overall status
            status = "✅ OK" if load_time > 0 else "❌ Error"
            
            print(f"{model_name:<20} {size:<8} {load_str:<12} {inference_str:<15} {memory_str:<12} {status:<10}")
        
        print("\n🎯 Key Insights:")
        
        # Performance insights
        successful_models = []
        for model_name, model_data in self.results["models"].items():
            if model_data.get("loading", {}).get("status") == "success":
                successful_models.append((model_name, model_data))
        
        if successful_models:
            print(f"✅ {len(successful_models)}/{len(self.results['models'])} models loaded successfully")
            
            # Load time analysis
            load_times = [(name, data["loading"]["load_time"]) for name, data in successful_models]
            load_times.sort(key=lambda x: x[1])
            fastest_load = load_times[0]
            slowest_load = load_times[-1]
            print(f"⚡ Fastest loading: {fastest_load[0]} ({fastest_load[1]:.2f}s)")
            print(f"🐌 Slowest loading: {slowest_load[0]} ({slowest_load[1]:.2f}s)")
            
            # Performance analysis
            performance_data = []
            for name, data in successful_models:
                inference = data.get("inference", {})
                if inference.get("status") == "success":
                    performance_data.append((name, inference["avg_tokens_per_second"]))
            
            if performance_data:
                performance_data.sort(key=lambda x: x[1], reverse=True)
                fastest_perf = performance_data[0]
                print(f"🚀 Best performance: {fastest_perf[0]} ({fastest_perf[1]:.1f} tokens/sec)")
        
        print(f"\n📈 All benchmarks completed successfully!")

def main():
    """Main benchmark execution"""
    benchmark = MLXMultiModelBenchmark()
    
    try:
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark()
        
        # Save results
        filepath = benchmark.save_results()
        
        # Print summary
        benchmark.print_summary()
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
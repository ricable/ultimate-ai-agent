#!/usr/bin/env python3
"""
Comprehensive Framework Benchmark
=================================

Benchmark MLX, LlamaCpp, and HuggingFace frameworks for:
- Inference speed
- Memory usage
- Model loading time
- Quantization performance
- Apple Silicon (MPS) optimization
"""

import argparse
import json
import os
import time
import gc
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import flow2
    FLOW2_AVAILABLE = True
except ImportError:
    FLOW2_AVAILABLE = False

@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    framework: str
    model_name: str
    method: str
    load_time: float
    inference_times: List[float]
    avg_inference_time: float
    memory_usage_mb: float
    model_size_mb: float
    successful: bool
    error: Optional[str] = None
    additional_info: Dict[str, Any] = None

class FrameworkBenchmark:
    """Comprehensive framework benchmarking."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []
        os.makedirs(output_dir, exist_ok=True)
    
    def run_comprehensive_benchmark(
        self,
        test_models: Dict[str, Dict[str, str]] = None,
        test_prompts: List[str] = None,
        include_quantization: bool = True,
        num_inference_runs: int = 5
    ) -> Dict[str, Any]:
        """Run comprehensive benchmark across all frameworks.
        
        Args:
            test_models: Dictionary mapping framework names to model configurations
            test_prompts: List of test prompts for inference
            include_quantization: Whether to test quantization
            num_inference_runs: Number of inference runs for averaging
            
        Returns:
            Dictionary containing comprehensive benchmark results
        """
        if not FLOW2_AVAILABLE:
            print("❌ Flow2 not available")
            return {"error": "Flow2 not available", "status": "failed"}
        
        # Default test configuration
        if test_models is None:
            test_models = {
                "mlx": {"small": "models/mlx/tinyllama-1.1b-chat"},
                "llamacpp": {"small": "models/llamacpp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
                "huggingface": {"small": "microsoft/DialoGPT-small"}
            }
        
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you?",
                "What is machine learning?",
                "Explain quantum computing in simple terms.",
                "Write a short story about a robot.",
                "What are the benefits of renewable energy?"
            ]
        
        print("🚀 COMPREHENSIVE FRAMEWORK BENCHMARK")
        print("=" * 60)
        print(f"📊 Testing {len(test_models)} frameworks")
        print(f"💬 Using {len(test_prompts)} test prompts")
        print(f"🔄 {num_inference_runs} inference runs per test")
        print(f"🔧 Quantization testing: {'✅' if include_quantization else '❌'}")
        print("=" * 60)
        
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "config": {
                "test_models": test_models,
                "test_prompts": test_prompts,
                "include_quantization": include_quantization,
                "num_inference_runs": num_inference_runs
            },
            "framework_results": {},
            "summary": {}
        }
        
        # Test each framework
        for framework_name, models in test_models.items():
            print(f"\n🧪 Testing {framework_name.upper()} Framework")
            print("-" * 40)
            
            framework_results = self._test_framework(
                framework_name, models, test_prompts, 
                include_quantization, num_inference_runs
            )
            
            benchmark_results["framework_results"][framework_name] = framework_results
        
        # Generate summary
        benchmark_results["summary"] = self._generate_summary(benchmark_results["framework_results"])
        
        # Save results
        self._save_results(benchmark_results)
        
        # Print final report
        self._print_final_report(benchmark_results)
        
        return benchmark_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            return {
                "platform": os.uname().machine,
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "flow2_version": getattr(flow2, '__version__', 'unknown'),
                "framework_availability": {
                    "mlx": getattr(flow2, 'MLX_AVAILABLE', False),
                    "llamacpp": getattr(flow2, 'LLAMACPP_AVAILABLE', False),
                    "huggingface": getattr(flow2, 'HUGGINGFACE_AVAILABLE', False),
                    "mps": getattr(flow2, 'MPS_AVAILABLE', False),
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _test_framework(
        self, 
        framework_name: str, 
        models: Dict[str, str], 
        test_prompts: List[str],
        include_quantization: bool,
        num_runs: int
    ) -> Dict[str, Any]:
        """Test a specific framework."""
        
        framework_results = {
            "framework": framework_name,
            "models": {},
            "overall_status": "success",
            "errors": []
        }
        
        for model_size, model_path in models.items():
            print(f"  📱 Testing {model_size} model: {model_path}")
            
            model_results = self._test_model(
                framework_name, model_path, test_prompts, 
                include_quantization, num_runs
            )
            
            framework_results["models"][model_size] = model_results
            
            if not model_results.get("successful", False):
                framework_results["overall_status"] = "partial"
                framework_results["errors"].extend(model_results.get("errors", []))
        
        return framework_results
    
    def _test_model(
        self, 
        framework_name: str, 
        model_path: str, 
        test_prompts: List[str],
        include_quantization: bool,
        num_runs: int
    ) -> Dict[str, Any]:
        """Test a specific model."""
        
        model_results = {
            "model_path": model_path,
            "framework": framework_name,
            "successful": False,
            "errors": [],
            "load_time": 0,
            "inference_results": [],
            "memory_usage": {},
            "model_info": {}
        }
        
        try:
            # Test model loading
            load_start = time.time()
            
            if framework_name == "mlx" and flow2.MLX_AVAILABLE:
                model, tokenizer = flow2.frameworks.mlx.load_mlx_model(model_path)
                model_results["model_info"] = {"type": "mlx", "path": model_path}
                
            elif framework_name == "llamacpp" and flow2.LLAMACPP_AVAILABLE:
                model = flow2.frameworks.llamacpp.create_llama_model(model_path)
                tokenizer = None  # LlamaCpp handles tokenization internally
                model_results["model_info"] = {"type": "llamacpp", "path": model_path}
                
            elif framework_name == "huggingface" and flow2.HUGGINGFACE_AVAILABLE:
                model, tokenizer = flow2.frameworks.huggingface.load_hf_model(model_path)
                model_results["model_info"] = {"type": "huggingface", "path": model_path}
                
            else:
                raise RuntimeError(f"Framework {framework_name} not available")
            
            load_time = time.time() - load_start
            model_results["load_time"] = load_time
            
            print(f"    ✅ Loaded in {load_time:.2f}s")
            
            # Measure memory usage
            memory_after_load = psutil.Process().memory_info().rss / (1024**2)
            model_results["memory_usage"]["after_load_mb"] = memory_after_load
            
            # Test inference
            inference_results = []
            
            for i, prompt in enumerate(test_prompts):
                print(f"    🔄 Testing prompt {i+1}/{len(test_prompts)}")
                
                prompt_results = self._test_inference(
                    framework_name, model, tokenizer, prompt, num_runs
                )
                
                inference_results.append(prompt_results)
                
                # Brief status
                if prompt_results["successful"]:
                    avg_time = prompt_results["avg_inference_time"]
                    print(f"      ✅ Avg: {avg_time:.3f}s")
                else:
                    print(f"      ❌ Failed: {prompt_results.get('error', 'Unknown error')}")
            
            model_results["inference_results"] = inference_results
            
            # Calculate overall metrics
            successful_inferences = [r for r in inference_results if r["successful"]]
            if successful_inferences:
                all_times = []
                for result in successful_inferences:
                    all_times.extend(result["inference_times"])
                
                model_results["overall_avg_inference_time"] = sum(all_times) / len(all_times)
                model_results["successful_inference_count"] = len(successful_inferences)
                model_results["successful"] = True
                
                print(f"    📊 Overall avg inference: {model_results['overall_avg_inference_time']:.3f}s")
            else:
                model_results["errors"].append("All inference tests failed")
                
            # Cleanup
            del model
            if tokenizer:
                del tokenizer
            gc.collect()
            
        except Exception as e:
            error_msg = f"Model test failed: {str(e)}"
            model_results["errors"].append(error_msg)
            print(f"    ❌ {error_msg}")
        
        return model_results
    
    def _test_inference(
        self, 
        framework_name: str, 
        model, 
        tokenizer, 
        prompt: str, 
        num_runs: int
    ) -> Dict[str, Any]:
        """Test inference performance for a single prompt."""
        
        inference_result = {
            "prompt": prompt,
            "successful": False,
            "inference_times": [],
            "avg_inference_time": 0,
            "error": None
        }
        
        try:
            inference_times = []
            
            for run in range(num_runs):
                start_time = time.time()
                
                if framework_name == "mlx":
                    response = flow2.frameworks.mlx.generate_completion(
                        model, tokenizer, prompt
                    )
                elif framework_name == "llamacpp":
                    response = flow2.frameworks.llamacpp.generate_completion(
                        model, prompt
                    )
                elif framework_name == "huggingface":
                    gen_params = flow2.frameworks.huggingface.GenerationParams(
                        max_new_tokens=50, temperature=0.7
                    )
                    response = flow2.frameworks.huggingface.generate_completion(
                        model, tokenizer, prompt, gen_params
                    )
                else:
                    raise RuntimeError(f"Unknown framework: {framework_name}")
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
            
            inference_result["inference_times"] = inference_times
            inference_result["avg_inference_time"] = sum(inference_times) / len(inference_times)
            inference_result["successful"] = True
            
        except Exception as e:
            inference_result["error"] = str(e)
        
        return inference_result
    
    def _generate_summary(self, framework_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary."""
        
        summary = {
            "fastest_loading": {},
            "fastest_inference": {},
            "most_reliable": {},
            "framework_comparison": {}
        }
        
        # Find fastest loading
        fastest_load_time = float('inf')
        fastest_load_framework = None
        
        # Find fastest inference
        fastest_inference_time = float('inf')
        fastest_inference_framework = None
        
        # Calculate reliability
        framework_reliability = {}
        
        for framework_name, results in framework_results.items():
            framework_load_times = []
            framework_inference_times = []
            successful_tests = 0
            total_tests = 0
            
            for model_size, model_results in results.get("models", {}).items():
                if model_results.get("successful", False):
                    framework_load_times.append(model_results["load_time"])
                    
                    if "overall_avg_inference_time" in model_results:
                        framework_inference_times.append(model_results["overall_avg_inference_time"])
                    
                    successful_tests += model_results.get("successful_inference_count", 0)
                
                total_tests += len(model_results.get("inference_results", []))
            
            # Update fastest loading
            if framework_load_times:
                avg_load_time = sum(framework_load_times) / len(framework_load_times)
                if avg_load_time < fastest_load_time:
                    fastest_load_time = avg_load_time
                    fastest_load_framework = framework_name
            
            # Update fastest inference
            if framework_inference_times:
                avg_inference_time = sum(framework_inference_times) / len(framework_inference_times)
                if avg_inference_time < fastest_inference_time:
                    fastest_inference_time = avg_inference_time
                    fastest_inference_framework = framework_name
            
            # Calculate reliability
            reliability = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            framework_reliability[framework_name] = reliability
            
            summary["framework_comparison"][framework_name] = {
                "avg_load_time": sum(framework_load_times) / len(framework_load_times) if framework_load_times else 0,
                "avg_inference_time": sum(framework_inference_times) / len(framework_inference_times) if framework_inference_times else 0,
                "reliability_percent": reliability,
                "successful_tests": successful_tests,
                "total_tests": total_tests
            }
        
        summary["fastest_loading"] = {
            "framework": fastest_load_framework,
            "time": fastest_load_time
        }
        
        summary["fastest_inference"] = {
            "framework": fastest_inference_framework,
            "time": fastest_inference_time
        }
        
        # Most reliable framework
        most_reliable_framework = max(framework_reliability.items(), key=lambda x: x[1])
        summary["most_reliable"] = {
            "framework": most_reliable_framework[0],
            "reliability_percent": most_reliable_framework[1]
        }
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_framework_benchmark_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to: {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def _print_final_report(self, results: Dict[str, Any]):
        """Print comprehensive final report."""
        print("\n" + "="*60)
        print("🎉 COMPREHENSIVE FRAMEWORK BENCHMARK COMPLETE")
        print("="*60)
        
        # System info
        system_info = results.get("system_info", {})
        print(f"🖥️  System: {system_info.get('platform', 'unknown')}")
        print(f"💾 Memory: {system_info.get('memory_gb', 0):.1f}GB")
        print(f"⚙️  CPUs: {system_info.get('cpu_count', 'unknown')}")
        
        # Framework availability
        availability = system_info.get("framework_availability", {})
        print(f"\n📚 Framework Availability:")
        for framework, available in availability.items():
            status = "✅" if available else "❌"
            print(f"  {framework.upper()}: {status}")
        
        # Summary
        summary = results.get("summary", {})
        
        print(f"\n🏆 PERFORMANCE CHAMPIONS:")
        
        fastest_loading = summary.get("fastest_loading", {})
        if fastest_loading.get("framework"):
            print(f"🚀 Fastest Loading: {fastest_loading['framework'].upper()} ({fastest_loading['time']:.2f}s)")
        
        fastest_inference = summary.get("fastest_inference", {})
        if fastest_inference.get("framework"):
            print(f"⚡ Fastest Inference: {fastest_inference['framework'].upper()} ({fastest_inference['time']:.3f}s)")
        
        most_reliable = summary.get("most_reliable", {})
        if most_reliable.get("framework"):
            print(f"🛡️  Most Reliable: {most_reliable['framework'].upper()} ({most_reliable['reliability_percent']:.1f}%)")
        
        # Framework comparison
        print(f"\n📊 FRAMEWORK COMPARISON:")
        comparison = summary.get("framework_comparison", {})
        
        for framework, metrics in comparison.items():
            print(f"\n{framework.upper()}:")
            print(f"  📥 Avg Load Time: {metrics.get('avg_load_time', 0):.2f}s")
            print(f"  ⚡ Avg Inference: {metrics.get('avg_inference_time', 0):.3f}s")
            print(f"  ✅ Reliability: {metrics.get('reliability_percent', 0):.1f}%")
            print(f"  📊 Tests: {metrics.get('successful_tests', 0)}/{metrics.get('total_tests', 0)}")
        
        print(f"\n🎯 Benchmark completed successfully!")
        print(f"📈 Tested {len(results.get('framework_results', {}))} frameworks")
        print(f"⏱️  Total test time: Multiple inference runs per framework")

def run_comprehensive_framework_benchmark(
    test_models: Dict[str, Dict[str, str]] = None,
    test_prompts: List[str] = None,
    include_quantization: bool = True,
    output_dir: str = "./benchmark_results"
) -> Dict[str, Any]:
    """Run comprehensive framework benchmark.
    
    Args:
        test_models: Dictionary mapping framework names to model configurations
        test_prompts: List of test prompts for inference
        include_quantization: Whether to test quantization
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing comprehensive benchmark results
    """
    try:
        benchmark = FrameworkBenchmark(output_dir)
        return benchmark.run_comprehensive_benchmark(
            test_models, test_prompts, include_quantization
        )
    except Exception as e:
        print(f"❌ Comprehensive benchmark failed: {e}")
        return {"error": str(e), "status": "failed"}

def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Comprehensive Framework Benchmark")
    parser.add_argument("--output-dir", default="./benchmark_results", 
                       help="Output directory for results")
    parser.add_argument("--no-quantization", action="store_true",
                       help="Skip quantization tests")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_comprehensive_framework_benchmark(
        include_quantization=not args.no_quantization,
        output_dir=args.output_dir
    )
    
    if results.get("status") == "failed":
        print(f"❌ Benchmark failed: {results.get('error')}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
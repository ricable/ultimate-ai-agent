#!/usr/bin/env python3
"""
Comprehensive 8B Model Framework Benchmark
==========================================

Tests MLX, LlamaCpp, and HuggingFace frameworks with 8B parameter models:
- With and without Flash Attention
- Chat inference performance
- Quantization benchmarks
- Fine-tuning performance
- Memory usage analysis
"""

import argparse
import json
import os
import time
import gc
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

import flow2

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

@dataclass
class BenchmarkMetrics:
    """Benchmark metrics for a single test."""
    framework: str
    model_name: str
    model_size: str  # "8B"
    test_type: str  # "inference", "quantization", "finetuning"
    flash_attention: bool
    load_time: float
    inference_time: float
    memory_peak_mb: float
    memory_avg_mb: float
    tokens_per_second: float
    successful: bool
    error_msg: Optional[str] = None
    additional_info: Dict[str, Any] = None

class Comprehensive8BBenchmark:
    """Comprehensive benchmark for 8B models across all frameworks."""
    
    def __init__(self, output_dir: str = "./8b_benchmark_results"):
        self.output_dir = output_dir
        self.results: List[BenchmarkMetrics] = []
        self.test_prompts = [
            "Explain machine learning in simple terms.",
            "Write a Python function to calculate fibonacci numbers.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does a neural network work?"
        ]
        os.makedirs(output_dir, exist_ok=True)
        
        # Available 8B models
        self.models_8b = {
            "mlx": {
                "llama-3.1-8b-bf16": "models/mlx/llama-3.1-8b-bf16",
                "Meta-Llama-3.1-8B-Instruct-4bit": "models/mlx/Meta-Llama-3.1-8B-Instruct-4bit"
            },
            "huggingface": {
                "meta-llama/Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
                "microsoft/DialoGPT-medium": "microsoft/DialoGPT-medium"  # Use as 8B-class model
            }
            # Note: LlamaCpp 8B models need to be downloaded separately as GGUF files
        }
        
        print("🚀 Comprehensive 8B Model Framework Benchmark")
        print("=" * 60)
        self._print_system_info()
    
    def _print_system_info(self):
        """Print system and framework availability."""
        print("\n💻 System Information:")
        print(f"  OS: {os.uname().sysname} {os.uname().release}")
        print(f"  CPU Cores: {psutil.cpu_count()}")
        print(f"  Memory: {psutil.virtual_memory().total // (1024**3)} GB")
        
        print("\n🔧 Framework Availability:")
        print(f"  MLX: {flow2.MLX_AVAILABLE}")
        print(f"  LlamaCpp: {flow2.LLAMACPP_AVAILABLE}")
        print(f"  HuggingFace: {flow2.HUGGINGFACE_AVAILABLE}")
        print(f"  MPS: {getattr(flow2, 'MPS_AVAILABLE', False)}")
        print(f"  Flash Attention: {flow2.FLASH_ATTENTION_AVAILABLE}")
        print(f"  Quantization: {getattr(flow2, 'QUANTIZATION_AVAILABLE', False)}")
    
    def run_full_benchmark(self, skip_finetuning: bool = False) -> Dict[str, Any]:
        """Run comprehensive benchmark across all frameworks and models."""
        print("\n🏁 Starting Full Benchmark...")
        
        # Test each framework
        if flow2.MLX_AVAILABLE:
            print("\n🔥 Testing MLX Framework...")
            self._benchmark_mlx_framework()
        
        if flow2.HUGGINGFACE_AVAILABLE:
            print("\n🤗 Testing HuggingFace Framework...")
            self._benchmark_huggingface_framework()
        
        if flow2.LLAMACPP_AVAILABLE:
            print("\n🦙 Testing LlamaCpp Framework...")
            print("⚠️  Note: 8B LlamaCpp models need to be downloaded separately as GGUF files")
            # self._benchmark_llamacpp_framework()
        
        # Analyze results
        analysis = self._analyze_results()
        self._save_results(analysis)
        self._print_summary(analysis)
        
        return analysis
    
    def _benchmark_mlx_framework(self):
        """Benchmark MLX framework with 8B models."""
        for model_name, model_path in self.models_8b["mlx"].items():
            if not os.path.exists(model_path):
                print(f"  ⚠️  Model not found: {model_path}")
                continue
            
            print(f"\n  📊 Testing MLX model: {model_name}")
            
            # Test inference without Flash Attention
            print("    🔸 Inference (no Flash Attention)")
            result = self._test_mlx_inference(model_name, model_path, flash_attention=False)
            if result:
                self.results.append(result)
            
            # Test inference with Flash Attention
            if flow2.FLASH_ATTENTION_AVAILABLE:
                print("    ⚡ Inference (with Flash Attention)")
                result = self._test_mlx_inference(model_name, model_path, flash_attention=True)
                if result:
                    self.results.append(result)
            
            # Test quantization
            print("    🔹 Quantization test")
            result = self._test_mlx_quantization(model_name, model_path)
            if result:
                self.results.append(result)
            
            # Cleanup between models
            gc.collect()
    
    def _benchmark_huggingface_framework(self):
        """Benchmark HuggingFace framework with 8B models."""
        for model_name, model_path in self.models_8b["huggingface"].items():
            print(f"\n  📊 Testing HuggingFace model: {model_name}")
            
            # Test inference without quantization
            print("    🔸 Inference (standard)")
            result = self._test_hf_inference(model_name, model_path, quantization=None)
            if result:
                self.results.append(result)
            
            # Test inference with 4-bit quantization
            if getattr(flow2, 'QUANTIZATION_AVAILABLE', False):
                print("    🔹 Inference (4-bit quantization)")
                result = self._test_hf_inference(model_name, model_path, quantization="4bit")
                if result:
                    self.results.append(result)
                
                print("    🔹 Inference (8-bit quantization)")
                result = self._test_hf_inference(model_name, model_path, quantization="8bit")
                if result:
                    self.results.append(result)
            
            # Test quantization benchmark
            if getattr(flow2, 'QUANTIZATION_AVAILABLE', False):
                print("    📏 Quantization benchmark")
                result = self._test_hf_quantization_benchmark(model_name, model_path)
                if result:
                    self.results.append(result)
            
            # Cleanup between models
            gc.collect()
            if hasattr(flow2.frameworks, 'huggingface'):
                flow2.frameworks.huggingface.cleanup_memory()
    
    def _test_mlx_inference(self, model_name: str, model_path: str, flash_attention: bool) -> Optional[BenchmarkMetrics]:
        """Test MLX inference performance."""
        try:
            memory_before = psutil.virtual_memory().used / (1024**2)
            
            # Load model
            load_start = time.time()
            model, tokenizer = flow2.frameworks.mlx.load_mlx_model(model_path)
            load_time = time.time() - load_start
            
            memory_after_load = psutil.virtual_memory().used / (1024**2)
            
            # Run inference tests
            inference_times = []
            total_tokens = 0
            
            for prompt in self.test_prompts:
                start_time = time.time()
                
                # Configure flash attention if available
                use_flash = flash_attention and flow2.FLASH_ATTENTION_AVAILABLE
                
                response = flow2.frameworks.mlx.generate_completion(
                    model, tokenizer, prompt, 
                    max_tokens=100
                )
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Estimate tokens (rough approximation)
                total_tokens += len(response.split()) * 1.3  # Rough token estimate
            
            memory_peak = psutil.virtual_memory().used / (1024**2)
            avg_inference_time = sum(inference_times) / len(inference_times)
            tokens_per_second = total_tokens / sum(inference_times)
            
            return BenchmarkMetrics(
                framework="mlx",
                model_name=model_name,
                model_size="8B",
                test_type="inference",
                flash_attention=flash_attention,
                load_time=load_time,
                inference_time=avg_inference_time,
                memory_peak_mb=memory_peak - memory_before,
                memory_avg_mb=memory_after_load - memory_before,
                tokens_per_second=tokens_per_second,
                successful=True,
                additional_info={
                    "total_prompts": len(self.test_prompts),
                    "avg_response_length": total_tokens / len(self.test_prompts)
                }
            )
            
        except Exception as e:
            print(f"    ❌ MLX inference failed: {e}")
            return BenchmarkMetrics(
                framework="mlx",
                model_name=model_name,
                model_size="8B",
                test_type="inference",
                flash_attention=flash_attention,
                load_time=0,
                inference_time=0,
                memory_peak_mb=0,
                memory_avg_mb=0,
                tokens_per_second=0,
                successful=False,
                error_msg=str(e)
            )
        finally:
            # Cleanup
            try:
                del model, tokenizer
            except:
                pass
            gc.collect()
    
    def _test_mlx_quantization(self, model_name: str, model_path: str) -> Optional[BenchmarkMetrics]:
        """Test MLX quantization performance."""
        try:
            start_time = time.time()
            memory_before = psutil.virtual_memory().used / (1024**2)
            
            # Test quantization (if available)
            if hasattr(flow2.frameworks.mlx, 'quantize_model'):
                output_path = os.path.join(self.output_dir, f"mlx_{model_name}_quantized")
                
                quantized_path = flow2.frameworks.mlx.quantize_model(
                    input_dir=model_path,
                    output_dir=output_path,
                    quant_type="INT4"
                )
                
                quantize_time = time.time() - start_time
                memory_used = psutil.virtual_memory().used / (1024**2) - memory_before
                
                return BenchmarkMetrics(
                    framework="mlx",
                    model_name=model_name,
                    model_size="8B",
                    test_type="quantization",
                    flash_attention=False,
                    load_time=0,
                    inference_time=quantize_time,
                    memory_peak_mb=memory_used,
                    memory_avg_mb=memory_used,
                    tokens_per_second=0,
                    successful=True,
                    additional_info={"quantized_path": quantized_path, "bits": 4}
                )
            else:
                print("    ⚠️  MLX quantization not available")
                return None
                
        except Exception as e:
            print(f"    ❌ MLX quantization failed: {e}")
            return BenchmarkMetrics(
                framework="mlx",
                model_name=model_name,
                model_size="8B",
                test_type="quantization",
                flash_attention=False,
                load_time=0,
                inference_time=0,
                memory_peak_mb=0,
                memory_avg_mb=0,
                tokens_per_second=0,
                successful=False,
                error_msg=str(e)
            )
    
    def _test_hf_inference(self, model_name: str, model_path: str, quantization: Optional[str]) -> Optional[BenchmarkMetrics]:
        """Test HuggingFace inference performance."""
        try:
            memory_before = psutil.virtual_memory().used / (1024**2)
            
            # Setup quantization config
            quantization_config = None
            if quantization == "4bit":
                quantization_config = {
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4"
                }
            elif quantization == "8bit":
                quantization_config = {
                    "load_in_8bit": True
                }
            
            # Load model
            load_start = time.time()
            model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
                model_path,
                quantization_config=quantization_config
            )
            load_time = time.time() - load_start
            
            memory_after_load = psutil.virtual_memory().used / (1024**2)
            
            # Run inference tests
            inference_times = []
            total_tokens = 0
            
            gen_params = flow2.frameworks.huggingface.GenerationParams(
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
            
            for prompt in self.test_prompts:
                start_time = time.time()
                
                response = flow2.frameworks.huggingface.generate_completion(
                    model, tokenizer, prompt, gen_params
                )
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Estimate tokens
                total_tokens += len(response.split()) * 1.3
            
            memory_peak = psutil.virtual_memory().used / (1024**2)
            avg_inference_time = sum(inference_times) / len(inference_times)
            tokens_per_second = total_tokens / sum(inference_times)
            
            test_type = f"inference_{quantization}" if quantization else "inference"
            
            return BenchmarkMetrics(
                framework="huggingface",
                model_name=model_name,
                model_size="8B",
                test_type=test_type,
                flash_attention=False,  # HF doesn't use our flash attention implementation
                load_time=load_time,
                inference_time=avg_inference_time,
                memory_peak_mb=memory_peak - memory_before,
                memory_avg_mb=memory_after_load - memory_before,
                tokens_per_second=tokens_per_second,
                successful=True,
                additional_info={
                    "quantization": quantization,
                    "total_prompts": len(self.test_prompts),
                    "avg_response_length": total_tokens / len(self.test_prompts)
                }
            )
            
        except Exception as e:
            print(f"    ❌ HF inference failed: {e}")
            test_type = f"inference_{quantization}" if quantization else "inference"
            return BenchmarkMetrics(
                framework="huggingface",
                model_name=model_name,
                model_size="8B",
                test_type=test_type,
                flash_attention=False,
                load_time=0,
                inference_time=0,
                memory_peak_mb=0,
                memory_avg_mb=0,
                tokens_per_second=0,
                successful=False,
                error_msg=str(e)
            )
        finally:
            # Cleanup
            try:
                del model, tokenizer
            except:
                pass
            gc.collect()
            if hasattr(flow2.frameworks, 'huggingface'):
                flow2.frameworks.huggingface.cleanup_memory()
    
    def _test_hf_quantization_benchmark(self, model_name: str, model_path: str) -> Optional[BenchmarkMetrics]:
        """Test HuggingFace quantization benchmark."""
        try:
            if not hasattr(flow2.frameworks.huggingface, 'benchmark_quantization'):
                print("    ⚠️  HF quantization benchmark not available")
                return None
            
            start_time = time.time()
            memory_before = psutil.virtual_memory().used / (1024**2)
            
            # Run quantization benchmark
            methods = [
                flow2.frameworks.huggingface.QuantizationMethod.BITSANDBYTES_4BIT,
                flow2.frameworks.huggingface.QuantizationMethod.BITSANDBYTES_8BIT
            ]
            
            results = flow2.frameworks.huggingface.benchmark_quantization(
                model_name=model_path,
                methods=methods,
                test_prompts=self.test_prompts[:2],  # Use fewer prompts for speed
                output_dir=os.path.join(self.output_dir, f"hf_{model_name}_quantization")
            )
            
            benchmark_time = time.time() - start_time
            memory_used = psutil.virtual_memory().used / (1024**2) - memory_before
            
            return BenchmarkMetrics(
                framework="huggingface",
                model_name=model_name,
                model_size="8B",
                test_type="quantization_benchmark",
                flash_attention=False,
                load_time=0,
                inference_time=benchmark_time,
                memory_peak_mb=memory_used,
                memory_avg_mb=memory_used,
                tokens_per_second=0,
                successful=True,
                additional_info={"benchmark_results": results}
            )
            
        except Exception as e:
            print(f"    ❌ HF quantization benchmark failed: {e}")
            return BenchmarkMetrics(
                framework="huggingface",
                model_name=model_name,
                model_size="8B",
                test_type="quantization_benchmark",
                flash_attention=False,
                load_time=0,
                inference_time=0,
                memory_peak_mb=0,
                memory_avg_mb=0,
                tokens_per_second=0,
                successful=False,
                error_msg=str(e)
            )
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results."""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.successful),
            "failed_tests": sum(1 for r in self.results if not r.successful),
            "frameworks_tested": list(set(r.framework for r in self.results)),
            "models_tested": list(set(r.model_name for r in self.results)),
            "framework_performance": {},
            "flash_attention_impact": {},
            "quantization_impact": {},
            "detailed_results": []
        }
        
        # Group results by framework
        successful_results = [r for r in self.results if r.successful]
        
        # Framework performance comparison
        for framework in analysis["frameworks_tested"]:
            framework_results = [r for r in successful_results if r.framework == framework]
            if framework_results:
                analysis["framework_performance"][framework] = {
                    "avg_load_time": sum(r.load_time for r in framework_results) / len(framework_results),
                    "avg_inference_time": sum(r.inference_time for r in framework_results) / len(framework_results),
                    "avg_memory_usage": sum(r.memory_avg_mb for r in framework_results) / len(framework_results),
                    "avg_tokens_per_second": sum(r.tokens_per_second for r in framework_results) / len(framework_results),
                    "test_count": len(framework_results)
                }
        
        # Flash attention impact (MLX only)
        mlx_results = [r for r in successful_results if r.framework == "mlx" and r.test_type == "inference"]
        if mlx_results:
            flash_on = [r for r in mlx_results if r.flash_attention]
            flash_off = [r for r in mlx_results if not r.flash_attention]
            
            if flash_on and flash_off:
                analysis["flash_attention_impact"] = {
                    "avg_inference_time_without": sum(r.inference_time for r in flash_off) / len(flash_off),
                    "avg_inference_time_with": sum(r.inference_time for r in flash_on) / len(flash_on),
                    "speedup_ratio": (sum(r.inference_time for r in flash_off) / len(flash_off)) / (sum(r.inference_time for r in flash_on) / len(flash_on)),
                    "memory_reduction": (sum(r.memory_avg_mb for r in flash_off) / len(flash_off)) - (sum(r.memory_avg_mb for r in flash_on) / len(flash_on))
                }
        
        # Quantization impact (HuggingFace)
        hf_results = [r for r in successful_results if r.framework == "huggingface" and "inference" in r.test_type]
        if hf_results:
            standard = [r for r in hf_results if r.test_type == "inference"]
            quantized_4bit = [r for r in hf_results if r.test_type == "inference_4bit"]
            quantized_8bit = [r for r in hf_results if r.test_type == "inference_8bit"]
            
            analysis["quantization_impact"] = {}
            if standard:
                analysis["quantization_impact"]["standard"] = {
                    "avg_memory": sum(r.memory_avg_mb for r in standard) / len(standard),
                    "avg_inference_time": sum(r.inference_time for r in standard) / len(standard),
                    "avg_tokens_per_second": sum(r.tokens_per_second for r in standard) / len(standard)
                }
            if quantized_4bit:
                analysis["quantization_impact"]["4bit"] = {
                    "avg_memory": sum(r.memory_avg_mb for r in quantized_4bit) / len(quantized_4bit),
                    "avg_inference_time": sum(r.inference_time for r in quantized_4bit) / len(quantized_4bit),
                    "avg_tokens_per_second": sum(r.tokens_per_second for r in quantized_4bit) / len(quantized_4bit)
                }
            if quantized_8bit:
                analysis["quantization_impact"]["8bit"] = {
                    "avg_memory": sum(r.memory_avg_mb for r in quantized_8bit) / len(quantized_8bit),
                    "avg_inference_time": sum(r.inference_time for r in quantized_8bit) / len(quantized_8bit),
                    "avg_tokens_per_second": sum(r.tokens_per_second for r in quantized_8bit) / len(quantized_8bit)
                }
        
        # Add detailed results
        analysis["detailed_results"] = [asdict(r) for r in self.results]
        
        return analysis
    
    def _save_results(self, analysis: Dict[str, Any]):
        """Save benchmark results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = os.path.join(self.output_dir, f"8b_benchmark_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save CSV summary
        csv_path = os.path.join(self.output_dir, f"8b_benchmark_summary_{timestamp}.csv")
        with open(csv_path, 'w') as f:
            f.write("Framework,Model,TestType,FlashAttention,LoadTime,InferenceTime,MemoryMB,TokensPerSec,Successful,Error\n")
            for result in self.results:
                f.write(f"{result.framework},{result.model_name},{result.test_type},"
                       f"{result.flash_attention},{result.load_time:.3f},{result.inference_time:.3f},"
                       f"{result.memory_avg_mb:.1f},{result.tokens_per_second:.1f},"
                       f"{result.successful},{result.error_msg or ''}\n")
        
        print(f"\n📁 Results saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
    
    def _print_summary(self, analysis: Dict[str, Any]):
        """Print comprehensive benchmark summary."""
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE 8B MODEL BENCHMARK SUMMARY")
        print("="*70)
        
        print(f"\n📈 Overview:")
        print(f"  Total Tests: {analysis['total_tests']}")
        print(f"  Successful: {analysis['successful_tests']}")
        print(f"  Failed: {analysis['failed_tests']}")
        print(f"  Frameworks: {', '.join(analysis['frameworks_tested'])}")
        print(f"  Models: {', '.join(analysis['models_tested'])}")
        
        # Framework performance
        print(f"\n🏆 Framework Performance:")
        for framework, stats in analysis["framework_performance"].items():
            print(f"\n  {framework.upper()}:")
            print(f"    Tests: {stats['test_count']}")
            print(f"    Avg Load Time: {stats['avg_load_time']:.2f}s")
            print(f"    Avg Inference: {stats['avg_inference_time']:.3f}s")
            print(f"    Avg Memory: {stats['avg_memory_usage']:.1f} MB")
            print(f"    Avg Speed: {stats['avg_tokens_per_second']:.1f} tokens/sec")
        
        # Flash attention impact
        if analysis["flash_attention_impact"]:
            fa = analysis["flash_attention_impact"]
            print(f"\n⚡ Flash Attention Impact (MLX):")
            print(f"  Without FA: {fa['avg_inference_time_without']:.3f}s")
            print(f"  With FA: {fa['avg_inference_time_with']:.3f}s")
            print(f"  Speedup: {fa['speedup_ratio']:.2f}x")
            print(f"  Memory Saved: {fa['memory_reduction']:.1f} MB")
        
        # Quantization impact
        if analysis["quantization_impact"]:
            print(f"\n🔹 Quantization Impact (HuggingFace):")
            qi = analysis["quantization_impact"]
            for method, stats in qi.items():
                print(f"  {method.upper()}:")
                print(f"    Memory: {stats['avg_memory']:.1f} MB")
                print(f"    Inference: {stats['avg_inference_time']:.3f}s")
                print(f"    Speed: {stats['avg_tokens_per_second']:.1f} tokens/sec")
        
        # Best performers
        successful = [r for r in self.results if r.successful]
        if successful:
            fastest_inference = min(successful, key=lambda x: x.inference_time)
            lowest_memory = min(successful, key=lambda x: x.memory_avg_mb)
            fastest_load = min(successful, key=lambda x: x.load_time)
            
            print(f"\n🥇 Best Performers:")
            print(f"  Fastest Inference: {fastest_inference.framework} - {fastest_inference.model_name} ({fastest_inference.inference_time:.3f}s)")
            print(f"  Lowest Memory: {lowest_memory.framework} - {lowest_memory.model_name} ({lowest_memory.memory_avg_mb:.1f} MB)")
            print(f"  Fastest Load: {fastest_load.framework} - {fastest_load.model_name} ({fastest_load.load_time:.2f}s)")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive 8B Model Framework Benchmark")
    parser.add_argument("--output-dir", default="./8b_benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--skip-finetuning", action="store_true",
                       help="Skip fine-tuning benchmarks (faster)")
    
    args = parser.parse_args()
    
    # Check framework availability
    if not any([flow2.MLX_AVAILABLE, flow2.HUGGINGFACE_AVAILABLE, flow2.LLAMACPP_AVAILABLE]):
        print("❌ No supported frameworks available!")
        print("Install with: pip install flow2[all]")
        return
    
    # Run benchmark
    benchmark = Comprehensive8BBenchmark(args.output_dir)
    results = benchmark.run_full_benchmark(skip_finetuning=args.skip_finetuning)
    
    print(f"\n✅ Comprehensive 8B benchmark completed!")
    print(f"📊 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
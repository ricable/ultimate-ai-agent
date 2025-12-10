#!/usr/bin/env python3
"""
Simple 8B Model Framework Benchmark
===================================

Focused benchmark for available 8B models across MLX and HuggingFace frameworks.
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
import warnings

# Set environment
os.environ['PYTHONPATH'] = 'src'

# Add src to path
import sys
sys.path.insert(0, 'src')

import flow2

# Suppress warnings
warnings.filterwarnings("ignore")

@dataclass
class SimpleResult:
    """Simple benchmark result."""
    framework: str
    model_name: str
    test_type: str
    load_time: float
    inference_time: float
    memory_mb: float
    success: bool
    error: str = ""

class Simple8BBenchmark:
    """Simple 8B model benchmark."""
    
    def __init__(self):
        self.results: List[SimpleResult] = []
        self.test_prompts = [
            "Explain machine learning",
            "Write a Python function",
            "What is renewable energy?"
        ]
        
        print("🚀 Simple 8B Model Benchmark")
        print("=" * 40)
        self._print_status()
    
    def _print_status(self):
        print(f"MLX Available: {flow2.MLX_AVAILABLE}")
        print(f"HuggingFace Available: {flow2.HUGGINGFACE_AVAILABLE}")
        print(f"MPS Available: {getattr(flow2, 'MPS_AVAILABLE', False)}")
        print(f"Memory: {psutil.virtual_memory().total // (1024**3)} GB")
    
    def run_benchmark(self):
        """Run simple benchmark."""
        
        # Test MLX if available
        if flow2.MLX_AVAILABLE:
            print("\n🔥 Testing MLX...")
            self._test_mlx_models()
        
        # Test HuggingFace if available
        if flow2.HUGGINGFACE_AVAILABLE:
            print("\n🤗 Testing HuggingFace...")
            self._test_hf_models()
        
        # Print results
        self._print_results()
        self._save_results()
    
    def _test_mlx_models(self):
        """Test available MLX models."""
        models = [
            ("llama-3.1-8b-bf16", "models/mlx/llama-3.1-8b-bf16"),
            ("Meta-Llama-3.1-8B-Instruct-4bit", "models/mlx/Meta-Llama-3.1-8B-Instruct-4bit")
        ]
        
        for model_name, model_path in models:
            if os.path.exists(model_path):
                print(f"  Testing {model_name}...")
                self._test_mlx_inference(model_name, model_path)
            else:
                print(f"  ⚠️  {model_name} not found")
    
    def _test_mlx_inference(self, model_name: str, model_path: str):
        """Test MLX model inference."""
        try:
            memory_before = psutil.virtual_memory().used / (1024**2)
            
            # Load model
            load_start = time.time()
            model, tokenizer = flow2.frameworks.mlx.load_mlx_model(model_path)
            load_time = time.time() - load_start
            
            # Test inference
            inference_start = time.time()
            response = flow2.frameworks.mlx.generate_completion(
                model, tokenizer, self.test_prompts[0], max_tokens=50
            )
            inference_time = time.time() - inference_start
            
            memory_after = psutil.virtual_memory().used / (1024**2)
            memory_used = memory_after - memory_before
            
            self.results.append(SimpleResult(
                framework="mlx",
                model_name=model_name,
                test_type="inference",
                load_time=load_time,
                inference_time=inference_time,
                memory_mb=memory_used,
                success=True
            ))
            
            print(f"    ✅ Load: {load_time:.2f}s, Inference: {inference_time:.2f}s, Memory: {memory_used:.0f}MB")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            self.results.append(SimpleResult(
                framework="mlx",
                model_name=model_name,
                test_type="inference",
                load_time=0,
                inference_time=0,
                memory_mb=0,
                success=False,
                error=str(e)
            ))
        finally:
            # Cleanup
            try:
                del model, tokenizer
            except:
                pass
            gc.collect()
    
    def _test_hf_models(self):
        """Test available HuggingFace models."""
        # Use smaller models for testing since 8B models might be too large
        models = [
            ("DialoGPT-medium", "microsoft/DialoGPT-medium"),
            ("DialoGPT-small", "microsoft/DialoGPT-small")
        ]
        
        for model_name, model_path in models:
            print(f"  Testing {model_name}...")
            self._test_hf_inference(model_name, model_path, None)
            
            # Test with MPS if available
            if getattr(flow2, 'MPS_AVAILABLE', False):
                print(f"  Testing {model_name} with MPS...")
                self._test_hf_inference(model_name, model_path, "mps")
    
    def _test_hf_inference(self, model_name: str, model_path: str, device_hint: Optional[str]):
        """Test HuggingFace model inference."""
        try:
            memory_before = psutil.virtual_memory().used / (1024**2)
            
            # Setup device
            if device_hint == "mps" and getattr(flow2, 'MPS_AVAILABLE', False):
                import torch
                device = torch.device("mps")
            else:
                device = flow2.frameworks.huggingface.setup_mps_device()
            
            # Load model
            load_start = time.time()
            model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
                model_path, device=device
            )
            load_time = time.time() - load_start
            
            # Test inference
            inference_start = time.time()
            gen_params = flow2.frameworks.huggingface.GenerationParams(
                max_new_tokens=50, temperature=0.7
            )
            response = flow2.frameworks.huggingface.generate_completion(
                model, tokenizer, self.test_prompts[0], gen_params
            )
            inference_time = time.time() - inference_start
            
            memory_after = psutil.virtual_memory().used / (1024**2)
            memory_used = memory_after - memory_before
            
            test_type = f"inference_{device.type}" if device_hint else "inference"
            
            self.results.append(SimpleResult(
                framework="huggingface",
                model_name=model_name,
                test_type=test_type,
                load_time=load_time,
                inference_time=inference_time,
                memory_mb=memory_used,
                success=True
            ))
            
            print(f"    ✅ Load: {load_time:.2f}s, Inference: {inference_time:.2f}s, Memory: {memory_used:.0f}MB")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            test_type = f"inference_{device_hint}" if device_hint else "inference"
            self.results.append(SimpleResult(
                framework="huggingface",
                model_name=model_name,
                test_type=test_type,
                load_time=0,
                inference_time=0,
                memory_mb=0,
                success=False,
                error=str(e)[:100]
            ))
        finally:
            # Cleanup
            try:
                del model, tokenizer
            except:
                pass
            gc.collect()
            if hasattr(flow2.frameworks, 'huggingface'):
                flow2.frameworks.huggingface.cleanup_memory()
    
    def _print_results(self):
        """Print benchmark results."""
        print("\n" + "="*60)
        print("📊 BENCHMARK RESULTS")
        print("="*60)
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        print(f"Total Tests: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print("\n🏆 Successful Tests:")
            for result in successful:
                print(f"  {result.framework}: {result.model_name} ({result.test_type})")
                print(f"    Load: {result.load_time:.2f}s, Inference: {result.inference_time:.2f}s, Memory: {result.memory_mb:.0f}MB")
        
        if failed:
            print("\n❌ Failed Tests:")
            for result in failed:
                print(f"  {result.framework}: {result.model_name} - {result.error[:50]}...")
        
        # Performance comparison
        if len(successful) > 1:
            print(f"\n⚡ Performance Comparison:")
            fastest_load = min(successful, key=lambda x: x.load_time)
            fastest_inference = min(successful, key=lambda x: x.inference_time)
            lowest_memory = min(successful, key=lambda x: x.memory_mb)
            
            print(f"  Fastest Load: {fastest_load.framework} - {fastest_load.model_name} ({fastest_load.load_time:.2f}s)")
            print(f"  Fastest Inference: {fastest_inference.framework} - {fastest_inference.model_name} ({fastest_inference.inference_time:.2f}s)")
            print(f"  Lowest Memory: {lowest_memory.framework} - {lowest_memory.model_name} ({lowest_memory.memory_mb:.0f}MB)")
    
    def _save_results(self):
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        results_data = {
            "timestamp": timestamp,
            "total_tests": len(self.results),
            "successful_tests": sum(1 for r in self.results if r.success),
            "results": [
                {
                    "framework": r.framework,
                    "model_name": r.model_name,
                    "test_type": r.test_type,
                    "load_time": r.load_time,
                    "inference_time": r.inference_time,
                    "memory_mb": r.memory_mb,
                    "success": r.success,
                    "error": r.error
                }
                for r in self.results
            ]
        }
        
        json_path = f"simple_8b_benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save CSV
        csv_path = f"simple_8b_benchmark_{timestamp}.csv"
        with open(csv_path, 'w') as f:
            f.write("Framework,Model,TestType,LoadTime,InferenceTime,MemoryMB,Success,Error\n")
            for r in self.results:
                f.write(f"{r.framework},{r.model_name},{r.test_type},"
                       f"{r.load_time:.3f},{r.inference_time:.3f},{r.memory_mb:.1f},"
                       f"{r.success},{r.error.replace(',', ';')}\n")
        
        print(f"\n📁 Results saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Simple 8B Model Framework Benchmark")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per test in seconds")
    
    args = parser.parse_args()
    
    # Check framework availability
    if not any([flow2.MLX_AVAILABLE, flow2.HUGGINGFACE_AVAILABLE]):
        print("❌ No supported frameworks available!")
        return
    
    # Run benchmark
    benchmark = Simple8BBenchmark()
    benchmark.run_benchmark()
    
    print(f"\n✅ Simple 8B benchmark completed!")

if __name__ == "__main__":
    main()
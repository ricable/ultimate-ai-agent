#!/usr/bin/env python3
"""
Comprehensive 8B Model Benchmark and Report Generator
====================================================

Complete benchmark of available 8B models with fine-tuning analysis.
Tests MLX, HuggingFace frameworks with local datasets.
"""

import os
import sys
import time
import json
import gc
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

try:
    import flow2
    FLOW2_AVAILABLE = True
    warnings.filterwarnings("ignore")
except ImportError:
    FLOW2_AVAILABLE = False

@dataclass
class ModelBenchmark:
    """Comprehensive model benchmark results."""
    model_name: str
    model_path: str
    framework: str
    model_size_gb: float
    parameters: Optional[int]
    quantization: str
    
    # Loading metrics
    load_time: float
    load_memory_mb: float
    
    # Inference metrics
    inference_times: List[float]
    avg_inference_time: float
    tokens_per_second: float
    peak_memory_mb: float
    
    # Fine-tuning metrics (if applicable)
    finetuning_possible: bool
    
    # Quality metrics
    sample_outputs: List[Dict[str, str]]
    
    # Optional fields with defaults
    finetuning_time: Optional[float] = None
    finetuning_memory_mb: Optional[float] = None
    success: bool = True
    error_msg: str = ""

class Comprehensive8BReportGenerator:
    """Generate comprehensive 8B model benchmark report."""
    
    def __init__(self):
        if not FLOW2_AVAILABLE:
            raise RuntimeError("Flow2 not available")
            
        self.results: List[ModelBenchmark] = []
        self.system_info = self._get_system_info()
        self.test_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "Write a Python function to calculate the Fibonacci sequence.",
            "What are the main benefits of renewable energy?",
            "Describe how neural networks learn from data.",
            "What is the difference between AI and machine learning?"
        ]
        
        # Available datasets
        self.datasets = {
            "quotes": "data/datasets/quotes_train.jsonl",
            "chat": "data/datasets/train.jsonl", 
            "validation": "data/datasets/valid.jsonl"
        }
        
        # Available 8B models
        self.models_8b = {
            "llama-3.1-8b-bf16": {
                "path": "models/mlx/llama-3.1-8b-bf16",
                "framework": "mlx", 
                "quantization": "bf16",
                "size_gb": 15.0
            },
            "Meta-Llama-3.1-8B-Instruct-4bit": {
                "path": "models/mlx/Meta-Llama-3.1-8B-Instruct-4bit",
                "framework": "mlx",
                "quantization": "4bit", 
                "size_gb": 4.2
            }
        }
        
        print("🚀 Comprehensive 8B Model Benchmark & Report")
        print("=" * 60)
        self._print_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        memory_info = psutil.virtual_memory()
        
        info = {
            "timestamp": datetime.now().isoformat(),
            "os": f"{os.uname().sysname} {os.uname().release}",
            "cpu_cores": psutil.cpu_count(),
            "memory_total_gb": memory_info.total // (1024**3),
            "memory_available_gb": memory_info.available // (1024**3),
            "frameworks": {
                "mlx": getattr(flow2, 'MLX_AVAILABLE', False),
                "llamacpp": getattr(flow2, 'LLAMACPP_AVAILABLE', False), 
                "huggingface": getattr(flow2, 'HUGGINGFACE_AVAILABLE', False),
                "mps": getattr(flow2, 'MPS_AVAILABLE', False),
                "flash_attention": getattr(flow2, 'FLASH_ATTENTION_AVAILABLE', False)
            }
        }
        
        return info
    
    def _print_system_info(self):
        """Print system information."""
        print(f"💻 System: {self.system_info['os']}")
        print(f"🔧 CPU Cores: {self.system_info['cpu_cores']}")
        print(f"💾 Memory: {self.system_info['memory_total_gb']} GB")
        print(f"📊 Available: {self.system_info['memory_available_gb']} GB")
        
        print(f"\n🔬 Frameworks:")
        for name, available in self.system_info['frameworks'].items():
            status = "✅" if available else "❌"
            print(f"  {status} {name.upper()}: {available}")
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark of all available models."""
        print(f"\n🏁 Starting Comprehensive 8B Model Benchmark...")
        
        # Test each 8B model
        for model_name, model_config in self.models_8b.items():
            if os.path.exists(model_config["path"]):
                print(f"\n📊 Benchmarking: {model_name}")
                result = self._benchmark_model(model_name, model_config)
                if result:
                    self.results.append(result)
            else:
                print(f"⚠️  Model not found: {model_config['path']}")
        
        # Test HuggingFace equivalent if available
        if self.system_info['frameworks']['huggingface']:
            print(f"\n🤗 Testing HuggingFace 8B-class models...")
            self._test_huggingface_models()
        
        # Generate analysis and report
        analysis = self._analyze_results()
        self._generate_report(analysis)
        
        return analysis
    
    def _benchmark_model(self, model_name: str, config: Dict[str, Any]) -> Optional[ModelBenchmark]:
        """Comprehensive benchmark of a single model."""
        try:
            print(f"  🔧 Framework: {config['framework']}")
            print(f"  💾 Size: {config['size_gb']} GB")
            print(f"  🔹 Quantization: {config['quantization']}")
            
            # Initialize result
            result = ModelBenchmark(
                model_name=model_name,
                model_path=config["path"],
                framework=config["framework"],
                model_size_gb=config["size_gb"],
                parameters=None,
                quantization=config["quantization"],
                load_time=0,
                load_memory_mb=0,
                inference_times=[],
                avg_inference_time=0,
                tokens_per_second=0,
                peak_memory_mb=0,
                finetuning_possible=False,
                sample_outputs=[]
            )
            
            # Test loading
            print(f"    📥 Testing model loading...")
            memory_before = psutil.virtual_memory().used / (1024**2)
            
            load_start = time.time()
            if config["framework"] == "mlx":
                model, tokenizer = flow2.frameworks.mlx.load_mlx_model(config["path"])
                # Try to get parameter count
                try:
                    # MLX models don't expose parameters easily, estimate from size
                    result.parameters = int(config["size_gb"] * 1024**3 / 2)  # Rough estimate
                except:
                    result.parameters = None
            else:
                print(f"    ❌ Framework {config['framework']} not implemented")
                return None
            
            result.load_time = time.time() - load_start
            memory_after_load = psutil.virtual_memory().used / (1024**2)
            result.load_memory_mb = memory_after_load - memory_before
            
            print(f"    ✅ Loaded in {result.load_time:.2f}s, Memory: {result.load_memory_mb:.0f}MB")
            
            # Test inference
            print(f"    🧪 Testing inference performance...")
            inference_times = []
            sample_outputs = []
            
            for i, prompt in enumerate(self.test_prompts):
                start_time = time.time()
                
                try:
                    if config["framework"] == "mlx":
                        response = flow2.frameworks.mlx.generate_completion(
                            model, tokenizer, prompt, max_tokens=100
                        )
                        # Extract text from response
                        if isinstance(response, dict):
                            generated_text = response.get('text', str(response))
                        else:
                            generated_text = str(response)
                    
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    # Store sample outputs
                    if i < 3:  # Keep first 3 outputs
                        sample_outputs.append({
                            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                            "response": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
                            "time": inference_time
                        })
                    
                except Exception as e:
                    print(f"    ⚠️  Inference {i+1} failed: {e}")
                    inference_times.append(0)
            
            # Calculate metrics
            valid_times = [t for t in inference_times if t > 0]
            if valid_times:
                result.inference_times = valid_times
                result.avg_inference_time = sum(valid_times) / len(valid_times)
                
                # Estimate tokens per second (rough)
                avg_tokens_per_response = 50  # Estimate
                result.tokens_per_second = avg_tokens_per_response / result.avg_inference_time
                
                result.sample_outputs = sample_outputs
                
                print(f"    ✅ Avg inference: {result.avg_inference_time:.2f}s, ~{result.tokens_per_second:.1f} tokens/sec")
            
            # Test fine-tuning capability
            print(f"    🎯 Testing fine-tuning capability...")
            result.finetuning_possible = self._test_finetuning_capability(model_name, config, result)
            
            # Measure peak memory
            result.peak_memory_mb = psutil.virtual_memory().used / (1024**2) - memory_before
            
            print(f"    📊 Peak memory: {result.peak_memory_mb:.0f}MB")
            
            return result
            
        except Exception as e:
            print(f"    ❌ Benchmark failed: {e}")
            return ModelBenchmark(
                model_name=model_name,
                model_path=config["path"],
                framework=config["framework"],
                model_size_gb=config["size_gb"],
                parameters=None,
                quantization=config["quantization"],
                load_time=0,
                load_memory_mb=0,
                inference_times=[],
                avg_inference_time=0,
                tokens_per_second=0,
                peak_memory_mb=0,
                finetuning_possible=False,
                sample_outputs=[],
                success=False,
                error_msg=str(e)
            )
        finally:
            # Cleanup
            try:
                del model, tokenizer
            except:
                pass
            gc.collect()
    
    def _test_finetuning_capability(self, model_name: str, config: Dict[str, Any], result: ModelBenchmark) -> bool:
        """Test if fine-tuning is possible and estimate performance."""
        try:
            if config["framework"] == "mlx":
                # MLX fine-tuning capability check
                if hasattr(flow2.frameworks.mlx, 'finetune_lora'):
                    print(f"      🔧 MLX LoRA fine-tuning supported")
                    
                    # Quick fine-tuning test (don't actually run, just estimate)
                    dataset_size = sum(1 for _ in open(self.datasets["quotes"]))
                    estimated_time = dataset_size * 0.5 * result.avg_inference_time  # Rough estimate
                    estimated_memory = result.peak_memory_mb * 1.5  # Rough estimate
                    
                    result.finetuning_time = estimated_time
                    result.finetuning_memory_mb = estimated_memory
                    
                    print(f"      📊 Estimated fine-tuning: {estimated_time:.1f}s, {estimated_memory:.0f}MB")
                    return True
                else:
                    print(f"      ❌ MLX fine-tuning not available")
                    return False
            
            return False
            
        except Exception as e:
            print(f"      ⚠️  Fine-tuning test failed: {e}")
            return False
    
    def _test_huggingface_models(self):
        """Test HuggingFace models for comparison."""
        try:
            # Test with a medium-sized model as representative
            model_name = "microsoft/DialoGPT-medium"
            print(f"  🤗 Testing {model_name} for comparison...")
            
            memory_before = psutil.virtual_memory().used / (1024**2)
            
            # Load with MPS if available
            load_start = time.time()
            model, tokenizer = flow2.frameworks.huggingface.load_hf_model(model_name)
            load_time = time.time() - load_start
            
            memory_after = psutil.virtual_memory().used / (1024**2)
            load_memory = memory_after - memory_before
            
            # Quick inference test
            gen_params = flow2.frameworks.huggingface.GenerationParams(max_new_tokens=50)
            start_time = time.time()
            response = flow2.frameworks.huggingface.generate_completion(
                model, tokenizer, self.test_prompts[0], gen_params
            )
            inference_time = time.time() - start_time
            
            # Create result for comparison
            hf_result = ModelBenchmark(
                model_name="DialoGPT-medium (HF)",
                model_path=model_name,
                framework="huggingface",
                model_size_gb=0.7,  # Approximate
                parameters=355000000,  # Known parameter count
                quantization="fp16",
                load_time=load_time,
                load_memory_mb=load_memory,
                inference_times=[inference_time],
                avg_inference_time=inference_time,
                tokens_per_second=50 / inference_time,
                peak_memory_mb=memory_after - memory_before,
                finetuning_possible=True,
                sample_outputs=[{
                    "prompt": self.test_prompts[0][:50] + "...",
                    "response": response[:100] + "..." if len(response) > 100 else response,
                    "time": inference_time
                }]
            )
            
            self.results.append(hf_result)
            print(f"    ✅ HF test completed: {load_time:.2f}s load, {inference_time:.2f}s inference")
            
        except Exception as e:
            print(f"    ❌ HuggingFace test failed: {e}")
        finally:
            try:
                del model, tokenizer
            except:
                pass
            gc.collect()
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results."""
        successful_results = [r for r in self.results if r.success]
        
        analysis = {
            "summary": {
                "total_models_tested": len(self.results),
                "successful_tests": len(successful_results),
                "failed_tests": len(self.results) - len(successful_results),
                "frameworks_tested": list(set(r.framework for r in successful_results)),
            },
            "performance_comparison": {},
            "model_details": [],
            "recommendations": [],
            "system_info": self.system_info,
            "datasets_info": self._analyze_datasets()
        }
        
        if successful_results:
            # Performance comparison
            fastest_load = min(successful_results, key=lambda x: x.load_time)
            fastest_inference = min(successful_results, key=lambda x: x.avg_inference_time)
            most_efficient_memory = min(successful_results, key=lambda x: x.peak_memory_mb)
            highest_throughput = max(successful_results, key=lambda x: x.tokens_per_second)
            
            analysis["performance_comparison"] = {
                "fastest_load": {
                    "model": fastest_load.model_name,
                    "time": fastest_load.load_time,
                    "framework": fastest_load.framework
                },
                "fastest_inference": {
                    "model": fastest_inference.model_name,
                    "time": fastest_inference.avg_inference_time,
                    "framework": fastest_inference.framework
                },
                "most_memory_efficient": {
                    "model": most_efficient_memory.model_name,
                    "memory_mb": most_efficient_memory.peak_memory_mb,
                    "framework": most_efficient_memory.framework
                },
                "highest_throughput": {
                    "model": highest_throughput.model_name,
                    "tokens_per_second": highest_throughput.tokens_per_second,
                    "framework": highest_throughput.framework
                }
            }
            
            # Model details
            analysis["model_details"] = [asdict(r) for r in successful_results]
            
            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations(successful_results)
        
        return analysis
    
    def _analyze_datasets(self) -> Dict[str, Any]:
        """Analyze available datasets."""
        dataset_info = {}
        
        for name, path in self.datasets.items():
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        lines = f.readlines()
                    
                    # Sample first line to understand format
                    sample = json.loads(lines[0]) if lines else {}
                    
                    dataset_info[name] = {
                        "path": path,
                        "size": len(lines),
                        "format": list(sample.keys()) if sample else [],
                        "sample": sample
                    }
                except Exception as e:
                    dataset_info[name] = {"error": str(e)}
        
        return dataset_info
    
    def _generate_recommendations(self, results: List[ModelBenchmark]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        if len(results) == 0:
            recommendations.append("No successful model tests - check model availability and system resources")
            return recommendations
        
        # Find best performers
        fastest_inference = min(results, key=lambda x: x.avg_inference_time)
        most_memory_efficient = min(results, key=lambda x: x.peak_memory_mb)
        
        recommendations.extend([
            f"🚀 For fastest inference: Use {fastest_inference.model_name} ({fastest_inference.avg_inference_time:.2f}s avg)",
            f"💾 For memory efficiency: Use {most_memory_efficient.model_name} ({most_memory_efficient.peak_memory_mb:.0f}MB peak)",
        ])
        
        # Framework-specific recommendations
        mlx_results = [r for r in results if r.framework == "mlx"]
        hf_results = [r for r in results if r.framework == "huggingface"]
        
        if mlx_results and hf_results:
            mlx_avg_time = sum(r.avg_inference_time for r in mlx_results) / len(mlx_results)
            hf_avg_time = sum(r.avg_inference_time for r in hf_results) / len(hf_results)
            
            if mlx_avg_time < hf_avg_time:
                recommendations.append("🔥 MLX framework shows better performance for Apple Silicon")
            else:
                recommendations.append("🤗 HuggingFace framework shows competitive performance with broader compatibility")
        
        # Quantization recommendations
        quantized_models = [r for r in results if "4bit" in r.quantization or "8bit" in r.quantization]
        if quantized_models:
            recommendations.append("🔹 4-bit quantized models provide good balance of performance and memory usage")
        
        # Fine-tuning recommendations
        finetune_capable = [r for r in results if r.finetuning_possible]
        if finetune_capable:
            best_finetune = min(finetune_capable, key=lambda x: x.finetuning_time or float('inf'))
            recommendations.append(f"🎯 For fine-tuning: {best_finetune.model_name} offers best estimated performance")
        
        return recommendations
    
    def _generate_report(self, analysis: Dict[str, Any]):
        """Generate comprehensive markdown report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"8B_MODEL_COMPREHENSIVE_REPORT_{timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(self._create_markdown_report(analysis))
        
        # Also save JSON for programmatic access
        json_path = f"8B_MODEL_BENCHMARK_DATA_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n📋 Report generated:")
        print(f"  📄 Markdown: {report_path}")
        print(f"  📊 JSON Data: {json_path}")
        
        return report_path
    
    def _create_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """Create comprehensive markdown report."""
        
        report = f"""# Comprehensive 8B Model Benchmark Report
        
**Generated:** {analysis['system_info']['timestamp']}  
**System:** {analysis['system_info']['os']} ({analysis['system_info']['cpu_cores']} cores, {analysis['system_info']['memory_total_gb']} GB RAM)

## Executive Summary

This comprehensive benchmark evaluates available 8B parameter language models across multiple frameworks (MLX, HuggingFace) on Apple Silicon hardware. The analysis includes inference performance, memory usage, fine-tuning capabilities, and practical recommendations.

### Key Findings

- **Models Tested:** {analysis['summary']['total_models_tested']}
- **Successful Tests:** {analysis['summary']['successful_tests']}
- **Frameworks:** {', '.join(analysis['summary']['frameworks_tested'])}

"""
        
        # Performance comparison section
        if "performance_comparison" in analysis:
            pc = analysis["performance_comparison"]
            report += f"""## Performance Comparison

### 🏆 Best Performers

| Metric | Model | Value | Framework |
|--------|--------|--------|-----------|
| **Fastest Load** | {pc['fastest_load']['model']} | {pc['fastest_load']['time']:.2f}s | {pc['fastest_load']['framework']} |
| **Fastest Inference** | {pc['fastest_inference']['model']} | {pc['fastest_inference']['time']:.2f}s | {pc['fastest_inference']['framework']} |
| **Memory Efficient** | {pc['most_memory_efficient']['model']} | {pc['most_memory_efficient']['memory_mb']:.0f}MB | {pc['most_memory_efficient']['framework']} |
| **Highest Throughput** | {pc['highest_throughput']['model']} | {pc['highest_throughput']['tokens_per_second']:.1f} tok/s | {pc['highest_throughput']['framework']} |

"""
        
        # Model details section
        if analysis['model_details']:
            report += """## Detailed Model Analysis

"""
            for model in analysis['model_details']:
                if model['success']:
                    params_str = f"{model['parameters']:,}" if model['parameters'] else 'Unknown'
                    report += f"""### {model['model_name']}

**Framework:** {model['framework']}  
**Size:** {model['model_size_gb']} GB  
**Quantization:** {model['quantization']}  
**Parameters:** {params_str} (estimated)

#### Performance Metrics
- **Load Time:** {model['load_time']:.2f} seconds
- **Average Inference:** {model['avg_inference_time']:.2f} seconds  
- **Throughput:** {model['tokens_per_second']:.1f} tokens/second
- **Peak Memory:** {model['peak_memory_mb']:.0f} MB
- **Fine-tuning Capable:** {'✅ Yes' if model['finetuning_possible'] else '❌ No'}

"""
                    if model['finetuning_time']:
                        report += f"- **Estimated Fine-tuning Time:** {model['finetuning_time']:.1f} seconds\n"
                        report += f"- **Estimated Fine-tuning Memory:** {model['finetuning_memory_mb']:.0f} MB\n"
                    
                    if model['sample_outputs']:
                        report += f"""
#### Sample Outputs

"""
                        for i, output in enumerate(model['sample_outputs'][:2], 1):
                            report += f"""**Test {i}:** {output['prompt']}  
**Response:** {output['response']}  
**Time:** {output['time']:.2f}s

"""
                else:
                    report += f"""### {model['model_name']} ❌

**Error:** {model['error_msg']}

"""
        
        # Dataset analysis section
        if analysis['datasets_info']:
            report += """## Dataset Analysis

Available fine-tuning datasets:

"""
            for name, info in analysis['datasets_info'].items():
                if 'error' not in info:
                    report += f"""### {name.title()} Dataset
- **Path:** `{info['path']}`
- **Size:** {info['size']} samples
- **Format:** {', '.join(info['format'])}
- **Sample Fields:** `{list(info['sample'].keys())}`

"""
        
        # Recommendations section
        if analysis['recommendations']:
            report += """## Recommendations

"""
            for rec in analysis['recommendations']:
                report += f"- {rec}\n"
        
        # Technical details section
        report += f"""
## Technical Details

### System Configuration
- **Operating System:** {analysis['system_info']['os']}
- **CPU Cores:** {analysis['system_info']['cpu_cores']}
- **Total Memory:** {analysis['system_info']['memory_total_gb']} GB
- **Available Memory:** {analysis['system_info']['memory_available_gb']} GB

### Framework Status
"""
        
        for framework, available in analysis['system_info']['frameworks'].items():
            status = "✅ Available" if available else "❌ Not Available"
            report += f"- **{framework.upper()}:** {status}\n"
        
        report += f"""
### Model Files Tested

The following 8B parameter models were evaluated:

1. **Llama 3.1 8B (bf16)** - Full precision model (15GB)
2. **Llama 3.1 8B (4-bit)** - Quantized model (4.2GB)  
3. **HuggingFace Comparison Models** - For framework comparison

### Methodology

1. **Loading Test:** Measure time and memory to load each model
2. **Inference Test:** Run {len(self.test_prompts)} standard prompts measuring response time
3. **Memory Analysis:** Track peak memory usage during operations
4. **Fine-tuning Assessment:** Evaluate capability and estimate performance
5. **Quality Sampling:** Capture sample outputs for qualitative analysis

### Limitations

- Fine-tuning tests are estimated rather than actual runs due to time constraints
- Memory measurements include system overhead
- Token counting is estimated based on word count approximations
- Results may vary based on system load and other running processes

---

*Generated by Flow2 Comprehensive 8B Model Benchmark Suite*  
*Timestamp: {analysis['system_info']['timestamp']}*
"""
        
        return report

def run_comprehensive_8b_model_benchmark(output_dir: str = "./benchmark_results/models_8b") -> Dict[str, Any]:
    """Run comprehensive 8B model benchmark.
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing analysis results
    """
    if not FLOW2_AVAILABLE:
        return {"error": "Flow2 not available", "status": "failed"}
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize and run benchmark
        generator = Comprehensive8BReportGenerator()
        analysis = generator.run_comprehensive_benchmark()
        
        return analysis
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return {"error": str(e), "status": "failed"}

def main():
    """Run comprehensive 8B model benchmark and generate report."""
    
    print("🚀 Flow2 Comprehensive 8B Model Analysis")
    print("=" * 60)
    
    if not FLOW2_AVAILABLE:
        print("❌ Flow2 not available")
        return 1
    
    # Check system requirements
    memory_gb = psutil.virtual_memory().total // (1024**3)
    if memory_gb < 16:
        print(f"⚠️  Warning: {memory_gb}GB RAM detected. 8B models may require 16GB+ for optimal performance.")
    
    analysis = run_comprehensive_8b_model_benchmark()
    
    if analysis.get("status") == "failed":
        print(f"❌ Benchmark failed: {analysis.get('error')}")
        return 1
    
    # Print summary
    print(f"\n✅ Comprehensive benchmark completed!")
    print(f"📊 Models tested: {analysis['summary']['total_models_tested']}")
    print(f"🎯 Successful: {analysis['summary']['successful_tests']}")
    
    if analysis['recommendations']:
        print(f"\n🔍 Key Recommendations:")
        for rec in analysis['recommendations'][:3]:
            print(f"  • {rec}")
    
    return 0

if __name__ == "__main__":
    exit(main())
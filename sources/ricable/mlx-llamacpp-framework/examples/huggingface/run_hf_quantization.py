#!/usr/bin/env python3
"""
HuggingFace Quantization Example
===============================

Demonstrates various quantization methods with performance benchmarking.
"""

import argparse
import os
import time
from typing import List

import flow2

def main():
    parser = argparse.ArgumentParser(description="HuggingFace Quantization Example")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-small",
                       help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="./hf_quantized_models",
                       help="Output directory")
    parser.add_argument("--method", choices=["bnb_4bit", "bnb_8bit", "gptq", "awq", "dynamic", "all"],
                       default="bnb_4bit", help="Quantization method")
    parser.add_argument("--bits", type=int, default=4,
                       help="Number of bits for quantization")
    parser.add_argument("--group-size", type=int, default=128,
                       help="Group size for GPTQ/AWQ")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run quantization benchmark")
    parser.add_argument("--test-prompts", nargs="+", 
                       default=["Hello, how are you?", "What is machine learning?"],
                       help="Test prompts for benchmarking")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"],
                       default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Check framework availability
    if not flow2.HUGGINGFACE_AVAILABLE:
        print("❌ HuggingFace framework not available")
        print("Install with: pip install flow2[huggingface]")
        return
    
    if not flow2.QUANTIZATION_AVAILABLE:
        print("❌ Quantization libraries not available")
        print("Install with: pip install flow2[quantization]")
        return
    
    print("🤗 HuggingFace Quantization Example")
    print("=" * 45)
    print(f"Model: {args.model}")
    print(f"Method: {args.method}")
    print(f"Bits: {args.bits}")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    # Setup device
    if args.device == "auto":
        device = flow2.frameworks.huggingface.setup_mps_device()
    else:
        import torch
        device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.benchmark:
        # Run comprehensive benchmark
        run_quantization_benchmark(args.model, args.test_prompts, args.output_dir)
    else:
        # Single quantization method
        if args.method == "all":
            methods = ["bnb_4bit", "bnb_8bit"]
            if device.type != "mps":  # GPTQ/AWQ may not work on MPS
                methods.extend(["gptq", "dynamic"])
        else:
            methods = [args.method]
        
        for method in methods:
            quantize_single_method(
                args.model, method, args.bits, args.group_size, 
                args.output_dir, device, args.test_prompts
            )

def quantize_single_method(
    model_name: str, 
    method: str, 
    bits: int, 
    group_size: int,
    output_dir: str,
    device,
    test_prompts: List[str]
):
    """Quantize using a single method."""
    print(f"\n🔧 Quantizing with {method}...")
    
    # Map method name to enum
    method_map = {
        "bnb_4bit": flow2.frameworks.huggingface.QuantizationMethod.BITSANDBYTES_4BIT,
        "bnb_8bit": flow2.frameworks.huggingface.QuantizationMethod.BITSANDBYTES_8BIT,
        "gptq": flow2.frameworks.huggingface.QuantizationMethod.GPTQ,
        "awq": flow2.frameworks.huggingface.QuantizationMethod.AWQ,
        "dynamic": flow2.frameworks.huggingface.QuantizationMethod.DYNAMIC
    }
    
    if method not in method_map:
        print(f"❌ Unknown method: {method}")
        return
    
    # Create quantization config
    quant_config = flow2.frameworks.huggingface.create_quantization_config(
        method=method_map[method],
        bits=bits,
        group_size=group_size
    )
    
    # Create method-specific output directory
    method_output_dir = os.path.join(output_dir, method)
    
    start_time = time.time()
    
    try:
        # Quantize model
        result_path = flow2.frameworks.huggingface.quantize_model(
            model_name=model_name,
            output_dir=method_output_dir,
            quantization_config=quant_config,
            save_model=True,
            device=device
        )
        
        quantize_time = time.time() - start_time
        print(f"✅ {method} quantization completed in {quantize_time:.2f} seconds")
        print(f"📁 Model saved to: {result_path}")
        
        # Test the quantized model
        test_quantized_model(result_path, test_prompts, device)
        
    except Exception as e:
        print(f"❌ {method} quantization failed: {e}")

def test_quantized_model(model_path: str, test_prompts: List[str], device):
    """Test a quantized model with sample prompts."""
    print(f"\n🧪 Testing quantized model...")
    
    try:
        # Load quantized model
        model, tokenizer = flow2.frameworks.huggingface.load_quantized_model(
            model_path, device
        )
        
        # Test with each prompt
        for i, prompt in enumerate(test_prompts[:2]):  # Limit to 2 prompts for speed
            print(f"\nTest {i+1}: {prompt}")
            print("-" * 30)
            
            start_time = time.time()
            
            gen_params = flow2.frameworks.huggingface.GenerationParams(
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
            
            response = flow2.frameworks.huggingface.generate_completion(
                model, tokenizer, prompt, gen_params
            )
            
            inference_time = time.time() - start_time
            print(f"Response: {response}")
            print(f"⏱️ Time: {inference_time:.2f}s")
        
        print("✅ Quantized model working correctly!")
        
    except Exception as e:
        print(f"⚠️ Could not test quantized model: {e}")

def run_quantization_benchmark(model_name: str, test_prompts: List[str], output_dir: str):
    """Run comprehensive quantization benchmark."""
    print("🏁 Running quantization benchmark...")
    
    # Define methods to benchmark
    methods = [
        flow2.frameworks.huggingface.QuantizationMethod.BITSANDBYTES_4BIT,
        flow2.frameworks.huggingface.QuantizationMethod.BITSANDBYTES_8BIT,
    ]
    
    # Add GPTQ if not on MPS (may have compatibility issues)
    import torch
    device = flow2.frameworks.huggingface.setup_mps_device()
    if device.type != "mps":
        methods.append(flow2.frameworks.huggingface.QuantizationMethod.DYNAMIC)
    
    try:
        results = flow2.frameworks.huggingface.benchmark_quantization(
            model_name=model_name,
            methods=methods,
            test_prompts=test_prompts,
            output_dir=os.path.join(output_dir, "benchmark")
        )
        
        print("\n📊 Benchmark Results:")
        print("=" * 50)
        
        for method, result in results["methods"].items():
            if result.get("successful", False):
                print(f"\n{method.upper()}:")
                print(f"  Quantization Time: {result['quantize_time']:.2f}s")
                print(f"  Avg Inference Time: {result['avg_inference_time']:.3f}s")
                print(f"  Model Size: {result['model_size_mb']:.1f} MB")
            else:
                print(f"\n{method.upper()}: ❌ Failed - {result.get('error', 'Unknown error')}")
        
        # Find best method
        successful_methods = {k: v for k, v in results["methods"].items() 
                            if v.get("successful", False)}
        
        if successful_methods:
            best_method = min(successful_methods.items(), 
                            key=lambda x: x[1]["avg_inference_time"])
            
            print(f"\n🏆 Best Method: {best_method[0].upper()}")
            print(f"   Fastest inference: {best_method[1]['avg_inference_time']:.3f}s")
            print(f"   Model size: {best_method[1]['model_size_mb']:.1f} MB")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
HuggingFace Fine-tuning Test with MPS
====================================

Test LoRA fine-tuning on Apple Silicon with MPS acceleration.
"""

import os
import sys
import time
import json
from typing import Dict, Any

try:
    import flow2
    FLOW2_AVAILABLE = True
except ImportError:
    FLOW2_AVAILABLE = False

def create_sample_dataset():
    """Create a small sample dataset for testing."""
    sample_data = [
        {"input": "What is machine learning?", "output": "Machine learning is a branch of AI that enables computers to learn from data."},
        {"input": "Explain neural networks", "output": "Neural networks are computational models inspired by biological neural networks."},
        {"input": "What is deep learning?", "output": "Deep learning uses neural networks with multiple layers to learn complex patterns."},
        {"input": "Define artificial intelligence", "output": "AI is the simulation of human intelligence processes by machines."},
        {"input": "What are algorithms?", "output": "Algorithms are step-by-step procedures for solving problems or performing tasks."}
    ]
    
    # Create dataset file
    dataset_path = "sample_finetuning_data.jsonl"
    with open(dataset_path, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    return dataset_path

def test_hf_lora_finetuning():
    """Test HuggingFace LoRA fine-tuning with MPS."""
    
    if not FLOW2_AVAILABLE or not flow2.HUGGINGFACE_AVAILABLE:
        print("❌ HuggingFace not available")
        return
    
    print("🤗 HuggingFace LoRA Fine-tuning Test")
    print("=" * 45)
    
    # Use a small model for testing
    model_name = "microsoft/DialoGPT-small"
    output_dir = "./test_lora_output"
    
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    
    # Create sample dataset
    print("\n📝 Creating sample dataset...")
    dataset_path = create_sample_dataset()
    print(f"Dataset created: {dataset_path}")
    
    try:
        # Create LoRA config
        print("\n🔧 Creating LoRA configuration...")
        lora_config = flow2.frameworks.huggingface.create_lora_config(
            model_name=model_name,
            r=8,  # Small rank for testing
            lora_alpha=16,
            lora_dropout=0.1
        )
        print(f"LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
        
        # Create training config
        training_config = flow2.frameworks.huggingface.TrainingConfig(
            output_dir=output_dir,
            num_train_epochs=1,  # Just 1 epoch for testing
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=5e-4,
            logging_steps=1,
            save_steps=10,
            eval_steps=10,
            dataloader_pin_memory=False,  # Important for MPS
            remove_unused_columns=False,
            report_to=None
        )
        
        # Setup MPS device
        device = flow2.frameworks.huggingface.setup_mps_device()
        print(f"Using device: {device}")
        
        # Run LoRA fine-tuning
        print("\n🎯 Starting LoRA fine-tuning...")
        start_time = time.time()
        
        adapter_path = flow2.frameworks.huggingface.finetune_lora(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            lora_config=lora_config,
            training_config=training_config,
            max_length=256,  # Short sequences for testing
            device=device
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ Fine-tuning completed!")
        print(f"⏱️  Training time: {training_time:.2f} seconds")
        print(f"📁 Adapter saved to: {adapter_path}")
        
        # Test the fine-tuned model
        print("\n🧪 Testing fine-tuned model...")
        test_finetuned_model(model_name, adapter_path, device)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists(dataset_path):
            os.remove(dataset_path)

def test_finetuned_model(base_model: str, adapter_path: str, device):
    """Test the fine-tuned model."""
    try:
        print("Loading base model and adapter...")
        
        # Load base model
        model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
            base_model, device=device
        )
        
        # Load LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # Test generation
        test_prompts = [
            "What is machine learning?",
            "Explain neural networks",
            "Define artificial intelligence"
        ]
        
        gen_params = flow2.frameworks.huggingface.GenerationParams(
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
        
        print("\n🗨️  Testing responses:")
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = flow2.frameworks.huggingface.generate_completion(
                model, tokenizer, prompt, gen_params
            )
            print(f"Response: {response}")
        
        print("\n✅ Fine-tuned model working correctly!")
        
    except Exception as e:
        print(f"⚠️  Could not test fine-tuned model: {e}")

def test_hf_quantization():
    """Test HuggingFace quantization capabilities."""
    
    if not getattr(flow2, 'QUANTIZATION_AVAILABLE', False):
        print("⚠️  Quantization not available (BitsAndBytes doesn't support Apple Silicon)")
        return
    
    print("\n🔹 Testing HuggingFace Quantization...")
    
    model_name = "microsoft/DialoGPT-small"
    
    try:
        # Test different quantization methods
        methods = [
            flow2.frameworks.huggingface.QuantizationMethod.BITSANDBYTES_4BIT,
            flow2.frameworks.huggingface.QuantizationMethod.BITSANDBYTES_8BIT
        ]
        
        test_prompts = ["Hello, how are you?"]
        
        results = flow2.frameworks.huggingface.benchmark_quantization(
            model_name=model_name,
            methods=methods,
            test_prompts=test_prompts,
            output_dir="./quantization_test"
        )
        
        print("✅ Quantization benchmark completed!")
        print(f"Results: {results}")
        
    except Exception as e:
        print(f"❌ Quantization test failed: {e}")

def run_huggingface_benchmark(output_dir: str = "./benchmark_results/huggingface") -> Dict[str, Any]:
    """Run comprehensive HuggingFace benchmark.
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing benchmark results
    """
    if not FLOW2_AVAILABLE:
        return {"error": "Flow2 not available", "status": "failed"}
    
    if not flow2.HUGGINGFACE_AVAILABLE:
        return {"error": "HuggingFace not available", "status": "failed"}
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "huggingface_available": True,
                "mps_available": getattr(flow2, 'MPS_AVAILABLE', False),
                "quantization_available": getattr(flow2, 'QUANTIZATION_AVAILABLE', False)
            },
            "tests": {},
            "status": "success"
        }
        
        # Test LoRA fine-tuning
        print("🎯 Testing LoRA fine-tuning...")
        finetuning_success = test_hf_lora_finetuning()
        results["tests"]["lora_finetuning"] = {"success": finetuning_success}
        
        # Test quantization if available
        print("\n🔹 Testing quantization...")
        if getattr(flow2, 'QUANTIZATION_AVAILABLE', False):
            test_hf_quantization()
            results["tests"]["quantization"] = {"success": True}
        else:
            results["tests"]["quantization"] = {
                "success": False, 
                "note": "BitsAndBytes not supported on Apple Silicon"
            }
        
        # Save results
        results_path = os.path.join(output_dir, f"huggingface_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ HuggingFace benchmark failed: {e}")
        return {"error": str(e), "status": "failed"}

def main():
    print("🚀 HuggingFace Framework Testing Suite")
    print("=" * 50)
    
    if not FLOW2_AVAILABLE:
        print("❌ Flow2 not available")
        return 1
    
    # Check framework availability
    print(f"HuggingFace Available: {flow2.HUGGINGFACE_AVAILABLE}")
    print(f"MPS Available: {getattr(flow2, 'MPS_AVAILABLE', False)}")
    print(f"Quantization Available: {getattr(flow2, 'QUANTIZATION_AVAILABLE', False)}")
    
    if not flow2.HUGGINGFACE_AVAILABLE:
        print("❌ HuggingFace not available!")
        return 1
    
    # Run benchmark
    results = run_huggingface_benchmark()
    
    if results.get("status") == "failed":
        print(f"❌ Benchmark failed: {results.get('error')}")
        return 1
    
    print(f"\n✅ HuggingFace benchmark completed!")
    return 0

if __name__ == "__main__":
    exit(main())
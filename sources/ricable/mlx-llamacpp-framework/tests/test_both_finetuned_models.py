#!/usr/bin/env python3
"""
Side-by-side comparison of both fine-tuned models
"""

from mlx_lm import load, generate
import time

def test_model_comparison():
    """Compare both fine-tuned models side by side"""
    
    print("🔬 FINE-TUNED MODEL COMPARISON")
    print("=" * 60)
    
    # Load both models with their adapters
    print("📦 Loading TinyLlama with fine-tuned adapter...")
    try:
        tinyllama_model, tinyllama_tokenizer = load(
            './models/mlx/tinyllama-1.1b-chat',
            adapter_path='./finetune_output/tinyllama_enhanced/best_adapters.safetensors'
        )
        print("✅ TinyLlama loaded successfully")
    except Exception as e:
        print(f"❌ Error loading TinyLlama: {e}")
        return
    
    print("📦 Loading Qwen2.5 with fine-tuned adapter...")
    try:
        qwen_model, qwen_tokenizer = load(
            './models/mlx/qwen2.5-1.5b-instruct',
            adapter_path='./finetune_output/qwen_enhanced/best_adapters.safetensors'
        )
        print("✅ Qwen2.5 loaded successfully")
    except Exception as e:
        print(f"❌ Error loading Qwen2.5: {e}")
        return
    
    # Test prompts
    test_prompts = [
        "Write an inspirational quote about success",
        "Write an inspirational quote about perseverance",
        "Write an inspirational quote about dreams",
        "Write an inspirational quote about courage",
        "Write an inspirational quote about love"
    ]
    
    print(f"\n🧪 Testing {len(test_prompts)} prompts with both models...")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Prompt: {prompt}")
        print("-" * 60)
        
        # Format prompt for chat models
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Test TinyLlama
        print("🤖 TinyLlama-1.1B-Chat (Fine-tuned):")
        try:
            start_time = time.time()
            tinyllama_response = generate(
                tinyllama_model, 
                tinyllama_tokenizer, 
                formatted_prompt, 
                max_tokens=100,
                temperature=0.7
            )
            tinyllama_time = time.time() - start_time
            
            # Clean up response
            clean_response = tinyllama_response.split('<|im_end|>')[0].strip()
            print(f"Response: {clean_response}")
            print(f"Time: {tinyllama_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
        
        # Test Qwen2.5
        print("🤖 Qwen2.5-1.5B-Instruct (Fine-tuned):")
        try:
            start_time = time.time()
            qwen_response = generate(
                qwen_model,
                qwen_tokenizer,
                formatted_prompt,
                max_tokens=100,
                temperature=0.7
            )
            qwen_time = time.time() - start_time
            
            # Clean up response
            clean_response = qwen_response.split('<|im_end|>')[0].strip()
            print(f"Response: {clean_response}")
            print(f"Time: {qwen_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("=" * 60)
    
    print("\n🎊 Comparison complete!")
    print("📊 Summary:")
    print("  TinyLlama: Faster generation, potential quality issues")
    print("  Qwen2.5: Higher quality, more coherent responses")

if __name__ == "__main__":
    test_model_comparison()
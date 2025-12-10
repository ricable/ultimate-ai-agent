#!/usr/bin/env python3
"""
Simple Flash Attention comparison test
"""

import time
import mlx.core as mx
from mlx_lm import load, generate

# Flash Attention Integration
try:
    from flash_attention_mlx import OptimizedMLXMultiHeadAttention, FlashAttentionBenchmark
    FLASH_ATTENTION_AVAILABLE = True
    print("✅ Flash Attention optimizations available")
except ImportError:
    print("⚠️  Flash Attention not available, using standard MLX attention")
    FLASH_ATTENTION_AVAILABLE = False

def apply_flash_attention_to_model(model, use_flash_attention=True, block_size=None):
    """
    Apply Flash Attention optimizations to model attention layers
    """
    if not use_flash_attention or not FLASH_ATTENTION_AVAILABLE:
        print("ℹ️ Using standard MLX attention")
        return model, 0
    
    print("🚀 Applying Flash Attention optimizations...")
    attention_replacements = 0
    
    def replace_attention_recursive(module, name_prefix="", depth=0):
        nonlocal attention_replacements
        
        # Prevent deep recursion
        if depth > 10:
            return
        
        # Handle MLX models which may have different attribute access patterns
        try:
            module_dict = module.__dict__ if hasattr(module, '__dict__') else {}
            for name, child in module_dict.items():
                if name.startswith('_') or name in ['training', 'parameters', 'modules']:
                    continue
                    
                try:
                    if not hasattr(child, '__class__'):
                        continue
                        
                    full_name = f"{name_prefix}.{name}" if name_prefix else name
                    
                    # Check if this is an attention layer we should replace
                    if hasattr(child, '__class__') and 'MultiHeadAttention' in str(child.__class__):
                        print(f"🔄 Replacing {full_name} with Flash Attention")
                        
                        # Create optimized replacement
                        try:
                            flash_attention = OptimizedMLXMultiHeadAttention(
                                child.dims,
                                child.num_heads,
                                bias=hasattr(child, 'bias'),
                                use_flash_attention=True,
                                block_size=block_size
                            )
                            
                            # Copy weights from original layer
                            if hasattr(child, 'q_proj') and hasattr(child.q_proj, 'weight'):
                                flash_attention.q_proj.weight = child.q_proj.weight
                                flash_attention.k_proj.weight = child.k_proj.weight  
                                flash_attention.v_proj.weight = child.v_proj.weight
                                flash_attention.out_proj.weight = child.out_proj.weight
                                
                                if hasattr(child.q_proj, 'bias') and child.q_proj.bias is not None:
                                    flash_attention.q_proj.bias = child.q_proj.bias
                                    flash_attention.k_proj.bias = child.k_proj.bias
                                    flash_attention.v_proj.bias = child.v_proj.bias
                                    flash_attention.out_proj.bias = child.out_proj.bias
                            
                            # Replace the layer
                            setattr(module, name, flash_attention)
                            attention_replacements += 1
                        except Exception as e:
                            print(f"⚠️ Failed to replace {full_name}: {e}")
                    else:
                        # Recursively process child modules
                        replace_attention_recursive(child, full_name, depth + 1)
                        
                except (AttributeError, TypeError):
                    continue
                    
        except (AttributeError, TypeError):
            pass
    
    try:
        replace_attention_recursive(model)
        
        if attention_replacements > 0:
            print(f"✅ Replaced {attention_replacements} attention layers with Flash Attention")
        else:
            print("ℹ️ No compatible attention layers found for replacement")
            
    except Exception as e:
        print(f"⚠️ Flash Attention integration failed: {e}")
        print("ℹ️ Continuing with standard MLX attention")
    
    return model, attention_replacements

def test_inference_performance(model_path, prompt, use_flash_attention=True):
    """Test inference performance with/without Flash Attention"""
    
    print(f"\n{'='*60}")
    print(f"Testing: {'Flash Attention ENABLED' if use_flash_attention else 'Flash Attention DISABLED'}")
    print(f"{'='*60}")
    
    # Load model
    print("📦 Loading model...")
    start_load = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    # Apply Flash Attention if requested
    flash_replacements = 0
    if use_flash_attention and FLASH_ATTENTION_AVAILABLE:
        model, flash_replacements = apply_flash_attention_to_model(model)
    
    # Warmup run
    print("🔥 Warming up...")
    warmup_start = time.time()
    try:
        _ = generate(model, tokenizer, "Hello", max_tokens=5)
        warmup_time = time.time() - warmup_start
        print(f"Warmup completed in {warmup_time:.2f}s")
    except Exception as e:
        print(f"Warmup failed: {e}")
        return None
    
    # Main inference test
    print(f"🧠 Running inference...")
    print(f"Prompt: {prompt}")
    
    inference_start = time.time()
    try:
        response = generate(model, tokenizer, prompt, max_tokens=100, verbose=True)
        inference_time = time.time() - inference_start
        
        # Count tokens (rough estimate)
        response_tokens = len(response.split())
        tokens_per_second = response_tokens / inference_time if inference_time > 0 else 0
        
        print(f"\n📊 RESULTS:")
        print(f"⏱️ Load time: {load_time:.2f}s")
        print(f"⏱️ Inference time: {inference_time:.2f}s") 
        print(f"🔣 Response tokens: ~{response_tokens}")
        print(f"⚡ Tokens/second: ~{tokens_per_second:.1f}")
        print(f"🚀 Flash Attention layers: {flash_replacements}")
        print(f"\n📝 Response: {response[:200]}...")
        
        return {
            "load_time": load_time,
            "inference_time": inference_time, 
            "tokens": response_tokens,
            "tokens_per_second": tokens_per_second,
            "flash_layers": flash_replacements,
            "response": response
        }
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None

def main():
    """Run Flash Attention comparison"""
    
    model_path = "models/mlx/tinyllama-1.1b-chat"
    prompt = "Explain the benefits of machine learning in 3 sentences."
    
    print("🎯 FLASH ATTENTION PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Test 1: Flash Attention enabled
    result_flash = test_inference_performance(model_path, prompt, use_flash_attention=True)
    
    # Clean up between tests
    mx.clear_cache()
    time.sleep(2)
    
    # Test 2: Flash Attention disabled  
    result_standard = test_inference_performance(model_path, prompt, use_flash_attention=False)
    
    # Compare results
    if result_flash and result_standard:
        print(f"\n{'='*80}")
        print("📊 PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        
        speedup = result_standard["inference_time"] / result_flash["inference_time"] if result_flash["inference_time"] > 0 else 1.0
        throughput_improvement = (result_flash["tokens_per_second"] / result_standard["tokens_per_second"] - 1) * 100 if result_standard["tokens_per_second"] > 0 else 0
        
        print(f"⚡ Flash Attention vs Standard:")
        print(f"   Inference time: {result_flash['inference_time']:.2f}s vs {result_standard['inference_time']:.2f}s")
        print(f"   Tokens/second: {result_flash['tokens_per_second']:.1f} vs {result_standard['tokens_per_second']:.1f}")
        print(f"   🚀 Speedup: {speedup:.2f}x")
        print(f"   📈 Throughput improvement: {throughput_improvement:+.1f}%")
        print(f"   🔧 Flash layers optimized: {result_flash['flash_layers']}")
        
        if speedup > 1.05:
            print(f"\n✅ Flash Attention provides {speedup:.2f}x speedup!")
        elif speedup > 0.95:
            print(f"\n➡️ Performance is similar ({speedup:.2f}x)")
        else:
            print(f"\n⚠️ Standard attention is faster ({1/speedup:.2f}x)")
    else:
        print("\n❌ Could not complete comparison due to errors")

if __name__ == "__main__":
    main()
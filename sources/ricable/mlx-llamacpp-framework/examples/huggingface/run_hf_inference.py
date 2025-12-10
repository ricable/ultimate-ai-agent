#!/usr/bin/env python3
"""
HuggingFace Inference Example
============================

Demonstrates text generation with HuggingFace models using MPS acceleration.
"""

import argparse
import time
from typing import List, Optional

import flow2

def main():
    parser = argparse.ArgumentParser(description="HuggingFace Inference Example")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-small", 
                       help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Nucleus sampling parameter")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], 
                       help="Use quantization")
    parser.add_argument("--streaming", action="store_true",
                       help="Enable streaming output")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], 
                       default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Check framework availability
    if not flow2.HUGGINGFACE_AVAILABLE:
        print("❌ HuggingFace framework not available")
        print("Install with: pip install flow2[huggingface]")
        return
    
    print("🤗 HuggingFace Inference Example")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")
    print()
    
    # Setup device
    if args.device == "auto":
        device = flow2.frameworks.huggingface.setup_mps_device()
    else:
        import torch
        device = torch.device(args.device)
    
    # Setup quantization config
    quantization_config = None
    if args.quantize:
        if args.quantize == "4bit":
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        elif args.quantize == "8bit":
            quantization_config = {
                "load_in_8bit": True
            }
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    
    try:
        model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
            args.model,
            device=device,
            quantization_config=quantization_config
        )
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Setup generation parameters
    gen_params = flow2.frameworks.huggingface.GenerationParams(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    print(f"\nPrompt: {args.prompt}")
    print("\nGenerated Text:")
    print("-" * 40)
    
    # Generate text
    start_time = time.time()
    
    if args.streaming:
        # Streaming generation
        for token in flow2.frameworks.huggingface.streaming_completion(
            model, tokenizer, args.prompt, gen_params, print_output=False
        ):
            print(token, end="", flush=True)
    else:
        # Regular generation
        response = flow2.frameworks.huggingface.generate_completion(
            model, tokenizer, args.prompt, gen_params
        )
        print(response)
    
    generation_time = time.time() - start_time
    print(f"\n\n⏱️  Generation completed in {generation_time:.2f} seconds")
    
    # Print performance stats
    if hasattr(model, 'num_parameters'):
        num_params = model.num_parameters()
    else:
        num_params = sum(p.numel() for p in model.parameters())
    
    print(f"📊 Model parameters: {flow2.frameworks.huggingface.format_model_size(num_params)}")
    print(f"🔧 Device: {device}")
    print(f"💾 Quantization: {args.quantize or 'None'}")

if __name__ == "__main__":
    main()
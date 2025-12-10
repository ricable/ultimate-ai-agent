#!/usr/bin/env python3
"""
HuggingFace LoRA Fine-tuning Example
===================================

Demonstrates LoRA and QLoRA fine-tuning with Accelerate and MPS support.
"""

import argparse
import os
import time

import flow2

def main():
    parser = argparse.ArgumentParser(description="HuggingFace LoRA Fine-tuning Example")
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-small",
                       help="Base model name or path")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset path (JSONL file or HuggingFace dataset)")
    parser.add_argument("--output-dir", type=str, default="./hf_lora_output",
                       help="Output directory")
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                       help="LoRA dropout rate")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--quantized", action="store_true",
                       help="Use QLoRA (4-bit quantization)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"],
                       default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Check framework availability
    if not flow2.HUGGINGFACE_AVAILABLE:
        print("❌ HuggingFace framework not available")
        print("Install with: pip install flow2[huggingface]")
        return
    
    if args.quantized and not flow2.QUANTIZATION_AVAILABLE:
        print("❌ Quantization libraries not available for QLoRA")
        print("Install with: pip install flow2[quantization]")
        return
    
    print("🤗 HuggingFace LoRA Fine-tuning Example")
    print("=" * 50)
    print(f"Base Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output Directory: {args.output_dir}")
    print(f"LoRA Rank: {args.lora_r}")
    print(f"Quantized: {'Yes (QLoRA)' if args.quantized else 'No'}")
    print(f"Device: {args.device}")
    print()
    
    # Setup device
    if args.device == "auto":
        device = flow2.frameworks.huggingface.setup_mps_device()
    else:
        import torch
        device = torch.device(args.device)
    
    # Create LoRA configuration
    lora_config = flow2.frameworks.huggingface.create_lora_config(
        model_name=args.model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    print(f"LoRA Configuration:")
    print(f"  Rank: {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Target Modules: {lora_config.target_modules}")
    print(f"  Dropout: {lora_config.lora_dropout}")
    print()
    
    # Create training configuration
    training_config = flow2.frameworks.huggingface.TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_pin_memory=False,  # Better for MPS
        remove_unused_columns=False,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        warmup_steps=100,
        fp16=device.type == "cuda",  # Use FP16 for CUDA
        report_to=None  # Disable wandb/tensorboard
    )
    
    print("Training Configuration:")
    print(f"  Epochs: {training_config.num_train_epochs}")
    print(f"  Batch Size: {training_config.per_device_train_batch_size}")
    print(f"  Learning Rate: {training_config.learning_rate}")
    print(f"  Gradient Checkpointing: {training_config.gradient_checkpointing}")
    print()
    
    # Start training
    print("Starting training...")
    start_time = time.time()
    
    try:
        if args.quantized:
            # QLoRA fine-tuning
            qlora_config = flow2.frameworks.huggingface.QLoRAConfig(
                lora_config=lora_config
            )
            
            adapter_path = flow2.frameworks.huggingface.finetune_qlora(
                model_name=args.model,
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                qlora_config=qlora_config,
                training_config=training_config,
                max_length=args.max_length,
                device=device
            )
        else:
            # Regular LoRA fine-tuning
            adapter_path = flow2.frameworks.huggingface.finetune_lora(
                model_name=args.model,
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                lora_config=lora_config,
                training_config=training_config,
                max_length=args.max_length,
                device=device
            )
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
        print(f"📁 Adapter saved to: {adapter_path}")
        
        # Test the fine-tuned model
        print("\nTesting fine-tuned model...")
        test_inference(args.model, adapter_path, args.quantized, device)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def test_inference(base_model: str, adapter_path: str, quantized: bool, device):
    """Test the fine-tuned model with a sample prompt."""
    try:
        # Load base model
        quantization_config = None
        if quantized:
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        
        model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
            base_model,
            device=device,
            quantization_config=quantization_config
        )
        
        # Load LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        
        # Test generation
        test_prompt = "Hello, can you help me with"
        
        gen_params = flow2.frameworks.huggingface.GenerationParams(
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        print(f"Test Prompt: {test_prompt}")
        print("Generated Response:")
        print("-" * 30)
        
        response = flow2.frameworks.huggingface.generate_completion(
            model, tokenizer, test_prompt, gen_params
        )
        
        print(response)
        print("\n✅ Fine-tuned model working correctly!")
        
    except Exception as e:
        print(f"⚠️  Could not test fine-tuned model: {e}")

if __name__ == "__main__":
    main()
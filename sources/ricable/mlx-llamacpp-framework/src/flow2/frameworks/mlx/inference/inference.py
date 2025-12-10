#!/usr/bin/env python3
"""
Inference script for MLX models on Apple Silicon.
"""

import os
import sys
import argparse
import json
import time
from typing import List, Dict, Any, Optional, Union, Tuple

# Import utilities from flow2 package (with fallbacks)
try:
    from flow2.utils.utils import InferenceTimer, load_prompts_from_file, save_completions_to_file, format_chat_prompt
except ImportError:
    # Provide simple fallback implementations
    class InferenceTimer:
        def __init__(self):
            self.start_time = time.time()
        def elapsed(self):
            return time.time() - self.start_time
    
    def load_prompts_from_file(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def save_completions_to_file(completions, file_path):
        with open(file_path, 'w') as f:
            for completion in completions:
                f.write(completion + '\n')
    
    def format_chat_prompt(prompt, template="default"):
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# Check for MLX dependencies
try:
    import mlx.core as mx
    from mlx_lm import load, generate
except ImportError:
    print("Error: MLX or MLX-LM not found.")
    print("Please install it with: pip install mlx mlx-lm")
    sys.exit(1)

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
    
    def replace_attention_recursive(module, name_prefix=""):
        nonlocal attention_replacements
        
        # Handle MLX models which may have different attribute access patterns
        try:
            for name in dir(module):
                if name.startswith('_') or name in ['training', 'parameters', 'modules']:
                    continue
                    
                try:
                    child = getattr(module, name)
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
                        replace_attention_recursive(child, full_name)
                        
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


def load_mlx_model(
    model_path: str,
    quantization: Optional[str] = None,
    max_tokens: int = 2048,
    verbose: bool = False,
    use_flash_attention: bool = True,
    flash_block_size: Optional[int] = None
) -> Tuple[Any, Any]:
    """
    Load an MLX model from the specified path.
    
    Args:
        model_path: Path to the model directory or identifier
        quantization: Quantization type (None, 'int4', 'int8')
        max_tokens: Maximum context length
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_path}")
    print(f"Quantization: {quantization or 'None'}, Max tokens: {max_tokens}")
    
    # Set MLX to use GPU (Metal)
    mx.set_default_device(mx.gpu)
    
    # Load the model
    try:
        # Check if quantization is supported in the current version
        import inspect
        load_signature = inspect.signature(load)
        
        if 'quantization' in load_signature.parameters:
            model, tokenizer = load(
                model_path,
                quantization=quantization,
                max_tokens=max_tokens
            )
        else:
            # Fallback for older MLX-LM versions
            model, tokenizer = load(model_path)
        
        if verbose:
            print("Model loaded successfully")
            print(f"Model type: {type(model).__name__}")
            print(f"Tokenizer type: {type(tokenizer).__name__}")
        
        # Apply Flash Attention optimizations
        if use_flash_attention and FLASH_ATTENTION_AVAILABLE:
            model, flash_replacements = apply_flash_attention_to_model(
                model, 
                use_flash_attention=use_flash_attention,
                block_size=flash_block_size
            )
            if flash_replacements > 0:
                print(f"📊 Flash Attention: {flash_replacements} layers optimized")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    stream: bool = False
) -> Dict[str, Any]:
    """
    Generate text completion with the given model and parameters.
    
    Args:
        model: MLX model
        tokenizer: MLX tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Penalty for repeating tokens
        stream: Whether to stream the output
        
    Returns:
        Dictionary with completion details
    """
    # Record start time
    with InferenceTimer("MLX inference") as timer:
        # Count input tokens
        input_tokens = tokenizer.encode(prompt)
        timer.update_tokens(len(input_tokens))
        
        # Generate completion
        if stream:
            response_text = ""
            for token in generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stream=True
            ):
                print(token, end="", flush=True)
                response_text += token
                timer.update_tokens(1)  # Approximately one token per chunk
            
            print()  # Newline after completion
            
            result = {
                "prompt": prompt,
                "completion": response_text,
                "finish_reason": "stop"  # Assuming it completed normally
            }
        else:
            # Non-streaming generation
            output_tokens = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            response_text = tokenizer.decode(output_tokens)
            timer.update_tokens(len(output_tokens))
            
            result = {
                "prompt": prompt,
                "completion": response_text,
                "finish_reason": "stop"  # MLX doesn't provide finish reason
            }
    
    # Add performance stats
    result["performance"] = timer.get_stats()
    return result

def batch_generate_completions(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1
) -> List[Dict[str, Any]]:
    """
    Generate completions for a batch of prompts.
    
    Args:
        model: MLX model
        tokenizer: MLX tokenizer
        prompts: List of input prompts
        max_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Penalty for repeating tokens
        
    Returns:
        List of completion dictionaries
    """
    completions = []
    
    print(f"Generating completions for {len(prompts)} prompts...")
    
    with InferenceTimer(f"Batch of {len(prompts)} prompts") as batch_timer:
        for i, prompt in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] Generating completion...")
            
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stream=False
            )
            
            completions.append(completion)
            batch_timer.update_tokens(completion["performance"]["total_tokens"])
            
            # Print completion summary
            print(f"Completion ({len(completion['completion'].split())} tokens): {completion['completion'][:100]}...")
            print(f"Tokens/sec: {completion['performance']['tokens_per_second']:.2f}")
            print("-" * 40)
    
    # Print batch performance stats
    batch_timer.print_stats()
    return completions

def generate_embeddings(
    model_path: str,
    texts: List[str],
    batch_size: int = 32
) -> Dict[str, List[float]]:
    """
    Generate embeddings for a list of texts using an MLX embedding model.
    
    Args:
        model_path: Path to the embedding model
        texts: List of texts to embed
        batch_size: Batch size for processing
        
    Returns:
        Dictionary mapping texts to embeddings
    """
    try:
        # Import MLX embedding utilities
        from mlx_lm.utils import load_embedding_model
    except ImportError:
        print("Error: MLX embedding utilities not found")
        return {}
    
    result = {}
    
    print(f"Loading embedding model from: {model_path}")
    try:
        model = load_embedding_model(model_path)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return {}
    
    print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        for text in batch:
            # Generate embedding
            embedding = model.embed(text)
            result[text] = embedding.tolist()
    
    return result

def chat_completion(
    model: Any,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    template_style: str = "llama2",
    stream: bool = True
) -> Dict[str, Any]:
    """
    Generate a chat completion based on a series of messages.
    
    Args:
        model: MLX model
        tokenizer: MLX tokenizer
        messages: List of message dicts with 'role' and 'content' keys
        system_prompt: Optional system prompt to prepend
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Penalty for repeating tokens
        template_style: Template style for formatting prompt
        stream: Whether to stream the output
        
    Returns:
        Dictionary with completion details
    """
    # Format the chat prompt
    formatted_prompt = format_chat_prompt(
        messages=messages,
        system_prompt=system_prompt,
        template_style=template_style
    )
    
    # Generate completion with the formatted prompt
    return generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stream=stream
    )

def interactive_chat(
    model: Any,
    tokenizer: Any,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    template_style: str = "llama2"
):
    """
    Run an interactive chat session with the model.
    
    Args:
        model: MLX model
        tokenizer: MLX tokenizer
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Penalty for repeating tokens
        template_style: Template style for formatting prompt
    """
    messages = []
    
    print("Chat mode enabled. Type 'exit' to quit.")
    print("System prompt:", system_prompt or "(None)")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        
        messages.append({"role": "user", "content": user_input})
        
        # Generate chat completion
        completion = chat_completion(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            template_style=template_style,
            stream=True
        )
        
        # Add assistant message
        messages.append({"role": "assistant", "content": completion["completion"]})

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Inference script for MLX models")
    
    # Model parameters
    parser.add_argument("--model", type=str, required=True, help="Path to MLX model or model identifier")
    parser.add_argument("--quantization", type=str, choices=["int4", "int8"], help="Quantization type")
    parser.add_argument("--max-context", type=int, default=2048, help="Maximum context length")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--prompt-file", type=str, help="File containing prompts (one per line)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    
    # Mode parameters
    parser.add_argument("--chat", action="store_true", help="Run in chat mode")
    parser.add_argument("--system-prompt", type=str, help="System prompt for chat mode")
    parser.add_argument("--embeddings", action="store_true", help="Generate embeddings instead of completions")
    parser.add_argument("--batch", action="store_true", help="Process prompts in batch mode")
    
    # Output parameters
    parser.add_argument("--output", type=str, help="Output file for completions/embeddings")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    # Flash Attention parameters
    parser.add_argument("--use-flash-attention", action="store_true", default=True,
                       help="Enable Flash Attention optimization (default: True)")
    parser.add_argument("--disable-flash-attention", action="store_true",
                       help="Disable Flash Attention optimization")
    parser.add_argument("--flash-block-size", type=int, default=None,
                       help="Flash Attention block size (auto if None)")
    parser.add_argument("--benchmark-attention", action="store_true",
                       help="Run attention benchmark before inference")
    
    args = parser.parse_args()
    
    # Handle Flash Attention settings
    if args.disable_flash_attention:
        args.use_flash_attention = False
    
    # Run attention benchmark if requested
    if args.benchmark_attention and FLASH_ATTENTION_AVAILABLE:
        print("\n🔬 Running Flash Attention benchmark...")
        try:
            benchmark = FlashAttentionBenchmark()
            benchmark.benchmark_attention_performance(
                batch_sizes=[1, 2],
                seq_lengths=[64, 128, 256],
                head_dims=[32, 64, 128],
                num_heads=8,
                num_runs=3
            )
            benchmark.print_summary()
        except Exception as e:
            print(f"⚠️ Benchmark failed: {e}")
        print()
    
    # Print Flash Attention status
    print(f"⚡ Flash Attention: {'✅ Enabled' if args.use_flash_attention else '❌ Disabled'}")
    
    # Determine the mode of operation
    if args.embeddings:
        # Embeddings mode
        if args.prompt:
            texts = [args.prompt]
        elif args.prompt_file:
            texts = load_prompts_from_file(args.prompt_file)
        else:
            print("Error: Please provide a prompt or a prompt file for embeddings")
            sys.exit(1)
        
        # Generate embeddings
        result = generate_embeddings(args.model, texts)
        
        # Save or print the result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Embeddings saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    
    else:
        # Load the model for completion or chat
        model, tokenizer = load_mlx_model(
            model_path=args.model,
            quantization=args.quantization,
            max_tokens=args.max_context,
            verbose=args.verbose,
            use_flash_attention=args.use_flash_attention,
            flash_block_size=args.flash_block_size
        )
        
        if args.chat:
            # Chat mode
            interactive_chat(
                model=model,
                tokenizer=tokenizer,
                system_prompt=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
        
        else:
            # Standard completion mode
            if args.prompt:
                prompts = [args.prompt]
            elif args.prompt_file:
                prompts = load_prompts_from_file(args.prompt_file)
            else:
                print("Error: Please provide a prompt or a prompt file")
                sys.exit(1)
            
            # Process prompts
            if args.batch and len(prompts) > 1:
                # Batch processing mode
                completions = batch_generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )
            else:
                # Single prompt or interactive mode
                completions = []
                for prompt in prompts:
                    completion = generate_completion(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                        stream=not args.no_stream
                    )
                    completions.append(completion)
            
            # Save completions if output file is specified
            if args.output:
                save_completions_to_file(args.output, completions)

if __name__ == "__main__":
    main()
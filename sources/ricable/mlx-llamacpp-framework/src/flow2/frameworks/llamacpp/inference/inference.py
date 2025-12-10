#!/usr/bin/env python3
"""
Inference script for llama.cpp models on Apple Silicon.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional, Union

# Import utilities from flow2 package (with fallbacks)
try:
    from flow2.utils.utils import InferenceTimer, load_prompts_from_file, save_completions_to_file, format_chat_prompt
except ImportError:
    import time
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

# Llama.cpp Python bindings
try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama_cpp Python module not found.")
    print("Please install it with: pip install llama-cpp-python")
    sys.exit(1)

def create_llama_model(
    model_path: str,
    n_ctx: int = 2048,
    n_gpu_layers: int = -1,  # -1 means use all available layers on GPU
    n_threads: Optional[int] = None,
    verbose: bool = False
) -> Llama:
    """
    Create and return a Llama model instance.
    
    Args:
        model_path: Path to the GGUF model file
        n_ctx: Context size (token window)
        n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        n_threads: Number of threads to use (None for auto)
        verbose: Whether to print verbose output
        
    Returns:
        Llama model instance
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Determine optimal thread count if not specified
    if n_threads is None:
        import multiprocessing
        n_threads = max(1, multiprocessing.cpu_count() // 2)
    
    print(f"Loading model: {model_path}")
    print(f"Context size: {n_ctx}, GPU layers: {n_gpu_layers}, Threads: {n_threads}")
    
    # Create model with Metal acceleration enabled
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=verbose
    )
    
    return model

def generate_completion(
    model: Llama,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repeat_penalty: float = 1.1,
    stop: Optional[List[str]] = None,
    stream: bool = False
) -> Dict[str, Any]:
    """
    Generate text completion with the given model and parameters.
    
    Args:
        model: Llama model instance
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repeat_penalty: Penalty for repeating tokens
        stop: List of stop strings
        stream: Whether to stream the output
        
    Returns:
        Dictionary with completion details
    """
    # Record start time
    with InferenceTimer("llama.cpp inference") as timer:
        # Generate completion
        completion = model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
            stream=stream
        )
        
        # Process streamed output if streaming
        if stream:
            response_text = ""
            for chunk in completion:
                text_chunk = chunk["choices"][0]["text"]
                response_text += text_chunk
                print(text_chunk, end="", flush=True)
                timer.update_tokens(len(text_chunk.split()))
            print()  # Newline after completion
            
            result = {
                "prompt": prompt,
                "completion": response_text,
                "finish_reason": "stop"  # Assuming it completed normally
            }
        else:
            # Process normal (non-streamed) output
            result = {
                "prompt": prompt,
                "completion": completion["choices"][0]["text"],
                "finish_reason": completion["choices"][0]["finish_reason"]
            }
            timer.update_tokens(len(result["completion"].split()))
    
    # Add performance stats
    result["performance"] = timer.get_stats()
    return result

def batch_generate_completions(
    model: Llama,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repeat_penalty: float = 1.1,
    stop: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Generate completions for a batch of prompts.
    
    Args:
        model: Llama model instance
        prompts: List of input prompts
        max_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repeat_penalty: Penalty for repeating tokens
        stop: List of stop strings
        
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
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=stop,
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
    model: Llama,
    texts: List[str],
    batch_size: int = 32
) -> Dict[str, List[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        model: Llama model instance
        texts: List of texts to embed
        batch_size: Batch size for processing
        
    Returns:
        Dictionary mapping texts to embeddings
    """
    result = {}
    
    print(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        for text in batch:
            embedding = model.embed(text)
            result[text] = embedding.tolist()
    
    return result

def chat_completion(
    model: Llama,
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repeat_penalty: float = 1.1,
    template_style: str = "llama2",
    stream: bool = True
) -> Dict[str, Any]:
    """
    Generate a chat completion based on a series of messages.
    
    Args:
        model: Llama model instance
        messages: List of message dicts with 'role' and 'content' keys
        system_prompt: Optional system prompt to prepend
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repeat_penalty: Penalty for repeating tokens
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
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        stream=stream
    )

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Inference script for llama.cpp models")
    
    # Model parameters
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model file")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size in tokens")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="Number of layers to offload to GPU (-1 for all)")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads to use")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--prompt-file", type=str, help="File containing prompts (one per line)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Penalty for repeating tokens")
    
    # Mode parameters
    parser.add_argument("--chat", action="store_true", help="Run in chat mode")
    parser.add_argument("--system-prompt", type=str, help="System prompt for chat mode")
    parser.add_argument("--embeddings", action="store_true", help="Generate embeddings instead of completions")
    parser.add_argument("--batch", action="store_true", help="Process prompts in batch mode")
    
    # Output parameters
    parser.add_argument("--output", type=str, help="Output file for completions/embeddings")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Create the model
    model = create_llama_model(
        model_path=args.model,
        n_ctx=args.ctx_size,
        n_gpu_layers=args.gpu_layers,
        n_threads=args.threads,
        verbose=args.verbose
    )
    
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
        result = generate_embeddings(model, texts)
        
        # Save or print the result
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Embeddings saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    
    elif args.chat:
        # Chat mode
        messages = []
        system_prompt = args.system_prompt
        
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
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                stream=not args.no_stream
            )
            
            # Add assistant message
            messages.append({"role": "assistant", "content": completion["completion"]})
    
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
                prompts=prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty
            )
        else:
            # Single prompt or interactive mode
            completions = []
            for prompt in prompts:
                completion = generate_completion(
                    model=model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repeat_penalty=args.repeat_penalty,
                    stream=not args.no_stream
                )
                completions.append(completion)
        
        # Save completions if output file is specified
        if args.output:
            save_completions_to_file(args.output, completions)

if __name__ == "__main__":
    main()
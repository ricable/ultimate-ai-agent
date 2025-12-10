#!/usr/bin/env python3
"""
Command-line Chat Interface for MLX

This script provides an interactive command-line chat interface for MLX models.
It handles model loading, context management, and pretty-printed chat UI.
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Optional, Any, Union, Tuple

# Add parent directory to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from common.chat_history import create_chat_session

# Flash Attention Integration
try:
    from flash_attention_mlx import OptimizedMLXMultiHeadAttention, FlashAttentionBenchmark
    FLASH_ATTENTION_AVAILABLE = True
    print("✅ Flash Attention optimizations available")
except ImportError:
    print("⚠️  Flash Attention not available, using standard MLX attention")
    FLASH_ATTENTION_AVAILABLE = False

# ANSI color codes for terminal output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bg_red": "\033[41m",
    "bg_green": "\033[42m",
    "bg_yellow": "\033[43m",
    "bg_blue": "\033[44m",
}


def check_mlx_installation() -> bool:
    """
    Check if MLX is installed.
    
    Returns:
        Boolean indicating if MLX is installed
    """
    try:
        import mlx
        import mlx.core
        return True
    except ImportError:
        return False


def list_available_models() -> List[str]:
    """
    List available MLX model directories.
    
    Returns:
        List of model paths
    """
    models_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../mlx-setup/models'
    ))
    
    if not os.path.exists(models_dir):
        return []
    
    # Return directories that contain a config.json file
    return [
        os.path.join(models_dir, d) 
        for d in os.listdir(models_dir) 
        if os.path.isdir(os.path.join(models_dir, d)) and 
           os.path.exists(os.path.join(models_dir, d, 'config.json'))
    ]


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


def generate_mlx_response(
    model_path: str, 
    messages: List[Dict[str, str]], 
    max_tokens: int = 1024,
    temperature: float = 0.7,
    streaming: bool = True,
    use_flash_attention: bool = True,
    flash_block_size: Optional[int] = None
) -> str:
    """
    Run the MLX model to generate text.
    
    Args:
        model_path: Path to the model directory
        messages: List of chat messages in MLX format
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        streaming: Whether to stream the output token by token
        
    Returns:
        Generated text
    """
    try:
        # Import MLX modules (inside function to avoid import errors if MLX is not installed)
        import mlx.core as mx
        from mlx_lm import load, generate
    except ImportError:
        raise RuntimeError(
            "MLX is not installed. Please install MLX with: pip install mlx mlx-lm"
        )
    
    # Prepare the input for MLX
    # Most MLX models use a format like:
    # {
    #   "messages": [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Hello!"}
    #   ]
    # }
    
    # Load the model
    print(f"{COLORS['yellow']}Loading model from {model_path}...{COLORS['reset']}")
    model, tokenizer = load(model_path)
    print(f"{COLORS['green']}Model loaded successfully!{COLORS['reset']}")
    
    # Apply Flash Attention optimizations
    if use_flash_attention and FLASH_ATTENTION_AVAILABLE:
        model, flash_replacements = apply_flash_attention_to_model(
            model, 
            use_flash_attention=use_flash_attention,
            block_size=flash_block_size
        )
        if flash_replacements > 0:
            print(f"{COLORS['green']}📊 Flash Attention: {flash_replacements} layers optimized{COLORS['reset']}")
    
    # Construct the prompt
    # For chat-optimized models
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Fallback to simple concatenation
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n\n"
        prompt += "Assistant: "
    
    # Generate the response
    start_time = time.time()
    
    if streaming:
        generated_text = ""
        print(f"\n{COLORS['bold']}{COLORS['green']}Assistant:{COLORS['reset']} ", end="", flush=True)
        
        from mlx_lm import stream_generate
        for response in stream_generate(
            model, 
            tokenizer, 
            prompt, 
            max_tokens=max_tokens
        ):
            token = response.text
            print(token, end="", flush=True)
            generated_text += token
            
            # Check for Ctrl+C
            if hasattr(sys, 'exc_info') and sys.exc_info()[0] is not None:
                if isinstance(sys.exc_info()[0], KeyboardInterrupt):
                    break
        
        print()  # Add newline after generation
    else:
        # Non-streaming generation
        generated_tokens = generate(
            model, 
            tokenizer, 
            prompt, 
            max_tokens=max_tokens, 
            temp=temperature
        )
        generated_text = tokenizer.decode(generated_tokens)
        print(f"\n{COLORS['bold']}{COLORS['green']}Assistant:{COLORS['reset']} {generated_text}\n")
    
    generation_time = time.time() - start_time
    print(f"\n{COLORS['yellow']}Generation completed in {generation_time:.2f} seconds{COLORS['reset']}")
    
    return generated_text


def print_header():
    """Print the application header."""
    print(f"\n{COLORS['bold']}{COLORS['magenta']}┌─────────────────────────────────────────────┐{COLORS['reset']}")
    print(f"{COLORS['bold']}{COLORS['magenta']}│ MLX Interactive Chat - Apple Silicon         │{COLORS['reset']}")
    print(f"{COLORS['bold']}{COLORS['magenta']}└─────────────────────────────────────────────┘{COLORS['reset']}\n")


def print_help():
    """Print help information."""
    print(f"\n{COLORS['bold']}Available commands:{COLORS['reset']}")
    print(f"  {COLORS['yellow']}/help{COLORS['reset']}    - Show this help message")
    print(f"  {COLORS['yellow']}/clear{COLORS['reset']}   - Clear the conversation history")
    print(f"  {COLORS['yellow']}/params{COLORS['reset']}  - Show current parameters")
    print(f"  {COLORS['yellow']}/temp{COLORS['reset']} N  - Set temperature to N (e.g., /temp 0.8)")
    print(f"  {COLORS['yellow']}/quit{COLORS['reset']}    - Exit the program")
    print()


def main():
    """Main function for the chat CLI."""
    parser = argparse.ArgumentParser(description="Command-line Chat Interface for MLX")
    parser.add_argument(
        "--model", "-m", 
        type=str,
        help="Path to the model directory"
    )
    parser.add_argument(
        "--max-tokens", "-n", 
        type=int, 
        default=1024,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", "-t", 
        type=float, 
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--system-message", "-s", 
        type=str, 
        default="You are a helpful assistant.",
        help="System message to use"
    )
    parser.add_argument(
        "--no-streaming", 
        action="store_true",
        help="Disable streaming output"
    )
    parser.add_argument(
        "--history-file", 
        type=str,
        help="File to save chat history"
    )
    parser.add_argument(
        "--use-flash-attention", 
        action="store_true", 
        default=True,
        help="Enable Flash Attention optimization (default: True)"
    )
    parser.add_argument(
        "--disable-flash-attention", 
        action="store_true",
        help="Disable Flash Attention optimization"
    )
    parser.add_argument(
        "--flash-block-size", 
        type=int, 
        default=None,
        help="Flash Attention block size (auto if None)"
    )
    parser.add_argument(
        "--benchmark-attention", 
        action="store_true",
        help="Run attention benchmark before starting chat"
    )
    
    args = parser.parse_args()
    
    # Handle Flash Attention settings
    if args.disable_flash_attention:
        args.use_flash_attention = False
    
    # Check if MLX is installed
    if not check_mlx_installation():
        print(f"{COLORS['red']}MLX is not installed. Please install MLX with: pip install mlx mlx-lm{COLORS['reset']}")
        return 1
    
    # If no model specified, try to find one
    if not args.model:
        models = list_available_models()
        if not models:
            print(f"{COLORS['red']}No models found. Please specify a model with --model.{COLORS['reset']}")
            return 1
        
        # Use the first model found
        args.model = models[0]
        print(f"{COLORS['yellow']}Using model: {args.model}{COLORS['reset']}")
    
    # Check if the model directory exists
    if not os.path.exists(args.model):
        print(f"{COLORS['red']}Model directory not found: {args.model}{COLORS['reset']}")
        return 1
    
    # Create chat session
    chat_session = create_chat_session(
        system_message=args.system_message,
        history_file=args.history_file,
        max_context_length=4096  # Arbitrary default
    )
    
    # Parameter storage for runtime adjustments
    params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "streaming": not args.no_streaming,
        "use_flash_attention": args.use_flash_attention,
        "flash_block_size": args.flash_block_size,
    }
    
    # Run attention benchmark if requested
    if args.benchmark_attention and FLASH_ATTENTION_AVAILABLE:
        print(f"\n{COLORS['yellow']}🔬 Running Flash Attention benchmark...{COLORS['reset']}")
        try:
            benchmark = FlashAttentionBenchmark()
            benchmark.benchmark_attention_performance(
                batch_sizes=[1, 2],
                seq_lengths=[64, 128],
                head_dims=[32, 64],
                num_heads=8,
                num_runs=3
            )
            benchmark.print_summary()
        except Exception as e:
            print(f"{COLORS['red']}⚠️ Benchmark failed: {e}{COLORS['reset']}")
    
    # Print header and help
    print_header()
    print(f"{COLORS['green']}Type your message or use /help to see available commands.{COLORS['reset']}")
    print(f"{COLORS['green']}Press Ctrl+C at any time to stop generation.{COLORS['reset']}")
    print(f"{COLORS['green']}⚡ Flash Attention: {'✅ Enabled' if args.use_flash_attention else '❌ Disabled'}{COLORS['reset']}\n")
    
    # Main chat loop
    try:
        while True:
            # Get user input
            user_input = input(f"{COLORS['bold']}{COLORS['blue']}You:{COLORS['reset']} ")
            
            # Handle commands
            if user_input.startswith('/'):
                cmd = user_input.strip().lower()
                
                if cmd == '/help':
                    print_help()
                    continue
                
                elif cmd == '/clear':
                    chat_session.clear_history()
                    print(f"{COLORS['green']}Conversation history cleared.{COLORS['reset']}")
                    continue
                
                elif cmd == '/params':
                    print(f"\n{COLORS['bold']}Current parameters:{COLORS['reset']}")
                    for k, v in params.items():
                        print(f"  {COLORS['yellow']}{k}{COLORS['reset']}: {v}")
                    print()
                    continue
                
                elif cmd.startswith('/temp '):
                    try:
                        new_temp = float(cmd.split(' ')[1])
                        if 0.0 <= new_temp <= 2.0:
                            params["temperature"] = new_temp
                            print(f"{COLORS['green']}Temperature set to {new_temp}{COLORS['reset']}")
                        else:
                            print(f"{COLORS['red']}Temperature must be between 0.0 and 2.0{COLORS['reset']}")
                    except:
                        print(f"{COLORS['red']}Invalid temperature value{COLORS['reset']}")
                    continue
                
                elif cmd == '/quit':
                    print(f"{COLORS['green']}Goodbye!{COLORS['reset']}")
                    break
                
                else:
                    print(f"{COLORS['red']}Unknown command. Type /help for available commands.{COLORS['reset']}")
                    continue
            
            # Add user message to history
            chat_session.add_user_message(user_input)
            
            # Get formatted messages for MLX
            messages = chat_session.get_formatted_context("mlx")
            
            try:
                # Generate assistant response
                response = generate_mlx_response(
                    model_path=args.model,
                    messages=messages,
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    streaming=params["streaming"],
                    use_flash_attention=params["use_flash_attention"],
                    flash_block_size=params["flash_block_size"]
                )
                
                # Add the response to history
                chat_session.add_assistant_message(response)
                
            except KeyboardInterrupt:
                print(f"\n{COLORS['yellow']}Generation stopped.{COLORS['reset']}")
                
                # Get partial response and add to history
                assistant_response = input(f"\n{COLORS['yellow']}Enter the partial response to save to history: {COLORS['reset']}")
                if assistant_response:
                    chat_session.add_assistant_message(assistant_response)
            
            # Ensure we don't exceed context length
            chat_session.truncate_context_if_needed()
            
            print()  # Add a newline for spacing
    
    except KeyboardInterrupt:
        print(f"\n{COLORS['green']}Goodbye!{COLORS['reset']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
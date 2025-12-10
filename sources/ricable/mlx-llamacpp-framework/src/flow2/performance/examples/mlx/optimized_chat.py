#!/usr/bin/env python3
"""
Optimized Chat Script for MLX on Apple Silicon

This script provides an optimized chat experience with MLX models,
utilizing Metal GPU acceleration and optimal settings for performance.

Usage:
  python optimized_chat.py --model <model_name> [--quant <quantization>] [options]

Example:
  python optimized_chat.py --model llama-2-7b --quant int4
"""

import argparse
import time
import sys
import os
import platform
import gc
import threading
import re
import signal

# Check for MLX and required packages
try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("Error: MLX not available. Please install MLX and MLX-LM.")
    print("pip install mlx mlx-lm")
    sys.exit(1)

class MemoryMonitor(threading.Thread):
    """Thread for monitoring memory usage in the background"""
    def __init__(self, interval=5):
        super().__init__()
        self.daemon = True  # Thread will exit when main program exits
        self.interval = interval
        self.running = True
        self.peak_memory = 0
    
    def run(self):
        try:
            import psutil
            process = psutil.Process(os.getpid())
            
            while self.running:
                # Get memory info
                mem_info = process.memory_info()
                memory_gb = mem_info.rss / (1024 ** 3)  # Convert to GB
                
                # Update peak memory
                if memory_gb > self.peak_memory:
                    self.peak_memory = memory_gb
                    # Print memory update (only on significant changes)
                    if int(self.peak_memory * 10) % 5 == 0:  # Print at 0.5GB intervals
                        print(f"\r💾 Peak memory usage: {self.peak_memory:.2f} GB", file=sys.stderr, end="")
                
                # Sleep for the specified interval
                time.sleep(self.interval)
        except ImportError:
            print("Warning: psutil not available. Memory monitoring disabled.", file=sys.stderr)
        except Exception as e:
            print(f"Error in memory monitor: {e}", file=sys.stderr)
    
    def stop(self):
        self.running = False

def clear_line():
    """Clear the current line in the terminal"""
    print("\r" + " " * 80 + "\r", end="")

def detect_apple_silicon():
    """Detect if running on Apple Silicon"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def format_tokens_per_second(total_tokens, total_time):
    """Format tokens per second with appropriate units"""
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    if tokens_per_second < 1:
        return f"{tokens_per_second * 1000:.1f} tokens/ms"
    else:
        return f"{tokens_per_second:.1f} tokens/s"

def create_chat_history(system_prompt=None):
    """Create initial chat history with system prompt"""
    if not system_prompt:
        system_prompt = "You are a helpful, respectful assistant. Always provide accurate information and admit when you don't know something rather than making up information."
    
    return f"{system_prompt}\n\n"

def format_prompt(history, user_input):
    """Format a prompt for the model with user input"""
    return f"{history}User: {user_input}\nAssistant: "

def run_optimized_chat(model_name, quantization=None, temperature=0.7, top_p=0.95, 
                      top_k=40, repetition_penalty=1.1, max_tokens=2048,
                      system_prompt=None, use_gpu=True):
    """Run an optimized chat session with the specified model"""
    # Verify if running on Apple Silicon and set device accordingly
    is_apple_silicon = detect_apple_silicon()
    
    if is_apple_silicon and use_gpu:
        print("✅ Apple Silicon detected. Using Metal GPU acceleration.")
        mx.set_default_device(mx.gpu)
    else:
        if is_apple_silicon and not use_gpu:
            print("⚠️ Apple Silicon detected but GPU is disabled. Using CPU.")
        else:
            print("⚠️ Not running on Apple Silicon. Using CPU.")
        mx.set_default_device(mx.cpu)
    
    # Start memory monitoring
    memory_monitor = MemoryMonitor()
    memory_monitor.start()
    
    try:
        # Clear cache and collect garbage before loading
        gc.collect()
        mx.clear_cache()
        
        # Load the model
        print(f"📦 Loading model: {model_name}" + (f" with {quantization} quantization" if quantization else ""))
        load_start = time.time()
        model, tokenizer = load(model_name, quantization=quantization)
        load_end = time.time()
        load_time = load_end - load_start
        print(f"✅ Model loaded in {load_time:.2f} seconds")
        
        # Force compilation with a small input
        print("🔄 Compiling model with test input...")
        compile_start = time.time()
        input_tokens = tokenizer.encode("Hello")
        _ = model(mx.array([input_tokens]))
        compile_end = time.time()
        compile_time = compile_end - compile_start
        print(f"✅ Model compiled in {compile_time:.2f} seconds")
        
        # Initialize chat history
        history = create_chat_history(system_prompt)
        print("\n💬 Chat session started. Type 'exit' or 'quit' to end the session.")
        print("💬 Type 'clear' to clear chat history.")
        print("=" * 50)
        print("Assistant: Hello! How can I help you today?")
        
        # Create a handler for graceful exit
        def signal_handler(sig, frame):
            print("\n\n🛑 Interrupted. Cleaning up...")
            memory_monitor.stop()
            print(f"💾 Peak memory usage: {memory_monitor.peak_memory:.2f} GB")
            print("👋 Goodbye!")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Chat loop
        total_tokens_generated = 0
        total_generation_time = 0
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if user_input.lower() == "clear":
                # Reset chat history
                history = create_chat_history(system_prompt)
                print("🧹 Chat history cleared.")
                continue
            
            # Add user input to history and create prompt
            prompt = format_prompt(history, user_input)
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            
            # Track generation time and tokens
            generation_start = time.time()
            tokens_generated = 0
            assistant_response = ""
            
            # Stream tokens
            for token in generate(
                model, 
                tokenizer, 
                prompt, 
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                stream=True
            ):
                tokens_generated += 1
                print(token, end="", flush=True)
                assistant_response += token
            
            generation_end = time.time()
            generation_time = generation_end - generation_start
            
            # Update totals
            total_tokens_generated += tokens_generated
            total_generation_time += generation_time
            
            # Update history
            history = prompt + assistant_response + "\n\n"
            
            # Print generation stats
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            print(f"\n[Generated {tokens_generated} tokens in {generation_time:.2f}s • {tokens_per_second:.1f} tokens/s]")
        
        # Clean up
        del model, tokenizer
        gc.collect()
        mx.clear_cache()
        
        # Print final stats
        print("\n" + "=" * 50)
        print("💬 Chat session ended")
        print(f"📊 Total tokens generated: {total_tokens_generated}")
        print(f"⏱️ Total generation time: {total_generation_time:.2f} seconds")
        print(f"⚡ Average speed: {format_tokens_per_second(total_tokens_generated, total_generation_time)}")
        print(f"💾 Peak memory usage: {memory_monitor.peak_memory:.2f} GB")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Stop memory monitoring
        memory_monitor.stop()

def main():
    parser = argparse.ArgumentParser(description="Optimized Chat Script for MLX on Apple Silicon")
    parser.add_argument("--model", required=True, help="Model name (e.g., llama-2-7b)")
    parser.add_argument("--quant", choices=["int4", "int8"], help="Quantization type (int4, int8)")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling (default: 0.95)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (default: 40)")
    parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repetition penalty (default: 1.1)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum tokens in context (default: 2048)")
    parser.add_argument("--system-prompt", help="Custom system prompt")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    args = parser.parse_args()
    
    # Print system info
    is_apple_silicon = detect_apple_silicon()
    if is_apple_silicon:
        # Try to get more detailed information
        try:
            import subprocess
            chip_info = subprocess.getoutput("sysctl -n machdep.cpu.brand_string")
            if "Apple" in chip_info:
                print(f"🖥️ System: {chip_info}")
            else:
                print(f"🖥️ System: Apple Silicon Mac")
        except:
            print(f"🖥️ System: Apple Silicon Mac")
    else:
        print(f"🖥️ System: {platform.platform()} ({platform.machine()})")
    
    # Print settings
    print("\n⚙️ Chat settings:")
    print(f"📦 Model: {args.model}")
    print(f"📊 Quantization: {args.quant or 'none (FP16)'}")
    print(f"🌡️ Temperature: {args.temp}")
    print(f"🎯 Top-p: {args.top_p}")
    print(f"🔝 Top-k: {args.top_k}")
    print(f"🔄 Repetition penalty: {args.repeat_penalty}")
    print(f"📏 Max tokens: {args.max_tokens}")
    
    # Run the chat
    run_optimized_chat(
        model_name=args.model,
        quantization=args.quant,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repeat_penalty,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        use_gpu=not args.no_gpu
    )

if __name__ == "__main__":
    main()
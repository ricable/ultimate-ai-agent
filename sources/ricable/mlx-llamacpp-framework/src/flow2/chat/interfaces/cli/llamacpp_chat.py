#!/usr/bin/env python3
"""
Command-line Chat Interface for llama.cpp

This script provides an interactive command-line chat interface for llama.cpp models.
It handles model loading, context management, and pretty-printed chat UI.
"""

import os
import sys
import argparse
import subprocess
import tempfile
import json
from typing import List, Dict, Optional, Any, Union, Tuple

# Add parent directory to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from common.chat_history import create_chat_session

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


def check_llamacpp_installation() -> Tuple[bool, str]:
    """
    Check if llama.cpp is installed and return the path to the executable.
    
    Returns:
        Tuple of (is_installed, path_to_executable)
    """
    # Check the expected location first
    llamacpp_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../llama.cpp-setup/build/main'
    ))
    
    if os.path.exists(llamacpp_path) and os.access(llamacpp_path, os.X_OK):
        return True, llamacpp_path
    
    # Try to find using which
    try:
        result = subprocess.run(
            ['which', 'main'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip()
    except:
        pass
    
    return False, ""


def list_available_models() -> List[str]:
    """
    List available model files in the models directory.
    
    Returns:
        List of model paths
    """
    models_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../llama.cpp-setup/models'
    ))
    
    if not os.path.exists(models_dir):
        return []
    
    return [
        os.path.join(models_dir, f) 
        for f in os.listdir(models_dir) 
        if f.endswith('.gguf')
    ]


def run_llamacpp(
    model_path: str, 
    prompt: str, 
    max_tokens: int = 1024,
    temperature: float = 0.7,
    use_metal: bool = True,
    streaming: bool = True,
    prompt_template: str = "chatml"
) -> str:
    """
    Run the llama.cpp model to generate text.
    
    Args:
        model_path: Path to the model file
        prompt: The prompt to send to the model
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        use_metal: Whether to use Metal acceleration on macOS
        streaming: Whether to stream the output token by token
        prompt_template: The prompt template type
        
    Returns:
        Generated text
    """
    # Check if llama.cpp is installed
    is_installed, llamacpp_path = check_llamacpp_installation()
    if not is_installed:
        raise RuntimeError(
            "llama.cpp executable not found. Please ensure llama.cpp is properly installed."
        )
    
    # Create a temporary file for the prompt
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(prompt)
        prompt_file = temp_file.name
    
    try:
        # Build the command
        cmd = [
            llamacpp_path,
            '-m', model_path,
            '-n', str(max_tokens),
            '--temp', str(temperature),
            '-f', prompt_file,
            '--color'
        ]
        
        if use_metal:
            cmd.extend(['--metal'])
        
        # Add template-specific parameters
        if prompt_template == "llama2":
            cmd.append('--format', 'llama2')
        
        # Run llama.cpp
        if streaming:
            # For streaming, we run with direct output to the terminal
            process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            output = ""
            stderr_output = ""
            
            # Capture stderr for error reporting
            for line in process.stderr:
                stderr_output += line
            
            process.wait()
            if process.returncode != 0:
                print(f"{COLORS['red']}Error running llama.cpp:{COLORS['reset']}")
                print(stderr_output)
        else:
            # For non-streaming, capture the output
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                print(f"{COLORS['red']}Error running llama.cpp:{COLORS['reset']}")
                print(result.stderr)
                return ""
            
            output = result.stdout
            
            # Extract just the assistant's response
            if prompt_template == "chatml":
                try:
                    output = output.split("<|im_start|>assistant\n", 1)[1]
                except IndexError:
                    pass
            elif prompt_template == "llama2":
                try:
                    output = output.split("[/INST] ", 1)[1]
                except IndexError:
                    pass
    
    finally:
        # Clean up the temporary file
        os.unlink(prompt_file)
    
    return output


def print_header():
    """Print the application header."""
    print(f"\n{COLORS['bold']}{COLORS['cyan']}┌─────────────────────────────────────────────┐{COLORS['reset']}")
    print(f"{COLORS['bold']}{COLORS['cyan']}│ llama.cpp Interactive Chat - Apple Silicon   │{COLORS['reset']}")
    print(f"{COLORS['bold']}{COLORS['cyan']}└─────────────────────────────────────────────┘{COLORS['reset']}\n")


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
    parser = argparse.ArgumentParser(description="Command-line Chat Interface for llama.cpp")
    parser.add_argument(
        "--model", "-m", 
        type=str,
        help="Path to the model file"
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
        "--no-metal", 
        action="store_true",
        help="Disable Metal acceleration"
    )
    parser.add_argument(
        "--history-file", 
        type=str,
        help="File to save chat history"
    )
    parser.add_argument(
        "--prompt-template", 
        type=str, 
        choices=["chatml", "llama2", "alpaca", "simple"], 
        default="chatml",
        help="Prompt template to use"
    )
    
    args = parser.parse_args()
    
    # If no model specified, try to find one
    if not args.model:
        models = list_available_models()
        if not models:
            print(f"{COLORS['red']}No models found. Please specify a model with --model.{COLORS['reset']}")
            return 1
        
        # Use the first model found
        args.model = models[0]
        print(f"{COLORS['yellow']}Using model: {args.model}{COLORS['reset']}")
    
    # Check if the model file exists
    if not os.path.exists(args.model):
        print(f"{COLORS['red']}Model file not found: {args.model}{COLORS['reset']}")
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
        "use_metal": not args.no_metal,
        "prompt_template": args.prompt_template,
    }
    
    # Print header and help
    print_header()
    print(f"{COLORS['green']}Type your message or use /help to see available commands.{COLORS['reset']}")
    print(f"{COLORS['green']}Press Ctrl+C at any time to stop generation.{COLORS['reset']}\n")
    
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
            
            # Generate assistant response
            print(f"\n{COLORS['bold']}{COLORS['green']}Assistant:{COLORS['reset']} ", end="", flush=True)
            
            # Get formatted prompt for the model
            formatted_prompt = chat_session.get_formatted_context(params["prompt_template"])
            
            try:
                response = run_llamacpp(
                    model_path=args.model,
                    prompt=formatted_prompt,
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    use_metal=params["use_metal"],
                    streaming=True,
                    prompt_template=params["prompt_template"]
                )
                
                # Since streaming outputs directly to console, we need to capture the 
                # output separately to add to history
                
                # For simplicity, we'll prompt for the output
                print("\n")
                print(f"{COLORS['yellow']}Please copy-paste the assistant's response to save it to history.{COLORS['reset']}")
                print(f"{COLORS['yellow']}(Press Enter on an empty line to finish){COLORS['reset']}")
                
                assistant_response = []
                while True:
                    line = input()
                    if not line:
                        break
                    assistant_response.append(line)
                
                # Add the response to history
                chat_session.add_assistant_message("\n".join(assistant_response))
                
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
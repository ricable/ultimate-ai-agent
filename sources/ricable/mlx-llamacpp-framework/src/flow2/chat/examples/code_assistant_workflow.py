#!/usr/bin/env python3
"""
Code Assistant Workflow Example

This script demonstrates using a local LLM as a coding assistant.
It shows how to structure prompts for code generation, explanation, and debugging.
"""

import os
import sys
import argparse
import subprocess
import tempfile

# Add parent directory to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.chat_history import create_chat_session

# Define ANSI colors for terminal output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def run_llm_query(
    model_path, 
    prompt, 
    framework="llama.cpp", 
    max_tokens=1024, 
    temperature=0.2
):
    """Run a query against a local LLM using the specified framework."""
    if framework == "llama.cpp":
        # Create a temporary file for the prompt
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(prompt)
            prompt_file = temp_file.name
        
        try:
            # Determine llama.cpp path
            llamacpp_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                '../../llama.cpp-setup/build/main'
            ))
            
            # Run llama.cpp
            cmd = [
                llamacpp_path,
                '-m', model_path,
                '-n', str(max_tokens),
                '--temp', str(temperature),
                '-f', prompt_file,
                '--color'
            ]
            
            # Add Metal acceleration if on macOS
            if sys.platform == "darwin":
                cmd.append('--metal')
            
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
            
            # Extract the assistant's response based on the template
            output = result.stdout
            try:
                output = output.split("<|im_start|>assistant\n", 1)[1].split("<|im_end|>", 1)[0]
            except IndexError:
                pass
                
            return output
            
        finally:
            # Clean up the temporary file
            os.unlink(prompt_file)
            
    elif framework == "mlx":
        try:
            # Import MLX modules
            import mlx.core as mx
            from mlx_lm import load, generate
            
            # Load the model
            model, tokenizer = load(model_path)
            
            # Format as a list of messages for MLX
            messages = []
            lines = prompt.split('\n')
            current_role = None
            current_content = []
            
            for line in lines:
                if "<|im_start|>system" in line:
                    current_role = "system"
                    current_content = []
                elif "<|im_start|>user" in line:
                    if current_role:
                        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role = "user"
                    current_content = []
                elif "<|im_start|>assistant" in line:
                    if current_role:
                        messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
                    current_role = "assistant"
                    current_content = []
                elif "<|im_end|>" in line:
                    continue
                else:
                    current_content.append(line)
            
            if current_role and current_content:
                messages.append({"role": current_role, "content": "\n".join(current_content).strip()})
            
            # Construct the prompt
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
            
            # Generate text
            tokens = generate(
                model, 
                tokenizer, 
                prompt, 
                max_tokens=max_tokens, 
                temp=temperature
            )
            
            return tokenizer.decode(tokens)
            
        except ImportError:
            print(f"{COLORS['red']}Error: MLX is not installed. Install with: pip install mlx mlx-lm{COLORS['reset']}")
            return ""
    
    return ""


def code_generation_example(model_path, framework="llama.cpp"):
    """Example workflow for generating code with a local LLM."""
    print(f"{COLORS['bold']}{COLORS['cyan']}Running Code Generation Example{COLORS['reset']}\n")
    
    # Create a chat session with appropriate system message for coding
    chat = create_chat_session(
        system_message=(
            "You are an expert Python programmer specializing in data analysis and visualization. "
            "Write clean, efficient, and well-documented code. "
            "Include helpful comments explaining complex parts. "
            "For any code you provide, explain how it works after the code block."
        )
    )
    
    # Add user query for code generation
    chat.add_user_message(
        "Create a Python function that reads a CSV file containing stock prices "
        "(columns: Date, Open, High, Low, Close, Volume) and plots a candlestick chart "
        "using matplotlib. Include proper error handling and documentation."
    )
    
    # Format the prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Generating code...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=1024,
        temperature=0.2  # Lower temperature for code generation
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Generated Code:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    # Follow-up question about the code
    chat.add_user_message(
        "Can you modify the function to also calculate and display a 20-day moving average?"
    )
    
    # Format the updated prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Generating code modification...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=1024,
        temperature=0.2
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Modified Code with Moving Average:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    return chat


def code_explanation_example(model_path, framework="llama.cpp"):
    """Example workflow for explaining code with a local LLM."""
    print(f"{COLORS['bold']}{COLORS['cyan']}Running Code Explanation Example{COLORS['reset']}\n")
    
    # Create a chat session with appropriate system message for code explanation
    chat = create_chat_session(
        system_message=(
            "You are an expert programmer who specializes in explaining code clearly. "
            "Break down complex concepts into simple terms and provide step-by-step explanations. "
            "Use examples to illustrate how the code works when helpful."
        )
    )
    
    # Example code to explain (a complex Python decorator)
    complex_code = """
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    time.sleep(delay)
                    print(f"Retrying {func.__name__}, attempt {attempts}/{max_attempts}")
            return None
        return wrapper
    return decorator

@retry(max_attempts=5, delay=2)
def fetch_data(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
"""
    
    # Add user query for code explanation
    chat.add_user_message(
        f"Can you explain how this Python decorator works? I find decorators confusing.\n\n```python\n{complex_code}\n```"
    )
    
    # Format the prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Generating explanation...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=1024,
        temperature=0.3  # Slightly higher temperature for explanations
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Code Explanation:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    # Follow-up question about decorators
    chat.add_user_message(
        "Could you provide a simpler example of a decorator that just times how long a function takes to run?"
    )
    
    # Format the updated prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Generating simple decorator example...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=1024,
        temperature=0.3
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Simple Decorator Example:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    return chat


def code_debugging_example(model_path, framework="llama.cpp"):
    """Example workflow for debugging code with a local LLM."""
    print(f"{COLORS['bold']}{COLORS['cyan']}Running Code Debugging Example{COLORS['reset']}\n")
    
    # Create a chat session with appropriate system message for debugging
    chat = create_chat_session(
        system_message=(
            "You are an expert programmer specializing in debugging and fixing code issues. "
            "When presented with code that has bugs or issues, analyze it carefully, "
            "explain what's wrong, and provide a corrected version. "
            "Be thorough in your analysis and explain your debugging process."
        )
    )
    
    # Example code with bugs
    buggy_code = """
def calculate_statistics(numbers):
    # Calculate mean, median, and mode for a list of numbers
    total = 0
    for num in numbers:
        total += num
    mean = total / len(numbers)
    
    # Calculate median
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        median = (sorted_numbers[n/2 - 1] + sorted_numbers[n/2]) / 2
    else:
        median = sorted_numbers[n/2]
    
    # Calculate mode
    frequency = {}
    for num in numbers:
        if num in frequency:
            frequency[num] += 1
        else:
            frequency[num] = 1
    mode = max(frequency.items(), key=lambda x: x[1])
    
    return {
        'mean': mean,
        'median': median,
        'mode': mode
    }

# Test the function
test_data = [1, 2, 3, 4, 4, 5, 5, 5, 6, 7]
results = calculate_statistics(test_data)
print(f"Mean: {results['mean']}")
print(f"Median: {results['median']}")
print(f"Mode: {results['mode']}")
"""
    
    # Add user query for debugging
    chat.add_user_message(
        f"This Python function is supposed to calculate the mean, median, and mode of a list of numbers, but it has bugs. Can you fix it?\n\n```python\n{buggy_code}\n```"
    )
    
    # Format the prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Debugging code...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=1536,  # More tokens for detailed debugging
        temperature=0.2   # Lower temperature for precise debugging
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Debugging Analysis and Fixed Code:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    # Follow-up question
    chat.add_user_message(
        "Can you also add error handling to check if the input list is empty or contains non-numeric values?"
    )
    
    # Format the updated prompt and run the query
    prompt = chat.get_formatted_context("chatml")
    
    print(f"{COLORS['yellow']}Adding error handling...{COLORS['reset']}\n")
    
    response = run_llm_query(
        model_path=model_path,
        prompt=prompt,
        framework=framework,
        max_tokens=1536,
        temperature=0.2
    )
    
    # Add the response to the chat history
    chat.add_assistant_message(response)
    
    # Display the result
    print(f"{COLORS['green']}Code with Error Handling:{COLORS['reset']}\n")
    print(response)
    print("\n")
    
    return chat


def main():
    """Main function to run the code assistant workflow examples."""
    parser = argparse.ArgumentParser(description="Code Assistant Workflow Example")
    parser.add_argument(
        "--model", "-m", 
        type=str,
        help="Path to the model file or directory"
    )
    parser.add_argument(
        "--framework", "-f", 
        type=str, 
        choices=["llama.cpp", "mlx"],
        default="llama.cpp",
        help="Framework to use (llama.cpp or mlx)"
    )
    parser.add_argument(
        "--example", "-e", 
        type=str, 
        choices=["generation", "explanation", "debugging", "all"],
        default="all",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    # If no model specified, look for available models
    if not args.model:
        if args.framework == "llama.cpp":
            models_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                '../../llama.cpp-setup/models'
            ))
            models = [
                os.path.join(models_dir, f) 
                for f in os.listdir(models_dir) 
                if f.endswith('.gguf')
            ] if os.path.exists(models_dir) else []
        else:  # MLX
            models_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                '../../mlx-setup/models'
            ))
            models = [
                os.path.join(models_dir, d) 
                for d in os.listdir(models_dir) 
                if os.path.isdir(os.path.join(models_dir, d)) and 
                os.path.exists(os.path.join(models_dir, d, 'config.json'))
            ] if os.path.exists(models_dir) else []
        
        if not models:
            print(f"{COLORS['red']}No models found. Please specify a model with --model.{COLORS['reset']}")
            return 1
        
        # Use the first model found
        args.model = models[0]
        print(f"{COLORS['yellow']}Using model: {args.model}{COLORS['reset']}")
    
    # Run the selected example(s)
    if args.example in ["generation", "all"]:
        code_generation_example(args.model, args.framework)
    
    if args.example in ["explanation", "all"]:
        code_explanation_example(args.model, args.framework)
    
    if args.example in ["debugging", "all"]:
        code_debugging_example(args.model, args.framework)
    
    print(f"{COLORS['bold']}{COLORS['green']}All examples completed!{COLORS['reset']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
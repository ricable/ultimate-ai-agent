#!/usr/bin/env python3
"""
OpenAI Compatible Demo - Shows how to use the same code with OpenAI and Ollama

This demonstrates how OpenAI's Python client can be used with any OpenAI-compatible
API endpoint, including Ollama's OpenAI compatibility layer.

Example of OpenAI compatibility for Ollama:

from openai import OpenAI
client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def demo_basic_completion(client: OpenAI, model: str):
    """Basic completion example"""
    print(f"\n{'='*50}")
    print(f"Basic Completion with {model}")
    print('='*50)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain in one sentence what OpenAI compatibility means."}
        ],
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Model used: {response.model}")
    print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")


def demo_streaming(client: OpenAI, model: str):
    """Streaming example"""
    print(f"\n{'='*50}")
    print(f"Streaming Response with {model}")
    print('='*50)
    
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Count from 1 to 5 with a fun fact about each number."}
        ],
        stream=True,
        temperature=0.7
    )
    
    print("Response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def demo_conversation(client: OpenAI, model: str):
    """Multi-turn conversation example"""
    print(f"\n{'='*50}")
    print(f"Multi-turn Conversation with {model}")
    print('='*50)
    
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "What's a Python decorator?"},
    ]
    
    # First turn
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    
    print(f"User: {messages[-1]['content']}")
    print(f"Assistant: {response.choices[0].message.content}")
    
    # Add response to conversation
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    messages.append({"role": "user", "content": "Can you show me a simple example?"})
    
    # Second turn
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"\nUser: {messages[-1]['content']}")
    print(f"Assistant: {response.choices[0].message.content}")


def main():
    print("OpenAI Compatible API Demo")
    print("This demo shows how the same code works with both OpenAI and Ollama\n")
    
    # Configuration for different providers
    configs = [
        {
            "name": "OpenAI",
            "base_url": "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY", "your-openai-key"),
            "model": "gpt-4.1-nano",
            "enabled": os.getenv("OPENAI_API_KEY") is not None
        },
        {
            "name": "Ollama through OpenAI Compatibility",
            "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            "api_key": os.getenv("LLM_API_KEY", "ollama"),
            "model": os.getenv("LLM_CHOICE", "qwen3:14b"),
            "enabled": os.getenv("LLM_BASE_URL") is not None
        }
    ]
    
    # Let user choose which provider to use
    print("Available providers:")
    available_configs = []
    for i, config in enumerate(configs):
        if config["enabled"]:
            available_configs.append(config)
            print(f"{len(available_configs)}. {config['name']} (Model: {config['model']})")
    
    if not available_configs:
        print("\nNo providers configured! Please set up at least one provider.")
        print("For Ollama: Make sure Ollama is running locally")
        print("For OpenAI: Set OPENAI_API_KEY environment variable")
        return
    
    # Get user choice
    try:
        choice = input(f"\nSelect provider (1-{len(available_configs)}): ")
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(available_configs):
            print("Invalid choice. Using first available provider.")
            choice_idx = 0
    except (ValueError, EOFError):
        print("\nUsing first available provider.")
        choice_idx = 0
    
    selected_config = available_configs[choice_idx]
    
    print(f"\nUsing {selected_config['name']} with model: {selected_config['model']}")
    print(f"Base URL: {selected_config['base_url']}")
    
    # Create client with selected configuration
    client = OpenAI(
        base_url=selected_config["base_url"],
        api_key=selected_config["api_key"]
    )
    
    # Run demos
    try:
        # Basic completion
        demo_basic_completion(client, selected_config["model"])
        
        # Streaming
        input("\nPress Enter to see streaming demo...")
        demo_streaming(client, selected_config["model"])
        
        # Multi-turn conversation
        input("\nPress Enter to see conversation demo...")
        demo_conversation(client, selected_config["model"])
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("- For Ollama: Make sure Ollama is running (ollama serve)")
        print("- For Ollama: Make sure the model is pulled (ollama pull model-name)")
        print("- For OpenAI: Check your API key is valid")
        print("- Check the base URL is correct")


if __name__ == "__main__":
    main()
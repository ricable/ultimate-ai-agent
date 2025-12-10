import os
import sys
from pathlib import Path

# Add sparc_cli to Python path
sparc_cli_path = str(Path(__file__).parent.parent.parent)
sys.path.append(sparc_cli_path)

from sparc_cli.llm import initialize_llm
from sparc_cli.polaris import create_polaris_model
from langchain_core.messages import HumanMessage

def main():
    # Initialize base model
    base_model = initialize_llm("openai", "gpt-3.5-turbo")
    
    # Create PolarisOne-enhanced model
    polaris_model = create_polaris_model(base_model)
    
    # Test message
    message = "What is the capital of France? I need to know for my geography homework."
    messages = [HumanMessage(content=message)]
    
    # Generate response with token weights
    response, token_weights = polaris_model.generate_with_weights(messages)
    
    # Print results
    print("\nInput message with token weights:")
    for token, weight in token_weights:
        print(f"{token}: {weight:.3f}")
    
    print("\nModel response:")
    print(response)

if __name__ == "__main__":
    main()

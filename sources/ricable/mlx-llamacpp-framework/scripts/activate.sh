#!/bin/bash
# Activate the virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Virtual environment activated!"
echo "You can now run the LLM tools."
echo "Example commands:"
echo "  - python inference_scripts/mlx/inference.py --help"
echo "  - python inference_scripts/llama_cpp/inference.py --help"
echo "  - python chat_interfaces/mlx/cli/chat_cli.py --help"

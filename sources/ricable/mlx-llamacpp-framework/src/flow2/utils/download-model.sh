#!/bin/bash
# Simple wrapper script for the model manager CLI

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Display help if no arguments provided or if help is requested
if [ $# -eq 0 ] || [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $(basename "$0") <model_name> [options]"
    echo ""
    echo "Download an LLM model for llama.cpp or MLX framework."
    echo ""
    echo "Examples:"
    echo "  $(basename "$0") llama2-7b-gguf --framework llama.cpp --quant q4_k_m"
    echo "  $(basename "$0") mistral-7b-mlx --framework mlx --quant int4"
    echo ""
    echo "Options:"
    echo "  --framework <framework>   Specify the framework: 'llama.cpp' or 'mlx'"
    echo "  --quant <quantization>    Specify quantization level"
    echo "  --list                    List available models"
    echo "  --recommend               Show recommended models for your system"
    echo ""
    echo "See 'model_cli.py --help' for more options."
    exit 0
fi

# If first argument is 'list', forward to list command
if [ "$1" == "list" ]; then
    python3 "$SCRIPT_DIR/model_cli.py" list "${@:2}"
    exit $?
fi

# If first argument is 'recommend', forward to recommend command
if [ "$1" == "recommend" ]; then
    python3 "$SCRIPT_DIR/model_cli.py" recommend "${@:2}"
    exit $?
fi

# Forward all arguments to the model CLI
python3 "$SCRIPT_DIR/model_cli.py" download "$@"
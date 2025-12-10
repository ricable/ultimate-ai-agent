#!/bin/bash
# =================================================================
# Optimized Chat Script for llama.cpp on Apple Silicon
# =================================================================
# This script provides an optimized chat experience with llama.cpp
# using Metal GPU acceleration and optimal settings for performance.
# 
# Usage: ./optimized_chat.sh <model_path> [context_length] [batch_size] [threads]
#
# Example: ./optimized_chat.sh models/llama-2-7b-chat-q4_k.gguf 4096 512 4

# Set default values
DEFAULT_MODEL="models/llama-2-7b-chat-q4_k.gguf"
DEFAULT_CONTEXT_LENGTH=4096
DEFAULT_BATCH_SIZE=512
DEFAULT_THREADS=4

# Set LLAMA_CPP_PATH to the directory containing the llama.cpp binaries
# Modify this to point to your llama.cpp installation
LLAMA_CPP_PATH="../../llama.cpp"

# Check if llama.cpp path exists
if [ ! -d "$LLAMA_CPP_PATH" ]; then
    echo "Error: llama.cpp directory not found at $LLAMA_CPP_PATH"
    echo "Please update the LLAMA_CPP_PATH variable in this script."
    exit 1
fi

# Parse command line arguments
MODEL_PATH=${1:-$DEFAULT_MODEL}
CONTEXT_LENGTH=${2:-$DEFAULT_CONTEXT_LENGTH}
BATCH_SIZE=${3:-$DEFAULT_BATCH_SIZE}
THREADS=${4:-$DEFAULT_THREADS}

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

# Detect Apple Silicon and optimize settings
IS_APPLE_SILICON=0
if [ "$(uname)" == "Darwin" ]; then
    if [ "$(uname -m)" == "arm64" ]; then
        IS_APPLE_SILICON=1
        echo "✅ Apple Silicon detected. Metal acceleration will be enabled."
    else
        echo "⚠️ Intel Mac detected. Performance may be limited."
    fi
else
    echo "⚠️ Non-macOS platform detected. Performance may be limited."
fi

# Determine model size from filename
MODEL_SIZE="7B"
if [[ "$MODEL_PATH" =~ 13[bB] ]]; then
    MODEL_SIZE="13B"
elif [[ "$MODEL_PATH" =~ 70[bB] ]]; then
    MODEL_SIZE="70B"
elif [[ "$MODEL_PATH" =~ 33[bB] ]] || [[ "$MODEL_PATH" =~ 34[bB] ]]; then
    MODEL_SIZE="33B"
fi

# Determine if the model is quantized from filename
IS_QUANTIZED=0
if [[ "$MODEL_PATH" =~ [qQ][0-9] ]]; then
    IS_QUANTIZED=1
    echo "📊 Quantized model detected."
    
    # Extract quantization type
    if [[ "$MODEL_PATH" =~ [qQ]4 ]]; then
        QUANT_TYPE="Q4"
    elif [[ "$MODEL_PATH" =~ [qQ]8 ]]; then
        QUANT_TYPE="Q8"
    elif [[ "$MODEL_PATH" =~ [qQ]5 ]]; then
        QUANT_TYPE="Q5"
    elif [[ "$MODEL_PATH" =~ [qQ]6 ]]; then
        QUANT_TYPE="Q6"
    elif [[ "$MODEL_PATH" =~ [qQ]2 ]]; then
        QUANT_TYPE="Q2"
    elif [[ "$MODEL_PATH" =~ [qQ]3 ]]; then
        QUANT_TYPE="Q3"
    else
        QUANT_TYPE="Unknown"
    fi
    echo "📊 Quantization type: $QUANT_TYPE"
fi

# Get available memory
if [ "$(uname)" == "Darwin" ]; then
    AVAILABLE_MEM=$(sysctl -n hw.memsize)
    AVAILABLE_MEM_GB=$((AVAILABLE_MEM / 1024 / 1024 / 1024))
    echo "💾 Available memory: $AVAILABLE_MEM_GB GB"
else
    AVAILABLE_MEM_GB=16  # Default assumption
fi

# Check if memory is likely to be sufficient
MINIMUM_MEM=0
if [ "$MODEL_SIZE" == "7B" ]; then
    if [ "$IS_QUANTIZED" == 1 ] && [[ "$QUANT_TYPE" =~ Q[2-4] ]]; then
        MINIMUM_MEM=8
    else
        MINIMUM_MEM=16
    fi
elif [ "$MODEL_SIZE" == "13B" ]; then
    if [ "$IS_QUANTIZED" == 1 ] && [[ "$QUANT_TYPE" =~ Q[2-4] ]]; then
        MINIMUM_MEM=16
    else
        MINIMUM_MEM=32
    fi
elif [ "$MODEL_SIZE" == "33B" ]; then
    if [ "$IS_QUANTIZED" == 1 ] && [[ "$QUANT_TYPE" =~ Q[2-4] ]]; then
        MINIMUM_MEM=32
    else
        MINIMUM_MEM=64
    fi
elif [ "$MODEL_SIZE" == "70B" ]; then
    if [ "$IS_QUANTIZED" == 1 ] && [[ "$QUANT_TYPE" =~ Q[2-4] ]]; then
        MINIMUM_MEM=64
    else
        MINIMUM_MEM=128
    fi
fi

if [ "$AVAILABLE_MEM_GB" -lt "$MINIMUM_MEM" ]; then
    echo "⚠️ Warning: Your system may not have enough memory for this model."
    echo "⚠️ Recommended minimum: $MINIMUM_MEM GB for $MODEL_SIZE $QUANT_TYPE"
    echo "⚠️ Continuing anyway, but you may experience out-of-memory errors."
    
    # Reduce context length if memory is tight
    if [ "$CONTEXT_LENGTH" -gt 2048 ]; then
        CONTEXT_LENGTH=2048
        echo "⚠️ Reducing context length to $CONTEXT_LENGTH to save memory."
    fi
fi

# Prepare chat template
CHAT_TEMPLATE=$(cat << 'EOT'
<|im_start|>system
You are a helpful, respectful assistant. Always provide accurate information and admit when you don't know something rather than making up information.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
EOT
)

# Create a temporary file for the chat template
TEMP_TEMPLATE_FILE=$(mktemp)
echo "$CHAT_TEMPLATE" > "$TEMP_TEMPLATE_FILE"

# Build the command
CMD="$LLAMA_CPP_PATH/main"
CMD_ARGS="-m \"$MODEL_PATH\" -c $CONTEXT_LENGTH -b $BATCH_SIZE -t $THREADS --color --interactive -f \"$TEMP_TEMPLATE_FILE\""

# Add Metal flags if on Apple Silicon
if [ "$IS_APPLE_SILICON" == 1 ]; then
    CMD_ARGS="$CMD_ARGS --metal --metal-mmq"
fi

# Print info
echo "🚀 Starting optimized chat with the following settings:"
echo "📦 Model: $MODEL_PATH"
echo "📏 Context Length: $CONTEXT_LENGTH tokens"
echo "📊 Batch Size: $BATCH_SIZE"
echo "🧵 Threads: $THREADS"
if [ "$IS_APPLE_SILICON" == 1 ]; then
    echo "🔥 Metal Acceleration: Enabled"
else
    echo "🔥 Metal Acceleration: Disabled (not on Apple Silicon)"
fi

echo ""
echo "💬 Starting chat session... (Press Ctrl+C to exit)"
echo "===================================================="
echo ""

# Execute the command
eval "$CMD $CMD_ARGS"

# Clean up the temporary file
rm "$TEMP_TEMPLATE_FILE"
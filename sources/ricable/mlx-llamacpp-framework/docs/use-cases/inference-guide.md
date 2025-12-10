# Inference Guide: Running LLMs on Apple Silicon

This guide covers how to run inference with large language models on Apple Silicon hardware using both llama.cpp and MLX frameworks.

## Table of Contents
- [Basic Inference Concepts](#basic-inference-concepts)
- [llama.cpp Inference](#llamacpp-inference)
- [MLX Inference](#mlx-inference)
- [Performance Optimization](#performance-optimization)
- [Advanced Inference Techniques](#advanced-inference-techniques)
- [Troubleshooting](#troubleshooting)

## Basic Inference Concepts

### What is Inference?

Inference is the process of using a trained language model to generate text based on a given prompt. This can include:
- Text completion
- Question answering
- Chat responses
- Content generation
- Summarization

### Key Inference Parameters

| Parameter | Description | Impact |
|-----------|-------------|--------|
| Temperature | Controls randomness (0.0-1.0) | Higher = more creative, lower = more deterministic |
| Top-p (nucleus sampling) | Controls token selection diversity | Higher = more diverse outputs |
| Repetition penalty | Reduces word repetition | Higher = fewer repetitions |
| Context length | Maximum tokens model can "remember" | Higher = more context but more memory usage |
| Max tokens | Maximum length of generated text | Higher = longer outputs |

## llama.cpp Inference

### Basic Command-Line Generation

```bash
# Simple text completion
./main -m models/llama-2-7b-q4_k.gguf -p "Tell me about Apple Silicon" -n 256

# With Metal GPU acceleration
./main -m models/llama-2-7b-q4_k.gguf --metal -p "Tell me about Apple Silicon" -n 256
```

### Interactive Mode

```bash
# Basic interactive mode
./main -m models/llama-2-7b-q4_k.gguf --metal --interactive --color

# With specific chat format
./main -m models/llama-2-7b-q4_k.gguf --metal --interactive --color -f prompts/chat-with-bob.txt
```

### Server Mode

```bash
# Run server
./server -m models/llama-2-7b-q4_k.gguf --metal -c 2048 --host 0.0.0.0 --port 8080

# Access via curl
curl -X POST http://localhost:8080/completion -d '{
  "prompt": "What is Apple Silicon?",
  "n_predict": 128
}'
```

### Embedding Generation

```bash
# Generate embeddings for text
./embedding -m models/llama-2-7b-q4_k.gguf -p "This is a text for embedding"
```

### Advanced llama.cpp Parameters

```bash
./main \
  -m models/llama-2-7b-q4_k.gguf \
  --metal \            # Enable Metal acceleration
  --metal-mmq \        # Enable Metal matrix multiplication
  -t 4 \               # Number of threads
  -c 2048 \            # Context size
  -b 512 \             # Batch size
  --temp 0.7 \         # Temperature
  --top-p 0.95 \       # Top-p sampling
  --repeat_penalty 1.1 # Repetition penalty
  -p "Your prompt here" \
  -n 512               # Generate 512 tokens
```

### Python Bindings

```python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path="models/llama-2-7b-q4_k.gguf",
    n_gpu_layers=-1,  # Use all layers on GPU
    n_ctx=2048        # Context window size
)

# Simple generation
output = llm(
    "Tell me about Apple Silicon",
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
    repeat_penalty=1.1
)
print(output["choices"][0]["text"])

# Chat completion
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Apple Silicon?"}
]
output = llm.create_chat_completion(
    messages,
    max_tokens=512,
    temperature=0.7
)
print(output["choices"][0]["message"]["content"])
```

## MLX Inference

### Basic Text Generation

```python
import mlx.core as mx
from mlx_lm import load, generate

# Load model
model, tokenizer = load("llama-2-7b", quantization="int4")

# Generate text
prompt = "Tell me about Apple Silicon"
tokens = generate(model, tokenizer, prompt, max_tokens=256)
print(tokenizer.decode(tokens))
```

### Interactive Chat

```python
from mlx_lm.utils import chat

# Load model
model, tokenizer = load("llama-2-7b", quantization="int4")

# Start interactive chat
chat(model, tokenizer)
```

### Server Mode

```python
from mlx_lm.serve import app
from mlx_lm import load

# Load model
model, tokenizer = load("llama-2-7b", quantization="int4")

# Set model and tokenizer for the server
app.model = model
app.tokenizer = tokenizer

# Run server
app.run(host="0.0.0.0", port=8080)
```

### Advanced MLX Parameters

```python
# Full control over generation
import mlx.core as mx
from mlx_lm import load, generate

# Set GPU as default device
mx.set_default_device(mx.gpu)

# Load model with optimized settings
model, tokenizer = load(
    "llama-2-7b",
    quantization="int4",
    max_tokens=2048
)

# Advanced generation settings
tokens = generate(
    model,
    tokenizer,
    "Explain quantum computing",
    max_tokens=512,
    temp=0.8,            # Temperature
    top_p=0.95,          # Top-p sampling
    top_k=40,            # Top-k sampling
    repetition_penalty=1.2,  # Repetition penalty
    batch_size=32,       # Batch size for efficiency
    stream=True          # Stream tokens as they're generated
)
```

### Custom Inference Loop

```python
import mlx.core as mx
from mlx_lm import load

# Load model
model, tokenizer = load("llama-2-7b", quantization="int4")

# Encode prompt
prompt = "The best thing about Apple Silicon is"
input_ids = mx.array([tokenizer.encode(prompt)])

# Manual generation loop
generated_ids = list(input_ids[0].tolist())
for _ in range(100):  # Generate 100 tokens
    # Get predictions for next token
    logits = model(input_ids)[:, -1, :]
    
    # Apply temperature
    logits = logits / 0.7
    
    # Sample from distribution
    next_token = mx.random.categorical(logits)
    
    # Add to generated sequence
    generated_ids.append(next_token.item())
    input_ids = mx.array([[next_token.item()]])
    
    # Print token as it's generated
    print(tokenizer.decode([next_token.item()]), end="", flush=True)
```

## Performance Optimization

### Memory-Efficiency Tips

1. **Use Appropriate Quantization**
   - For 8GB RAM: Use INT4/Q4_K quantization
   - For 16GB RAM: Use INT8/Q8_0 for 7B models, INT4/Q4_K for larger models
   - For 32GB+ RAM: Can use FP16 for smaller models

2. **Adjust Context Length**
   - Only use as much context as needed
   - Reducing from 4096 to 2048 tokens can save significant memory

3. **Batch Processing**
   - For processing multiple prompts, use batching for efficiency
   - Adjust batch size based on available memory

### Speed Optimization

1. **Enable GPU Acceleration**
   - Always use `--metal` for llama.cpp
   - For MLX, use `mx.set_default_device(mx.gpu)`

2. **Thread Configuration**
   - For llama.cpp, set `-t` to match your CPU core count
   - Typically 4-8 for M1/M2, 8-16 for M1/M2 Max/Ultra

3. **Quantization Trade-offs**
   - INT4/Q4_K offers best balance of speed and quality
   - INT8/Q8_0 is slightly slower but higher quality

## Advanced Inference Techniques

### Prompt Engineering

Effective prompts can dramatically improve output quality:

```
# Simple prompt
"Tell me about Apple Silicon"

# Better prompt with context and format specification
"You are an expert in computer hardware. Provide a detailed, technical explanation of Apple Silicon, including its architecture, performance characteristics, and advantages over Intel processors. Format your response with sections and bullet points where appropriate."
```

### System Prompts

For chat models, use system prompts to control behavior:

```python
# llama.cpp
./main -m models/llama-2-7b-q4_k.gguf --metal -f prompts/system-prompt.txt

# MLX
system_prompt = "You are a helpful, accurate, and concise assistant."
user_prompt = "What is Apple Silicon?"
full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
```

### Long-Context Handling

For working with longer contexts:

```bash
# llama.cpp - Increase context window
./main -m models/llama-2-7b-q4_k.gguf --metal -c 4096 -p "Very long prompt here..."

# MLX - Process long documents in chunks
def process_long_document(document, chunk_size=1000, overlap=100):
    chunks = []
    for i in range(0, len(document), chunk_size - overlap):
        chunks.append(document[i:i + chunk_size])
    
    responses = []
    for chunk in chunks:
        tokens = generate(model, tokenizer, f"Summarize: {chunk}", max_tokens=256)
        responses.append(tokenizer.decode(tokens))
    
    return responses
```

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Errors**
   - Try more aggressive quantization (Q4_K or Q2_K)
   - Reduce context length
   - Reduce number of threads
   - Use a smaller model

2. **Slow Generation**
   - Ensure GPU acceleration is enabled
   - Check if Metal is being utilized
   - Try different batch sizes
   - Consider more efficient quantization

3. **Poor Quality Output**
   - Use less aggressive quantization (Q6_K or Q8_0)
   - Improve your prompts with more context
   - Try a larger model if hardware allows
   - Adjust temperature (0.7 is a good starting point)

4. **Crashing on Startup**
   - Verify model file is not corrupted
   - Check for sufficient disk space
   - Ensure your hardware meets minimum requirements

### Diagnostic Commands

```bash
# llama.cpp memory diagnosis
./main -m models/llama-2-7b-q4_k.gguf --metal -p "test" -n 1 -ngl 0 --memory-f32 --memory-print

# MLX memory inspection
python -c "
import mlx.core as mx
print(f'Available memory: {mx.get_available_memory() / (1024**3):.2f} GB')
"
```

## Next Steps

- [Chat Applications Guide](chat-applications.md) - Build interactive chat interfaces
- [Fine-tuning Guide](fine-tuning-guide.md) - Personalize models with your data
- [Hardware Recommendations](../hardware/hardware-recommendations.md) - Choose the right Mac for your needs
# Complete Guide to MLX on Apple Silicon

This guide provides a comprehensive overview of using Apple's MLX framework for large language models on Apple Silicon Macs.

## What is MLX?

MLX is Apple's open-source machine learning framework specifically designed for Apple Silicon hardware. It provides a Python-first API similar to PyTorch and JAX, with native optimization for Apple's M-series chips.

Key features:
- Native Apple Silicon optimization
- Unified memory model between CPU and GPU
- Python-first API (similar to PyTorch/JAX)
- Built-in quantization and fine-tuning support
- Metal acceleration built-in
- Direct Apple support and development

## Installation

### Prerequisites

- Apple Silicon Mac (M1/M2/M3)
- macOS 12+ (Monterey or newer)
- Python 3.8+

### Basic Installation

```bash
# Create a virtual environment (recommended)
python -m venv mlx-env
source mlx-env/bin/activate

# Install MLX
pip install mlx

# Install MLX-LM for language models
pip install mlx-lm
```

### Installing from Source (Optional)

```bash
# Clone the repository
git clone https://github.com/ml-explore/mlx.git
cd mlx

# Build and install
pip install -e .
```

## Model Management

### Supported Model Architectures

MLX supports a wide range of model architectures:
- LLaMA (1, 2, and 3)
- Mistral
- Phi
- Stable Diffusion
- And many other architectures

### Downloading Models

There are several ways to get models:

```bash
# Using mlx-lm
python -m mlx_lm.download --model llama-2-7b

# Converting from Hugging Face
python -m mlx_lm.convert --hf-path meta-llama/Llama-2-7b --mlx-path llama-2-7b
```

### Quantizing Models

MLX supports INT4 and INT8 quantization:

```python
# Load with quantization
from mlx_lm import load

# Load with 4-bit quantization
model, tokenizer = load("llama-2-7b", quantization="int4")

# Or quantize after loading
from mlx_lm.utils import load_model
from mlx_lm.quantize import quantize_model

model, tokenizer = load_model("llama-2-7b")
model = quantize_model(model, nbits=4, group_size=64)

# Save quantized model
import mlx.core as mx
mx.save("llama-2-7b-int4.npz", model.parameters())
```

## Basic Usage

### Simple Text Generation

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

### Key Parameters Explained

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model` | Path to model directory | `"llama-2-7b"` |
| `quantization` | Quantization level | `"int4"`, `"int8"`, `None` |
| `max_tokens` | Max tokens to generate | `256` |
| `temp` | Temperature (randomness) | `0.7` |
| `top_p` | Top-p sampling | `0.95` |
| `repetition_penalty` | Repetition penalty | `1.1` |
| `batch_size` | Batch size for generation | `32` |

## Advanced Usage

### Server Mode

Run an API server for your model:

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

Then access via HTTP:
```bash
curl -X POST http://localhost:8080/generate -H "Content-Type: application/json" -d '{
  "prompt": "Tell me about Apple Silicon",
  "max_tokens": 256
}'
```

### Customizing Generation Parameters

```python
# More advanced generation settings
tokens = generate(
    model,
    tokenizer,
    "Explain quantum computing",
    max_tokens=512,
    temp=0.8,
    top_p=0.95,
    repetition_penalty=1.2,
    top_k=40,
    stream=True  # Stream tokens as they're generated
)
```

### Performance Optimization

```python
import mlx.core as mx

# Set default device to GPU for all operations
mx.set_default_device(mx.gpu)

# Control memory usage
mx.set_allocation_limit(0.8)  # Use up to 80% of available memory

# Use optimized settings for generation
tokens = generate(
    model,
    tokenizer,
    prompt,
    max_tokens=512,
    batch_size=32,  # Larger batch size for throughput
    temp=0.7
)
```

## Fine-tuning

MLX offers multiple fine-tuning approaches, from full fine-tuning to more efficient methods like LoRA and QLoRA.

### Preparing Data

Create a dataset in the appropriate format:

```python
# Example dataset format
examples = [
    {"prompt": "What is Apple Silicon?", "response": "Apple Silicon refers to..."},
    {"prompt": "Explain quantum computing", "response": "Quantum computing is..."}
]

# Process data
def prepare_data(examples):
    inputs = []
    targets = []
    for ex in examples:
        prompt_ids = tokenizer.encode(ex["prompt"])
        response_ids = tokenizer.encode(ex["response"])
        
        input_ids = prompt_ids + response_ids
        target_ids = [-100] * len(prompt_ids) + response_ids  # -100 means ignore for loss
        
        inputs.append(input_ids)
        targets.append(target_ids)
    
    return inputs, targets
```

### Full Fine-tuning

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Load model
from mlx_lm.utils import load_model
model, tokenizer = load_model("llama-2-7b")

# Setup optimizer
optimizer = optim.AdamW(learning_rate=2e-5)

# Define loss function
def loss_fn(model, inputs, targets):
    logits = model(inputs)
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), 
        targets.reshape(-1),
        ignore_index=-100
    )

# Training loop
def train_step(model, inputs, targets):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, inputs, targets)
    optimizer.update(model, grads)
    return loss

# Run training
for epoch in range(epochs):
    for batch in dataset:
        inputs, targets = batch
        loss = train_step(model, mx.array(inputs), mx.array(targets))
        print(f"Loss: {loss}")

# Save model
mx.save("fine_tuned_model.npz", model.parameters())
```

### LoRA Fine-tuning

```python
from mlx_lm.lora import apply_lora

# Load model
model, tokenizer = load_model("llama-2-7b")

# Apply LoRA
model = apply_lora(
    model,
    r=8,  # LoRA rank
    alpha=16,  # LoRA alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Continue with training as above, but only LoRA parameters will be updated
# This requires much less memory
```

### QLoRA (Quantized LoRA)

```python
# Load quantized model
model, tokenizer = load_model("llama-2-7b", quantization="int4")

# Apply LoRA to quantized model
model = apply_lora(
    model, 
    r=8, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Continue with training as above
# This requires the least memory of all fine-tuning methods
```

## Memory Usage Guide

| Quantization | 7B Model | 13B Model | 33B Model | 70B Model |
|--------------|----------|-----------|-----------|-----------|
| None (FP16) | ~14GB | ~26GB | ~65GB | ~140GB |
| INT8 | ~8GB | ~13GB | ~32GB | ~70GB |
| INT4 | ~4GB | ~7GB | ~16GB | ~35GB |

### Fine-tuning Memory Requirements

| Method | 7B Model | 13B Model | 33B Model | 70B Model |
|--------|----------|-----------|-----------|-----------|
| Full Fine-tuning | 32GB+ | 64GB+ | 128GB+ | Not practical |
| LoRA (r=16) | 16GB+ | 32GB+ | 64GB+ | 128GB+ |
| QLoRA (INT8) | 12GB+ | 24GB+ | 48GB+ | 96GB+ |
| QLoRA (INT4) | 8GB+ | 16GB+ | 32GB+ | 64GB+ |

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Error: "Memory allocation failed"
   - Solution: Use quantization or reduce batch size

2. **Installation Problems**
   - Error: "Module not found"
   - Solution: Verify you're using a compatible Python version (3.8+)

3. **Slow Performance**
   - Problem: Inference seems CPU-bound
   - Solution: Ensure GPU is being used with `mx.set_default_device(mx.gpu)`

### Performance Issues

1. **Optimizing for Speed**
   - Use quantization (int4 offers good balance)
   - Increase batch size for throughput
   - Ensure Metal GPU is being used

2. **Reducing Memory Usage**
   - Use int4 quantization
   - Reduce context length
   - For fine-tuning, use QLoRA instead of full fine-tuning

## Multi-modal Capabilities

MLX also supports vision + language models:

```python
from mlx_lm import load
import mlx.core as mx

# Load a multimodal model
model, processor = load("llava-7b")

# Process image and text
image = mx.array(load_image("example.jpg"))
text = "What's in this image?"

# Generate caption
outputs = model.generate(processor, text, image, max_tokens=256)
print(processor.decode(outputs))
```

## Resources

- [Official MLX Repository](https://github.com/ml-explore/mlx)
- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX-LM Repository](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [MLX Discord Community](https://discord.gg/mlx)
- [Apple Developer Documentation](https://developer.apple.com/machine-learning/)

## Advanced Topics

- [Custom Model Architecture Implementation](../advanced/custom-models.md)
- [Integration with Other ML Workflows](../advanced/application-integration.md)
- [Comparison with Other Frameworks](framework-comparison.md)
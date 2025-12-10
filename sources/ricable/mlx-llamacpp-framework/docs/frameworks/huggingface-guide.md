# Complete Guide to HuggingFace Transformers with MPS

This guide provides a comprehensive overview of using HuggingFace Transformers with MPS (Metal Performance Shaders) acceleration on Apple Silicon Macs.

## What is HuggingFace?

HuggingFace is the leading open-source platform for machine learning, providing state-of-the-art natural language processing models, datasets, and tools. The Transformers library is the core component for working with pre-trained language models.

Key features:
- Massive model ecosystem (100,000+ models)
- Production-ready training and inference
- Advanced fine-tuning capabilities (LoRA, QLoRA, full fine-tuning)
- Comprehensive quantization support
- MPS acceleration for Apple Silicon
- Accelerate library for distributed training
- Text Generation Inference (TGI) for production

## Installation

### Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4) for MPS acceleration
- macOS 12+ (Monterey or newer)
- Python 3.8+
- 16GB+ RAM (32GB+ recommended for larger models)

### Basic Installation

```bash
# Create a virtual environment (recommended)
python -m venv hf-env
source hf-env/bin/activate

# Install core HuggingFace libraries
pip install transformers>=4.35.0
pip install torch>=2.0.0
pip install tokenizers>=0.14.0

# Install optional dependencies for advanced features
pip install accelerate>=0.24.0    # For distributed training
pip install peft>=0.6.0          # For parameter-efficient fine-tuning
pip install bitsandbytes>=0.41.0 # For quantization (limited Apple Silicon support)
pip install datasets>=2.14.0     # For dataset management
```

### Flow2 Integration

```bash
# Install Flow2 with HuggingFace support
pip install -e .[huggingface]

# Or install with all dependencies
pip install -e .[all]
```

## Model Management

### Loading Models

```python
import flow2

# Basic model loading with MPS acceleration
model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
    "microsoft/DialoGPT-medium", 
    device="mps"
)

# With specific configuration
model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
    "microsoft/DialoGPT-large",
    device="mps",
    torch_dtype="float16",  # Use half precision for memory efficiency
    trust_remote_code=True
)
```

### Model Categories

#### Conversational Models
- **microsoft/DialoGPT-small/medium/large**: Optimized for chat
- **microsoft/BlenderBot-400M-distill**: Facebook's conversational AI
- **facebook/blenderbot-3B**: Larger conversational model

#### Instruction-Following Models
- **microsoft/DialoGPT-medium**: Good balance of size and capability
- **google/flan-t5-base**: Google's instruction-tuned model
- **huggingface/CodeBERTa-small-v1**: Code-focused model

#### Large Language Models (8B+)
- **meta-llama/Llama-2-7b-chat-hf**: Meta's Llama 2 (requires access)
- **mistralai/Mistral-7B-Instruct-v0.1**: Mistral's instruction model
- **microsoft/phi-2**: Microsoft's 2.7B parameter model

## Inference and Generation

### Basic Text Generation

```python
from flow2.frameworks.huggingface import GenerationParams

# Set up generation parameters
gen_params = GenerationParams(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Generate response
response = flow2.frameworks.huggingface.generate_completion(
    model, tokenizer, 
    "Explain quantum computing in simple terms.", 
    gen_params
)
print(response)
```

### Streaming Generation

```python
# Stream responses for real-time applications
for chunk in flow2.frameworks.huggingface.streaming_completion(
    model, tokenizer, 
    "Write a story about AI and humanity.", 
    gen_params
):
    print(chunk, end="", flush=True)
```

### Batch Processing

```python
# Process multiple prompts efficiently
prompts = [
    "What is machine learning?",
    "Explain neural networks",
    "Define artificial intelligence"
]

responses = flow2.frameworks.huggingface.batch_generate(
    model, tokenizer, prompts, gen_params
)

for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

## Fine-tuning

### LoRA Fine-tuning

```python
# Create LoRA configuration
lora_config = flow2.frameworks.huggingface.create_lora_config(
    model_name="microsoft/DialoGPT-medium",
    r=16,                    # Rank of adaptation
    lora_alpha=32,          # LoRA scaling parameter
    lora_dropout=0.1,       # Dropout probability
    target_modules=["c_attn", "c_proj"]  # Target modules for LoRA
)

# Set up training configuration
training_config = flow2.frameworks.huggingface.TrainingConfig(
    output_dir="./lora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-4,
    fp16=False,  # Disable for MPS compatibility
    dataloader_pin_memory=False  # Important for MPS
)

# Run LoRA fine-tuning
adapter_path = flow2.frameworks.huggingface.finetune_lora(
    model_name="microsoft/DialoGPT-medium",
    dataset_path="path/to/your/dataset.jsonl",
    output_dir="./lora_output",
    lora_config=lora_config,
    training_config=training_config,
    device="mps"
)
```

### QLoRA Fine-tuning (Quantized LoRA)

```python
# QLoRA with 4-bit quantization (where supported)
qlora_config = flow2.frameworks.huggingface.QLoRAConfig(
    lora_config=lora_config,
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

# Note: BitsAndBytes has limited Apple Silicon support
adapter_path = flow2.frameworks.huggingface.finetune_qlora(
    model_name="microsoft/DialoGPT-large",
    dataset_path="path/to/your/dataset.jsonl",
    output_dir="./qlora_output",
    qlora_config=qlora_config,
    training_config=training_config
)
```

### Dataset Format

Your training data should be in JSONL format:

```json
{"text": "### User\nWhat is AI?\n\n### Assistant\nAI is artificial intelligence..."}
{"text": "### User\nExplain ML\n\n### Assistant\nMachine learning is..."}
```

Or with separate input/output:

```json
{"input": "What is AI?", "output": "AI is artificial intelligence..."}
{"input": "Explain ML", "output": "Machine learning is..."}
```

## Quantization

### Dynamic Quantization

```python
# Basic quantization for memory efficiency
quantized_model = flow2.frameworks.huggingface.quantize_model(
    model, 
    method=flow2.frameworks.huggingface.QuantizationMethod.DYNAMIC
)
```

### BitsAndBytes Quantization (Limited Apple Silicon Support)

```python
# 4-bit quantization (where supported)
quantized_model = flow2.frameworks.huggingface.load_quantized_model(
    "microsoft/DialoGPT-large",
    quantization_method=flow2.frameworks.huggingface.QuantizationMethod.BITSANDBYTES_4BIT,
    device="mps"
)
```

### Quantization Benchmarking

```python
# Compare quantization methods
results = flow2.frameworks.huggingface.benchmark_quantization(
    model_name="microsoft/DialoGPT-medium",
    methods=[
        flow2.frameworks.huggingface.QuantizationMethod.DYNAMIC,
        flow2.frameworks.huggingface.QuantizationMethod.BITSANDBYTES_4BIT
    ],
    test_prompts=["Hello, how are you?", "What is AI?"],
    output_dir="./quantization_benchmark"
)
```

## Apple Silicon Optimization

### MPS Setup

```python
# Automatic MPS device setup
device = flow2.frameworks.huggingface.setup_mps_device()
print(f"Using device: {device}")

# Manual MPS configuration
import torch
if torch.backends.mps.is_available():
    device = torch.device("mps")
    # Enable MPS fallback for unsupported operations
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

### Memory Optimization

```python
# Memory-efficient model loading
model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
    "microsoft/DialoGPT-large",
    device="mps",
    torch_dtype=torch.float16,  # Use half precision
    low_cpu_mem_usage=True,     # Reduce CPU memory usage
    device_map="auto"           # Automatic device mapping
)

# Clear cache when needed
torch.mps.empty_cache()
```

### Performance Tips

1. **Use Float16**: Reduces memory usage by ~50%
```python
model = model.half().to("mps")
```

2. **Gradient Checkpointing**: Trade compute for memory
```python
model.gradient_checkpointing_enable()
```

3. **Batch Size Tuning**: Start small and increase
```python
training_config.per_device_train_batch_size = 1
training_config.gradient_accumulation_steps = 8  # Effective batch size = 8
```

## Production Deployment

### Model Serving

```python
# Simple inference server
from flow2.frameworks.huggingface import create_inference_server

server = create_inference_server(
    model_name="microsoft/DialoGPT-medium",
    device="mps",
    max_concurrent_requests=4
)
server.start()
```

### Chat Interface

```python
# Interactive chat application
from flow2.chat.interfaces.cli.hf_chat import HuggingFaceChatInterface

chat = HuggingFaceChatInterface(
    model_name="microsoft/DialoGPT-medium",
    device="mps"
)
chat.start_chat()
```

## Common Use Cases

### 1. Conversational AI

```python
# Load a conversational model
model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
    "microsoft/DialoGPT-medium",
    device="mps"
)

# Set up for dialogue
gen_params = GenerationParams(
    max_new_tokens=150,
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
```

### 2. Question Answering

```python
# Use instruction-tuned model
model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
    "google/flan-t5-base",
    device="mps"
)

# Format questions properly
question = "Question: What is the capital of France? Answer:"
response = flow2.frameworks.huggingface.generate_completion(
    model, tokenizer, question, gen_params
)
```

### 3. Text Summarization

```python
# Use T5 for summarization
model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
    "t5-small",
    device="mps"
)

# Format for summarization
text = "summarize: " + long_text
summary = flow2.frameworks.huggingface.generate_completion(
    model, tokenizer, text, gen_params
)
```

## Troubleshooting

### Common Issues

1. **MPS Compatibility**
```python
# Some operations may not be supported on MPS
# Enable fallback to CPU
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

2. **Memory Issues**
```python
# Reduce batch size
training_config.per_device_train_batch_size = 1

# Use gradient accumulation
training_config.gradient_accumulation_steps = 8

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

3. **Tokenizer Padding**
```python
# Set pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

4. **Model Loading Errors**
```python
# Try loading with trust_remote_code
model, tokenizer = flow2.frameworks.huggingface.load_hf_model(
    model_name,
    trust_remote_code=True,
    device="mps"
)
```

## Model Information and Utilities

### Get Model Information

```python
# Get detailed model information
info = flow2.frameworks.huggingface.get_model_info("microsoft/DialoGPT-medium")
print(f"Parameters: {info.num_parameters:,}")
print(f"Model type: {info.model_type}")
print(f"Architecture: {info.architecture}")
```

### Memory Cleanup

```python
# Clean up memory after use
flow2.frameworks.huggingface.cleanup_memory()

# Or manually
import torch
import gc

del model, tokenizer
gc.collect()
torch.mps.empty_cache()
```

## Integration with Flow2

### Benchmarking

```python
# Run comprehensive HuggingFace benchmark
from flow2.performance.benchmark.huggingface import run_huggingface_benchmark

results = run_huggingface_benchmark(
    output_dir="./hf_benchmark_results"
)
```

### Framework Comparison

```python
# Compare with other frameworks
from flow2.performance.benchmark import run_comprehensive_framework_benchmark

results = run_comprehensive_framework_benchmark(
    test_models={
        "huggingface": {"small": "microsoft/DialoGPT-small"},
        "mlx": {"small": "models/mlx/tinyllama-1.1b-chat"},
        "llamacpp": {"small": "models/llamacpp/tinyllama.gguf"}
    }
)
```

## Best Practices

### 1. Model Selection
- Start with smaller models (DialoGPT-small) for development
- Use medium models for production with good hardware
- Consider instruction-tuned models for better task performance

### 2. Memory Management
- Use float16 for inference to save memory
- Implement proper cleanup in production
- Monitor memory usage with Activity Monitor

### 3. Fine-tuning Strategy
- Start with LoRA for parameter efficiency
- Use QLoRA for very large models (where supported)
- Validate on held-out data regularly

### 4. Performance Optimization
- Use MPS acceleration when available
- Batch multiple requests when possible
- Cache frequently used models in memory

### 5. Production Considerations
- Implement proper error handling
- Add request rate limiting
- Monitor GPU/CPU usage
- Use appropriate logging

## Advanced Topics

### Custom Model Architectures

```python
# Load custom models
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("path/to/custom/model")
model = AutoModelForCausalLM.from_pretrained(
    "path/to/custom/model",
    config=config,
    torch_dtype=torch.float16
).to("mps")
```

### Distributed Training

```python
# Use Accelerate for multi-GPU training (when available)
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```

### Model Merging

```python
# Merge LoRA adapters back to base model
merged_model = flow2.frameworks.huggingface.merge_lora_adapter(
    base_model="microsoft/DialoGPT-medium",
    adapter_path="./lora_output",
    output_path="./merged_model"
)
```

This guide provides comprehensive coverage of HuggingFace Transformers with MPS acceleration. For the latest updates and additional examples, refer to the [Flow2 documentation](../README.md) and [HuggingFace documentation](https://huggingface.co/docs/transformers/).
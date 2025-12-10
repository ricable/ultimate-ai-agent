# Fine-tuning LLMs on Apple Silicon

This guide covers how to fine-tune large language models on Apple Silicon hardware using both llama.cpp and MLX frameworks, with a focus on memory-efficient techniques.

## Table of Contents
- [Understanding Fine-tuning](#understanding-fine-tuning)
- [Fine-tuning Methods](#fine-tuning-methods)
- [Data Preparation](#data-preparation)
- [llama.cpp Fine-tuning](#llamacpp-fine-tuning)
- [MLX Fine-tuning](#mlx-fine-tuning)
- [Evaluating Fine-tuned Models](#evaluating-fine-tuned-models)
- [Memory and Performance Considerations](#memory-and-performance-considerations)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)

## Understanding Fine-tuning

### What is Fine-tuning?

Fine-tuning is the process of further training a pre-trained language model on a specific dataset to adapt it for particular tasks, domains, or styles. Unlike training from scratch, fine-tuning starts with the knowledge already in the model and refines it.

### When to Fine-tune

Fine-tuning is beneficial when:
- You need the model to follow specific instructions or patterns
- You want to adapt to specialized domain knowledge
- You need consistent style or format in responses
- You want to improve performance on specific tasks
- You need to align model behavior with specific guidelines

### Fine-tuning vs. Prompt Engineering

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Fine-tuning** | More consistent behavior<br>Better specialized knowledge<br>Improved task performance | Requires training data<br>More resource intensive<br>Risk of overfitting | Specialized applications<br>Consistent usage patterns<br>Domain adaptation |
| **Prompt Engineering** | No training required<br>Easily modified<br>Zero resource cost | Less consistent<br>Limited by model's knowledge<br>Prompt length limits | Quick adjustments<br>General-purpose use<br>Exploration |

## Fine-tuning Methods

### Available Techniques

| Method | Description | Memory Required | Quality | Use When |
|--------|-------------|----------------|---------|----------|
| **Full Fine-tuning** | Updates all model parameters | Very High | Excellent | You have ample RAM and need the best quality |
| **LoRA (Low-Rank Adaptation)** | Updates low-rank matrices instead of full weights | Moderate | Very Good | You have moderate RAM and need good quality |
| **QLoRA** | Combines quantization with LoRA | Low | Good | You have limited RAM but still need decent quality |
| **Prefix/Prompt Tuning** | Learns optimal prompts/prefixes | Very Low | Acceptable | You have severe RAM constraints |

### Framework Support

| Method | llama.cpp | MLX |
|--------|-----------|-----|
| Full Fine-tuning | ❌ | ✓ |
| LoRA | ✓ | ✓ |
| QLoRA | ✓ | ✓ |
| Prefix/Prompt Tuning | ❌ | ✓ |

### Hardware Requirements

| Method | 7B Model | 13B Model | 33B Model | 70B Model |
|--------|----------|-----------|-----------|-----------|
| Full Fine-tuning | 32GB+ RAM | 64GB+ RAM | 128GB+ RAM | Not practical |
| LoRA (r=16) | 16GB+ RAM | 32GB+ RAM | 64GB+ RAM | 128GB+ RAM |
| QLoRA (INT8) | 12GB+ RAM | 24GB+ RAM | 48GB+ RAM | 96GB+ RAM |
| QLoRA (INT4) | 8GB+ RAM | 16GB+ RAM | 32GB+ RAM | 64GB+ RAM |

## Data Preparation

### Dataset Format

Fine-tuning datasets typically use one of these formats:

#### 1. JSONL Format (most common)
```jsonl
{"prompt": "What is Apple Silicon?", "response": "Apple Silicon refers to the custom ARM-based processors designed by Apple for their Mac computers and iPad tablets."}
{"prompt": "What are the advantages of Apple Silicon?", "response": "The advantages include superior performance-per-watt, integrated GPU and Neural Engine, unified memory architecture, and native optimization for macOS."}
```

#### 2. Instruction Format
```jsonl
{"instruction": "Explain what Apple Silicon is", "output": "Apple Silicon refers to the custom ARM-based processors..."}
{"instruction": "List the advantages of Apple Silicon", "output": "The advantages include superior performance-per-watt..."}
```

#### 3. Chat Format
```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is Apple Silicon?"}, {"role": "assistant", "content": "Apple Silicon refers to..."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What are the advantages of Apple Silicon?"}, {"role": "assistant", "content": "The advantages include..."}]}
```

### Dataset Size Guidelines

| Fine-tuning Goal | Minimum Examples | Recommended Examples |
|------------------|------------------|----------------------|
| Style adaptation | 10-50 | 100-200 |
| Task adaptation | 50-100 | 500-1000 |
| Domain knowledge | 100-500 | 1000+ |
| Comprehensive | 500+ | 5000+ |

### Data Preparation Tools

```python
# Converting text pairs to JSONL
import json

def text_to_jsonl(input_file, output_file):
    pairs = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            prompt = lines[i].strip()
            response = lines[i+1].strip()
            pairs.append({"prompt": prompt, "response": response})
    
    with open(output_file, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

# Dataset splitting
def split_dataset(input_file, train_file, val_file, split=0.9):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Shuffle
    import random
    random.shuffle(lines)
    
    # Split
    split_idx = int(len(lines) * split)
    train_data = lines[:split_idx]
    val_data = lines[split_idx:]
    
    # Write files
    with open(train_file, 'w') as f:
        f.writelines(train_data)
    
    with open(val_file, 'w') as f:
        f.writelines(val_data)

# Data filtering
def filter_dataset(input_file, output_file, min_length=10, max_length=2048):
    filtered = []
    with open(input_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            prompt_len = len(example.get("prompt", ""))
            response_len = len(example.get("response", ""))
            
            if min_length <= prompt_len <= max_length and min_length <= response_len <= max_length:
                filtered.append(line)
    
    with open(output_file, 'w') as f:
        f.writelines(filtered)
```

### Example Dataset Preparation

```bash
# Prepare a dataset from text files
python -c "
import json

# Sample data (in real use, you'd read from files)
examples = [
    ('What is Apple Silicon?', 'Apple Silicon refers to the custom ARM-based processors...'),
    ('What are the advantages of Apple Silicon?', 'The advantages include superior performance-per-watt...')
]

# Convert to JSONL
with open('data/train.jsonl', 'w') as f:
    for prompt, response in examples:
        f.write(json.dumps({'prompt': prompt, 'response': response}) + '\n')

# Split into train/val
import random
with open('data/train.jsonl', 'r') as f:
    lines = f.readlines()

random.shuffle(lines)
split = int(len(lines) * 0.9)

with open('data/train_split.jsonl', 'w') as f:
    f.writelines(lines[:split])

with open('data/val.jsonl', 'w') as f:
    f.writelines(lines[split:])
"
```

## llama.cpp Fine-tuning

### LoRA Fine-tuning

```bash
# Ensure you've built the fine-tuning tools
cd llama.cpp
cmake --build . --config Release --target llama-finetune

# Run fine-tuning
./llama-finetune \
  --model-base ./models/llama-2-7b-q4_0.gguf \
  --lora-rank 8 \
  --lora-layers all \
  --data-train ./data/train.jsonl \
  --data-val ./data/val.jsonl \
  --lora-out ./lora-finetune.bin
```

#### Key Parameters Explained

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--model-base` | Base model path | Path to .gguf model |
| `--lora-rank` | Rank of LoRA matrices | 4, 8, 16, 32 (higher = more capacity) |
| `--lora-layers` | Which layers to adapt | "all", "attn", "ffn" |
| `--data-train` | Training data | Path to training JSONL |
| `--data-val` | Validation data | Path to validation JSONL |
| `--lora-out` | Output path | Path to save LoRA weights |
| `--batch-size` | Batch size | 1-32 (depends on RAM) |
| `--epochs` | Training epochs | 1-5 |
| `--learning-rate` | Learning rate | 1e-4 to 5e-4 |

### Using Fine-tuned Models

```bash
# Run inference with fine-tuned model
./main \
  --model ./models/llama-2-7b-q4_0.gguf \
  --lora ./lora-finetune.bin \
  --prompt "What is the advantage of Apple Silicon?"
```

### QLoRA with llama.cpp

```bash
# Quantize model first if needed
python convert.py --outtype q4_k --outfile models/llama-2-7b-q4_k.gguf meta-llama/Llama-2-7b

# Run fine-tuning on quantized model
./llama-finetune \
  --model-base ./models/llama-2-7b-q4_k.gguf \
  --lora-rank 8 \
  --lora-layers all \
  --data-train ./data/train.jsonl \
  --data-val ./data/val.jsonl \
  --lora-out ./lora-finetune-q4.bin
```

### Python API for llama.cpp Fine-tuning

```python
from llama_cpp import Llama

# Load base model
llm = Llama(
    model_path="models/llama-2-7b-q4_0.gguf",
    n_gpu_layers=-1
)

# Load with LoRA weights for inference
llm = Llama(
    model_path="models/llama-2-7b-q4_0.gguf",
    lora_path="lora-finetune.bin", 
    n_gpu_layers=-1
)

# Generate with fine-tuned model
output = llm(
    "What is the advantage of Apple Silicon?",
    max_tokens=512
)
print(output["choices"][0]["text"])
```

## MLX Fine-tuning

### Full Fine-tuning

```python
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load

# Load model
model, tokenizer = load("llama-2-7b")

# Load dataset
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl("data/train.jsonl")
val_data = load_jsonl("data/val.jsonl")

# Prepare training data
def prepare_batch(examples, tokenizer, max_length=512):
    prompts = [ex["prompt"] for ex in examples]
    responses = [ex["response"] for ex in examples]
    
    input_ids = []
    labels = []
    
    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt)
        response_ids = tokenizer.encode(response)
        
        # Combine prompt and response
        input_seq = prompt_ids + response_ids
        
        # Create labels: -100 for prompt (ignored in loss), actual tokens for response
        label_seq = [-100] * len(prompt_ids) + response_ids
        
        # Truncate if needed
        if len(input_seq) > max_length:
            input_seq = input_seq[:max_length]
            label_seq = label_seq[:max_length]
        
        input_ids.append(input_seq)
        labels.append(label_seq)
    
    # Pad sequences
    max_len = max(len(seq) for seq in input_ids)
    input_ids = [seq + [tokenizer.pad_id] * (max_len - len(seq)) for seq in input_ids]
    labels = [seq + [-100] * (max_len - len(seq)) for seq in labels]
    
    return mx.array(input_ids), mx.array(labels)

# Define loss function
def loss_fn(model, inputs, targets):
    logits = model(inputs)
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    
    # Create a mask for non-padding tokens
    mask = targets != -100
    
    # Apply mask
    logits = logits[mask]
    targets = targets[mask]
    
    # Compute cross entropy loss
    return nn.losses.cross_entropy(logits, targets)

# Setup optimizer
optimizer = optim.AdamW(learning_rate=2e-5)

# Training loop
def train_step(model, inputs, targets):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, inputs, targets)
    optimizer.update(model, grads)
    return loss

# Train the model
batch_size = 4  # Adjust based on your RAM
num_epochs = 3

for epoch in range(num_epochs):
    # Shuffle data
    indices = np.random.permutation(len(train_data))
    total_loss = 0.0
    batches = 0
    
    for i in range(0, len(train_data), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_data = [train_data[idx] for idx in batch_indices]
        
        inputs, targets = prepare_batch(batch_data, tokenizer)
        loss = train_step(model, inputs, targets)
        
        total_loss += loss
        batches += 1
        
        if batches % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batches}, Loss: {loss}")
    
    avg_loss = total_loss / batches
    print(f"Epoch {epoch+1} complete, Avg Loss: {avg_loss}")
    
    # Validate
    val_indices = np.random.permutation(len(val_data))
    val_loss = 0.0
    val_batches = 0
    
    for i in range(0, len(val_data), batch_size):
        batch_indices = val_indices[i:i+batch_size]
        batch_data = [val_data[idx] for idx in batch_indices]
        
        inputs, targets = prepare_batch(batch_data, tokenizer)
        val_loss += loss_fn(model, inputs, targets)
        val_batches += 1
    
    avg_val_loss = val_loss / val_batches
    print(f"Validation Loss: {avg_val_loss}")

# Save the model
mx.save("fine_tuned_model.npz", model.parameters())
```

### LoRA Fine-tuning with MLX

```python
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.lora import apply_lora

# Load model
model, tokenizer = load("llama-2-7b")

# Apply LoRA
model = apply_lora(
    model,
    r=8,  # LoRA rank
    alpha=16,  # LoRA alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Load and prepare data as in full fine-tuning example
# ...

# Setup optimizer - only LoRA parameters will be updated
optimizer = optim.AdamW(learning_rate=5e-4)

# Training loop
# Same as full fine-tuning but much more memory efficient
# ...

# Save only the LoRA weights
lora_params = {k: v for k, v in model.parameters().items() if "lora" in k}
mx.save("lora_weights.npz", lora_params)
```

### QLoRA with MLX

```python
# Load quantized model
model, tokenizer = load("llama-2-7b", quantization="int4")

# Apply LoRA to quantized model
model = apply_lora(
    model,
    r=8,
    alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Continue with training as in LoRA example
# This requires the least amount of memory
# ...
```

### Using Fine-tuned MLX Models

```python
from mlx_lm import load, generate
import mlx.core as mx

# For full fine-tuned model
model, tokenizer = load("llama-2-7b")
model.update(mx.load("fine_tuned_model.npz"))

# For LoRA fine-tuned model
model, tokenizer = load("llama-2-7b")
model = apply_lora(model, r=8, alpha=16)
model.update(mx.load("lora_weights.npz"))

# Generate with fine-tuned model
prompt = "What is the advantage of Apple Silicon?"
tokens = generate(model, tokenizer, prompt, max_tokens=512)
print(tokenizer.decode(tokens))
```

## Evaluating Fine-tuned Models

### Basic Evaluation Metrics

```python
def evaluate_model(model, tokenizer, eval_data, max_tokens=512):
    results = {
        "perplexity": 0,
        "exact_match": 0,
        "bleu_score": 0,
        "examples": []
    }
    
    for example in eval_data:
        prompt = example["prompt"]
        ground_truth = example["response"]
        
        # Generate response
        tokens = generate(model, tokenizer, prompt, max_tokens=max_tokens)
        response = tokenizer.decode(tokens)
        
        # Store example
        results["examples"].append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "prediction": response
        })
        
        # Calculate metrics
        # ... (implement perplexity, BLEU, etc.)
    
    # Calculate averages
    # ...
    
    return results
```

### Human Evaluation Template

Create a simple form for human evaluation:

```
# Model Evaluation Form

Model: [model name]
Evaluator: [name]
Date: [date]

For each example, rate the following criteria from 1-5:
1 = Poor, 2 = Fair, 3 = Good, 4 = Very Good, 5 = Excellent

## Example 1
Prompt: [prompt]
Response: [generated response]

- Accuracy: [1-5]
- Relevance: [1-5]
- Coherence: [1-5]
- Completeness: [1-5]
- Overall quality: [1-5]

Comments:
[comments]

## Example 2
...
```

### A/B Testing

```python
def ab_test(model_a, model_b, tokenizer, test_prompts, human_evaluator=None):
    results = []
    
    for prompt in test_prompts:
        # Generate from both models
        tokens_a = generate(model_a, tokenizer, prompt, max_tokens=512)
        response_a = tokenizer.decode(tokens_a)
        
        tokens_b = generate(model_b, tokenizer, prompt, max_tokens=512)
        response_b = tokenizer.decode(tokens_b)
        
        # Randomly order responses
        import random
        if random.random() < 0.5:
            order = ["A", "B"]
            ordered_responses = [response_a, response_b]
        else:
            order = ["B", "A"]
            ordered_responses = [response_b, response_a]
        
        if human_evaluator:
            # Present to human evaluator
            winner = human_evaluator(prompt, ordered_responses[0], ordered_responses[1])
            actual_winner = model_a if (winner == "A" and order[0] == "A") or (winner == "B" and order[0] == "B") else model_b
        else:
            # Automated evaluation
            # ...
            actual_winner = None
        
        results.append({
            "prompt": prompt,
            "winner": actual_winner,
            "responses": {
                "model_a": response_a,
                "model_b": response_b
            }
        })
    
    return results
```

## Memory and Performance Considerations

### Memory-Efficient Training

1. **Gradient Checkpointing**:
   Trades computation for memory by recomputing activations during backward pass

2. **Mixed Precision Training**:
   Use lower precision (fp16) for most operations while maintaining stability

3. **Layer Freezing**:
   Only train a subset of the model's layers, keeping others frozen

4. **Small Batch Sizes**:
   Start with batch size of 1 and increase if memory allows

5. **Gradient Accumulation**:
   Update weights after accumulating gradients from several forward/backward passes

### Performance Optimization

```python
# MLX performance optimization example
import mlx.core as mx

# Set GPU as default device
mx.set_default_device(mx.gpu)

# Limit memory usage
mx.set_allocation_limit(0.9)  # Use up to 90% of available memory

# Use half precision
def to_fp16(model):
    for name, param in model.parameters().items():
        if param.dtype == mx.float32:
            model.update_parameter(name, param.astype(mx.float16))
    return model

model = to_fp16(model)

# Gradient accumulation
def train_with_accumulation(model, data_loader, accumulation_steps=4):
    optimizer = optim.AdamW(learning_rate=2e-5)
    accumulated_grads = None
    
    for i, batch in enumerate(data_loader):
        inputs, targets = batch
        loss, grads = nn.value_and_grad(model, loss_fn)(model, inputs, targets)
        
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            for k in accumulated_grads:
                accumulated_grads[k] += grads[k]
        
        if (i + 1) % accumulation_steps == 0:
            # Scale gradients
            for k in accumulated_grads:
                accumulated_grads[k] /= accumulation_steps
            
            # Update weights
            optimizer.update(model, accumulated_grads)
            accumulated_grads = None
            
            print(f"Step {i+1}, Loss: {loss}")
```

### Overnight Training

For longer training jobs that might run overnight:

```python
# Checkpoint saving
def save_checkpoint(model, optimizer, epoch, step, loss, path):
    checkpoint = {
        "model": model.parameters(),
        "optimizer": optimizer.state,
        "epoch": epoch,
        "step": step,
        "loss": loss
    }
    mx.save(path, checkpoint)

# Checkpoint loading
def load_checkpoint(model, optimizer, path):
    checkpoint = mx.load(path)
    model.update(checkpoint["model"])
    optimizer.state = checkpoint["optimizer"]
    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]

# Training loop with checkpointing
for epoch in range(start_epoch, num_epochs):
    # Training code
    # ...
    
    # Save checkpoint every N steps
    if step % save_every == 0:
        save_checkpoint(
            model, 
            optimizer, 
            epoch, 
            step, 
            loss, 
            f"checkpoint_epoch{epoch}_step{step}.npz"
        )
```

## Common Use Cases

### 1. Domain Adaptation

Fine-tuning a model to understand domain-specific terminology and knowledge.

```python
# Example dataset for medical domain adaptation
medical_examples = [
    {"prompt": "What is hypertension?", "response": "Hypertension, or high blood pressure, is a condition where the force of blood against artery walls is consistently too high..."},
    {"prompt": "What causes myocardial infarction?", "response": "Myocardial infarction (heart attack) is typically caused by a blockage in one or more of the coronary arteries..."},
    # More examples...
]
```

### 2. Instruction Following

Improving a model's ability to follow specific instructions.

```python
# Example dataset for instruction tuning
instruction_examples = [
    {"instruction": "Summarize the following text", "input": "Long text here...", "output": "Concise summary here..."},
    {"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"},
    # More examples...
]
```

### 3. Style Adaptation

Teaching a model to respond in a particular style or tone.

```python
# Example dataset for creative writing style
creative_examples = [
    {"prompt": "Describe a sunset", "response": "The sun descended like a weary traveler, painting the sky with strokes of amber and crimson..."},
    {"prompt": "Describe a city at night", "response": "The city transformed as darkness fell, a constellation of artificial stars igniting one by one..."},
    # More examples...
]
```

### 4. Behavior Alignment

Aligning model behavior with specific ethical guidelines or company policies.

```python
# Example dataset for alignment
alignment_examples = [
    {"prompt": "How can I hack into someone's email?", "response": "I cannot and will not provide instructions for illegal activities like unauthorized access to accounts..."},
    {"prompt": "Write a fake news article", "response": "I cannot generate misleading or false content designed to deceive readers..."},
    # More examples...
]
```

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Errors**
   - Switch to a more memory-efficient method (QLoRA)
   - Reduce batch size to 1
   - Reduce model size or use more aggressive quantization
   - Implement gradient accumulation

2. **Training Instability**
   - Lower the learning rate (try 1e-5 instead of 5e-4)
   - Add gradient clipping
   - Use a warm-up period for learning rate
   - Check data for outliers or problematic examples

3. **Poor Performance on Target Task**
   - Increase dataset size or diversity
   - Ensure data quality and relevance
   - Try a larger LoRA rank
   - Train for more epochs
   - Use a larger or higher-quality base model

4. **Slow Training**
   - Ensure GPU acceleration is enabled
   - Optimize batch size for your hardware
   - Use FP16 training where possible
   - Apply selective layer freezing

### Diagnostic Tips

```python
# Check memory usage
import mlx.core as mx
print(f"Available memory: {mx.get_available_memory() / (1024**3):.2f} GB")

# Test model loading
try:
    model, tokenizer = load("llama-2-7b")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Verify data loading
with open("data/train.jsonl", "r") as f:
    line = f.readline()
    try:
        example = json.loads(line)
        print(f"Data format: {example.keys()}")
    except:
        print("Error parsing data")

# Test small training run
small_data = train_data[:10]
try:
    inputs, targets = prepare_batch(small_data, tokenizer)
    loss = loss_fn(model, inputs, targets)
    print(f"Training test successful, loss: {loss}")
except Exception as e:
    print(f"Training test failed: {e}")
```

## Next Steps

- [Evaluating Models](../advanced/model-evaluation.md) - Comprehensive evaluation techniques
- [Deploying Fine-tuned Models](../advanced/deployment-guide.md) - Using your models in production
- [Hardware Recommendations](../hardware/hardware-recommendations.md) - Choosing the right Mac for fine-tuning
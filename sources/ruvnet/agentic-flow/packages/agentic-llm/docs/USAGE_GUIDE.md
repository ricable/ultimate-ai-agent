# Usage Guide - Agentic LLM Training System

Practical guide for using the Phi-4 training, quantization, and deployment system.

## Table of Contents

1. [Training Workflow](#training-workflow)
2. [Quantization Workflow](#quantization-workflow)
3. [Validation Workflow](#validation-workflow)
4. [Benchmarking Workflow](#benchmarking-workflow)
5. [Complete Pipeline](#complete-pipeline)

---

## Training Workflow

### 1. Configure Training

Edit `configs/training_config.json` or modify `training/train.py`:

```json
{
  "training": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "learning_rate": 2e-05
  },
  "lora": {
    "r": 16,
    "alpha": 32
  }
}
```

### 2. Start Training

**On Fly.io (GPU):**
```bash
# Deploy and start training
./deployment/deploy.sh

# Monitor progress
flyctl logs -a agentic-llm-phi4 -f
```

**Locally (with GPU):**
```bash
# Build and run
docker build -t agentic-llm:latest .
docker run --gpus all \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -e HF_TOKEN=$HF_TOKEN \
  agentic-llm:latest
```

**Direct Python:**
```bash
python3 training/train.py
```

### 3. Monitor Training

**WandB Dashboard:**
- Navigate to https://wandb.ai/your-project
- View real-time metrics, loss curves, GPU utilization

**TensorBoard:**
```bash
tensorboard --logdir /app/checkpoints/logs
```

**Logs:**
```bash
# View training logs
tail -f /app/checkpoints/logs/training.log
```

### 4. Training Outputs

```
checkpoints/
├── checkpoint-100/       # Intermediate checkpoint
├── checkpoint-200/
├── checkpoint-300/
├── final_model/         # Best model (selected by early stopping)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── tokenizer_config.json
│   └── ...
├── logs/
│   └── events.out.tfevents.xxx
└── training_metrics.json
```

---

## Quantization Workflow

### 1. Run Quantization Pipeline

After training completes:

```bash
python3 quantization/quantize.py \
  --model-path /app/checkpoints/final_model \
  --output-dir /app/models/quantized \
  --validate
```

### 2. Quantization Outputs

```
models/quantized/
├── phi4_fp32.onnx              # 14GB - Original ONNX
├── phi4_int8.onnx              # 3.5GB - INT8 quantized (2-3x faster)
├── phi4_int4.onnx              # 1.75GB - INT4 quantized (3-4x faster)
└── quantization_results.json   # Detailed metrics
```

### 3. Validate Quantization Quality

```bash
# Check quality metrics
cat /app/models/quantized/quantization_results.json
```

Expected output:
```json
{
  "benchmarks": {
    "int8": {
      "model_size_mb": 3500,
      "load_time_s": 0.5
    },
    "int4": {
      "model_size_mb": 1750,
      "load_time_s": 0.3
    }
  }
}
```

---

## Validation Workflow

### 1. Run Overfitting Detection

```bash
python3 validation/validator.py \
  --model-path /app/checkpoints/final_model \
  --output-dir /app/validation/results
```

### 2. Review Results

**Validation Metrics:**
```json
{
  "train_loss": 0.45,
  "val_loss": 0.52,
  "test_loss": 0.54,
  "overfitting_ratio": 0.12,
  "is_overfitting": false
}
```

**Visual Analysis:**
```bash
# Open learning curves
open /app/validation/results/learning_curves.png
```

### 3. Interpret Results

**Overfitting Ratio:**
- < 0.10: Excellent generalization ✅
- 0.10-0.15: Good (acceptable) ✅
- 0.15-0.25: Moderate overfitting ⚠️
- > 0.25: Severe overfitting ❌

**Actions:**
- If overfitting: Reduce epochs, increase dropout, add regularization
- If underfitting: Increase capacity, train longer, reduce regularization

---

## Benchmarking Workflow

### 1. Compare Model Variants

```bash
python3 benchmarks/benchmark.py \
  --baseline /app/checkpoints/final_model \
  --quantized-int8 /app/models/quantized/phi4_int8.onnx \
  --quantized-int4 /app/models/quantized/phi4_int4.onnx \
  --output-dir /app/benchmarks/results
```

### 2. Review Benchmark Report

```bash
cat /app/benchmarks/results/benchmark_report.json
```

Example output:
```json
{
  "summary": {
    "int8": {
      "speedup": "2.50x",
      "size_reduction": "75.0%",
      "memory_reduction": "66.7%",
      "throughput_improvement": "2.50x"
    },
    "int4": {
      "speedup": "3.33x",
      "size_reduction": "87.5%",
      "memory_reduction": "79.2%",
      "throughput_improvement": "3.30x"
    }
  }
}
```

### 3. Analyze Visual Comparisons

```bash
open /app/benchmarks/results/benchmark_comparison.png
```

Charts include:
- Model size comparison
- Latency comparison
- Throughput comparison
- Tool calling accuracy

---

## Complete Pipeline

### End-to-End Example

```bash
#!/bin/bash
# complete_pipeline.sh - Full training to deployment

set -e

echo "🚀 Starting complete Agentic LLM pipeline..."

# Step 1: Train model
echo "📚 Step 1: Training..."
python3 training/train.py

# Step 2: Validate
echo "✅ Step 2: Validating..."
python3 validation/validator.py \
  --model-path /app/checkpoints/final_model \
  --output-dir /app/validation/results

# Step 3: Quantize
echo "⚡ Step 3: Quantizing..."
python3 quantization/quantize.py \
  --model-path /app/checkpoints/final_model \
  --output-dir /app/models/quantized \
  --validate

# Step 4: Benchmark
echo "📊 Step 4: Benchmarking..."
python3 benchmarks/benchmark.py \
  --baseline /app/checkpoints/final_model \
  --quantized-int8 /app/models/quantized/phi4_int8.onnx \
  --quantized-int4 /app/models/quantized/phi4_int4.onnx \
  --output-dir /app/benchmarks/results

echo "✨ Pipeline complete!"
echo "📁 Results in:"
echo "  - /app/checkpoints/final_model"
echo "  - /app/models/quantized/"
echo "  - /app/validation/results/"
echo "  - /app/benchmarks/results/"
```

### Running the Pipeline

```bash
# Make executable
chmod +x complete_pipeline.sh

# Run on Fly.io
flyctl ssh console -a agentic-llm-phi4 -C "./complete_pipeline.sh"

# Run locally
./complete_pipeline.sh
```

---

## Common Tasks

### Task 1: Fine-tune on Custom Dataset

```python
# In training/train.py, modify MCPToolDatasetBuilder:

def load_custom_dataset(self, dataset_path: str):
    """Load custom training data"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    examples = []
    for item in data:
        examples.append({
            "text": f"User: {item['input']}\nAssistant: {item['output']}"
        })

    return Dataset.from_dict({"text": [ex["text"] for ex in examples]})
```

### Task 2: Adjust Overfitting Prevention

```python
# More aggressive prevention:
config = AgenticTrainingConfig(
    early_stopping_patience=2,      # Stop earlier
    weight_decay=0.05,               # Stronger regularization
    lora_dropout=0.1,                # Higher dropout
    validation_split=0.2             # More validation data
)
```

### Task 3: Optimize for Speed

```python
# Faster training:
config = AgenticTrainingConfig(
    per_device_train_batch_size=8,  # Larger batches
    gradient_accumulation_steps=2,   # Less accumulation
    gradient_checkpointing=False,    # Faster but more memory
    fp16=True,                       # Mixed precision
    eval_steps=200                   # Less frequent eval
)
```

### Task 4: Export for Claude Agent SDK

```bash
# After quantization:
cp /app/models/quantized/phi4_int8.onnx \
   /app/claude-agent-sdk/models/

# Update SDK config
cat > /app/claude-agent-sdk/config.json <<EOF
{
  "model_path": "models/phi4_int8.onnx",
  "provider": "CUDAExecutionProvider",
  "max_tokens": 2048
}
EOF
```

### Task 5: A/B Test Models

```bash
# Compare baseline vs optimized
python3 benchmarks/benchmark.py \
  --baseline /app/checkpoints/baseline_model \
  --quantized-int8 /app/checkpoints/optimized_model \
  --output-dir /app/benchmarks/ab_test
```

---

## Troubleshooting

### Issue: Training OOM (Out of Memory)

**Solution 1: Reduce batch size**
```python
per_device_train_batch_size=2
gradient_accumulation_steps=8  # Keep effective batch size
```

**Solution 2: Enable gradient checkpointing**
```python
gradient_checkpointing=True
```

**Solution 3: Use smaller LoRA rank**
```python
lora_r=8
lora_alpha=16
```

### Issue: Overfitting Detected

**Solution 1: Increase regularization**
```python
weight_decay=0.05
lora_dropout=0.15
```

**Solution 2: More validation data**
```python
validation_split=0.2
```

**Solution 3: Early stopping**
```python
early_stopping_patience=2
early_stopping_threshold=0.005
```

### Issue: Slow Training

**Solution 1: Optimize data loading**
```python
num_workers=4
prefetch_factor=2
pin_memory=True
```

**Solution 2: Reduce evaluation frequency**
```python
eval_steps=500
logging_steps=100
```

**Solution 3: Use larger GPU**
```bash
flyctl scale vm a100-80gb -a agentic-llm-phi4
```

### Issue: Poor Tool Calling Accuracy

**Solution 1: Add more training examples**
```python
# Increase synthetic examples
mcp_tool_focus=True
synthetic_examples=500
```

**Solution 2: Adjust temperature**
```python
# During inference
temperature=0.1  # More deterministic
top_p=0.9
```

**Solution 3: Fine-tune on real MCP logs**
```bash
# Export real Claude Code interactions
# Use as training data
```

---

## Best Practices

### Training

1. **Start small**: Train on subset first to validate pipeline
2. **Monitor closely**: Use WandB/TensorBoard from the start
3. **Save checkpoints**: Don't rely only on final model
4. **Test early**: Validate on small test set during training
5. **Track experiments**: Use consistent naming and tags

### Quantization

1. **Validate quality**: Always run validation after quantization
2. **Test inference**: Verify ONNX model works before deploying
3. **Compare formats**: Try both INT8 and INT4, pick best tradeoff
4. **Benchmark thoroughly**: Measure real-world performance

### Deployment

1. **Use volumes**: Store models on persistent volumes
2. **Auto-scale**: Enable auto-stop to save costs
3. **Monitor costs**: Set budget alerts
4. **Backup models**: Create volume snapshots regularly
5. **Version control**: Tag models with version numbers

---

## Next Steps

After completing the pipeline:

1. **Integrate with Claude Agent SDK**: Deploy quantized model
2. **Create inference API**: Expose model via REST API
3. **Build evaluation suite**: Automated quality testing
4. **Set up CI/CD**: Automated retraining pipeline
5. **Monitor production**: Track real-world performance

---

**Ready to optimize your Phi-4 model? Start with the training workflow!**

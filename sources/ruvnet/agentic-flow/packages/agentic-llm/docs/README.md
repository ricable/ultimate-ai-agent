# Agentic LLM Training System for Phi-4

GPU-accelerated training, quantization, and deployment system for optimizing Microsoft's Phi-4 model for Claude Agent SDK and MCP tools.

## 🎯 Features

- **GPU Training on Fly.io** - NVIDIA A100 40GB GPU support
- **Overfitting Prevention** - Early stopping, validation monitoring, learning curves
- **ONNX Quantization** - INT8/INT4 quantization for efficient inference
- **MCP Tool Optimization** - Fine-tuned for Claude SDK tool calling
- **Comprehensive Benchmarking** - Performance, accuracy, and memory metrics
- **Modular Architecture** - Clean separation of training, validation, and deployment

## 📁 Project Structure

```
docker/agentic-llm/
├── training/           # GPU training system
│   └── train.py       # Main training script with LoRA and overfitting prevention
├── validation/        # Validation and overfitting detection
│   └── validator.py   # Comprehensive validation suite
├── quantization/      # ONNX quantization pipeline
│   └── quantize.py    # INT8/INT4 quantization
├── deployment/        # Fly.io deployment scripts
│   └── deploy.sh      # Automated deployment to GPU instances
├── benchmarks/        # Performance benchmarking
│   └── benchmark.py   # Comparative benchmarking system
├── docs/              # Documentation
├── configs/           # Configuration files
├── models/            # Model storage (persistent volume)
├── Dockerfile         # GPU-enabled container
└── fly.toml           # Fly.io configuration
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install Fly.io CLI
curl -L https://fly.io/install.sh | sh

# Login to Fly.io
flyctl auth login

# Verify CUDA availability (for local testing)
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Deploy to Fly.io

```bash
cd docker/agentic-llm

# Make deployment script executable
chmod +x deployment/deploy.sh

# Deploy with GPU
./deployment/deploy.sh
```

The script will:
- ✅ Create A100 GPU instance
- ✅ Set up persistent volumes (100GB models, 50GB checkpoints)
- ✅ Configure secrets (HF_TOKEN, WANDB_API_KEY)
- ✅ Deploy and start training

### 3. Monitor Training

```bash
# View real-time logs
flyctl logs -a agentic-llm-phi4

# SSH into container
flyctl ssh console -a agentic-llm-phi4

# Check status
flyctl status -a agentic-llm-phi4
```

## 🎓 Training Pipeline

### Local Training (Development)

```bash
# Build Docker image
docker build -t agentic-llm:latest .

# Run training with GPU
docker run --gpus all \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -e HF_TOKEN=your_token \
  -e WANDB_API_KEY=your_key \
  agentic-llm:latest
```

### Training Configuration

Edit training parameters in `training/train.py`:

```python
@dataclass
class AgenticTrainingConfig:
    # Model
    model_name: str = "microsoft/Phi-4"

    # Training hyperparameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Overfitting prevention
    early_stopping_patience: int = 3
    validation_split: float = 0.15
    weight_decay: float = 0.01

    # Quantization
    use_4bit: bool = True
```

### Overfitting Prevention

The system includes multiple mechanisms:

1. **Early Stopping** - Monitors validation loss with patience
2. **Validation Split** - 15% validation, 5% test data
3. **Weight Decay** - L2 regularization (0.01)
4. **Dropout** - LoRA dropout (0.05)
5. **Learning Curves** - Automatic plotting and analysis
6. **Generalization Gap Monitoring** - Tracks train/val difference

## 🔧 Quantization Pipeline

### Run Quantization

```bash
# After training completes
python3 quantization/quantize.py \
  --model-path /app/checkpoints/final_model \
  --output-dir /app/models/quantized \
  --validate
```

### Quantization Levels

| Format | Size Reduction | Speed Improvement | Accuracy Loss |
|--------|---------------|-------------------|---------------|
| FP32   | Baseline      | 1x                | 0%            |
| INT8   | ~75%          | 2-3x              | <2%           |
| INT4   | ~87.5%        | 3-4x              | <5%           |

### Output Files

```
models/quantized/
├── phi4_fp32.onnx              # Original ONNX
├── phi4_int8.onnx              # INT8 quantized
├── phi4_int4.onnx              # INT4 quantized
└── quantization_results.json   # Benchmark results
```

## 📊 Benchmarking

### Run Comprehensive Benchmarks

```bash
python3 benchmarks/benchmark.py \
  --baseline /app/checkpoints/final_model \
  --quantized-int8 /app/models/quantized/phi4_int8.onnx \
  --quantized-int4 /app/models/quantized/phi4_int4.onnx \
  --output-dir /app/benchmarks/results
```

### Metrics Measured

- **Performance**
  - Inference latency (ms)
  - Throughput (tokens/sec)
  - Memory usage (MB)

- **Quality**
  - Perplexity
  - MCP tool calling accuracy
  - Response quality

- **Efficiency**
  - Model size (MB)
  - GPU utilization
  - Cost per 1M tokens

### Benchmark Output

```
benchmarks/results/
├── benchmark_report.json        # Detailed metrics
├── benchmark_comparison.png     # Visual comparisons
└── performance_summary.txt      # Human-readable summary
```

## 🛡️ Validation System

### Run Validation

```bash
python3 validation/validator.py \
  --model-path /app/checkpoints/final_model \
  --output-dir /app/validation/results
```

### Validation Outputs

- **Overfitting Detection**
  - Train vs validation loss curves
  - Overfitting ratio
  - Generalization gap

- **Quality Metrics**
  - Test set perplexity
  - Tool calling accuracy
  - Response coherence

- **Visualizations**
  - Learning curves
  - Loss distributions
  - Metric trends

## 🎯 MCP Tool Optimization

### Training Data

The system includes synthetic MCP tool calling examples:

```python
# Tool calling patterns
"User: Search for authentication functions\nAssistant: <tool_use>grep(pattern='def.*auth', path='.')</tool_use>"

# Agentic workflow patterns
"User: Implement API endpoint\nAssistant: Breaking this down: 1) Create file, 2) Implement logic, 3) Write tests..."
```

### Evaluation

Tool calling accuracy is measured against expected patterns:
- ✅ Correct tool selection
- ✅ Proper tool syntax
- ✅ Appropriate parameters
- ✅ Multi-step planning

## 💰 Cost Analysis

### Fly.io GPU Costs

| Instance Type | GPU        | vCPU | RAM   | Cost/hour | Cost/day |
|---------------|------------|------|-------|-----------|----------|
| a100-40gb     | A100 40GB  | 8    | 32GB  | ~$3.00    | ~$72.00  |
| a100-80gb     | A100 80GB  | 12   | 48GB  | ~$4.50    | ~$108.00 |

### Training Cost Estimates

- **Phi-4 Fine-tuning**: 2-6 hours = $6-18
- **Quantization**: 1-2 hours = $3-6
- **Benchmarking**: 0.5-1 hour = $1.50-3
- **Total**: ~$10.50-27 per iteration

### Cost Optimization

```bash
# Scale down when not training
flyctl scale count 0 -a agentic-llm-phi4

# Scale up for training
flyctl scale count 1 -a agentic-llm-phi4

# Auto-pause after training
flyctl apps update --auto-stop-machines=true -a agentic-llm-phi4
```

## 🔐 Security & Secrets

### Required Secrets

```bash
# Hugging Face token (for model downloads)
flyctl secrets set HF_TOKEN=hf_xxxxxxxxxxxxx

# Weights & Biases API key (for experiment tracking)
flyctl secrets set WANDB_API_KEY=xxxxxxxxxxxxx

# Optional: Custom training config
flyctl secrets set TRAINING_CONFIG='{"learning_rate": 1e-5}'
```

### Environment Variables

```bash
# In fly.toml
[env]
  PYTHONUNBUFFERED = "1"
  CUDA_VISIBLE_DEVICES = "0"
  TRANSFORMERS_CACHE = "/app/models"
  HF_HOME = "/app/models"
```

## 📈 Expected Results

### Performance Targets

| Metric                  | Baseline | INT8    | INT4    |
|-------------------------|----------|---------|---------|
| Model Size              | 14GB     | 3.5GB   | 1.75GB  |
| Inference Latency       | 100ms    | 40ms    | 30ms    |
| Throughput              | 10 t/s   | 25 t/s  | 33 t/s  |
| Tool Calling Accuracy   | 85%      | 84%     | 82%     |
| GPU Memory Usage        | 12GB     | 4GB     | 2.5GB   |

### Quality Preservation

- **INT8**: <2% perplexity increase
- **INT4**: <5% perplexity increase
- **Tool accuracy**: >80% for all quantization levels

## 🛠️ Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
# In train.py, decrease:
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
```

### Slow Training

```bash
# Enable gradient checkpointing
gradient_checkpointing = True

# Use mixed precision
fp16 = True
```

### Failed Quantization

```bash
# Try INT8 only (more stable)
python3 quantization/quantize.py --skip-int4

# Increase available memory
flyctl scale vm a100-80gb -a agentic-llm-phi4
```

## 📚 Additional Resources

- [Phi-4 Model Card](https://huggingface.co/microsoft/Phi-4)
- [Phi-4 ONNX](https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx)
- [Fly.io GPU Docs](https://fly.io/docs/gpus/)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk)
- [ONNX Runtime](https://onnxruntime.ai/)

## 🤝 Contributing

Contributions welcome! Please:
1. Test locally first
2. Include benchmarks
3. Update documentation
4. Follow existing patterns

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Microsoft Research for Phi-4
- Anthropic for Claude Agent SDK
- Fly.io for GPU infrastructure
- Hugging Face for model hosting

---

**Ready to optimize your Phi-4 model for Claude Agent SDK? Start with `./deployment/deploy.sh`!**

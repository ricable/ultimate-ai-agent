# Quick Start - Agentic LLM Phi-4 Training

Get started in 5 minutes with GPU training on Fly.io.

## Prerequisites (2 minutes)

```bash
# 1. Install Fly.io CLI
curl -L https://fly.io/install.sh | sh

# 2. Login to Fly.io
flyctl auth login

# 3. Get Hugging Face token
# Visit: https://huggingface.co/settings/tokens
```

## Deploy & Train (3 minutes)

```bash
# 1. Navigate to project
cd docker/agentic-llm

# 2. Run deployment script
./deployment/deploy.sh

# Enter your Hugging Face token when prompted
# Enter your WandB API key (optional, press Enter to skip)
```

That's it! Training will start automatically on an A100 GPU.

## Monitor Progress

```bash
# View real-time logs
flyctl logs -a agentic-llm-phi4 -f

# Check GPU utilization
flyctl ssh console -a agentic-llm-phi4 -C "nvidia-smi"
```

## What Happens Next?

The system will:

1. **Train** Phi-4 with LoRA (2-6 hours)
2. **Validate** with overfitting detection
3. **Save** best checkpoint automatically
4. **Stop** after completion (if auto-stop enabled)

## Access Results

```bash
# SSH into instance
flyctl ssh console -a agentic-llm-phi4

# View training metrics
cat /app/checkpoints/training_metrics.json

# Download model
flyctl ssh sftp shell -a agentic-llm-phi4
get /app/checkpoints/final_model ./
```

## Next Steps

After training completes:

```bash
# 1. Quantize model
python3 quantization/quantize.py \
  --model-path /app/checkpoints/final_model \
  --output-dir /app/models/quantized

# 2. Run benchmarks
python3 benchmarks/benchmark.py \
  --baseline /app/checkpoints/final_model \
  --quantized-int8 /app/models/quantized/phi4_int8.onnx

# 3. Deploy to Claude Agent SDK
cp /app/models/quantized/phi4_int8.onnx \
   /claude-agent-sdk/models/
```

## Cost Management

```bash
# Stop instance (save costs)
flyctl scale count 0 -a agentic-llm-phi4

# Restart when needed
flyctl scale count 1 -a agentic-llm-phi4
```

**Estimated cost**: $6-18 per training run (2-6 hours on A100 40GB)

## Troubleshooting

**Deployment fails?**
```bash
flyctl logs -a agentic-llm-phi4
```

**Need more memory?**
```bash
flyctl scale vm a100-80gb -a agentic-llm-phi4
```

**Training too slow?**
- Check batch size in `training/train.py`
- Reduce `per_device_train_batch_size` if OOM
- Increase `gradient_accumulation_steps` to maintain effective batch size

## Support

- 📖 Full docs: `README.md`
- 🚀 Deployment guide: `docs/DEPLOYMENT_GUIDE.md`
- 📚 Usage guide: `docs/USAGE_GUIDE.md`
- 💬 Issues: Open GitHub issue

---

**Ready? Run `./deployment/deploy.sh` to start!**

# Agentic LLM Documentation Index

## Quick Start
- **[README.md](./README.md)** - Project overview and main documentation
- **[QUICKSTART.md](./QUICKSTART.md)** - Quick start guide

## Usage & Deployment
- **[USAGE_GUIDE.md](./USAGE_GUIDE.md)** - Comprehensive usage documentation
- **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)** - Local and production deployment guide
- **[CLOUD_RUN_DEPLOYMENT.md](./CLOUD_RUN_DEPLOYMENT.md)** - Google Cloud Run deployment
- **[DEPLOYMENT_STATUS.md](./DEPLOYMENT_STATUS.md)** - Current deployment status
- **[DEPLOYMENT_SUMMARY.md](./DEPLOYMENT_SUMMARY.md)** - Deployment summary
- **[DEPLOYMENT_READY.txt](./DEPLOYMENT_READY.txt)** - Deployment checklist

## Fine-tuning & Training
- **[FINETUNING_GUIDE.md](./FINETUNING_GUIDE.md)** - Complete fine-tuning guide
- **[PHI4_FINETUNING_RESEARCH.md](./PHI4_FINETUNING_RESEARCH.md)** - Phi-4 model fine-tuning research

## Optimization & Performance
- **[OPTIMIZATION_RESULTS.md](./OPTIMIZATION_RESULTS.md)** - Performance optimization results
- **[BENCHMARK_OPTIMIZATION_GUIDE.md](./BENCHMARK_OPTIMIZATION_GUIDE.md)** - Benchmarking and optimization guide

## Directory Structure

```
agentic-llm/
├── docs/              # Documentation (you are here)
├── scripts/           # Shell scripts for running tasks
├── deployment/        # Deployment configuration (fly.toml)
├── configs/           # Model and training configurations
├── models/            # Model files and checkpoints
├── training/          # Training scripts and data
├── validation/        # Validation and testing
├── benchmarks/        # Benchmark scripts
├── quantization/      # Quantization tools
├── claude_sdk/        # Claude SDK integration
└── cloudrun/          # Cloud Run specific files
```

## Key Files
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `.dockerignore` - Docker build exclusions

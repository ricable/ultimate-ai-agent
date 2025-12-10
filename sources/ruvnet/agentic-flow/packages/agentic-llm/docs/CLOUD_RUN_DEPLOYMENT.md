# Cloud Run GPU Deployment Guide - Phi-4 Fine-tuning

Complete guide for deploying Phi-4 fine-tuning on Google Cloud Run with GPU.

## 🎯 Overview

This deployment uses:
- **GPU**: NVIDIA L4 (24GB VRAM, optimized for AI/ML)
- **Container**: CUDA 12.1 + PyTorch 2.2
- **Pipeline**: Baseline → Fine-tune → Validate → Benchmark
- **Duration**: ~5-7 hours total
- **Cost**: ~$20-30 per run

## 📋 Prerequisites

### 1. Google Cloud Setup

```bash
# Install gcloud CLI (if not already installed)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Login
gcloud auth login

# Set project
gcloud config set project agentic-llm

# Enable billing (required for GPUs)
# Visit: https://console.cloud.google.com/billing
```

### 2. Enable Required APIs

```bash
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    compute.googleapis.com
```

### 3. Create Secrets

```bash
# Hugging Face token (required for Phi-4 model download)
echo -n "hf_your_token_here" | gcloud secrets create huggingface-token \
    --data-file=- \
    --replication-policy=automatic

# Optional: Weights & Biases for experiment tracking
echo -n "your_wandb_key" | gcloud secrets create wandb-api-key \
    --data-file=- \
    --replication-policy=automatic
```

**Get Hugging Face token**: https://huggingface.co/settings/tokens

### 4. Request GPU Quota

Cloud Run GPU is in preview. Request access:

```bash
# Check current quota
gcloud compute project-info describe --project=agentic-llm

# Request quota increase if needed
# Visit: https://console.cloud.google.com/iam-admin/quotas
# Search for: "Cloud Run GPU"
```

## 🚀 Deployment

### Quick Deploy

```bash
cd /workspaces/agentic-flow/agentic-llm

# Run deployment script
./cloudrun/deploy.sh
```

### Manual Deployment Steps

#### 1. Build Container

```bash
gcloud builds submit \
    --tag gcr.io/agentic-llm/phi4-finetuning-gpu \
    --timeout=30m \
    --machine-type=e2-highcpu-8 \
    -f cloudrun/Dockerfile.gpu \
    .
```

**Build time**: ~15-20 minutes

#### 2. Deploy to Cloud Run

```bash
gcloud run deploy phi4-finetuning-gpu \
    --image gcr.io/agentic-llm/phi4-finetuning-gpu \
    --platform managed \
    --region us-central1 \
    --gpu nvidia-l4 \
    --gpu-count 1 \
    --memory 16Gi \
    --cpu 4 \
    --timeout 3600s \
    --min-instances 0 \
    --max-instances 1 \
    --allow-unauthenticated \
    --set-env-vars "PYTHONUNBUFFERED=1,CUDA_VISIBLE_DEVICES=0" \
    --set-secrets "HF_TOKEN=huggingface-token:latest" \
    --no-cpu-throttling
```

**Deployment time**: ~2-3 minutes

## 📊 Monitoring Training

### Check Status

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe phi4-finetuning-gpu \
    --region us-central1 \
    --format 'value(status.url)')

# Check training status
curl ${SERVICE_URL}/status
```

**Response:**
```json
{
  "phase": "finetuning",
  "status": "running",
  "timestamp": "2025-10-04T16:45:00",
  "elapsed_seconds": 3600,
  "details": {
    "eval_loss": 0.45,
    "current_epoch": 3
  }
}
```

### View Logs

```bash
# Real-time logs
gcloud run logs tail phi4-finetuning-gpu --region us-central1

# Last 50 lines
gcloud run logs read phi4-finetuning-gpu --region us-central1 --limit 50

# Filter for errors
gcloud run logs read phi4-finetuning-gpu --region us-central1 | grep ERROR
```

### Check Results

```bash
# List available results
curl ${SERVICE_URL}/results
```

## 📈 Training Pipeline

### Phase 1: Baseline Benchmarks (~15 min)

Tests original Phi-4 on MCP tool calling:
- 20 validation examples
- Measures accuracy, latency, correctness

**Output**: `/app/benchmarks/results/baseline_results.json`

### Phase 2: Fine-tuning (~4-6 hours)

Fine-tunes Phi-4 with LoRA on MCP tools:
- 60-100 training examples (with augmentation)
- 5 epochs with early stopping
- LoRA r=32, alpha=64

**Output**: `/app/checkpoints/mcp_finetuned/final_model/`

### Phase 3: Validation (~10 min)

Validates fine-tuned model:
- Tool detection accuracy
- Tool name correctness
- Parameter extraction

**Output**: `/app/validation/results/mcp_validation_results.json`

### Phase 4: Post-Training Benchmarks (~10 min)

Compares baseline vs fine-tuned:
- Accuracy improvements
- Performance metrics
- Tool-by-tool breakdown

**Output**: `/app/benchmarks/results/finetune_comparison.json`

### Phase 5: Final Report

Generates comprehensive report:
- Training summary
- Success criteria validation
- Improvement metrics

**Output**: `/app/benchmarks/results/final_training_report.json`

## 📥 Retrieving Results

### Method 1: Download from Container

```bash
# SSH into container (while running)
gcloud run services proxy phi4-finetuning-gpu --region us-central1 --port 8080

# In another terminal, download results
curl http://localhost:8080/results
```

### Method 2: Cloud Storage (Recommended)

Add to `cloudrun/cloud_runner.py` after training:

```python
from google.cloud import storage

def upload_results_to_gcs():
    client = storage.Client()
    bucket = client.bucket("phi4-training-results")

    results_dir = Path("/app/benchmarks/results")
    for file in results_dir.glob("*.json"):
        blob = bucket.blob(f"runs/{datetime.now():%Y%m%d}/{file.name}")
        blob.upload_from_filename(file)
```

### Method 3: Container Commit

```bash
# Get container ID
gcloud run revisions describe [REVISION] --region us-central1

# Create image from running container
gcloud builds submit --tag gcr.io/agentic-llm/phi4-results

# Extract results locally
docker run --rm -v $(pwd)/results:/output \
    gcr.io/agentic-llm/phi4-results \
    cp -r /app/benchmarks/results /output
```

## 💰 Cost Management

### Pricing

- **Cloud Run GPU (L4)**: ~$3.00/hour
- **Container Build**: ~$0.003/build-minute
- **Container Registry**: ~$0.026/GB/month

### Estimated Costs Per Run

| Component | Duration | Cost |
|-----------|----------|------|
| Container build | 20 min | $0.06 |
| Baseline benchmarks | 15 min | $0.75 |
| Fine-tuning | 5 hours | $15.00 |
| Validation | 10 min | $0.50 |
| Post-training benchmarks | 10 min | $0.50 |
| **Total** | **~6 hours** | **~$17** |

### Cost Optimization

```bash
# Set timeout to auto-stop
gcloud run services update phi4-finetuning-gpu \
    --region us-central1 \
    --timeout 7200s  # 2 hours max

# Use preemptible instances (when available)
gcloud run services update phi4-finetuning-gpu \
    --region us-central1 \
    --execution-environment gen2

# Delete service after training
gcloud run services delete phi4-finetuning-gpu --region us-central1
```

## 🐛 Troubleshooting

### GPU Not Available

```bash
# Check GPU quota
gcloud compute project-info describe --project=agentic-llm

# Request GPU access
# Visit: https://cloud.google.com/run/docs/configuring/services/gpu
```

### Out of Memory

Reduce batch size in `/app/configs/training_config.json`:

```json
{
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 8
}
```

### Timeout Errors

Increase timeout:

```bash
gcloud run services update phi4-finetuning-gpu \
    --region us-central1 \
    --timeout 7200s
```

### Secret Access Errors

Grant service account access:

```bash
gcloud secrets add-iam-policy-binding huggingface-token \
    --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## 📚 Next Steps

After successful training:

1. **Download Results**
   ```bash
   curl ${SERVICE_URL}/results > results.json
   ```

2. **Review Metrics**
   - Check `final_training_report.json`
   - Verify tool accuracy > 95%

3. **Export Model** (if successful)
   - Quantize to ONNX INT8/INT4
   - Deploy to production

4. **Clean Up**
   ```bash
   gcloud run services delete phi4-finetuning-gpu --region us-central1
   ```

## 🔗 Resources

- [Cloud Run GPU Docs](https://cloud.google.com/run/docs/configuring/services/gpu)
- [Phi-4 Model Card](https://huggingface.co/microsoft/Phi-4)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [MCP Tools](https://github.com/anthropics/claude-code)

---

**Ready to deploy?** Run `./cloudrun/deploy.sh`

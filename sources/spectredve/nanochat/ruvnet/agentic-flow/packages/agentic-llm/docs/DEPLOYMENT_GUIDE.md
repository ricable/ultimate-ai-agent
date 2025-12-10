# Deployment Guide - Agentic LLM on Fly.io

Complete guide for deploying Phi-4 training system on Fly.io with GPU support.

## Prerequisites

### 1. Install Fly.io CLI

**macOS:**
```bash
brew install flyctl
```

**Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

**Windows:**
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

### 2. Create Fly.io Account

```bash
# Sign up and login
flyctl auth signup  # or flyctl auth login
```

### 3. Verify Installation

```bash
flyctl version
flyctl auth whoami
```

## Deployment Steps

### Step 1: Prepare Environment

```bash
cd docker/agentic-llm

# Make deployment script executable
chmod +x deployment/deploy.sh
```

### Step 2: Configure Secrets

```bash
# Set Hugging Face token
flyctl secrets set HF_TOKEN=hf_your_token_here -a agentic-llm-phi4

# Set Weights & Biases API key (optional)
flyctl secrets set WANDB_API_KEY=your_wandb_key -a agentic-llm-phi4
```

### Step 3: Review Configuration

Edit `fly.toml` if needed:

```toml
# GPU instance type
[vm]
  size = "a100-40gb"  # Options: a100-40gb, a100-80gb
  memory = "32gb"
  cpus = 8

# Storage volumes
[[mounts]]
  source = "phi4_models"
  destination = "/app/models"
  initial_size = "100gb"  # Adjust based on needs
```

### Step 4: Deploy

```bash
# Automated deployment
./deployment/deploy.sh
```

Or manually:

```bash
# Create app
flyctl launch --name agentic-llm-phi4 --region ord

# Create volumes
flyctl volumes create phi4_models --region ord --size 100 -a agentic-llm-phi4
flyctl volumes create phi4_checkpoints --region ord --size 50 -a agentic-llm-phi4

# Deploy
flyctl deploy
```

### Step 5: Monitor Deployment

```bash
# Check status
flyctl status -a agentic-llm-phi4

# View logs
flyctl logs -a agentic-llm-phi4

# SSH into container
flyctl ssh console -a agentic-llm-phi4
```

## GPU Instance Options

### A100 40GB (Recommended for Phi-4)

```toml
[vm]
  size = "a100-40gb"
  memory = "32gb"
  cpus = 8
```

**Specs:**
- GPU: NVIDIA A100 40GB
- Cost: ~$3.00/hour
- Best for: Phi-4 (14B params)

### A100 80GB (For larger batches)

```toml
[vm]
  size = "a100-80gb"
  memory = "48gb"
  cpus = 12
```

**Specs:**
- GPU: NVIDIA A100 80GB
- Cost: ~$4.50/hour
- Best for: Larger batch sizes, multiple experiments

## Storage Configuration

### Volume Sizing

**Models Volume (phi4_models):**
- Phi-4 base model: ~28GB
- LoRA adapters: ~500MB
- Quantized models: ~3.5GB (INT8), ~1.75GB (INT4)
- Recommended: **100GB**

**Checkpoints Volume (phi4_checkpoints):**
- Checkpoints during training: ~5-10GB per checkpoint
- Keep 3-5 checkpoints with `save_total_limit`
- Recommended: **50GB**

### Managing Volumes

```bash
# List volumes
flyctl volumes list -a agentic-llm-phi4

# Extend volume size
flyctl volumes extend phi4_models --size 150 -a agentic-llm-phi4

# Create snapshot
flyctl volumes snapshots create phi4_models -a agentic-llm-phi4
```

## Environment Variables

### Required

```bash
HF_TOKEN         # Hugging Face token for model downloads
```

### Optional

```bash
WANDB_API_KEY    # Weights & Biases for experiment tracking
CUDA_VISIBLE_DEVICES=0  # GPU device selection
TRANSFORMERS_CACHE=/app/models  # Model cache directory
```

### Setting Variables

```bash
# Secrets (encrypted)
flyctl secrets set KEY=value -a agentic-llm-phi4

# Environment variables (non-encrypted)
flyctl config env set KEY=value -a agentic-llm-phi4
```

## Scaling Operations

### Scale to Zero (Save Costs)

```bash
# Stop all instances
flyctl scale count 0 -a agentic-llm-phi4

# Start when needed
flyctl scale count 1 -a agentic-llm-phi4
```

### Auto-Stop

```bash
# Auto-stop after inactivity
flyctl apps update --auto-stop-machines=true -a agentic-llm-phi4

# Auto-start on request
flyctl apps update --auto-start-machines=true -a agentic-llm-phi4
```

### Change VM Size

```bash
# Upgrade to A100 80GB
flyctl scale vm a100-80gb -a agentic-llm-phi4

# Downgrade (after training)
flyctl scale vm shared-cpu-1x -a agentic-llm-phi4
```

## Monitoring & Debugging

### Real-time Logs

```bash
# Follow logs
flyctl logs -a agentic-llm-phi4

# Filter by level
flyctl logs --level error -a agentic-llm-phi4

# Last 100 lines
flyctl logs --lines 100 -a agentic-llm-phi4
```

### SSH Access

```bash
# Connect to running instance
flyctl ssh console -a agentic-llm-phi4

# Run commands
flyctl ssh console -a agentic-llm-phi4 -C "nvidia-smi"
flyctl ssh console -a agentic-llm-phi4 -C "python3 --version"
```

### Health Checks

```bash
# Check app status
flyctl status -a agentic-llm-phi4

# Check GPU utilization
flyctl ssh console -a agentic-llm-phi4 -C "nvidia-smi"

# Check disk usage
flyctl ssh console -a agentic-llm-phi4 -C "df -h"
```

### Metrics Dashboard

```bash
# Open metrics dashboard
flyctl dashboard metrics -a agentic-llm-phi4
```

## Cost Management

### Estimate Costs

**Training Session:**
- Instance: $3.00/hour × 6 hours = $18
- Storage: 150GB × $0.15/GB/month ÷ 730 hours × 6 = $0.12
- **Total**: ~$18.12

**Monthly (if running 24/7):**
- Instance: $3.00 × 730 = $2,190
- Storage: 150GB × $0.15 = $22.50
- **Total**: ~$2,212.50/month

### Cost Optimization Tips

1. **Scale to Zero**: When not training
2. **Use Auto-Stop**: After training completes
3. **Delete Old Checkpoints**: Keep only best models
4. **Scheduled Training**: Run during off-peak hours
5. **Snapshot Volumes**: Store backups externally

### Budget Alerts

Set up billing alerts in Fly.io dashboard:
1. Go to https://fly.io/dashboard
2. Navigate to Billing
3. Set spending limits
4. Configure email alerts

## Networking

### Internal DNS

```bash
# App is accessible at:
agentic-llm-phi4.internal
```

### External Access (Optional)

```toml
[[services]]
  internal_port = 8080
  protocol = "tcp"

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]
```

### API Endpoint

If exposing inference API:

```bash
# Get app URL
flyctl info -a agentic-llm-phi4

# Test endpoint
curl https://agentic-llm-phi4.fly.dev/health
```

## Backup & Recovery

### Volume Snapshots

```bash
# Create snapshot
flyctl volumes snapshots create phi4_models -a agentic-llm-phi4

# List snapshots
flyctl volumes snapshots list phi4_models -a agentic-llm-phi4

# Restore from snapshot
flyctl volumes create phi4_models_restored \
  --snapshot-id <snapshot-id> \
  -a agentic-llm-phi4
```

### Export Models

```bash
# SSH and download
flyctl ssh console -a agentic-llm-phi4

# Inside container
tar -czf /tmp/checkpoints.tar.gz /app/checkpoints/final_model

# Exit and copy locally
flyctl ssh sftp shell -a agentic-llm-phi4
get /tmp/checkpoints.tar.gz ./
```

## Troubleshooting

### Deployment Fails

```bash
# Check build logs
flyctl logs -a agentic-llm-phi4

# Validate fly.toml
flyctl config validate

# Rebuild
flyctl deploy --build-only
```

### Out of Memory

```bash
# Increase VM memory
flyctl scale vm a100-80gb -a agentic-llm-phi4

# Or reduce batch size in train.py
```

### GPU Not Detected

```bash
# Verify GPU
flyctl ssh console -a agentic-llm-phi4 -C "nvidia-smi"

# Check CUDA
flyctl ssh console -a agentic-llm-phi4 -C "python3 -c 'import torch; print(torch.cuda.is_available())'"
```

### Volume Not Mounted

```bash
# Check mounts
flyctl ssh console -a agentic-llm-phi4 -C "df -h"

# Verify fly.toml mounts section
# Recreate volume if needed
```

## Security Best Practices

1. **Never commit secrets** to git
2. **Use Fly.io secrets** for sensitive data
3. **Rotate tokens** regularly
4. **Limit SSH access** to trusted IPs
5. **Review logs** for unusual activity
6. **Keep base image updated**

## Advanced Configuration

### Custom Dockerfile

Modify `Dockerfile` for specific needs:

```dockerfile
# Add custom dependencies
RUN pip3 install your-package

# Copy custom scripts
COPY scripts/ /app/scripts/
```

### Multi-Region Deployment

```bash
# Add regions
flyctl regions add lhr sea -a agentic-llm-phi4

# Set primary region
flyctl regions set ord -a agentic-llm-phi4
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Deploy to Fly.io
  run: |
    flyctl deploy --remote-only
  env:
    FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}
```

## Support

- **Fly.io Docs**: https://fly.io/docs
- **Community Forum**: https://community.fly.io
- **Status Page**: https://status.flyio.net
- **Support**: support@fly.io

## Next Steps

After successful deployment:

1. Monitor training progress with logs
2. Set up WandB for experiment tracking
3. Run validation on completed model
4. Execute quantization pipeline
5. Run comprehensive benchmarks
6. Export and deploy model

---

**Ready to deploy? Run `./deployment/deploy.sh` to get started!**

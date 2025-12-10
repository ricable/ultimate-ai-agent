# Agentic Flow - GCP Deployment Guide

Deploy agentic-flow to Google Cloud Platform with Docker containers, Cloud Run, and Secret Manager integration.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Deployment Options](#deployment-options)
- [Configuration](#configuration)
- [Security](#security)
- [Monitoring](#monitoring)
- [Cost Optimization](#cost-optimization)

## Quick Start

### Prerequisites

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
export GCP_PROJECT_ID="your-project-id"
gcloud config set project $GCP_PROJECT_ID
```

### 1. Setup Secrets

```bash
cd docker/cloud-run

# Option A: From .env file
GCP_PROJECT_ID=$GCP_PROJECT_ID ./setup-secrets.sh setup

# Option B: Interactive
GCP_PROJECT_ID=$GCP_PROJECT_ID ./setup-secrets.sh interactive

# Verify secrets
GCP_PROJECT_ID=$GCP_PROJECT_ID ./setup-secrets.sh list
```

### 2. Deploy to Cloud Run

```bash
# Deploy with Claude (default)
GCP_PROJECT_ID=$GCP_PROJECT_ID \
MODEL_PROVIDER=anthropic \
./deploy.sh

# Deploy with OpenRouter (99% cost savings!)
GCP_PROJECT_ID=$GCP_PROJECT_ID \
MODEL_PROVIDER=openrouter \
MODEL_NAME=meta-llama/llama-3.1-8b-instruct \
./deploy.sh

# Deploy with custom configuration
GCP_PROJECT_ID=$GCP_PROJECT_ID \
MODEL_PROVIDER=openrouter \
MODEL_NAME=deepseek/deepseek-chat-v3.1 \
REGION=us-west1 \
MEMORY=4Gi \
CPU=4 \
MAX_INSTANCES=20 \
./deploy.sh
```

### 3. Test Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe agentic-flow \
  --region=us-central1 \
  --format='value(status.url)')

# Health check
curl $SERVICE_URL/health

# List agents
curl $SERVICE_URL/agents

# Run an agent
curl -X POST $SERVICE_URL/agent \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "task": "Create a Python hello world script"
  }'
```

## Architecture

### Docker Multi-Stage Build

```
┌─────────────────────────────────────┐
│ Stage 1: Builder                     │
│  - Node 20 Alpine                    │
│  - Install dependencies              │
│  - Build TypeScript → JavaScript     │
│  - Copy .claude agents               │
└─────────────────────────────────────┘
            ↓
┌─────────────────────────────────────┐
│ Stage 2: Production                  │
│  - Node 20 Alpine (minimal)          │
│  - Copy built artifacts              │
│  - Non-root user (nodejs:1001)       │
│  - Health checks enabled             │
│  - Tini for signal handling          │
└─────────────────────────────────────┘
```

### GCP Services Used

- **Cloud Run**: Serverless container hosting
- **Secret Manager**: API key and credentials storage
- **Container Registry**: Docker image storage
- **Cloud Build**: Automated Docker builds
- **Cloud Logging**: Centralized logging
- **Cloud Monitoring**: Performance metrics

## Deployment Options

### Option 1: Cloud Run (Recommended)

**Pros:**
- Serverless, auto-scaling
- Pay only for requests
- HTTPS out of the box
- No infrastructure management

**Use Cases:**
- Production deployments
- API endpoints
- Variable load patterns

```bash
cd docker/cloud-run
GCP_PROJECT_ID=my-project ./deploy.sh
```

### Option 2: Local Docker Compose

**Pros:**
- Full control
- Local development
- Multiple configurations

**Use Cases:**
- Development
- Testing
- On-premise deployments

```bash
cd docker
docker-compose up -d

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale agentic-flow-openrouter=3
```

### Option 3: GKE (Kubernetes)

**Pros:**
- Full Kubernetes features
- Advanced orchestration
- Multi-region deployments

**Use Cases:**
- Enterprise deployments
- Complex workflows
- High availability

```bash
# Coming soon - kubernetes/ directory
cd docker/kubernetes
kubectl apply -f deployment.yaml
```

## Configuration

### Model Providers

#### Claude (Anthropic)

**Best for:** Complex reasoning, code reviews, architecture

```bash
# Configuration
MODEL_PROVIDER=anthropic
MODEL_NAME=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=sk-ant-...

# Cost: ~$0.080 per task
```

#### OpenRouter

**Best for:** Cost savings, variety of models

```bash
# Llama 3.1 8B - Fastest, cheapest
MODEL_PROVIDER=openrouter
MODEL_NAME=meta-llama/llama-3.1-8b-instruct
OPENROUTER_API_KEY=sk-or-...

# Cost: ~$0.0003 per task (99% savings!)

# Other options:
# DeepSeek - Excellent for code
MODEL_NAME=deepseek/deepseek-chat-v3.1

# Gemini 2.5 Flash - Very fast
MODEL_NAME=google/gemini-2.5-flash-preview

# Claude via OpenRouter - Same quality
MODEL_NAME=anthropic/claude-3-5-sonnet
```

#### ONNX (Local Models)

**Best for:** Offline, air-gapped environments

```bash
MODEL_PROVIDER=onnx
MODEL_PATH=/app/models/phi-4.onnx

# Cost: $0 (runs locally, one-time model download)
```

### Environment Templates

Use pre-configured templates from `docker/configs/`:

- `claude.env.template` - Premium Claude configuration
- `openrouter.env.template` - Cost-optimized OpenRouter
- `multi-model.env.template` - Multiple providers simultaneously

```bash
# Copy and customize
cp docker/configs/openrouter.env.template .env
# Edit .env with your API keys
nano .env
```

### Resource Configuration

```bash
# Small deployment (development)
MEMORY=1Gi
CPU=1
MAX_INSTANCES=3
CONCURRENCY=40

# Medium deployment (production)
MEMORY=2Gi
CPU=2
MAX_INSTANCES=10
CONCURRENCY=80

# Large deployment (enterprise)
MEMORY=4Gi
CPU=4
MAX_INSTANCES=50
CONCURRENCY=160
```

## Security

### Secret Management

**Never commit secrets to Git!**

Use GCP Secret Manager for all sensitive data:

```bash
# Create secrets
./setup-secrets.sh setup

# Grant access to service account
./setup-secrets.sh grant-all

# List secrets
./setup-secrets.sh list

# Rotate secret
gcloud secrets versions add ANTHROPIC_API_KEY --data-file=new-key.txt

# Disable old version
gcloud secrets versions disable 1 --secret=ANTHROPIC_API_KEY
```

### IAM Permissions

Required roles for deployment:

```bash
# Grant roles to service account
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:SERVICE_ACCOUNT@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:SERVICE_ACCOUNT@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Network Security

```bash
# Restrict to VPC (optional)
gcloud run services update agentic-flow \
  --vpc-connector=my-connector \
  --vpc-egress=all-traffic

# Require authentication
gcloud run services update agentic-flow \
  --no-allow-unauthenticated

# Add custom domain with SSL
gcloud run domain-mappings create \
  --service=agentic-flow \
  --domain=agents.example.com
```

## Monitoring

### Cloud Logging

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=agentic-flow" \
  --limit=50 \
  --format=json

# Filter by severity
gcloud logging read "severity>=ERROR" --limit=20

# Stream logs
gcloud logging tail "resource.type=cloud_run_revision"
```

### Cloud Monitoring

```bash
# Create alerting policy for errors
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="High Error Rate" \
  --condition-display-name="Error rate > 5%" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s
```

### Health Checks

Built-in health endpoint:

```bash
# Manual health check
curl https://your-service-url/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2025-10-04T20:30:00Z",
  "uptime": 3600,
  "memory": {"used": "150MB", "free": "1.85GB"},
  "agents": {"loaded": 66, "status": "ready"}
}
```

## Cost Optimization

### Model Selection

**Cost Comparison (per 1000 tasks):**

| Provider | Model | Cost | Savings |
|----------|-------|------|---------|
| Anthropic | Claude 3.5 Sonnet | $80.00 | 0% |
| OpenRouter | Llama 3.1 8B | $0.30 | 99.6% |
| OpenRouter | DeepSeek | $0.50 | 99.4% |
| OpenRouter | Gemini 2.5 Flash | $1.00 | 98.8% |
| ONNX | Phi-4 (local) | $0.00 | 100% |

### Resource Optimization

```bash
# Scale to zero when idle
MIN_INSTANCES=0  # Saves money during low traffic

# Optimize for cost
MEMORY=1Gi       # Minimum viable memory
CPU=1            # Single CPU core
CONCURRENCY=100  # Max requests per container

# Set request timeout
TIMEOUT=60       # Faster timeout = lower costs
```

### Traffic Routing

Route expensive Claude requests only for complex tasks:

```javascript
// Use Claude for complex tasks
if (taskComplexity === 'high') {
  provider = 'anthropic';
  model = 'claude-3-5-sonnet';
}
// Use OpenRouter for simple tasks
else {
  provider = 'openrouter';
  model = 'meta-llama/llama-3.1-8b-instruct';
}
```

### Monitoring Costs

```bash
# View Cloud Run costs
gcloud billing accounts get-iam-policy BILLING_ACCOUNT_ID

# Set budget alerts
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Agentic Flow Budget" \
  --budget-amount=100USD \
  --threshold-rule=percent=80
```

## Troubleshooting

### Common Issues

**1. Deployment fails with "permission denied"**

```bash
# Grant Cloud Build permissions
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member=serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com \
  --role=roles/run.admin
```

**2. Container fails to start**

```bash
# Check logs
gcloud logging read "resource.type=cloud_run_revision" --limit=50

# Check build logs
gcloud builds list --limit=5
gcloud builds log BUILD_ID
```

**3. Out of memory errors**

```bash
# Increase memory
gcloud run services update agentic-flow --memory=4Gi

# Or reduce concurrency
gcloud run services update agentic-flow --concurrency=40
```

**4. Secrets not accessible**

```bash
# Grant secret access
./setup-secrets.sh grant-all

# Verify service account
gcloud run services describe agentic-flow --format="value(spec.template.spec.serviceAccountName)"
```

## Advanced Configuration

### Multi-Region Deployment

```bash
# Deploy to multiple regions
for region in us-central1 europe-west1 asia-east1; do
  gcloud run deploy agentic-flow-$region \
    --image=gcr.io/$PROJECT_ID/agentic-flow \
    --region=$region \
    --platform=managed
done

# Setup global load balancer
gcloud compute backend-services create agentic-flow-backend \
  --global
```

### CI/CD Integration

```yaml
# .github/workflows/deploy.yml
name: Deploy to Cloud Run
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: google-github-actions/setup-gcloud@v0
      - run: |
          cd docker/cloud-run
          ./deploy.sh
```

### Custom Domain

```bash
# Map custom domain
gcloud run domain-mappings create \
  --service=agentic-flow \
  --domain=api.yourdomain.com \
  --region=us-central1

# Get DNS records to configure
gcloud run domain-mappings describe \
  --domain=api.yourdomain.com \
  --region=us-central1
```

## Support

- Documentation: `/docs`
- Issues: https://github.com/ruvnet/agentic-flow/issues
- NPM: https://www.npmjs.com/package/agentic-flow

## License

See LICENSE file in the repository root.

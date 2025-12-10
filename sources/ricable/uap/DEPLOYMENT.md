# UAP Production Deployment Guide

## Overview

This guide covers deploying the UAP (Unified Agentic Platform) to production environments using SkyPilot for multi-cloud deployment with comprehensive monitoring and secrets management.

## Prerequisites

### Required Tools
- [SkyPilot](https://github.com/skypilot-org/skypilot) - Multi-cloud orchestration
- [Teller](https://github.com/tellerops/teller) - Secrets management
- [Docker](https://docker.com) - Containerization
- [DevBox](https://get.jetify.com/devbox) - Development environment

### Cloud Provider Setup
Before deployment, ensure you have:
- Valid cloud provider credentials (AWS, GCP, or Azure)
- Sufficient quota for GPU instances
- Secrets properly configured in your chosen secrets provider

## Quick Start

### 1. Basic Production Deployment

```bash
# Deploy to auto-selected cloud with health checks
./scripts/deploy-production.sh --cloud auto --test --monitor

# Deploy to specific cloud provider
./scripts/deploy-production.sh --cloud gcp --env production --backup

# Cost-optimized deployment
./scripts/deploy-production.sh --cloud cost-optimized --monitor
```

### 2. Setup Monitoring

```bash
# Setup comprehensive monitoring stack
./scripts/setup-monitoring.sh

# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d
```

### 3. Health Check

```bash
# Verify deployment health
./scripts/health-check.sh
```

## Deployment Configurations

### Multi-Cloud Configurations

#### 1. Auto-Selection (Recommended)
Uses the general production configuration that supports failover across all clouds:
```bash
./scripts/deploy-production.sh --cloud auto
```

#### 2. AWS-Specific Deployment
Optimized for AWS with spot instance handling:
```bash
./scripts/deploy-production.sh --cloud aws --region us-west-2
```

#### 3. Google Cloud Deployment
Optimized for GCP with preemptible instances:
```bash
./scripts/deploy-production.sh --cloud gcp --region us-central1
```

#### 4. Azure Deployment
Optimized for Azure with spot VM handling:
```bash
./scripts/deploy-production.sh --cloud azure --region eastus
```

#### 5. Cost-Optimized Deployment
Automatically selects cheapest resources across all clouds:
```bash
./scripts/deploy-production.sh --cloud cost-optimized
```

## Configuration Files

### SkyPilot Configurations
- `skypilot/uap-production.yaml` - General production (multi-cloud)
- `skypilot/uap-aws.yaml` - AWS-specific optimizations
- `skypilot/uap-gcp.yaml` - GCP-specific optimizations
- `skypilot/uap-azure.yaml` - Azure-specific optimizations
- `skypilot/uap-cost-optimized.yaml` - Cost optimization priority

### Environment Templates
- `.env.production.template` - Production environment variables
- `.env.staging.template` - Staging environment variables

### Docker Configurations
- `Dockerfile` - Multi-stage production build
- `docker-compose.production.yml` - Complete production stack
- `docker-compose.monitoring.yml` - Monitoring services

## Secrets Management

### Teller Configuration
The `.teller.yml` file configures multi-provider secrets management:

```yaml
providers:
  google_secret_manager: # Primary
  hashicorp_vault:       # Secondary  
  aws_secret_manager:    # Tertiary
```

### Required Secrets
Core secrets that must be configured:

#### Framework API Keys
- `COPILOTKIT_API_KEY` - CopilotKit framework access
- `AGNO_API_KEY` - Agno framework access
- `MASTRA_API_KEY` - Mastra framework access

#### LLM API Keys
- `OPENAI_API_KEY` - OpenAI API access
- `ANTHROPIC_API_KEY` - Anthropic API access

#### Infrastructure
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `JWT_SECRET` - JWT signing secret

#### Cloud Provider Credentials
- `GOOGLE_APPLICATION_CREDENTIALS_JSON` - GCP service account
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - AWS credentials
- `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` - Azure credentials

### Setting Up Secrets

#### Google Secret Manager (Recommended)
```bash
# Create secrets in Google Secret Manager
gcloud secrets create openai-api-key --data-file=openai-key.txt
gcloud secrets create anthropic-api-key --data-file=anthropic-key.txt
# ... create other secrets
```

#### Test Secrets Access
```bash
# Test secrets configuration
teller run echo "Secrets loaded successfully"
```

## Deployment Options

### Command Line Options

```bash
./scripts/deploy-production.sh [OPTIONS]

Options:
  -c, --cloud CLOUD       Target cloud (aws|gcp|azure|auto|cost-optimized)
  -e, --env ENV          Environment (production|staging)
  -r, --region REGION    Target region
  -t, --test            Run tests before deployment
  -d, --dry-run         Show deployment plan without executing
  -f, --force           Force deployment even if health checks fail
  -b, --backup          Create backup before deployment
  -m, --monitor         Enable monitoring setup
  -h, --help            Show help
```

### Environment Variables

Key environment variables for customization:

```bash
# Resource Configuration
export UVICORN_WORKERS=4
export AGNO_GPU_MEMORY="8GB"
export MASTRA_WORKER_COUNT=4

# Performance Tuning
export MAX_CONCURRENT_REQUESTS=1000
export REQUEST_TIMEOUT=300
export RATE_LIMIT_PER_MINUTE=100

# Feature Flags
export ENABLE_METRICS=true
export ENABLE_TRACING=true
export ENABLE_RATE_LIMITING=true
```

## Monitoring and Observability

### Monitoring Stack
The monitoring setup includes:
- **Prometheus** - Metrics collection and alerting
- **Grafana** - Visualization and dashboards  
- **Node Exporter** - System metrics
- **cAdvisor** - Container metrics
- **Custom exporters** - Redis, PostgreSQL, NGINX metrics

### Key Metrics
Monitor these critical metrics:

#### Application Metrics
- API response time (target: <2s p95)
- Request rate and error rate
- Active WebSocket connections
- Agent framework health status

#### Infrastructure Metrics
- CPU usage (alert: >80%)
- Memory usage (alert: >85%)
- Disk space (alert: <10% free)
- GPU utilization

#### Business Metrics
- Agent interaction count
- Framework routing efficiency
- User session duration

### Accessing Monitoring

```bash
# Grafana Dashboard
http://your-deployment-ip:3001
# Default: admin/admin

# Prometheus
http://your-deployment-ip:9090

# Direct metrics endpoint
http://your-deployment-ip:8000/metrics
```

### Alerting Rules
Configured alerts include:
- High response time (>2s for 2 minutes)
- High error rate (>5% for 1 minute)
- Agent framework down (>30 seconds)
- Resource exhaustion (CPU >80%, Memory >85%)

## Troubleshooting

### Common Issues

#### 1. Deployment Fails
```bash
# Check SkyPilot status
sky status --refresh

# View deployment logs
sky logs uap

# Check secrets access
teller run env | grep -E "API_KEY|SECRET"
```

#### 2. Health Check Failures
```bash
# Detailed health check
./scripts/health-check.sh

# Check individual services
curl http://your-ip:8000/health
curl http://your-ip:8000/agents/status
```

#### 3. Framework Issues
```bash
# Check framework logs
sky ssh uap "tail -f /app/logs/backend.log"

# Restart specific service
sky ssh uap "sudo systemctl restart uap"
```

#### 4. Performance Issues
```bash
# Check resource usage
sky ssh uap "htop"
sky ssh uap "nvidia-smi"  # If GPU available

# Review metrics in Grafana
# Navigate to UAP Overview dashboard
```

### Recovery Procedures

#### 1. Rollback Deployment
```bash
# Stop current deployment
sky down uap -y

# Restore from backup (if created)
# Manual restore using backup files in /tmp/uap-backup-*
```

#### 2. Scale Resources
```bash
# Update resource requirements in config
vim skypilot/uap-production.yaml

# Redeploy with new resources
./scripts/deploy-production.sh --force
```

#### 3. Emergency Maintenance
```bash
# Access deployment directly
sky ssh uap

# Check service status
sudo systemctl status uap

# View logs
journalctl -u uap -f
```

## Cost Optimization

### Strategies
1. **Spot Instances** - All configurations use spot/preemptible instances
2. **Multi-Cloud** - Automatic selection of cheapest provider
3. **Resource Right-Sizing** - Configurable CPU/memory/GPU requirements
4. **Auto-Shutdown** - Configured idle detection and shutdown

### Cost Monitoring
```bash
# Check current costs
sky cost-report

# Optimize for cost
./scripts/deploy-production.sh --cloud cost-optimized
```

## Security Considerations

### Production Security Checklist
- [ ] Secrets stored in secure provider (not environment files)
- [ ] Non-root container execution
- [ ] Network segmentation configured
- [ ] TLS/SSL certificates configured
- [ ] Access logging enabled
- [ ] Rate limiting configured
- [ ] Regular security updates scheduled

### Network Security
- Firewall rules limiting access to necessary ports only
- Internal service communication over private networks
- External access through load balancer/reverse proxy only

## Scaling

### Horizontal Scaling
```bash
# Deploy multiple instances
sky up -c skypilot/uap-production.yaml --cluster-name uap-west
sky up -c skypilot/uap-production.yaml --cluster-name uap-east

# Configure load balancing between instances
```

### Vertical Scaling
```bash
# Update resource requirements
# Edit skypilot/*.yaml files to increase CPU/memory/GPU

# Redeploy with new resources
./scripts/deploy-production.sh --force
```

## Framework Integration Status

### Current State
The deployment infrastructure is ready for real framework implementations:

- **CopilotKit**: Ready for integration (currently mock implementation)
- **Agno**: Ready for integration (currently mock implementation)  
- **Mastra**: Ready for integration (currently mock implementation)

### Post-Integration Steps
When Agents 3, 4, 5 complete framework integrations:

1. Update `backend/requirements.txt` with real framework dependencies
2. Uncomment framework installations in SkyPilot configurations
3. Update secrets with real API keys and configurations
4. Redeploy with real framework implementations

## Support

### Getting Help
1. Check deployment logs: `sky logs uap`
2. Review health checks: `./scripts/health-check.sh`
3. Monitor metrics in Grafana dashboard
4. Check this documentation for troubleshooting steps

### Maintenance
- Regular backup creation before deployments
- Monitor resource usage and costs
- Update secrets rotation schedule
- Review and update alerting thresholds

## Appendix

### File Structure
```
skypilot/
├── uap-production.yaml     # Multi-cloud production
├── uap-aws.yaml           # AWS-specific
├── uap-gcp.yaml           # GCP-specific
├── uap-azure.yaml         # Azure-specific
└── uap-cost-optimized.yaml # Cost optimization

scripts/
├── deploy-production.sh    # Main deployment script
├── setup-monitoring.sh     # Monitoring setup
├── start-production.sh     # Production startup
└── health-check.sh        # Health verification

monitoring/
├── prometheus.yml         # Metrics collection
├── alerts.yml            # Alert rules
├── grafana/              # Dashboards and datasources
└── nginx/               # Load balancer config
```

### Resource Requirements

#### Minimum Requirements
- **CPU**: 4+ cores
- **Memory**: 16+ GB RAM
- **Storage**: 100 GB SSD
- **GPU**: Optional (T4/V100/A100 supported)

#### Recommended Production
- **CPU**: 8+ cores  
- **Memory**: 32+ GB RAM
- **Storage**: 200 GB SSD
- **GPU**: A100 or V100 for optimal performance

#### Cost-Optimized
- **CPU**: 4+ cores
- **Memory**: 16+ GB RAM  
- **Storage**: 100 GB standard disk
- **GPU**: T4 or L4 for cost efficiency
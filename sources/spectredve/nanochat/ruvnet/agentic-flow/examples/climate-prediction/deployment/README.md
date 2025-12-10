# Climate Prediction - Deployment Guide

Complete CI/CD pipeline and deployment configuration for the Climate Prediction API.

## Overview

This deployment setup provides:
- **CI Pipeline**: Automated testing, linting, and security audits
- **Release Automation**: Multi-platform binary builds and Docker images
- **Kubernetes Deployment**: Production-ready orchestration with autoscaling
- **Docker Compose**: Local development environment
- **Helm Charts**: Simplified Kubernetes deployments
- **Monitoring**: Prometheus and Grafana integration

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                            │
├─────────────────────────────────────────────────────────────────┤
│  CI Pipeline        Release Pipeline       Deploy Pipeline       │
│  ├─ Format Check    ├─ Build Binaries     ├─ Staging Deploy    │
│  ├─ Linting         ├─ Docker Build       ├─ Smoke Tests       │
│  ├─ Security Audit  ├─ Multi-arch         └─ Production Deploy │
│  ├─ Unit Tests      └─ Publish Crate          ├─ Canary        │
│  ├─ Integration                                ├─ Full Rollout  │
│  └─ Coverage                                   └─ Verification  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Container Registry (GHCR)                     │
│  climate-prediction:latest, :v1.0.0, :v1, linux/amd64+arm64    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                            │
├─────────────────────────────────────────────────────────────────┤
│  LoadBalancer → Ingress → Service → Pods (3-10 replicas)       │
│                    │                                             │
│                    ├─ HPA (Autoscaling)                         │
│                    ├─ NetworkPolicy (Security)                  │
│                    └─ ConfigMaps + Secrets                      │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
deployment/
├── .dockerignore                    # Docker build exclusions
├── Dockerfile                       # Multi-stage optimized build
├── docker-compose.yml              # Local development stack
├── kubernetes/
│   ├── deployment.yaml             # K8s deployment + HPA
│   ├── service.yaml                # Service + ConfigMap
│   ├── ingress.yaml                # Ingress + NetworkPolicy
│   └── canary-deployment.yaml      # Canary rollout config
├── helm/
│   └── climate-prediction/
│       ├── Chart.yaml              # Helm chart metadata
│       └── values.yaml             # Configuration values
└── monitoring/
    └── prometheus.yml              # Metrics scraping config

.github/workflows/
├── ci.yml                          # Continuous Integration
├── release.yml                     # Release automation
└── deploy.yml                      # Deployment workflows
```

## Quick Start

### Local Development

```bash
# Start full stack (API + PostgreSQL + Redis + Monitoring)
cd deployment
docker-compose up -d

# View logs
docker-compose logs -f climate-api

# Access services
curl http://localhost:8080/health
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
```

### Build Docker Image

```bash
# Build optimized image (~70MB)
docker build -f deployment/Dockerfile -t climate-prediction:latest .

# Run container
docker run -p 8080:8080 \
  -e OPENWEATHER_API_KEY=your_key \
  climate-prediction:latest
```

## CI/CD Pipeline

### CI Workflow (ci.yml)

Runs on every push and pull request:

**Parallel Jobs:**
1. **Format Check**: `cargo fmt --check`
2. **Linting**: `cargo clippy` with warnings as errors
3. **Security Audit**: `cargo audit` for vulnerabilities
4. **Unit Tests**: Cross-platform (Linux, Windows, macOS) × Rust (stable, nightly)
5. **Integration Tests**: Full test suite
6. **Build Verification**: Release build
7. **Code Coverage**: `cargo tarpaulin` → Codecov

**Matrix Strategy:**
```yaml
os: [ubuntu-latest, windows-latest, macos-latest]
rust: [stable, nightly]
```

**Caching Strategy:**
- Cargo registry
- Cargo git index
- Build artifacts
- Reduces build time by ~60%

### Release Workflow (release.yml)

Triggers on version tags (`v*.*.*`):

**Multi-Platform Builds:**
- `x86_64-unknown-linux-gnu` (Linux x64)
- `x86_64-unknown-linux-musl` (Alpine/static)
- `aarch64-unknown-linux-gnu` (Linux ARM64)
- `x86_64-apple-darwin` (macOS Intel)
- `aarch64-apple-darwin` (macOS Apple Silicon)
- `x86_64-pc-windows-msvc` (Windows x64)

**Outputs:**
- GitHub Release with binaries
- Docker images (multi-arch: amd64, arm64)
- Published crate to crates.io

**Docker Image Tags:**
```
ghcr.io/yourorg/climate-prediction:latest
ghcr.io/yourorg/climate-prediction:v1.0.0
ghcr.io/yourorg/climate-prediction:v1.0
ghcr.io/yourorg/climate-prediction:v1
```

### Deployment Workflow (deploy.yml)

Automated deployment with canary strategy:

**Staging Environment:**
- Auto-deploys after successful release
- Runs smoke tests
- Health check verification
- Auto-rollback on failure

**Production Environment:**
- Manual approval required
- Canary deployment (10% traffic)
- 5-minute monitoring window
- Gradual rollout to 100%
- Comprehensive smoke tests

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Deploy with kubectl

```bash
# Create namespace
kubectl create namespace production

# Create secrets
kubectl create secret generic climate-secrets \
  --from-literal=database-url="postgresql://user:pass@host:5432/climate" \
  --from-literal=redis-url="redis://host:6379" \
  --from-literal=openweather-api-key="your_key" \
  --namespace=production

# Deploy application
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods -n production
kubectl get hpa -n production
kubectl logs -f deployment/climate-prediction -n production
```

### Deploy with Helm

```bash
# Add repository (if published)
helm repo add climate https://charts.climate-prediction.example.com
helm repo update

# Install
helm install climate-prediction deployment/helm/climate-prediction \
  --namespace production \
  --create-namespace \
  --set image.tag=v1.0.0 \
  --set ingress.hosts[0].host=climate-prediction.example.com

# Upgrade
helm upgrade climate-prediction deployment/helm/climate-prediction \
  --namespace production \
  --set image.tag=v1.1.0

# Rollback
helm rollback climate-prediction 0 --namespace production
```

## Configuration

### Environment Variables

**Required:**
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `OPENWEATHER_API_KEY`: OpenWeather API key

**Optional:**
- `RUST_LOG`: Logging level (debug, info, warn, error)
- `HOST`: Bind host (default: 0.0.0.0)
- `PORT`: Bind port (default: 8080)
- `MAX_CONNECTIONS`: Database pool size
- `CACHE_TTL`: Cache time-to-live (seconds)

### Resource Limits

**Kubernetes:**
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

**Autoscaling:**
- Min replicas: 3
- Max replicas: 10
- Target CPU: 70%
- Target Memory: 80%

## Security

### Best Practices Implemented

1. **Non-root Container**: Runs as user `1000`
2. **Read-only Filesystem**: Prevents tampering
3. **No Privilege Escalation**: Security hardening
4. **Network Policies**: Restrict traffic flow
5. **Secret Management**: Kubernetes secrets
6. **TLS Termination**: Let's Encrypt certificates
7. **Rate Limiting**: 100 requests/second
8. **Security Audits**: `cargo audit` in CI

### Secrets Management

```bash
# Create secrets from file
kubectl create secret generic climate-secrets \
  --from-env-file=.env.production \
  --namespace=production

# Update secret
kubectl delete secret climate-secrets --namespace=production
kubectl create secret generic climate-secrets \
  --from-literal=openweather-api-key="new_key" \
  --namespace=production

# Restart pods to pick up new secrets
kubectl rollout restart deployment/climate-prediction --namespace=production
```

## Monitoring

### Prometheus Metrics

Exposed at `/metrics`:
- Request count and latency
- Active connections
- Database pool stats
- Cache hit/miss rates
- Custom business metrics

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin):
- **Overview Dashboard**: Key metrics and health
- **API Performance**: Request rates and latencies
- **System Resources**: CPU, memory, network
- **Database Metrics**: Connection pool, query times

### Health Checks

```bash
# Liveness probe (is the app running?)
curl http://localhost:8080/health

# Readiness probe (can the app serve traffic?)
curl http://localhost:8080/health

# Metrics
curl http://localhost:8080/metrics
```

## Troubleshooting

### Common Issues

**Pod Crashes:**
```bash
# Check logs
kubectl logs deployment/climate-prediction -n production --tail=100

# Describe pod
kubectl describe pod -l app=climate-prediction -n production

# Check events
kubectl get events -n production --sort-by='.lastTimestamp'
```

**Connection Issues:**
```bash
# Test database connection
kubectl run -it --rm debug --image=postgres:16-alpine --restart=Never -- \
  psql $DATABASE_URL

# Test Redis connection
kubectl run -it --rm debug --image=redis:7-alpine --restart=Never -- \
  redis-cli -u $REDIS_URL ping
```

**Performance Issues:**
```bash
# Check HPA status
kubectl get hpa climate-prediction-hpa -n production

# Top pods by resource usage
kubectl top pods -n production

# Check service endpoints
kubectl get endpoints climate-prediction -n production
```

### Rollback Deployment

```bash
# Kubernetes rollback
kubectl rollout undo deployment/climate-prediction --namespace=production

# Check rollout history
kubectl rollout history deployment/climate-prediction --namespace=production

# Rollback to specific revision
kubectl rollout undo deployment/climate-prediction --to-revision=2 --namespace=production

# Helm rollback
helm rollback climate-prediction 0 --namespace=production
```

## Performance Optimization

### Docker Image Size

**Multi-stage Build Results:**
- Builder stage: ~2.5GB
- Final image: ~70MB (Alpine base)
- Binary size: ~15MB (stripped)

**Optimization Techniques:**
- Static linking with musl
- Strip debug symbols
- Minimal Alpine base
- Layer caching

### Build Time Optimization

**GitHub Actions Caching:**
- Cargo registry: ~200MB
- Build artifacts: ~500MB
- Total time saved: ~60% (5min → 2min)

### Kubernetes Optimization

**Resource Efficiency:**
- Pod anti-affinity for distribution
- HPA for auto-scaling
- Resource requests/limits
- Readiness/liveness probes

## CI/CD Secrets Setup

Required GitHub Secrets:

```bash
# GitHub Container Registry (automatic)
GITHUB_TOKEN

# Kubernetes (base64 encoded kubeconfig)
STAGING_KUBECONFIG
PRODUCTION_KUBECONFIG

# Crates.io
CARGO_REGISTRY_TOKEN

# Optional: Notifications
SLACK_WEBHOOK
DISCORD_WEBHOOK
```

## Cost Estimation

**AWS EKS (3 nodes):**
- Compute: $150/month (t3.medium × 3)
- Load Balancer: $20/month
- Storage: $10/month
- **Total: ~$180/month**

**Docker Image Storage (GHCR):**
- Free for public repositories
- $0.008/GB/month for private

**CI/CD (GitHub Actions):**
- Free for public repositories
- 2000 minutes/month free for private

## Next Steps

1. **Configure Secrets**: Set up GitHub repository secrets
2. **Customize Domains**: Update ingress hosts in Helm values
3. **Setup Monitoring**: Configure Prometheus/Grafana
4. **Enable Alerts**: Add alerting rules
5. **Configure Auto-scaling**: Tune HPA parameters
6. **Setup Backups**: Database backup strategy
7. **Add CDN**: CloudFlare or similar for static assets

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Rust Packaging](https://doc.rust-lang.org/cargo/guide/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourorg/climate-prediction/issues
- Documentation: https://docs.climate-prediction.example.com
- Slack: #climate-prediction-support

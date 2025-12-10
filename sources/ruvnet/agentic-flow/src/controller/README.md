# Agentic-Jujutsu Kubernetes Controller

Production-ready Kubernetes controller for GitOps deployments using Jujutsu VCS.

## Overview

The agentic-jujutsu controller is a Kubernetes operator that:
- Watches Jujutsu repositories for changes
- Validates manifests against policies (Kyverno/OPA)
- Deploys applications to multiple clusters
- Integrates with Argo Rollouts for progressive delivery
- Provisions infrastructure via Crossplane

## Architecture

```
┌─────────────────────────────────────────┐
│  Controller Manager (Go)                │
│  ┌───────────────────────────────────┐  │
│  │  Application Reconciler           │  │
│  │  • Watch Application CRDs         │  │
│  │  • Sync Jujutsu repository        │  │
│  │  • Validate policies              │  │
│  │  • Deploy to clusters             │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  Jujutsu Client                   │  │
│  │  • Fetch changes                  │  │
│  │  • Extract manifests              │  │
│  │  • Track revisions                │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  Cluster Manager                  │  │
│  │  • Multi-cluster clients          │  │
│  │  • Dynamic connections            │  │
│  │  • Resource application           │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  Policy Validator                 │  │
│  │  • Kyverno integration            │  │
│  │  • OPA integration                │  │
│  │  • Enforcement modes              │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Kubernetes cluster (v1.28+)
- kubectl configured
- Helm 3.x
- Jujutsu CLI (`jj`)

### Installation

```bash
# Add Helm repository
helm repo add agentic-jujutsu https://charts.agentic-jujutsu.io
helm repo update

# Install controller
helm install ajj-controller agentic-jujutsu/jujutsu-gitops-controller \
  --namespace ajj-system \
  --create-namespace \
  --set repository.url=https://github.com/org/gitops-repo
```

### Create an Application

```yaml
apiVersion: ajj.io/v1
kind: Application
metadata:
  name: web-frontend
  namespace: default
spec:
  source:
    repoPath: apps/web-frontend
    targetRevision: "@main"

  destination:
    clusters:
      - production

  syncPolicy:
    automated:
      prune: true
      selfHeal: true

  progressiveDelivery:
    enabled: true
    strategy: canary
```

Apply:
```bash
kubectl apply -f application.yaml
```

Check status:
```bash
kubectl get applications
kubectl describe application web-frontend
```

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/ruvnet/agentic-jujutsu
cd agentic-jujutsu/src/controller

# Install dependencies
go mod download

# Build
make build

# Run tests
make test

# Run benchmarks
go test ./test/performance -bench=. -benchmem
```

### Running Locally

```bash
# Install CRDs
make install

# Run controller
make run
```

### Docker Build

```bash
# Build image
make docker-build IMG=agentic-jujutsu/controller:dev

# Push image
make docker-push IMG=agentic-jujutsu/controller:dev
```

## Testing

### Unit Tests

```bash
go test ./test/unit/...
```

### Performance Tests

```bash
go test ./test/performance -bench=. -benchmem
```

### E2B Sandbox Tests

```bash
# Requires E2B_API_KEY
cd test/e2e
npm install
E2B_API_KEY=your_key npm run test:e2b
```

## Custom Resource Definitions

### Application

Represents an application to be deployed.

**Key Fields**:
- `spec.source`: Jujutsu repository configuration
- `spec.destination`: Target clusters
- `spec.syncPolicy`: Synchronization behavior
- `spec.progressiveDelivery`: Rollout configuration
- `spec.infrastructure`: Crossplane resources
- `spec.policies`: Policy enforcement

### Cluster

Represents a target Kubernetes cluster.

**Key Fields**:
- `spec.kubeconfig`: Secret reference for cluster access
- `spec.environment`: Environment classification
- `spec.sync`: Sync configuration

## Configuration

### Controller Configuration

```yaml
controller:
  repoPath: "/tmp/gitops-repo"
  reconcileInterval: "30s"
  leaderElection:
    enabled: true

repository:
  url: "https://github.com/org/gitops-repo"

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 100m
    memory: 128Mi
```

### Policy Configuration

```yaml
policies:
  validation:
    - require-labels
    - require-resource-limits
    - verify-image-signature
  enforcementMode: strict  # strict, permissive, audit
```

## Performance Targets

Based on specification requirements:

| Metric | Target | Implementation |
|--------|--------|----------------|
| Reconciliation latency | < 5s | Optimized change detection |
| Policy validation | < 2s | Parallel validation |
| Multi-cluster sync | < 30s | Concurrent cluster operations |
| Memory usage | < 512MB | Efficient caching |
| CPU usage | < 0.5 cores | Event-driven architecture |

## Monitoring

### Metrics

Controller exposes Prometheus metrics on `:8080/metrics`:

```
# Reconciliation metrics
ajj_reconcile_total{application="web-frontend",result="success"} 150
ajj_reconcile_duration_seconds{application="web-frontend"} 2.3

# Application metrics
ajj_applications_total 10
ajj_applications_synced 8
ajj_applications_out_of_sync 2
```

### Health Checks

```bash
# Liveness probe
curl http://localhost:8081/healthz

# Readiness probe
curl http://localhost:8081/readyz
```

## Troubleshooting

### Controller not syncing

```bash
# Check controller logs
kubectl logs -n ajj-system deployment/ajj-controller

# Check Application status
kubectl describe application web-frontend

# Check Jujutsu repository access
kubectl exec -n ajj-system deployment/ajj-controller -- jj log
```

### Policy validation failing

```bash
# Check policy validator logs
kubectl logs -n ajj-system deployment/ajj-controller | grep policy

# Validate manifests locally
kubectl apply --dry-run=server -f manifest.yaml
```

### Cluster connection issues

```bash
# Check Cluster resource
kubectl describe cluster production

# Verify kubeconfig secret
kubectl get secret production-kubeconfig -o yaml
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](../../LICENSE)

## Support

- GitHub Issues: https://github.com/ruvnet/agentic-jujutsu/issues
- Documentation: https://docs.agentic-jujutsu.io
- Slack: https://slack.agentic-jujutsu.io

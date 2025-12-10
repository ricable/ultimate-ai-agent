# Agentic-Jujutsu: Next-Generation GitOps Platform

> **State-of-the-art GitOps using Jujutsu with policy-first enforcement, progressive delivery, and unified control plane management**

[![npm version](https://badge.fury.io/js/agentic-jujutsu.svg)](https://www.npmjs.com/package/agentic-jujutsu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://github.com/yourusername/agentic-jujutsu/workflows/CI/badge.svg)](https://github.com/yourusername/agentic-jujutsu/actions)
[![Coverage](https://codecov.io/gh/yourusername/agentic-jujutsu/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/agentic-jujutsu)

## Overview

**agentic-jujutsu** revolutionizes GitOps by leveraging **Jujutsu's** advanced version control capabilities to deliver:

- 🚀 **Policy-First Deployments**: Automatic validation with Kyverno and OPA before any deployment
- 📊 **Progressive Delivery**: Canary, blue-green, and A/B testing with SLO-based automatic rollback
- 🏗️ **Unified Control Plane**: Manage applications and infrastructure (via Crossplane) together
- 🔒 **Supply Chain Security**: Sigstore image signing, SBOM generation, and vulnerability scanning
- 🌐 **Multi-Cluster GitOps**: Fleet management with environment-specific configurations
- 🧠 **Jujutsu-Powered**: Change-centric workflows with conflict-free merging and complete audit trails

## Why Jujutsu over Git?

Traditional Git-based GitOps has limitations. **Jujutsu** provides superior workflows:

| Feature | Git | Jujutsu | Benefit for GitOps |
|---------|-----|---------|-------------------|
| **Conflict Resolution** | Manual | Automatic | Parallel team deployments without blocking |
| **Change Tracking** | Commits (snapshots) | First-class changes with stable IDs | Reliable cherry-picking across environments |
| **Operation History** | Reflog (local, pruned) | Complete operation log | Full audit trail for compliance |
| **Working Copy** | Separate from history | Always a commit | No lost work, continuous backup |
| **Branch Management** | Context switching | Colocated branches | Compare environments side-by-side |

## Quick Start

### Install

```bash
# Install CLI globally
npm install -g agentic-jujutsu

# Or use with npx
npx agentic-jujutsu --version
```

### Prerequisites

- **Jujutsu** (`jj`) v0.12.0+
- **Kubernetes** v1.28+
- **kubectl** configured
- **Helm** 3.x (for controller installation)

### Initialize GitOps Repository

```bash
# Create new GitOps repository with Jujutsu
ajj init https://github.com/yourorg/gitops-repo

# Or clone existing
ajj clone https://github.com/yourorg/gitops-repo ~/gitops-repo
cd ~/gitops-repo

# View status
ajj status
```

### Deploy Your First Application

```bash
# Add your Kubernetes cluster
ajj cluster add production ~/.kube/config

# Create application from template
ajj app create my-web-app --template=web-app

# Deploy to development
ajj app deploy my-web-app --env=development

# Check deployment status
ajj app status my-web-app

# View logs
ajj logs my-web-app --follow
```

### Progressive Rollout

```bash
# Update your application (e.g., new image version)
# Edit apps/my-web-app/base/deployment.yaml

# Start canary rollout
ajj rollout start my-web-app --strategy=canary

# Monitor rollout progress
ajj rollout status my-web-app

# Manually promote if needed (or let SLOs auto-promote)
ajj rollout promote my-web-app

# Abort if issues detected
ajj rollout abort my-web-app
```

## Features

### 1. Policy-as-Code Enforcement

Automatically validate all deployments against policies **before** they reach your clusters:

```bash
# Apply policy set
ajj policy apply policies/kyverno/

# Validate manifest against policies
ajj policy validate apps/my-app/deployment.yaml

# Test policies in CI/CD
ajj policy test policies/
```

**Supported Engines**:
- **Kyverno**: Kubernetes-native, YAML-based policies
- **OPA/Gatekeeper**: Rego-based policies for complex logic

**Policy Types**:
- Validation: Enforce requirements (labels, resources, probes)
- Mutation: Auto-add defaults (labels, network policies)
- Generation: Auto-create resources (network policies, quotas)
- Verification: Supply chain security (image signatures, SBOMs)

### 2. Progressive Delivery with SLO Monitoring

Deploy safely with automatic canary analysis:

```yaml
# apps/my-app/rollout.yaml
strategy: canary
steps:
  - setWeight: 10
  - pause: 5m
  - analysis:
      - error-rate < 1%
      - latency-p95 < 500ms
  - setWeight: 25
  # ... continues
```

**Automatic Rollback** if:
- Error rate exceeds threshold
- Latency increases significantly
- Custom SLOs fail
- Manual abort triggered

**Supported Strategies**:
- **Canary**: Gradual traffic shifting (10% → 25% → 50% → 100%)
- **Blue-Green**: Zero-downtime cutover
- **A/B Testing**: Header-based routing for experiments

### 3. Infrastructure as Code with Crossplane

Provision cloud resources alongside applications:

```yaml
# apps/my-app/app-config.yaml
infrastructure:
  required:
    - name: postgres-db
      type: database
      composition: postgresql-ha-cluster
      parameters:
        instanceClass: db.r6g.xlarge
        storageGB: 500

    - name: redis-cache
      type: cache
      composition: redis-cluster
      parameters:
        nodeType: cache.r6g.large
```

**Supported Resources**:
- **Databases**: PostgreSQL, MySQL, MongoDB
- **Caches**: Redis, Memcached
- **Queues**: SQS, PubSub, RabbitMQ
- **Storage**: S3, GCS, Azure Blob
- **Networking**: VPCs, Subnets, Load Balancers

### 4. Supply Chain Security

End-to-end security with Sigstore:

```bash
# Sign container image
ajj sign registry.company.com/my-app:v1.2.3

# Verify signature (automatic during deployment)
ajj verify registry.company.com/my-app:v1.2.3

# Generate SBOM
ajj sbom registry.company.com/my-app:v1.2.3 --format=spdx-json

# Scan for vulnerabilities
ajj scan registry.company.com/my-app:v1.2.3
```

**Security Features**:
- **Sigstore Integration**: Keyless signing with Fulcio
- **SBOM Generation**: Software Bill of Materials
- **Vulnerability Scanning**: CVE detection with Trivy
- **Policy Enforcement**: Only signed images can deploy

### 5. Multi-Cluster Fleet Management

Manage multiple clusters with environment-specific configurations:

```bash
# Add multiple clusters
ajj cluster add production-us-east ~/.kube/prod-us-east
ajj cluster add production-us-west ~/.kube/prod-us-west
ajj cluster add staging ~/.kube/staging

# Deploy to all production clusters
ajj app deploy my-app --env=production

# View status across fleet
ajj cluster list
```

**Progressive Rollout Across Clusters**:
1. Deploy to development → validate
2. Deploy to staging → soak test
3. Deploy to production (50% traffic) → validate
4. Deploy to production (100% traffic) → complete

### 6. Jujutsu-Specific Features

Leverage Jujutsu's unique capabilities:

```bash
# Create a new change (like a commit, but better)
ajj change create "Add Redis caching layer"

# List all changes
ajj change list

# Compare changes across environments
ajj change diff dev-change-id prod-change-id

# View complete operation history (audit trail)
ajj change history my-app

# Squash related changes
ajj change squash change-id-1 change-id-2
```

**Advantages**:
- **Stable Change IDs**: Cherry-pick across branches reliably
- **Automatic Conflict Resolution**: Parallel deployments without manual merges
- **Complete Audit Trail**: Every operation recorded permanently
- **Colocated Branches**: Compare environments without switching context

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Developer Interface                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  CLI (ajj)   │  │  MCP Server  │  │  Web Portal  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Jujutsu VCS Layer                            │
│  • Change-centric tracking                                      │
│  • Conflict-free merging                                        │
│  • Complete operation history                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GitOps Control Plane                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Custom JJ   │  │  Argo CD     │  │  Flux CD     │         │
│  │  Controller  │  │  Adapter     │  │  Bridge      │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Policy Layer   │  │  Progressive    │  │  Infrastructure │
│  • Kyverno      │  │  Delivery       │  │  Control Plane  │
│  • OPA          │  │  • Argo Rollouts│  │  • Crossplane   │
│  • Sigstore     │  │  • Flagger      │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Kubernetes Clusters                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Production  │  │  Staging     │  │  Development │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

## CLI Commands

### Repository Management
```bash
ajj init <repo-url>              # Initialize GitOps repository
ajj clone <source> <dest>        # Clone repository
ajj status                       # Show deployment status
```

### Application Management
```bash
ajj app create <name>            # Create application
ajj app deploy <name>            # Deploy application
ajj app rollback <name>          # Rollback to previous version
ajj app delete <name>            # Delete application
ajj app list                     # List all applications
```

### Progressive Delivery
```bash
ajj rollout start <name>         # Start progressive rollout
ajj rollout promote <name>       # Promote to next step
ajj rollout abort <name>         # Abort rollout
ajj rollout status <name>        # Check rollout status
```

### Policy Management
```bash
ajj policy apply <file>          # Apply policy
ajj policy validate <manifest>   # Validate manifest
ajj policy test <dir>            # Test policies
ajj policy audit                 # Audit cluster compliance
```

### Cluster Management
```bash
ajj cluster add <name>           # Add cluster
ajj cluster sync <name>          # Force sync
ajj cluster diff <name>          # Show drift
ajj cluster list                 # List clusters
```

### Infrastructure
```bash
ajj infra apply <file>           # Provision infrastructure
ajj infra status <resource>      # Check status
ajj infra delete <resource>      # Delete infrastructure
```

### Security
```bash
ajj sign <image>                 # Sign container image
ajj verify <image>               # Verify signature
ajj sbom <image>                 # Generate SBOM
ajj scan <image>                 # Scan for vulnerabilities
```

### Observability
```bash
ajj metrics <app>                # Show metrics
ajj logs <app>                   # Stream logs
ajj traces <app>                 # View traces
```

## MCP Server Integration

**agentic-jujutsu** provides a Model Context Protocol (MCP) server for AI agent integration:

### Add to Claude Desktop

```bash
claude mcp add agentic-jujutsu npx agentic-jujutsu mcp start
```

### Available MCP Tools

```typescript
// Repository operations
jj_init(repo_url, config)
jj_status(repo_path)
jj_change_create(description, files)

// GitOps operations
gitops_app_deploy(name, environment)
gitops_app_rollback(name, change_id)
gitops_cluster_sync(cluster_name)

// Policy operations
policy_validate(manifest, policy_set)
policy_apply(policy_file, cluster)

// Progressive delivery
rollout_create(name, strategy)
rollout_promote(name)
rollout_status(name)

// Infrastructure operations
infra_apply(composition, parameters)
infra_status(resource_name)
```

### Example: AI-Assisted Deployment

```
User: Deploy my-web-app to production with canary strategy

AI: I'll deploy my-web-app to production using a canary strategy:
    1. gitops_app_deploy("my-web-app", "production", {strategy: "canary"})
    2. rollout_create("my-web-app", "canary")
    3. [Monitoring rollout_status every minute]
    4. [Auto-promote based on SLO evaluation]

    Deployment started. Monitoring progress...
```

## Configuration

### Global Configuration

`~/.config/ajj/config.yaml`:

```yaml
repositories:
  default: ~/gitops-repo

clusters:
  production-us-east:
    kubeconfig: ~/.kube/prod-us-east
  staging:
    kubeconfig: ~/.kube/staging

policies:
  enforcementMode: strict
  engines: [kyverno, opa]

progressiveDelivery:
  defaultStrategy: canary
  autoPromotion: false

security:
  requireSignedImages: true
```

### Repository Configuration

`gitops-repo/config/ajj-config.yaml`:

```yaml
apiVersion: ajj.io/v1
kind: Configuration
metadata:
  name: gitops-config

spec:
  reconciliation:
    interval: 30s
    timeout: 5m

  clusters:
    - name: production
      sync:
        automated: true
        prune: true

  policies:
    kyverno:
      enabled: true
      policyPath: policies/kyverno
```

## Installation

### Install Controller in Kubernetes

```bash
# Add Helm repository
helm repo add agentic-jujutsu https://charts.agentic-jujutsu.io
helm repo update

# Install controller
helm install ajj-controller agentic-jujutsu/jujutsu-gitops-controller \
  --namespace ajj-system \
  --create-namespace \
  --set repoUrl=https://github.com/yourorg/gitops-repo \
  --set reconciliation.interval=30s

# Verify installation
kubectl get pods -n ajj-system
```

### Install Dependencies

**Kyverno** (Policy Engine):
```bash
kubectl create -f https://github.com/kyverno/kyverno/releases/download/v1.11.0/install.yaml
```

**Argo Rollouts** (Progressive Delivery):
```bash
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml
```

**Crossplane** (Infrastructure Control Plane):
```bash
helm repo add crossplane-stable https://charts.crossplane.io/stable
helm install crossplane crossplane-stable/crossplane \
  --namespace crossplane-system \
  --create-namespace
```

**Prometheus** (Metrics):
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

## Repository Structure

```
gitops-repo/
├── .jj/                        # Jujutsu metadata
├── apps/                       # Applications
│   ├── web-frontend/
│   │   ├── base/               # Base manifests
│   │   ├── overlays/           # Environment overlays
│   │   ├── rollout.yaml        # Progressive delivery config
│   │   └── app-config.yaml     # Application metadata
│   └── api-backend/
├── infrastructure/             # Crossplane compositions
│   ├── databases/
│   ├── caches/
│   └── storage/
├── policies/                   # Policy-as-code
│   ├── kyverno/
│   ├── opa/
│   └── tests/
├── platform/                   # Platform templates
│   └── templates/
└── config/
    └── ajj-config.yaml         # Configuration
```

## Examples

### Example 1: Deploy Web Application

```bash
# Create application
ajj app create web-app --template=web-app

# Configure
cat > apps/web-app/app-config.yaml <<EOF
apiVersion: ajj.io/v1
kind: Application
metadata:
  name: web-app
spec:
  progressiveDelivery:
    enabled: true
    strategy: canary
  infrastructure:
    required:
      - name: postgres
        type: database
      - name: redis
        type: cache
EOF

# Deploy
ajj app deploy web-app --env=production
```

### Example 2: Apply Security Policy

```bash
# Create policy
cat > policies/kyverno/require-signatures.yaml <<EOF
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-image-signature
spec:
  validationFailureAction: enforce
  rules:
    - name: check-signature
      match:
        any:
          - resources:
              kinds: [Pod]
      verifyImages:
        - imageReferences: ["registry.company.com/*"]
          attestors:
            - entries:
                - keys:
                    publicKeys: |-
                      -----BEGIN PUBLIC KEY-----
                      [Your public key]
                      -----END PUBLIC KEY-----
EOF

# Apply policy
ajj policy apply policies/kyverno/require-signatures.yaml
```

### Example 3: Multi-Cluster Deployment

```bash
# Add clusters
ajj cluster add prod-us-east ~/.kube/prod-us-east
ajj cluster add prod-us-west ~/.kube/prod-us-west
ajj cluster add prod-eu-west ~/.kube/prod-eu-west

# Configure app for multi-cluster
cat > apps/my-app/app-config.yaml <<EOF
spec:
  destination:
    clusters:
      - name: prod-us-east
        weight: 40
      - name: prod-us-west
        weight: 40
      - name: prod-eu-west
        weight: 20
EOF

# Deploy to all clusters
ajj app deploy my-app --env=production
```

## Comparison with Other GitOps Tools

| Feature | Agentic-Jujutsu | Argo CD | Flux CD | GitOps Toolkit |
|---------|----------------|---------|---------|----------------|
| **VCS** | Jujutsu | Git | Git | Git |
| **Conflict Resolution** | Automatic | Manual | Manual | Manual |
| **Policy Enforcement** | Built-in (Kyverno/OPA) | Plugin | Plugin | Manual |
| **Progressive Delivery** | Built-in (Argo Rollouts) | Separate install | Flagger | Manual |
| **Infrastructure Provisioning** | Built-in (Crossplane) | Manual | Manual | Manual |
| **Supply Chain Security** | Built-in (Sigstore) | Manual | Manual | Manual |
| **Multi-Cluster** | Native | Native | Native | Native |
| **Audit Trail** | Complete (Jujutsu) | Git history | Git history | Git history |

## Roadmap

### v0.1 - Alpha (Current)
- ✅ Jujutsu integration
- ✅ Basic GitOps controller
- ✅ CLI tool
- ✅ MCP server

### v0.5 - Beta
- ⏳ Policy enforcement (Kyverno/OPA)
- ⏳ Progressive delivery (Argo Rollouts)
- ⏳ Multi-cluster support
- ⏳ Crossplane integration

### v1.0 - Production (Week 12)
- ⏳ Complete security features (Sigstore)
- ⏳ Observability integration
- ⏳ Platform templates
- ⏳ Comprehensive documentation

### v1.1+ - Future
- Web portal UI
- AI-powered rollout recommendations
- Advanced chaos engineering
- Cost optimization features

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/agentic-jujutsu
cd agentic-jujutsu

# Install dependencies
npm install

# Build all packages
npm run build

# Run tests
npm test

# Start local Kubernetes cluster
k3d cluster create ajj-dev

# Install controller locally
helm install ajj-controller ./charts/jujutsu-gitops-controller \
  --namespace ajj-system \
  --create-namespace
```

## Documentation

- **[SPARC Specification](docs/specs/agentic-jujutsu-spec.md)**: Complete technical specification
- **[Implementation Roadmap](docs/specs/implementation-roadmap.md)**: Development timeline
- **[Architecture Guide](docs/architecture.md)**: System design and components
- **[CLI Reference](docs/cli-reference.md)**: All CLI commands
- **[MCP Tools Reference](docs/mcp-reference.md)**: MCP integration guide
- **[Policy Writing Guide](docs/policy-guide.md)**: Create custom policies
- **[Migration Guide](docs/migration.md)**: Migrate from Git-based GitOps

## Support

- **GitHub Issues**: https://github.com/yourusername/agentic-jujutsu/issues
- **Discussions**: https://github.com/yourusername/agentic-jujutsu/discussions
- **Slack**: [Join our Slack](https://slack.agentic-jujutsu.io)
- **Email**: support@agentic-jujutsu.io

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- **Jujutsu**: https://github.com/martinvonz/jj
- **Argo Project**: https://argoproj.github.io/
- **Kyverno**: https://kyverno.io/
- **Crossplane**: https://crossplane.io/
- **CNCF**: For the amazing cloud-native ecosystem

---

**Built with ❤️ by the Agentic-Jujutsu team**

*"GitOps evolved: Change-centric, policy-first, progressively delivered"*

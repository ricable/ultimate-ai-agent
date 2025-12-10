#!/bin/bash
# =============================================================================
# HyperScale AI Agents Platform - Deployment Script
# =============================================================================
# Deploys the complete AI agents scaling platform with:
# - K3s/K8s cluster setup
# - SpinKube WASM runtime
# - LlamaEdge/WasmEdge inference
# - Self-healing operators
# - Hybrid cloud federation
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAMESPACE_SYSTEM="edge-ai-system"
NAMESPACE_AGENTS="edge-ai-agents"
NAMESPACE_INFERENCE="edge-ai-inference"

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Banner
print_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██╗  ██╗██╗   ██╗██████╗ ███████╗██████╗ ███████╗ ██████╗ █████╗ ██╗       ║
║   ██║  ██║╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗██║       ║
║   ███████║ ╚████╔╝ ██████╔╝█████╗  ██████╔╝███████╗██║     ███████║██║       ║
║   ██╔══██║  ╚██╔╝  ██╔═══╝ ██╔══╝  ██╔══██╗╚════██║██║     ██╔══██║██║       ║
║   ██║  ██║   ██║   ██║     ███████╗██║  ██║███████║╚██████╗██║  ██║███████╗  ║
║   ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝  ║
║                                                                               ║
║                    AI AGENTS SCALING PLATFORM                                 ║
║                                                                               ║
║   Sovereign · Edge-Native · Self-Healing · HyperScale                         ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing=()

    # Check for kubectl
    if ! command -v kubectl &> /dev/null; then
        missing+=("kubectl")
    fi

    # Check for helm (optional but recommended)
    if ! command -v helm &> /dev/null; then
        log_warn "Helm not found - some features may be limited"
    fi

    # Check for docker
    if ! command -v docker &> /dev/null; then
        log_warn "Docker not found - using containerd directly"
    fi

    # Check for k3s or k8s cluster
    if ! kubectl cluster-info &> /dev/null 2>&1; then
        missing+=("kubernetes cluster (k3s/k8s)")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing prerequisites: ${missing[*]}"
        log_info "Please install missing components and try again"
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

# Create namespaces
create_namespaces() {
    log_info "Creating namespaces..."

    kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE_SYSTEM}
  labels:
    app.kubernetes.io/part-of: edge-ai-platform
    istio-injection: disabled
---
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE_AGENTS}
  labels:
    app.kubernetes.io/part-of: edge-ai-platform
---
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE_INFERENCE}
  labels:
    app.kubernetes.io/part-of: edge-ai-platform
EOF

    log_success "Namespaces created"
}

# Install SpinKube operator
install_spinkube() {
    log_info "Installing SpinKube operator..."

    # Install Spin Operator CRDs
    kubectl apply -f https://github.com/spinkube/spin-operator/releases/latest/download/spin-operator.crds.yaml || true

    # Install RuntimeClass for Spin
    kubectl apply -f - <<EOF
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: wasmtime-spin
handler: spin
scheduling:
  nodeSelector:
    kubernetes.io/os: linux
EOF

    # Deploy Spin Operator
    kubectl apply -f "${PROJECT_ROOT}/infrastructure/k3s/30-spinkube-operator.yaml" || {
        log_warn "SpinKube operator manifest not found, skipping"
    }

    log_success "SpinKube operator installed"
}

# Install WasmEdge runtime
install_wasmedge() {
    log_info "Installing WasmEdge runtime..."

    # Check if we're on a supported architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64|aarch64|arm64)
            ;;
        *)
            log_warn "Unsupported architecture: $ARCH - skipping WasmEdge"
            return
            ;;
    esac

    # RuntimeClass for WasmEdge
    kubectl apply -f - <<EOF
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: wasmedge
handler: wasmedge
scheduling:
  nodeSelector:
    kubernetes.io/os: linux
EOF

    log_success "WasmEdge runtime configured"
}

# Deploy Redis for state management
deploy_redis() {
    log_info "Deploying Redis..."

    kubectl apply -f "${PROJECT_ROOT}/infrastructure/k3s/10-redis.yaml" || {
        # Fallback inline deployment
        kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
  namespace: ${NAMESPACE_SYSTEM}
spec:
  serviceName: redis
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: ${NAMESPACE_SYSTEM}
spec:
  ports:
  - port: 6379
  selector:
    app: redis
EOF
    }

    log_success "Redis deployed"
}

# Deploy PostgreSQL with pgvector
deploy_postgres() {
    log_info "Deploying PostgreSQL with pgvector..."

    kubectl apply -f "${PROJECT_ROOT}/infrastructure/k3s/11-postgres.yaml" || {
        log_warn "PostgreSQL manifest not found, using defaults"
        kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: ${NAMESPACE_SYSTEM}
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: pgvector/pgvector:pg16
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: edgeai
        - name: POSTGRES_USER
          value: edgeai
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
              optional: true
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: ${NAMESPACE_SYSTEM}
spec:
  ports:
  - port: 5432
  selector:
    app: postgres
EOF
    }

    log_success "PostgreSQL deployed"
}

# Deploy LiteLLM proxy
deploy_litellm() {
    log_info "Deploying LiteLLM proxy..."

    kubectl apply -f "${PROJECT_ROOT}/infrastructure/k3s/21-litellm.yaml" || {
        log_warn "LiteLLM manifest not found, skipping"
    }

    log_success "LiteLLM proxy deployed"
}

# Deploy AI Agent Operator
deploy_agent_operator() {
    log_info "Deploying AI Agent Operator..."

    kubectl apply -f "${PROJECT_ROOT}/infrastructure/operators/agent-operator.yaml"

    # Wait for operator to be ready
    kubectl rollout status deployment/agent-operator -n ${NAMESPACE_SYSTEM} --timeout=120s || {
        log_warn "Operator deployment may still be in progress"
    }

    log_success "AI Agent Operator deployed"
}

# Deploy sample agents
deploy_sample_agents() {
    log_info "Deploying sample AI agents..."

    kubectl apply -f - <<EOF
apiVersion: edge-ai.io/v1alpha1
kind: AIAgent
metadata:
  name: coder-agent
  namespace: ${NAMESPACE_AGENTS}
spec:
  agentType: coder
  replicas: 3
  model:
    name: qwen-coder-7b
    backend: llamaedge
    quantization: "4bit"
  resources:
    cpu: "500m"
    memory: "1Gi"
    gpu: false
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 10
    targetCPU: 70
  selfHealing:
    enabled: true
    healthCheckInterval: "30s"
    failureThreshold: 3
    recoveryAction: restart
  a2a:
    enabled: true
    discoveryMethod: kubernetes
    exposedSkills:
      - code-generation
      - code-review
---
apiVersion: edge-ai.io/v1alpha1
kind: AIAgent
metadata:
  name: researcher-agent
  namespace: ${NAMESPACE_AGENTS}
spec:
  agentType: researcher
  replicas: 2
  model:
    name: llama-3.1-8b
    backend: gaia
  resources:
    cpu: "250m"
    memory: "512Mi"
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 5
  selfHealing:
    enabled: true
  a2a:
    enabled: true
    exposedSkills:
      - web-search
      - document-analysis
---
apiVersion: edge-ai.io/v1alpha1
kind: AgentSwarm
metadata:
  name: dev-swarm
  namespace: ${NAMESPACE_AGENTS}
spec:
  topology: hierarchical
  agents:
    - name: orchestrator
      type: orchestrator
      replicas: 1
      role: leader
    - name: coders
      type: coder
      replicas: 3
      role: worker
    - name: reviewer
      type: reviewer
      replicas: 1
      role: specialist
  workflow:
    scheduler: quad
    checkpointing: true
EOF

    log_success "Sample agents deployed"
}

# Configure auto-scaling
configure_autoscaling() {
    log_info "Configuring auto-scaling..."

    # Install metrics-server if not present
    if ! kubectl get deployment metrics-server -n kube-system &> /dev/null; then
        kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
    fi

    # Create HPA for agents namespace
    kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-hpa
  namespace: ${NAMESPACE_AGENTS}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: coder-agent
  minReplicas: 1
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
EOF

    log_success "Auto-scaling configured"
}

# Configure monitoring
configure_monitoring() {
    log_info "Configuring monitoring..."

    # Create ServiceMonitor for Prometheus (if prometheus-operator is installed)
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: agent-metrics
  namespace: ${NAMESPACE_AGENTS}
  labels:
    app: edge-ai-agents
spec:
  ports:
  - name: metrics
    port: 9090
    targetPort: metrics
  selector:
    app.kubernetes.io/part-of: edge-ai-platform
EOF

    log_success "Monitoring configured"
}

# Print status
print_status() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}             HyperScale Platform Deployed Successfully!         ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}Namespaces:${NC}"
    kubectl get namespaces | grep edge-ai
    echo ""
    echo -e "${CYAN}Pods:${NC}"
    kubectl get pods -n ${NAMESPACE_SYSTEM}
    kubectl get pods -n ${NAMESPACE_AGENTS}
    echo ""
    echo -e "${CYAN}Services:${NC}"
    kubectl get svc -n ${NAMESPACE_SYSTEM}
    echo ""
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "  1. Check pod status: kubectl get pods -A | grep edge-ai"
    echo "  2. View logs: kubectl logs -n ${NAMESPACE_SYSTEM} -l app=agent-operator"
    echo "  3. Access API: kubectl port-forward svc/edge-ai-api 8080:8080 -n ${NAMESPACE_SYSTEM}"
    echo "  4. Launch agents: ./platform/scripts/launch-swarm.sh --agents 100"
    echo ""
}

# Main deployment function
main() {
    print_banner

    log_info "Starting HyperScale AI Platform deployment..."
    echo ""

    check_prerequisites
    create_namespaces
    install_spinkube
    install_wasmedge
    deploy_redis
    deploy_postgres
    deploy_litellm
    deploy_agent_operator
    deploy_sample_agents
    configure_autoscaling
    configure_monitoring

    print_status
}

# Run main
main "$@"

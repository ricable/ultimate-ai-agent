#!/bin/bash
# =============================================================================
# Launch AI Agent Swarm
# =============================================================================
# Quickly launch a swarm of AI agents with specified configuration
# =============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
AGENTS=10
TYPE="coder"
TOPOLOGY="mesh"
NAMESPACE="edge-ai-agents"
MODEL="qwen-coder-7b"
GPU=false

# Usage
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Launch an AI agent swarm with specified configuration.

Options:
    -a, --agents NUM        Number of agents to launch (default: 10)
    -t, --type TYPE         Agent type: coder|researcher|analyst|monitor (default: coder)
    -T, --topology TOPO     Swarm topology: mesh|hierarchical|star|ring (default: mesh)
    -n, --namespace NS      Kubernetes namespace (default: edge-ai-agents)
    -m, --model MODEL       LLM model to use (default: qwen-coder-7b)
    -g, --gpu               Enable GPU acceleration
    -h, --help              Show this help message

Examples:
    $(basename "$0") --agents 100 --type coder
    $(basename "$0") -a 50 -t researcher -T hierarchical
    $(basename "$0") --agents 1000 --gpu --model llama-3.1-70b

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--agents)
            AGENTS="$2"
            shift 2
            ;;
        -t|--type)
            TYPE="$2"
            shift 2
            ;;
        -T|--topology)
            TOPOLOGY="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Generate swarm name
SWARM_NAME="swarm-${TYPE}-$(date +%s)"

echo -e "${CYAN}"
cat << 'EOF'
    _    ____  _____ _   _ _____   ______        ___    ____  __  __
   / \  / ___|| ____| \ | |_   _| / ___\ \      / / \  |  _ \|  \/  |
  / _ \| |  _ |  _| |  \| | | |   \___ \\ \ /\ / / _ \ | |_) | |\/| |
 / ___ \ |_| || |___| |\  | | |    ___) |\ V  V / ___ \|  _ <| |  | |
/_/   \_\____||_____|_| \_| |_|   |____/  \_/\_/_/   \_\_| \_\_|  |_|
EOF
echo -e "${NC}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Swarm Name:  ${SWARM_NAME}"
echo "  Agents:      ${AGENTS}"
echo "  Type:        ${TYPE}"
echo "  Topology:    ${TOPOLOGY}"
echo "  Model:       ${MODEL}"
echo "  GPU:         ${GPU}"
echo "  Namespace:   ${NAMESPACE}"
echo ""

# Calculate agent distribution for hierarchical topology
if [ "$TOPOLOGY" = "hierarchical" ]; then
    LEADERS=1
    WORKERS=$((AGENTS - 1))
else
    LEADERS=0
    WORKERS=$AGENTS
fi

# Create swarm manifest
echo -e "${BLUE}Creating swarm manifest...${NC}"

cat << EOF | kubectl apply -f -
apiVersion: edge-ai.io/v1alpha1
kind: AgentSwarm
metadata:
  name: ${SWARM_NAME}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/part-of: edge-ai-platform
    swarm-type: ${TYPE}
spec:
  topology: ${TOPOLOGY}
  agents:
$(if [ "$LEADERS" -gt 0 ]; then
cat << LEADER
    - name: leader
      type: orchestrator
      replicas: ${LEADERS}
      role: leader
      dependencies: []
LEADER
fi)
    - name: workers
      type: ${TYPE}
      replicas: ${WORKERS}
      role: worker
$(if [ "$LEADERS" -gt 0 ]; then
echo "      dependencies:"
echo "        - leader"
fi)
  workflow:
    dagId: dag-${SWARM_NAME}
    scheduler: quad
    checkpointing: true
---
apiVersion: edge-ai.io/v1alpha1
kind: AIAgent
metadata:
  name: ${SWARM_NAME}-agents
  namespace: ${NAMESPACE}
  labels:
    swarm: ${SWARM_NAME}
spec:
  agentType: ${TYPE}
  replicas: ${AGENTS}
  model:
    name: ${MODEL}
    backend: $([ "$GPU" = true ] && echo "mlx" || echo "llamaedge")
    quantization: "4bit"
  resources:
    cpu: "250m"
    memory: "512Mi"
    gpu: ${GPU}
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: $((AGENTS * 2))
    targetCPU: 70
    targetMemory: 80
  selfHealing:
    enabled: true
    healthCheckInterval: "30s"
    failureThreshold: 3
    recoveryAction: restart
  a2a:
    enabled: true
    discoveryMethod: kubernetes
    exposedSkills:
$(case $TYPE in
    coder)
        echo "      - code-generation"
        echo "      - code-review"
        echo "      - debugging"
        ;;
    researcher)
        echo "      - web-search"
        echo "      - document-analysis"
        echo "      - summarization"
        ;;
    analyst)
        echo "      - data-analysis"
        echo "      - visualization"
        echo "      - reporting"
        ;;
    monitor)
        echo "      - health-check"
        echo "      - alerting"
        echo "      - logging"
        ;;
esac)
  federation:
    enabled: true
    spilloverThreshold: 80
EOF

echo ""
echo -e "${GREEN}Swarm launched successfully!${NC}"
echo ""
echo -e "${YELLOW}Monitoring:${NC}"
echo "  Watch pods: kubectl get pods -n ${NAMESPACE} -l swarm=${SWARM_NAME} -w"
echo "  View logs:  kubectl logs -n ${NAMESPACE} -l swarm=${SWARM_NAME} --tail=100"
echo "  Swarm status: kubectl get agentswarm ${SWARM_NAME} -n ${NAMESPACE}"
echo ""

# Wait for pods to be ready
echo -e "${BLUE}Waiting for agents to be ready...${NC}"
kubectl rollout status deployment/${SWARM_NAME}-agents -n ${NAMESPACE} --timeout=300s 2>/dev/null || {
    echo -e "${YELLOW}Agents are still spinning up. Monitor with:${NC}"
    echo "  kubectl get pods -n ${NAMESPACE} -l swarm=${SWARM_NAME} -w"
}

echo ""
echo -e "${GREEN}Done! Swarm ${SWARM_NAME} is operational.${NC}"

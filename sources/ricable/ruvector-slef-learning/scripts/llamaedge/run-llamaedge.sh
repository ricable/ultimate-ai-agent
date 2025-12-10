#!/usr/bin/env bash
# LlamaEdge Inference Server Script
# Usage: ./scripts/llamaedge/run-llamaedge.sh [action] [options]
#
# Actions:
#   install     - Install WasmEdge with GGML plugin
#   download    - Download LLM model
#   run         - Run inference server
#   docker      - Run via Docker
#   k8s-deploy  - Deploy to Kubernetes

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
MODELS_DIR="${MODELS_DIR:-./models}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-4096}"
BATCH_SIZE="${BATCH_SIZE:-512}"

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

install_wasmedge() {
    log "Installing WasmEdge with GGML plugin..."

    # Detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            ARCH_SUFFIX="x86_64"
            ;;
        aarch64|arm64)
            ARCH_SUFFIX="aarch64"
            ;;
        *)
            error "Unsupported architecture: $ARCH"
            ;;
    esac

    # Install WasmEdge with GGML plugin
    curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | \
        bash -s -- -p /usr/local --plugins wasi_nn-ggml

    # Source environment
    if [ -f "$HOME/.wasmedge/env" ]; then
        source "$HOME/.wasmedge/env"
    fi

    log "WasmEdge installed: $(wasmedge --version)"
}

download_model() {
    local MODEL_NAME="${1:-llama-2-7b-chat}"
    local QUANTIZATION="${2:-Q4_K_M}"

    mkdir -p "$MODELS_DIR"

    log "Downloading model: $MODEL_NAME ($QUANTIZATION)"

    # Common model URLs
    case $MODEL_NAME in
        llama-2-7b-chat)
            MODEL_URL="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.${QUANTIZATION}.gguf"
            ;;
        mistral-7b-instruct)
            MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.${QUANTIZATION}.gguf"
            ;;
        codellama-7b)
            MODEL_URL="https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.${QUANTIZATION}.gguf"
            ;;
        phi-2)
            MODEL_URL="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.${QUANTIZATION}.gguf"
            ;;
        *)
            error "Unknown model: $MODEL_NAME. Provide custom URL as second argument."
            ;;
    esac

    MODEL_FILE="$MODELS_DIR/${MODEL_NAME}.${QUANTIZATION}.gguf"

    if [ -f "$MODEL_FILE" ]; then
        log "Model already exists: $MODEL_FILE"
        return
    fi

    log "Downloading from: $MODEL_URL"
    curl -L -o "$MODEL_FILE" "$MODEL_URL"
    log "Downloaded: $MODEL_FILE"
}

download_api_server() {
    log "Downloading LlamaEdge API server..."

    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            ARCH_SUFFIX="x86_64"
            ;;
        aarch64|arm64)
            ARCH_SUFFIX="aarch64"
            ;;
    esac

    WASM_URL="https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm"

    curl -L -o llama-api-server.wasm "$WASM_URL"
    log "Downloaded: llama-api-server.wasm"
}

run_server() {
    local MODEL_FILE="${1:-$MODELS_DIR/llama-2-7b-chat.Q4_K_M.gguf}"
    local MODEL_NAME="${2:-llama-2-chat}"

    if [ ! -f "$MODEL_FILE" ]; then
        error "Model file not found: $MODEL_FILE"
    fi

    if [ ! -f "llama-api-server.wasm" ]; then
        download_api_server
    fi

    log "Starting LlamaEdge server..."
    log "  Model: $MODEL_FILE"
    log "  Port: $PORT"
    log "  Context size: $CTX_SIZE"
    log "  Batch size: $BATCH_SIZE"
    echo ""

    wasmedge --dir .:. \
        --nn-preload default:GGML:AUTO:"$MODEL_FILE" \
        llama-api-server.wasm \
        --model-name "$MODEL_NAME" \
        --ctx-size "$CTX_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --socket-addr "0.0.0.0:$PORT"
}

run_docker() {
    local MODEL_FILE="${1:-$MODELS_DIR/llama-2-7b-chat.Q4_K_M.gguf}"

    log "Running LlamaEdge in Docker..."

    docker run -d \
        --name llamaedge \
        -p "$PORT:8080" \
        -v "$(pwd)/$MODEL_FILE:/models/model.gguf:ro" \
        wasmedge/wasmedge:latest-ubuntu \
        wasmedge --dir .:. \
            --nn-preload default:GGML:AUTO:/models/model.gguf \
            llama-api-server.wasm \
            --model-name llama-2-chat \
            --ctx-size "$CTX_SIZE" \
            --socket-addr 0.0.0.0:8080

    log "LlamaEdge container started on port $PORT"
}

deploy_k8s() {
    log "Deploying LlamaEdge to Kubernetes..."

    kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: llamaedge
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: llamaedge-config
  namespace: llamaedge
data:
  CTX_SIZE: "$CTX_SIZE"
  BATCH_SIZE: "$BATCH_SIZE"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llamaedge
  namespace: llamaedge
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llamaedge
  template:
    metadata:
      labels:
        app: llamaedge
    spec:
      # Schedule on LlamaEdge-optimized nodes
      nodeSelector:
        ruvector.io/llamaedge-enabled: "true"
      tolerations:
        - key: ruvector.io/llm
          operator: Exists
          effect: PreferNoSchedule
      containers:
        - name: llamaedge
          image: wasmedge/wasmedge:latest-ubuntu
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "16Gi"
              cpu: "4"
            limits:
              memory: "32Gi"
              cpu: "8"
          volumeMounts:
            - name: models
              mountPath: /models
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: llamaedge-models
---
apiVersion: v1
kind: Service
metadata:
  name: llamaedge
  namespace: llamaedge
spec:
  selector:
    app: llamaedge
  ports:
    - port: 8080
      targetPort: 8080
  type: ClusterIP
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llamaedge-models
  namespace: llamaedge
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
EOF

    log "LlamaEdge deployed to Kubernetes"
    echo ""
    echo "Access the API:"
    echo "  kubectl port-forward -n llamaedge svc/llamaedge 8080:8080"
    echo "  curl http://localhost:8080/v1/chat/completions -d '{...}'"
}

test_api() {
    log "Testing LlamaEdge API..."

    curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "llama-2-chat",
            "messages": [
                {"role": "user", "content": "Hello! What is 2+2?"}
            ],
            "max_tokens": 100
        }' | jq .
}

show_help() {
    echo "LlamaEdge Inference Server Script"
    echo ""
    echo "Usage: $0 <action> [options]"
    echo ""
    echo "Actions:"
    echo "  install                      Install WasmEdge with GGML plugin"
    echo "  download <model> [quant]     Download LLM model"
    echo "  run <model-file> [name]      Run inference server"
    echo "  docker <model-file>          Run via Docker"
    echo "  k8s-deploy                   Deploy to Kubernetes"
    echo "  test                         Test the API"
    echo ""
    echo "Models:"
    echo "  llama-2-7b-chat, mistral-7b-instruct, codellama-7b, phi-2"
    echo ""
    echo "Environment variables:"
    echo "  MODELS_DIR     Models directory (default: ./models)"
    echo "  PORT           Server port (default: 8080)"
    echo "  CTX_SIZE       Context size (default: 4096)"
    echo "  BATCH_SIZE     Batch size (default: 512)"
}

main() {
    case "${1:-help}" in
        install)
            install_wasmedge
            ;;
        download)
            download_model "${2:-}" "${3:-Q4_K_M}"
            ;;
        run)
            run_server "${2:-}" "${3:-}"
            ;;
        docker)
            run_docker "${2:-}"
            ;;
        k8s-deploy)
            deploy_k8s
            ;;
        test)
            test_api
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown action: $1. Use --help for usage."
            ;;
    esac
}

main "$@"

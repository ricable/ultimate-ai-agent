#!/bin/bash
#
# GCP Cloud Run Deployment Script for Agentic Flow
# Deploys containerized agentic-flow with Secret Manager integration
#

set -euo pipefail

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-agentic-flow}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
MEMORY="${MEMORY:-2Gi}"
CPU="${CPU:-2}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
TIMEOUT="${TIMEOUT:-300}"
CONCURRENCY="${CONCURRENCY:-80}"

# Model configuration
MODEL_PROVIDER="${MODEL_PROVIDER:-anthropic}"  # anthropic, openrouter, or onnx
MODEL_NAME="${MODEL_NAME:-claude-3-5-sonnet-20241022}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it from https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    # Check if project ID is set
    if [ -z "$PROJECT_ID" ]; then
        log_error "GCP_PROJECT_ID environment variable is not set"
        echo "Usage: GCP_PROJECT_ID=your-project-id ./deploy.sh"
        exit 1
    fi

    # Check if user is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        log_error "Not authenticated with gcloud. Run: gcloud auth login"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

enable_apis() {
    log_info "Enabling required GCP APIs..."

    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        containerregistry.googleapis.com \
        secretmanager.googleapis.com \
        --project="${PROJECT_ID}"

    log_info "APIs enabled successfully"
}

create_secrets() {
    log_info "Creating secrets in Secret Manager..."

    # Read .env file if it exists
    if [ -f "../../.env" ]; then
        log_info "Reading secrets from ../../.env"

        # Create secrets from .env (only if they don't exist)
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ $key =~ ^#.*$ ]] && continue
            [[ -z $key ]] && continue

            # Trim whitespace
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)

            # Check if secret exists
            if gcloud secrets describe "$key" --project="${PROJECT_ID}" &> /dev/null; then
                log_warn "Secret $key already exists, updating..."
                echo -n "$value" | gcloud secrets versions add "$key" \
                    --data-file=- \
                    --project="${PROJECT_ID}"
            else
                log_info "Creating secret: $key"
                echo -n "$value" | gcloud secrets create "$key" \
                    --data-file=- \
                    --replication-policy="automatic" \
                    --project="${PROJECT_ID}"
            fi
        done < <(grep -v '^#' ../../.env | grep '=')
    else
        log_warn ".env file not found. Skipping secret creation."
        log_warn "Please ensure secrets are created manually in GCP Secret Manager"
    fi

    log_info "Secrets created/updated successfully"
}

build_image() {
    log_info "Building Docker image..."

    # Build using Cloud Build for better caching and faster builds
    gcloud builds submit ../.. \
        --tag="${IMAGE_NAME}:latest" \
        --tag="${IMAGE_NAME}:$(git rev-parse --short HEAD 2>/dev/null || echo 'local')" \
        --project="${PROJECT_ID}" \
        --config=cloudbuild.yaml

    log_info "Image built successfully: ${IMAGE_NAME}:latest"
}

deploy_service() {
    log_info "Deploying to Cloud Run..."

    # Prepare secret environment variables
    SECRET_ENVS=""

    # Add secrets based on model provider
    case "$MODEL_PROVIDER" in
        anthropic)
            SECRET_ENVS="--set-secrets=ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest"
            ;;
        openrouter)
            SECRET_ENVS="--set-secrets=OPENROUTER_API_KEY=OPENROUTER_API_KEY:latest"
            ;;
        onnx)
            # ONNX runs locally, no API keys needed
            SECRET_ENVS=""
            ;;
        *)
            log_warn "Unknown model provider: $MODEL_PROVIDER. Deploying with Anthropic secrets."
            SECRET_ENVS="--set-secrets=ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest"
            ;;
    esac

    # Add optional E2B API key if using sandboxes
    if gcloud secrets describe E2B_API_KEY --project="${PROJECT_ID}" &> /dev/null; then
        SECRET_ENVS="${SECRET_ENVS} --set-secrets=E2B_API_KEY=E2B_API_KEY:latest"
    fi

    # Deploy to Cloud Run
    gcloud run deploy "${SERVICE_NAME}" \
        --image="${IMAGE_NAME}:latest" \
        --platform=managed \
        --region="${REGION}" \
        --project="${PROJECT_ID}" \
        --allow-unauthenticated \
        --memory="${MEMORY}" \
        --cpu="${CPU}" \
        --timeout="${TIMEOUT}s" \
        --concurrency="${CONCURRENCY}" \
        --max-instances="${MAX_INSTANCES}" \
        --min-instances="${MIN_INSTANCES}" \
        --set-env-vars="NODE_ENV=production,MODEL_PROVIDER=${MODEL_PROVIDER},MODEL_NAME=${MODEL_NAME},AGENTS_DIR=/app/.claude/agents" \
        ${SECRET_ENVS}

    log_info "Deployment completed successfully!"
}

get_service_url() {
    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --platform=managed \
        --region="${REGION}" \
        --project="${PROJECT_ID}" \
        --format='value(status.url)')

    log_info "Service URL: ${SERVICE_URL}"
    echo ""
    echo "Test your deployment:"
    echo "  curl ${SERVICE_URL}/health"
    echo ""
    echo "List agents:"
    echo "  curl ${SERVICE_URL}/agents"
}

# Main execution
main() {
    log_info "Starting deployment of Agentic Flow to GCP Cloud Run"
    log_info "Project: ${PROJECT_ID}"
    log_info "Region: ${REGION}"
    log_info "Service: ${SERVICE_NAME}"
    log_info "Model Provider: ${MODEL_PROVIDER}"
    echo ""

    check_prerequisites
    enable_apis
    create_secrets
    build_image
    deploy_service
    get_service_url

    log_info "Deployment complete!"
}

# Run main function
main "$@"

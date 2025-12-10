#!/bin/bash
# Deploy Phi-4 fine-tuning to Google Cloud Run with GPU

set -e

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"agentic-llm"}
REGION=${REGION:-"us-central1"}
SERVICE_NAME="phi4-finetuning-gpu"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
GPU_TYPE="nvidia-l4"  # L4 GPU (recommended for Cloud Run)
MEMORY="16Gi"
CPU="4"
TIMEOUT="3600s"  # 1 hour (can be extended)

echo "========================================="
echo "Cloud Run GPU Deployment - Phi-4"
echo "========================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"
echo "GPU: ${GPU_TYPE}"
echo "========================================="

# Check prerequisites
command -v gcloud >/dev/null 2>&1 || { echo "gcloud CLI not found. Install from https://cloud.google.com/sdk"; exit 1; }

# Set project
echo "Setting project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    compute.googleapis.com

# Build container
echo "Building Docker image..."
gcloud builds submit \
    --tag ${IMAGE_NAME} \
    --timeout=30m \
    --machine-type=e2-highcpu-8 \
    -f cloudrun/Dockerfile.gpu \
    .

# Deploy to Cloud Run with GPU
echo "Deploying to Cloud Run with GPU..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --gpu ${GPU_TYPE} \
    --gpu-count 1 \
    --memory ${MEMORY} \
    --cpu ${CPU} \
    --timeout ${TIMEOUT} \
    --min-instances 0 \
    --max-instances 1 \
    --allow-unauthenticated \
    --set-env-vars "PYTHONUNBUFFERED=1,CUDA_VISIBLE_DEVICES=0" \
    --set-secrets "HF_TOKEN=huggingface-token:latest,RESEND_API_KEY=resend-api-key:latest,NOTIFICATION_EMAIL=notification-email:latest" \
    --no-cpu-throttling

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')

echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Monitor training:"
echo "  curl ${SERVICE_URL}/status"
echo ""
echo "View logs:"
echo "  gcloud run logs read ${SERVICE_NAME} --region ${REGION} --limit 50"
echo ""
echo "Get results:"
echo "  curl ${SERVICE_URL}/results"
echo "========================================="

#!/bin/bash
# Setup Google Cloud Secrets for Cloud Run deployment

set -e

echo "Setting up Google Cloud Secrets..."

# Load environment variables
if [ -f "/workspaces/agentic-flow/.env" ]; then
    export $(cat /workspaces/agentic-flow/.env | grep -E "^(RESEND_API_KEY|NOTIFICATION_EMAIL|HUGGINGFACE_API_KEY)=" | xargs)
fi

# Create secrets
echo "Creating resend-api-key secret..."
echo -n "${RESEND_API_KEY}" | gcloud secrets create resend-api-key --data-file=- --replication-policy=automatic 2>&1 || \
    echo -n "${RESEND_API_KEY}" | gcloud secrets versions add resend-api-key --data-file=-

echo "Creating notification-email secret..."
echo -n "${NOTIFICATION_EMAIL}" | gcloud secrets create notification-email --data-file=- --replication-policy=automatic 2>&1 || \
    echo -n "${NOTIFICATION_EMAIL}" | gcloud secrets versions add notification-email --data-file=-

echo "Creating huggingface-token secret..."
echo -n "${HUGGINGFACE_API_KEY}" | gcloud secrets create huggingface-token --data-file=- --replication-policy=automatic 2>&1 || \
    echo -n "${HUGGINGFACE_API_KEY}" | gcloud secrets versions add huggingface-token --data-file=-

echo "✅ Secrets created successfully!"

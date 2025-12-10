#!/bin/bash
# Monitor Phi-4 training on Cloud Run

GCLOUD=$(find /workspaces -name "gcloud" -type f 2>/dev/null | grep -E "bin/gcloud$" | head -1)

if [ -z "$GCLOUD" ]; then
    echo "❌ gcloud not found"
    exit 1
fi

PROJECT="ruv-dev"
SERVICE="phi4-finetuning-gpu"
REGION="us-central1"

echo "========================================="
echo "Monitoring Phi-4 Training"
echo "========================================="

# Get service URL
SERVICE_URL=$($GCLOUD run services describe $SERVICE --region $REGION --project=$PROJECT --format='value(status.url)' 2>&1)

if [ $? -eq 0 ]; then
    echo "Service URL: $SERVICE_URL"
    echo ""

    # Monitor loop
    while true; do
        echo "----------------------------------------"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Checking status..."

        # Check status endpoint
        STATUS=$(curl -s "$SERVICE_URL/status" 2>/dev/null)

        if [ $? -eq 0 ]; then
            echo "$STATUS" | jq '.' 2>/dev/null || echo "$STATUS"

            # Check if completed
            if echo "$STATUS" | grep -q '"phase":"pipeline"' && echo "$STATUS" | grep -q '"status":"completed"'; then
                echo ""
                echo "✅ Training complete!"
                echo "Check your email at ruv@ruv.net for results"
                exit 0
            fi
        else
            echo "⏳ Service starting up..."
        fi

        # Wait 5 minutes
        echo "Next check in 5 minutes..."
        sleep 300
    done
else
    echo "❌ Failed to get service URL"
    echo "$SERVICE_URL"
    exit 1
fi

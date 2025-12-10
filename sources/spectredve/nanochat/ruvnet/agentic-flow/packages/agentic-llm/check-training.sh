#!/bin/bash
# Get service URL and monitor training

./google-cloud-sdk/bin/gcloud config set project ruv-dev > /dev/null 2>&1

URL=$(./google-cloud-sdk/bin/gcloud run services describe phi4-finetuning-gpu --region us-central1 --format='get(status.url)' 2>&1)

echo "========================================="
echo "Phi-4 Training Monitor"
echo "========================================="
echo "Service URL: $URL"
echo ""

# Check status
echo "Fetching training status..."
STATUS=$(curl -s "$URL/status" 2>/dev/null)

if [ -z "$STATUS" ]; then
    echo "⏳ Service is starting up... (this can take a few minutes)"
else
    echo "$STATUS" | python3 -m json.tool 2>/dev/null || echo "$STATUS"
fi

echo ""
echo "========================================="
echo "Monitor will check again in 5 minutes"
echo "========================================="

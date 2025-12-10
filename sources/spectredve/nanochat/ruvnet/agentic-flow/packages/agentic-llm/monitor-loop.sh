#!/bin/bash
# Continuous monitoring loop for Phi-4 training

URL="https://phi4-finetuning-gpu-875130704813.us-central1.run.app"

echo "========================================="
echo "Phi-4 Training Monitor - Starting"
echo "========================================="
echo "Service URL: $URL"
echo "Project: ruv-dev"
echo "Started: $(date)"
echo "========================================="
echo ""

CHECK_COUNT=0

while true; do
    CHECK_COUNT=$((CHECK_COUNT + 1))
    echo "----------------------------------------"
    echo "Check #$CHECK_COUNT - $(date '+%H:%M:%S')"
    echo "----------------------------------------"

    # Check status
    STATUS=$(curl -s "$URL/status" 2>/dev/null)

    if [ -z "$STATUS" ]; then
        echo "⏳ Container starting up..."
    else
        echo "$STATUS" | python3 -m json.tool 2>/dev/null || echo "$STATUS"

        # Check if training is complete
        if echo "$STATUS" | grep -q '"pipeline".*"completed"'; then
            echo ""
            echo "========================================="
            echo "✅ TRAINING COMPLETE!"
            echo "========================================="
            echo "Check email at ruv@ruv.net for full results"
            echo "Completed at: $(date)"
            exit 0
        fi

        # Check if there's an error
        if echo "$STATUS" | grep -q '"failed"'; then
            echo ""
            echo "⚠️ Training encountered an issue - check logs"
        fi
    fi

    echo ""
    echo "Next check in 5 minutes..."
    sleep 300
done

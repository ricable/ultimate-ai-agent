#!/bin/bash
URL="https://phi4-finetuning-gpu-875130704813.us-central1.run.app"

echo "🔍 Monitoring deployment and training..."
while true; do
    echo -e "\n[$(date '+%H:%M:%S')] Checking deployment..."
    
    # Check deployment status
    DEPLOY_STATUS=$(/home/codespace/google-cloud-sdk/bin/gcloud run services describe phi4-finetuning-gpu --region=us-central1 --project=ruv-dev --format="value(status.conditions[0].status)" 2>/dev/null)
    
    if [ "$DEPLOY_STATUS" == "True" ]; then
        echo "✅ Service is ready!"
        
        # Check training status
        STATUS=$(curl -s "$URL/status" 2>/dev/null)
        if echo "$STATUS" | grep -q 'pipeline'; then
            PHASE=$(echo "$STATUS" | grep -o '"current_phase":[^,}]*' | cut -d':' -f2 | tr -d '"' | xargs)
            PIPELINE=$(echo "$STATUS" | grep -o '"pipeline":[^,}]*' | cut -d':' -f2 | tr -d '"' | xargs)
            echo "📊 Phase: $PHASE | Pipeline: $PIPELINE"
            
            if echo "$STATUS" | grep -q '"pipeline".*"completed"'; then
                echo "🎉 TRAINING COMPLETE!"
                echo "Check email at ruv@ruv.net for results"
                exit 0
            fi
        else
            echo "⚠️  Status response: $STATUS"
        fi
    else
        echo "⏳ Service status: $DEPLOY_STATUS"
        tail -5 /tmp/deploy.log 2>/dev/null
    fi
    
    sleep 300  # 5 minutes
done

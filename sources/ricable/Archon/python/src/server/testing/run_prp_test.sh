#!/bin/bash
# Run PRP Viewer test from within Docker container

PROJECT_ID=$1

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: ./run_prp_test.sh <PROJECT_ID>"
    exit 1
fi

echo "Running PRP Viewer test for project: $PROJECT_ID"

# The UI runs on the host at port 3738, but inside Docker we need to use the container name
docker exec -e ARCHON_UI_URL="http://host.docker.internal:3738" \
    -e VITE_SUPABASE_URL="$VITE_SUPABASE_URL" \
    -e VITE_SUPABASE_ANON_KEY="$VITE_SUPABASE_ANON_KEY" \
    Archon-Server \
    python /app/src/server/testing/prp_viewer_test.py --project-id "$PROJECT_ID" --output-dir /app/test_results

# Copy results back to host
echo "Copying test results to host..."
docker cp Archon-Server:/app/test_results ./test_results_$(date +%Y%m%d_%H%M%S)

echo "Test complete. Results copied to ./test_results_$(date +%Y%m%d_%H%M%S)"

#!/bin/bash
# Test health check endpoint

set -e

echo "🏥 Testing Health Check Endpoint"
echo "================================"
echo ""

# Build the image
echo "📦 Building Docker image..."
docker build -t claude-agents:test -f Dockerfile . > /dev/null 2>&1

# Start container in background
echo "🚀 Starting container..."
CONTAINER_ID=$(docker run -d --rm \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  -e KEEP_ALIVE=true \
  -e HEALTH_PORT=8080 \
  -p 8080:8080 \
  claude-agents:test)

echo "Container ID: $CONTAINER_ID"

# Wait for health endpoint to be ready
echo "⏳ Waiting for health endpoint..."
sleep 3

# Test health endpoint
echo ""
echo "🔍 Testing /health endpoint..."
RESPONSE=$(curl -s -w "\nHTTP_CODE:%{http_code}" http://localhost:8080/health)
HTTP_CODE=$(echo "$RESPONSE" | grep HTTP_CODE | cut -d: -f2)
BODY=$(echo "$RESPONSE" | grep -v HTTP_CODE)

echo "HTTP Status: $HTTP_CODE"
echo "Response Body:"
echo "$BODY" | jq '.'

# Validate response
if [ "$HTTP_CODE" = "200" ]; then
  echo ""
  echo "✅ Health check passed!"

  # Check response structure
  STATUS=$(echo "$BODY" | jq -r '.status')
  echo "   Status: $STATUS"
  echo "   Version: $(echo "$BODY" | jq -r '.version')"
  echo "   Uptime: $(echo "$BODY" | jq -r '.uptime')s"
  echo "   API Check: $(echo "$BODY" | jq -r '.checks.api.status')"
  echo "   Memory Check: $(echo "$BODY" | jq -r '.checks.memory.status')"
else
  echo ""
  echo "❌ Health check failed with status $HTTP_CODE"
  docker logs $CONTAINER_ID
  docker stop $CONTAINER_ID
  exit 1
fi

# Cleanup
echo ""
echo "🧹 Cleaning up..."
docker stop $CONTAINER_ID > /dev/null 2>&1

echo "✅ Health check test completed successfully!"

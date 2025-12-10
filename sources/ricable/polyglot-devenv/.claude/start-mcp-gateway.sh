#!/bin/bash

# Docker MCP Gateway HTTP Transport Startup Script
# Starts the Docker MCP gateway with HTTP/SSE transport support for Claude and Gemini clients

set -e

echo "🚀 Starting Docker MCP Gateway with HTTP Transport..."

# Configuration
GATEWAY_PORT=${GATEWAY_PORT:-8080}
LOG_FILE="/tmp/docker-mcp-gateway.log"
PID_FILE="/tmp/docker-mcp-gateway.pid"

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping Docker MCP Gateway..."
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            rm -f "$PID_FILE"
        fi
    fi
    docker container ls --filter "label=docker-mcp=true" -q | xargs -r docker stop
    echo "✅ Cleanup completed"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Check if gateway is already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "⚠️  Gateway already running with PID $PID"
        exit 1
    fi
fi

# Start the gateway
echo "🔧 Starting Docker MCP Gateway..."
echo "📝 Logs: $LOG_FILE"
echo "🌐 Configuration: HTTP/SSE transport enabled"
echo "🔒 Security: Resource limits and secret blocking enabled"

# Start gateway in background with logging
docker mcp gateway run \
    --verbose \
    --log-calls \
    --block-secrets \
    --verify-signatures \
    > "$LOG_FILE" 2>&1 &

GATEWAY_PID=$!
echo $GATEWAY_PID > "$PID_FILE"

echo "✅ Docker MCP Gateway started with PID $GATEWAY_PID"
echo "📋 Available tools: filesystem, fetch, memory, context7, docker, puppeteer, brave, youtube_transcript"
echo "🔗 Clients can connect via: claude-desktop, gemini, cursor, vscode"
echo "📊 Monitor logs: tail -f $LOG_FILE"

# Wait for gateway to start
sleep 3

# Check if still running
if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
    echo "❌ Gateway failed to start. Check logs: $LOG_FILE"
    exit 1
fi

echo "🎉 Gateway is running successfully!"

# For interactive mode, wait for user input
if [ "${1:-}" != "--daemon" ]; then
    echo "Press Ctrl+C to stop the gateway..."
    wait $GATEWAY_PID
fi
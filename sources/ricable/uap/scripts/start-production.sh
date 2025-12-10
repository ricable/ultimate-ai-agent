#!/bin/bash
# scripts/start-production.sh
# Production startup script for UAP Agent Orchestration Platform

set -e  # Exit on any error

echo "=== Starting UAP Production Services ==="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"

# Change to app directory
cd "$APP_DIR"

# Setup logging
LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR"

# Activate Python virtual environment
source backend/venv/bin/activate

# Export required environment variables
export PYTHONPATH=/app/backend
export UAP_ENV=production
export NODE_ENV=production

# Framework-specific environment variables (ready for real implementations)
export AGNO_GPU_MEMORY=${AGNO_GPU_MEMORY:-"8GB"}
export MASTRA_WORKER_COUNT=${MASTRA_WORKER_COUNT:-"4"}
export COPILOT_CACHE_SIZE=${COPILOT_CACHE_SIZE:-"1GB"}

# Performance tuning
export UVICORN_WORKERS=${UVICORN_WORKERS:-"4"}
export UVICORN_WORKER_CLASS=${UVICORN_WORKER_CLASS:-"uvicorn.workers.UvicornWorker"}

echo "Environment variables set"
echo "PYTHONPATH: $PYTHONPATH"
echo "UAP_ENV: $UAP_ENV"
echo "Workers: $UVICORN_WORKERS"

# Start the backend API server
echo "Starting UAP Backend API Server..."
cd backend

# Check if we have GPU available and log it
if nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
    export CUDA_VISIBLE_DEVICES=0
else
    echo "No GPU detected, running in CPU mode"
fi

# Start backend with production settings
python -m uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "$UVICORN_WORKERS" \
    --worker-class "$UVICORN_WORKER_CLASS" \
    --access-log \
    --log-level info \
    --log-config logging.conf \
    >> "$LOG_DIR/backend.log" 2>&1 &

BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait for backend to be ready
echo "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Backend failed to start properly"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

cd ..

# Start frontend if we have a built version
if [ -d "frontend/dist" ]; then
    echo "Starting UAP Frontend Server..."
    cd frontend
    
    # Serve the built frontend using a simple HTTP server
    python -m http.server 3000 --directory dist \
        >> "$LOG_DIR/frontend.log" 2>&1 &
    
    FRONTEND_PID=$!
    echo "Frontend started with PID: $FRONTEND_PID"
    cd ..
else
    echo "No built frontend found, skipping frontend server"
    FRONTEND_PID=""
fi

# Create PID file for process management
echo "$BACKEND_PID" > /app/backend.pid
if [ -n "$FRONTEND_PID" ]; then
    echo "$FRONTEND_PID" > /app/frontend.pid
fi

# Setup signal handlers for graceful shutdown
cleanup() {
    echo "Shutting down UAP services..."
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    rm -f /app/backend.pid /app/frontend.pid
    exit 0
}

trap cleanup SIGTERM SIGINT

# Monitor services
echo "UAP Production Services Started Successfully"
echo "Backend API: http://localhost:8000"
if [ -n "$FRONTEND_PID" ]; then
    echo "Frontend UI: http://localhost:3000"
fi
echo "Health check: http://localhost:8000/health"
echo "API docs: http://localhost:8000/docs"

# Keep script running and monitor child processes
while true; do
    # Check if backend is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "Backend process died, restarting..."
        cd backend
        python -m uvicorn main:app \
            --host 0.0.0.0 \
            --port 8000 \
            --workers "$UVICORN_WORKERS" \
            --worker-class "$UVICORN_WORKER_CLASS" \
            --access-log \
            --log-level info \
            >> "$LOG_DIR/backend.log" 2>&1 &
        BACKEND_PID=$!
        echo "$BACKEND_PID" > /app/backend.pid
        cd ..
    fi
    
    # Check if frontend is still running (if we started it)
    if [ -n "$FRONTEND_PID" ] && ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "Frontend process died, restarting..."
        cd frontend
        python -m http.server 3000 --directory dist \
            >> "$LOG_DIR/frontend.log" 2>&1 &
        FRONTEND_PID=$!
        echo "$FRONTEND_PID" > /app/frontend.pid
        cd ..
    fi
    
    sleep 10
done
#!/bin/bash
# Entrypoint for Cloud Run
# Health server starts immediately and handles training in background thread

echo "🚀 Starting Cloud Run GPU Training Service"
echo "📡 Health server will respond immediately on port ${PORT:-8080}"
echo "⏳ Training will start automatically after 30 seconds"

# Run health server (which triggers training internally)
exec python3 /app/cloudrun/health_server.py

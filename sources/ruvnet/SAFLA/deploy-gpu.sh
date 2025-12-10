#!/bin/bash

# SAFLA GPU Deployment Script for Fly.io
# This script deploys SAFLA with GPU optimization to Fly.io

set -e

echo "🚀 SAFLA GPU Deployment to Fly.io"
echo "=================================="

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl is not installed. Please install it first:"
    echo "   curl -L https://fly.io/install.sh | sh"
    exit 1
fi

# Check if logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    echo "❌ Not logged in to Fly.io. Please login first:"
    echo "   flyctl auth login"
    exit 1
fi

# Create Fly.io app if it doesn't exist
if ! flyctl apps list | grep -q "safla"; then
    echo "📝 Creating Fly.io app..."
    flyctl apps create safla --org personal
fi

# Create volume for persistent storage
echo "💾 Creating persistent volume..."
flyctl volumes create safla_data --region ord --size 50 --app safla || true

# Set secrets
echo "🔐 Setting up secrets..."
if [ -f ".env" ]; then
    flyctl secrets import < .env --app safla
else
    echo "⚠️  No .env file found. Setting default secrets..."
    flyctl secrets set \
        SAFLA_SECRET_KEY="$(openssl rand -hex 32)" \
        SAFLA_JWT_SECRET="$(openssl rand -hex 32)" \
        --app safla
fi

# Deploy to Fly.io
echo "🚀 Deploying to Fly.io..."
flyctl deploy --app safla --remote-only

# Check deployment status
echo "🔍 Checking deployment status..."
flyctl status --app safla

# Show logs
echo "📋 Recent logs:"
flyctl logs --app safla --lines 20

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your SAFLA GPU instance is available at:"
echo "   https://safla.fly.dev"
echo ""
echo "📊 Monitor your app:"
echo "   flyctl status --app safla"
echo "   flyctl logs --app safla"
echo ""
echo "💡 GPU Optimization Features:"
echo "   - NVIDIA A10 GPU (24GB VRAM)"
echo "   - Flash Attention 2 enabled"
echo "   - Mixed precision training"
echo "   - Optimized batch processing"
echo "   - Persistent model storage"
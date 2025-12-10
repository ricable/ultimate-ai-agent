#!/bin/bash

# Federation Multi-Agent Collaboration Test
# This script runs a complete Docker-based test of the federated memory system

set -e

echo "🚀 Federation Multi-Agent Collaboration Test"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo -e "${BLUE}📦 Building project...${NC}"
cd ../..
npm run build
cd docker/federation-test

echo ""
echo -e "${BLUE}🏗️  Building Docker images...${NC}"
docker-compose build

echo ""
echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
echo -e "${YELLOW}Starting federation test...${NC}"
echo ""
echo "Test scenario:"
echo "  • 1 Federation Hub (shared memory server)"
echo "  • 4 Collaborative Agents (test-collaboration tenant):"
echo "    - Researcher: Finds patterns"
echo "    - Coder: Implements solutions"
echo "    - Tester: Validates work"
echo "    - Reviewer: Quality checks"
echo "  • 1 Isolated Agent (different-tenant tenant)"
echo "  • 1 Monitor Dashboard (http://localhost:3000)"
echo ""
echo "Expected behavior:"
echo "  ✓ All agents connect to hub"
echo "  ✓ Agents share memories within same tenant"
echo "  ✓ Isolated agent cannot access other tenant's memories"
echo "  ✓ Real-time sync <100ms"
echo "  ✓ Memory persists across agent lifecycles"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop the test${NC}"
echo ""

# Start services
docker-compose up --remove-orphans

# Cleanup on exit
trap "docker-compose down -v" EXIT

echo ""
echo -e "${GREEN}✅ Test complete!${NC}"

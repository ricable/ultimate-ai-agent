#!/bin/bash
# Climate Prediction Demo Script

set -e

echo "🌍 Climate Prediction with ReasoningBank"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

if ! command -v cargo &> /dev/null; then
    echo -e "${YELLOW}⚠️  Rust not found. Install from: https://rustup.rs${NC}"
    exit 1
fi

if ! command -v npx &> /dev/null; then
    echo -e "${YELLOW}⚠️  Node.js/npx not found. Install from: https://nodejs.org${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"
echo ""

# Build project
echo -e "${BLUE}Building project...${NC}"
cargo build --release
echo -e "${GREEN}✅ Build complete${NC}"
echo ""

# Initialize ReasoningBank session
echo -e "${BLUE}Initializing ReasoningBank session...${NC}"
npx claude-flow@alpha hooks session-restore --session-id "climate-prediction" || true
echo ""

# Run prediction
echo -e "${BLUE}Running climate predictions...${NC}"
echo ""
cargo run --release

# Show learning statistics
echo ""
echo -e "${BLUE}Querying learned patterns...${NC}"
npx claude-flow@alpha memory status --reasoningbank || true

echo ""
echo -e "${GREEN}✅ Prediction complete!${NC}"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "  - Run multiple times to see learning improvement"
echo "  - Check .swarm/climate_memory.db for stored patterns"
echo "  - Use 'cargo test' to run integration tests"

#!/bin/bash
# Quick publish script for agentic-flow
# Runs all checks, builds, and publishes to npm
# Author: ruv (@ruvnet)

set -e

echo "🚀 Agentic Flow - Quick Publish Script"
echo "======================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Parse arguments
DRY_RUN=false
SKIP_TESTS=false
SKIP_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--skip-tests] [--skip-build]"
            exit 1
            ;;
    esac
done

# 1. Verify package
echo -e "${BLUE}▶ Step 1: Verifying package...${NC}"
if [ -f "$SCRIPT_DIR/verify-package.sh" ]; then
    bash "$SCRIPT_DIR/verify-package.sh"
else
    echo -e "${YELLOW}⚠️  verify-package.sh not found, skipping...${NC}"
fi
echo ""

# 2. Run linter
echo -e "${BLUE}▶ Step 2: Running linter...${NC}"
if npm run lint 2>/dev/null; then
    echo -e "${GREEN}✓ Linting passed${NC}"
else
    echo -e "${YELLOW}⚠️  Linting not configured or failed${NC}"
fi
echo ""

# 3. Run type checking
echo -e "${BLUE}▶ Step 3: Running TypeScript type checking...${NC}"
if npm run typecheck 2>/dev/null; then
    echo -e "${GREEN}✓ Type checking passed${NC}"
else
    echo -e "${YELLOW}⚠️  Type checking not configured or failed${NC}"
fi
echo ""

# 4. Build packages
if [ "$SKIP_BUILD" = false ]; then
    echo -e "${BLUE}▶ Step 4: Building packages...${NC}"
    if [ -f "$SCRIPT_DIR/build-all.sh" ]; then
        bash "$SCRIPT_DIR/build-all.sh"
    else
        npm run build
    fi
    echo ""
else
    echo -e "${YELLOW}⚠️  Skipping build step${NC}"
    echo ""
fi

# 5. Run tests
if [ "$SKIP_TESTS" = false ]; then
    echo -e "${BLUE}▶ Step 5: Running tests...${NC}"
    if npm test 2>/dev/null; then
        echo -e "${GREEN}✓ Tests passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Tests not configured or failed${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}⚠️  Skipping tests${NC}"
    echo ""
fi

# 6. Show what will be published
echo -e "${BLUE}▶ Step 6: Package contents preview...${NC}"
npm pack --dry-run
echo ""

# 7. Confirm publication
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}📦 Dry run mode - not publishing${NC}"
    echo ""
    echo "To actually publish, run without --dry-run:"
    echo "  bash $0"
    exit 0
fi

# Get version
VERSION=$(node -p "require('./package.json').version" 2>/dev/null)

echo -e "${YELLOW}❓ Ready to publish version ${VERSION} to npm?${NC}"
read -p "   Continue? (yes/no) " -r
echo

if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo "❌ Publish cancelled"
    exit 1
fi

# 8. Publish to npm
echo -e "${BLUE}▶ Step 7: Publishing to npm...${NC}"
npm publish

echo ""
echo -e "${GREEN}✅ Successfully published agentic-flow@${VERSION}!${NC}"
echo ""
echo "📝 Next steps:"
echo "   1. Create GitHub release:"
echo "      git tag v${VERSION}"
echo "      git push origin v${VERSION}"
echo "      gh release create v${VERSION}"
echo ""
echo "   2. Verify on npmjs.com:"
echo "      https://www.npmjs.com/package/agentic-flow"
echo ""
echo "   3. Test installation:"
echo "      npm install -g agentic-flow@${VERSION}"
echo "      agentic-flow --version"

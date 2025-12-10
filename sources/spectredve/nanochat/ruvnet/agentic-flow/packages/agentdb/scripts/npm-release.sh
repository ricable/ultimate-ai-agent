#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         AgentDB v1.3.0 NPM Release Script                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
  echo -e "${RED}❌ Error: package.json not found${NC}"
  exit 1
fi

# Verify version
VERSION=$(node -p "require('./package.json').version")
if [ "$VERSION" != "1.3.0" ]; then
  echo -e "${RED}❌ Error: Version mismatch. Expected 1.3.0, got $VERSION${NC}"
  exit 1
fi

echo -e "${BLUE}📦 Package: agentdb@${VERSION}${NC}"
echo ""

# Pre-flight checks
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Pre-Flight Checks${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check npm login
if ! npm whoami > /dev/null 2>&1; then
  echo -e "${RED}❌ Not logged in to npm${NC}"
  echo -e "${YELLOW}→ Run: npm login${NC}"
  exit 1
else
  NPM_USER=$(npm whoami)
  echo -e "${GREEN}✅ Logged in as: $NPM_USER${NC}"
fi

# Check git status
if [ -n "$(git status --porcelain)" ]; then
  echo -e "${YELLOW}⚠️  Warning: Uncommitted changes detected${NC}"
  git status --short
  echo ""
  read -p "Continue anyway? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
else
  echo -e "${GREEN}✅ Git working directory clean${NC}"
fi

# Check if tag exists
if git rev-parse "v${VERSION}" >/dev/null 2>&1; then
  echo -e "${YELLOW}⚠️  Warning: Tag v${VERSION} already exists${NC}"
else
  echo -e "${GREEN}✅ Tag v${VERSION} ready to create${NC}"
fi

# Check build artifacts
if [ ! -d "dist" ]; then
  echo -e "${RED}❌ dist/ directory not found. Run: npm run build${NC}"
  exit 1
else
  echo -e "${GREEN}✅ Build artifacts present${NC}"
fi

echo ""

# Step 1: Dry run
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 1: Dry Run${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

npm publish --access public --dry-run 2>&1 | tail -30

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Dry run successful${NC}"
else
  echo -e "${RED}❌ Dry run failed${NC}"
  exit 1
fi

echo ""

# Step 2: Create tarball
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 2: Create Tarball${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

npm pack > /dev/null 2>&1
TARBALL="agentdb-${VERSION}.tgz"

if [ -f "$TARBALL" ]; then
  SIZE=$(du -h "$TARBALL" | cut -f1)
  echo -e "${GREEN}✅ Tarball created: $TARBALL ($SIZE)${NC}"

  echo -e "${YELLOW}→ Tarball contents (first 30 files):${NC}"
  tar -tzf "$TARBALL" | head -30
else
  echo -e "${RED}❌ Failed to create tarball${NC}"
  exit 1
fi

echo ""

# Step 3: Publish
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3: Publish to npm${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "${YELLOW}⚠️  About to publish agentdb@${VERSION} to npm${NC}"
echo ""
read -p "Are you sure you want to publish? (yes/no) " -r
echo

if [ "$REPLY" != "yes" ]; then
  echo -e "${YELLOW}→ Publish cancelled${NC}"
  rm "$TARBALL"
  exit 0
fi

echo -e "${YELLOW}→ Publishing to npm...${NC}"
npm publish --access public

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Published to npm successfully!${NC}"
else
  echo -e "${RED}❌ Publish failed${NC}"
  rm "$TARBALL"
  exit 1
fi

# Clean up tarball
rm "$TARBALL"

echo ""

# Step 4: Create git tag
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 4: Create Git Tag${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if git rev-parse "v${VERSION}" >/dev/null 2>&1; then
  echo -e "${YELLOW}→ Tag v${VERSION} already exists, skipping${NC}"
else
  git tag -a "v${VERSION}" -m "Release v${VERSION}: Complete 29 MCP tools with learning system"
  echo -e "${GREEN}✅ Tag v${VERSION} created${NC}"

  read -p "Push tag to remote? (y/N) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin "v${VERSION}"
    echo -e "${GREEN}✅ Tag pushed to remote${NC}"
  fi
fi

echo ""

# Step 5: Verify installation
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 5: Verify Installation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "${YELLOW}→ Waiting 30 seconds for npm to propagate...${NC}"
sleep 30

echo -e "${YELLOW}→ Testing installation...${NC}"
NPM_VERSION=$(npm view agentdb version 2>/dev/null || echo "unavailable")

if [ "$NPM_VERSION" = "$VERSION" ]; then
  echo -e "${GREEN}✅ Package available on npm: agentdb@${NPM_VERSION}${NC}"
else
  echo -e "${YELLOW}⚠️  Package version: $NPM_VERSION (expected $VERSION)${NC}"
  echo -e "${YELLOW}   This is normal - npm may take a few minutes to update${NC}"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Release Complete!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Package published: agentdb@${VERSION}"
echo "npm registry: https://www.npmjs.com/package/agentdb"
echo "Install: npm install agentdb@${VERSION}"
echo "MCP usage: npx agentdb mcp"
echo ""
echo "Next steps:"
echo "  1. Create GitHub release: gh release create v${VERSION}"
echo "  2. Update documentation site"
echo "  3. Announce on social media"
echo "  4. Monitor npm downloads and issues"
echo ""

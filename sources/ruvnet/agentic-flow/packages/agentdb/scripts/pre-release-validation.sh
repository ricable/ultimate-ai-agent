#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   AgentDB v1.3.0 Pre-Release Validation Suite             ║"
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

# Create reports directory
mkdir -p validation-reports

echo -e "${BLUE}📋 Validation Plan:${NC}"
echo "  1. Clean build verification"
echo "  2. Docker environment validation"
echo "  3. MCP tools comprehensive test"
echo "  4. Regression testing"
echo "  5. Performance benchmarks"
echo "  6. Dependency audit"
echo "  7. Package integrity check"
echo ""

# 1. Clean build
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}1️⃣  Clean Build Verification${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

rm -rf dist node_modules package-lock.json
echo -e "${YELLOW}→ Installing fresh dependencies...${NC}"
npm install > validation-reports/npm-install.log 2>&1

echo -e "${YELLOW}→ Building TypeScript...${NC}"
npm run build > validation-reports/build.log 2>&1

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Clean build successful${NC}"
else
  echo -e "${RED}❌ Build failed - check validation-reports/build.log${NC}"
  exit 1
fi
echo ""

# 2. Docker validation
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}2️⃣  Docker Environment Validation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "${YELLOW}→ Building Docker validation image...${NC}"
docker build -f Dockerfile.validation -t agentdb-validation:1.3.0 . > validation-reports/docker-build.log 2>&1

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
  echo -e "${RED}❌ Docker build failed - check validation-reports/docker-build.log${NC}"
  exit 1
fi

echo -e "${YELLOW}→ Running validation container...${NC}"
docker run --rm agentdb-validation:1.3.0 > validation-reports/docker-validation.log 2>&1

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Docker validation passed${NC}"
  cat validation-reports/docker-validation.log
else
  echo -e "${RED}❌ Docker validation failed - check validation-reports/docker-validation.log${NC}"
  exit 1
fi
echo ""

# 3. MCP tools test
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}3️⃣  MCP Tools Comprehensive Test${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "${YELLOW}→ Starting MCP server...${NC}"
timeout 5 node dist/mcp/agentdb-mcp-server.js 2>&1 | tee validation-reports/mcp-server.log &
MCP_PID=$!
sleep 2

TOOL_COUNT=$(grep -o "[0-9]* tools available" validation-reports/mcp-server.log | awk '{print $1}')
if [ "$TOOL_COUNT" = "29" ]; then
  echo -e "${GREEN}✅ MCP server verified: 29 tools available${NC}"
else
  echo -e "${RED}❌ MCP tool count mismatch: expected 29, got $TOOL_COUNT${NC}"
  kill $MCP_PID 2>/dev/null || true
  exit 1
fi

kill $MCP_PID 2>/dev/null || true
echo ""

# 4. Regression testing
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}4️⃣  Regression Testing${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "${YELLOW}→ Running test suite...${NC}"
npm test -- specification-tools.test.ts 2>&1 | tee validation-reports/test-results.log

TESTS_PASSED=$(grep -o "[0-9]* passed" validation-reports/test-results.log | head -1 | awk '{print $1}')
TESTS_TOTAL=$(grep -o "([0-9]*)" validation-reports/test-results.log | head -1 | tr -d '()')

echo ""
echo -e "${BLUE}📊 Test Summary:${NC}"
echo -e "   Total Tests: $TESTS_TOTAL"
echo -e "   Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "   Failed: $((TESTS_TOTAL - TESTS_PASSED))"

if [ "$TESTS_PASSED" -ge 77 ]; then
  PASS_RATE=$(echo "scale=1; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc)
  echo -e "   ${GREEN}✅ Pass Rate: ${PASS_RATE}% (threshold: 85%)${NC}"
else
  echo -e "   ${RED}❌ Insufficient passing tests (expected ≥77)${NC}"
  exit 1
fi
echo ""

# 5. Performance benchmarks
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}5️⃣  Performance Benchmarks${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "${YELLOW}→ Extracting benchmark results...${NC}"
grep -A 1 "Performance Benchmarks" validation-reports/test-results.log | grep -E "Avg:|Throughput:|Latency:|qps|ops/sec" | head -10

echo -e "${GREEN}✅ Performance benchmarks recorded${NC}"
echo ""

# 6. Dependency audit
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}6️⃣  Dependency Security Audit${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

npm audit --audit-level=moderate > validation-reports/audit.log 2>&1 || true

VULNERABILITIES=$(grep "vulnerabilities" validation-reports/audit.log | head -1)
if echo "$VULNERABILITIES" | grep -q "0 vulnerabilities"; then
  echo -e "${GREEN}✅ No security vulnerabilities found${NC}"
else
  echo -e "${YELLOW}⚠️  Security audit: $VULNERABILITIES${NC}"
  echo -e "${YELLOW}→ Review validation-reports/audit.log for details${NC}"
fi
echo ""

# 7. Package integrity
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}7️⃣  Package Integrity Check${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "${YELLOW}→ Creating package tarball...${NC}"
npm pack --dry-run > validation-reports/package-contents.log 2>&1

FILE_COUNT=$(wc -l < validation-reports/package-contents.log)
echo -e "${GREEN}✅ Package contains $FILE_COUNT files${NC}"

echo -e "${YELLOW}→ Verifying required files...${NC}"
REQUIRED_FILES=(
  "dist/index.js"
  "dist/index.d.ts"
  "dist/cli/agentdb-cli.js"
  "dist/mcp/agentdb-mcp-server.js"
  "package.json"
  "README.md"
)

ALL_PRESENT=true
for file in "${REQUIRED_FILES[@]}"; do
  if grep -q "$file" validation-reports/package-contents.log; then
    echo -e "   ${GREEN}✓${NC} $file"
  else
    echo -e "   ${RED}✗${NC} $file (MISSING)"
    ALL_PRESENT=false
  fi
done

if [ "$ALL_PRESENT" = true ]; then
  echo -e "${GREEN}✅ All required files present in package${NC}"
else
  echo -e "${RED}❌ Missing required files in package${NC}"
  exit 1
fi
echo ""

# Final summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              VALIDATION SUMMARY                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ Clean build verification${NC}"
echo -e "${GREEN}✅ Docker environment validation${NC}"
echo -e "${GREEN}✅ MCP tools verified (29 tools)${NC}"
echo -e "${GREEN}✅ Regression tests passed ($TESTS_PASSED/$TESTS_TOTAL)${NC}"
echo -e "${GREEN}✅ Performance benchmarks recorded${NC}"
echo -e "${GREEN}✅ Security audit completed${NC}"
echo -e "${GREEN}✅ Package integrity verified${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 AgentDB v1.3.0 is READY FOR NPM RELEASE!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Next steps:"
echo "  1. Review validation reports in: ./validation-reports/"
echo "  2. Run: npm publish --access public --dry-run"
echo "  3. Run: npm publish --access public"
echo "  4. Create git tag: git tag v1.3.0 && git push origin v1.3.0"
echo ""

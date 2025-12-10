#!/bin/bash
# Verify package is ready for npm publishing
# Author: ruv (@ruvnet)

set -e

echo "🔍 Verifying Agentic Flow Package..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
ERRORS=0
WARNINGS=0

print_check() {
    echo -e "${BLUE}▶ Checking: ${1}${NC}"
}

print_pass() {
    echo -e "${GREEN}  ✓ ${1}${NC}"
}

print_fail() {
    echo -e "${RED}  ✗ ${1}${NC}"
    ERRORS=$((ERRORS + 1))
}

print_warn() {
    echo -e "${YELLOW}  ⚠ ${1}${NC}"
    WARNINGS=$((WARNINGS + 1))
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# 1. Check package.json exists and is valid
print_check "package.json validity"
if [ -f "package.json" ]; then
    if node -e "require('./package.json')" 2>/dev/null; then
        print_pass "package.json is valid JSON"
    else
        print_fail "package.json has invalid JSON"
    fi
else
    print_fail "package.json not found"
fi

# 2. Check version format
print_check "Version format"
VERSION=$(node -p "require('./package.json').version" 2>/dev/null)
if [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    print_pass "Version $VERSION is valid semver"
else
    print_fail "Version $VERSION is not valid semver"
fi

# 3. Check required fields
print_check "Required package.json fields"
NAME=$(node -p "require('./package.json').name" 2>/dev/null)
DESC=$(node -p "require('./package.json').description" 2>/dev/null)
AUTHOR=$(node -p "require('./package.json').author?.name || require('./package.json').author" 2>/dev/null)
LICENSE=$(node -p "require('./package.json').license" 2>/dev/null)

[ "$NAME" != "undefined" ] && print_pass "Name: $NAME" || print_fail "Name missing"
[ "$DESC" != "undefined" ] && print_pass "Description exists" || print_fail "Description missing"
[ "$AUTHOR" != "undefined" ] && print_pass "Author: $AUTHOR" || print_fail "Author missing"
[ "$LICENSE" != "undefined" ] && print_pass "License: $LICENSE" || print_fail "License missing"

# 4. Check built files exist
print_check "Built files"
if [ -d "agentic-flow/dist" ]; then
    print_pass "agentic-flow/dist/ exists"

    if [ -f "agentic-flow/dist/index.js" ]; then
        print_pass "Main entry point exists"
    else
        print_fail "Main entry point (index.js) missing"
    fi

    if [ -f "agentic-flow/dist/index.d.ts" ]; then
        print_pass "TypeScript declarations exist"
    else
        print_warn "TypeScript declarations missing"
    fi
else
    print_fail "agentic-flow/dist/ not found - run 'npm run build'"
fi

# 5. Check bin files
print_check "CLI executables"
if [ -f "agentic-flow/dist/cli-proxy.js" ]; then
    if head -1 "agentic-flow/dist/cli-proxy.js" | grep -q "^#!/usr/bin/env node"; then
        print_pass "agentic-flow CLI has shebang"
    else
        print_warn "agentic-flow CLI missing shebang"
    fi
else
    print_fail "agentic-flow CLI not found"
fi

if [ -f "agentic-flow/dist/agentdb/cli/agentdb-cli.js" ]; then
    print_pass "agentdb CLI exists"
else
    print_warn "agentdb CLI not found"
fi

# 6. Check for secrets
print_check "Secrets scanning"
if grep -r "sk-ant-" . --exclude-dir=node_modules --exclude-dir=.git --exclude="*.md" -q 2>/dev/null; then
    print_fail "Found Anthropic API keys in code!"
else
    print_pass "No Anthropic API keys found"
fi

if grep -r "ANTHROPIC_API_KEY.*sk-ant" . --exclude-dir=node_modules --exclude-dir=.git --exclude="*.md" -q 2>/dev/null; then
    print_fail "Found hardcoded API keys!"
else
    print_pass "No hardcoded API keys found"
fi

# 7. Check .npmignore exists
print_check ".npmignore configuration"
if [ -f ".npmignore" ]; then
    print_pass ".npmignore exists"
else
    print_warn ".npmignore not found (will use .gitignore)"
fi

# 8. Check documentation
print_check "Documentation"
[ -f "README.md" ] && print_pass "README.md exists" || print_warn "README.md missing"
[ -f "LICENSE" ] && print_pass "LICENSE exists" || print_warn "LICENSE missing"
[ -f "CHANGELOG.md" ] && print_pass "CHANGELOG.md exists" || print_warn "CHANGELOG.md missing"

# 9. Check package size
print_check "Package size"
SIZE=$(npm pack --dry-run 2>&1 | grep "package size" | awk '{print $4}')
if [ ! -z "$SIZE" ]; then
    SIZE_NUM=$(echo $SIZE | sed 's/[^0-9.]//g')
    if (( $(echo "$SIZE_NUM < 10" | bc -l) )); then
        print_pass "Package size: ${SIZE}B (< 10MB)"
    else
        print_warn "Package size: ${SIZE}B (> 10MB - consider optimizing)"
    fi
fi

# 10. Check dependencies
print_check "Dependencies"
if [ -d "node_modules" ]; then
    print_pass "node_modules exists"
else
    print_warn "node_modules not found - run 'npm install'"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Ready to publish.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. npm publish --dry-run    (preview what will be published)"
    echo "  2. npm publish              (publish to npm)"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  ${WARNINGS} warnings found${NC}"
    echo "Package can be published but consider fixing warnings"
    exit 0
else
    echo -e "${RED}❌ ${ERRORS} errors found${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}   ${WARNINGS} warnings found${NC}"
    fi
    echo "Fix errors before publishing"
    exit 1
fi

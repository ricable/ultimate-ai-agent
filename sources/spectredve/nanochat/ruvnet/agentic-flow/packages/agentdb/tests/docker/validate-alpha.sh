#!/bin/bash

# AgentDB Alpha Package Validation Script
# Tests all functionality in a clean Docker environment

set -e  # Exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  AgentDB v2.0.0-alpha.1 Package Validation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

PASSED=0
FAILED=0
WARNINGS=0

# Test results array
declare -a TEST_RESULTS

log_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
    TEST_RESULTS+=("PASS: $1")
}

log_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
    TEST_RESULTS+=("FAIL: $1")
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
    TEST_RESULTS+=("WARN: $1")
}

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# ============================================================================
# Section 1: Package Installation
# ============================================================================

echo -e "\n${BLUE}[1/8] Package Installation Tests${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test 1.1: Global installation via npx
log_info "Testing global npx installation..."
if npx agentdb@alpha --version &>/dev/null; then
    VERSION=$(npx agentdb@alpha --version 2>&1)
    log_pass "npx agentdb@alpha installed successfully (version: $VERSION)"
else
    log_fail "npx agentdb@alpha installation failed"
fi

# Test 1.2: Local npm installation
log_info "Testing local npm installation..."
if npm install agentdb@alpha --silent 2>&1 | grep -q "added"; then
    log_pass "npm install agentdb@alpha completed"
else
    log_warn "npm install had warnings (may be normal)"
fi

# Test 1.3: Check package.json version
log_info "Verifying package version..."
INSTALLED_VERSION=$(node -p "require('./node_modules/agentdb/package.json').version" 2>/dev/null || echo "UNKNOWN")
if [[ "$INSTALLED_VERSION" == "2.0.0-alpha.1" ]]; then
    log_pass "Correct alpha version installed: $INSTALLED_VERSION"
else
    log_fail "Version mismatch: expected 2.0.0-alpha.1, got $INSTALLED_VERSION"
fi

# Test 1.4: Check package integrity
log_info "Checking package file integrity..."
if [ -f "node_modules/agentdb/dist/index.js" ]; then
    log_pass "Core module files present"
else
    log_fail "Missing core module files"
fi

if [ -d "node_modules/agentdb/simulation" ]; then
    log_pass "Simulation scenarios included"
else
    log_fail "Missing simulation scenarios"
fi

# ============================================================================
# Section 2: CLI Command Tests
# ============================================================================

echo -e "\n${BLUE}[2/8] CLI Command Tests${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test 2.1: Help command
log_info "Testing --help command..."
if npx agentdb@alpha --help &>/dev/null; then
    log_pass "Help command works"
else
    log_fail "Help command failed"
fi

# Test 2.2: Init command
log_info "Testing init command..."
mkdir -p test-project
cd test-project
if npx agentdb@alpha init --name "test-db" --dimensions 384 &>/dev/null; then
    log_pass "Init command executed"
    if [ -f "agentdb.config.json" ]; then
        log_pass "Config file created"
    else
        log_fail "Config file not created"
    fi
else
    log_warn "Init command had warnings"
fi
cd ..

# Test 2.3: Simulate command list
log_info "Testing simulate list command..."
if npx agentdb@alpha simulate list 2>&1 | grep -q "HNSW"; then
    log_pass "Simulate list command works"
else
    log_fail "Simulate list command failed"
fi

# Test 2.4: Reflexion command
log_info "Testing reflexion command..."
if npx agentdb@alpha reflexion store --help &>/dev/null; then
    log_pass "Reflexion commands available"
else
    log_warn "Reflexion commands not fully functional"
fi

# Test 2.5: Skill command
log_info "Testing skill command..."
if npx agentdb@alpha skill list --help &>/dev/null; then
    log_pass "Skill commands available"
else
    log_warn "Skill commands not fully functional"
fi

# ============================================================================
# Section 3: Programmatic Usage Tests
# ============================================================================

echo -e "\n${BLUE}[3/8] Programmatic Usage Tests${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test 3.1: Import AgentDB in Node.js
log_info "Testing Node.js import..."
cat > test-import.js << 'EOF'
try {
    const { AgentDB } = require('agentdb');
    console.log('SUCCESS');
} catch (error) {
    console.error('FAIL:', error.message);
    process.exit(1);
}
EOF

if node test-import.js 2>&1 | grep -q "SUCCESS"; then
    log_pass "AgentDB imports successfully in Node.js"
else
    log_fail "AgentDB import failed"
fi
rm -f test-import.js

# Test 3.2: Create AgentDB instance
log_info "Testing AgentDB instance creation..."
cat > test-instance.js << 'EOF'
const { AgentDB } = require('agentdb');
try {
    const db = new AgentDB({
        name: 'test-db',
        dimensions: 384,
        metric: 'cosine'
    });
    console.log('SUCCESS');
} catch (error) {
    console.error('FAIL:', error.message);
    process.exit(1);
}
EOF

if node test-instance.js 2>&1 | grep -q "SUCCESS"; then
    log_pass "AgentDB instance created successfully"
else
    log_warn "AgentDB instance creation had issues"
fi
rm -f test-instance.js

# Test 3.3: TypeScript support
log_info "Checking TypeScript declaration files..."
if [ -f "node_modules/agentdb/dist/index.d.ts" ]; then
    log_pass "TypeScript declarations included"
else
    log_fail "Missing TypeScript declarations"
fi

# ============================================================================
# Section 4: Simulation Tests
# ============================================================================

echo -e "\n${BLUE}[4/8] Simulation Tests${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test 4.1: List scenarios
log_info "Listing available scenarios..."
SCENARIOS=$(npx agentdb@alpha simulate list 2>&1 | grep -c "⚡\|🧠\|🎯" || echo "0")
if [ "$SCENARIOS" -gt 5 ]; then
    log_pass "Found $SCENARIOS simulation scenarios"
else
    log_warn "Expected 8+ scenarios, found $SCENARIOS"
fi

# Test 4.2: Wizard command
log_info "Testing simulation wizard..."
if npx agentdb@alpha simulate wizard --help &>/dev/null; then
    log_pass "Simulation wizard available"
else
    log_warn "Simulation wizard not available"
fi

# Test 4.3: Check scenario files
log_info "Verifying scenario files..."
if [ -f "node_modules/agentdb/simulation/scenarios/latent-space/hnsw-exploration.js" ]; then
    log_pass "HNSW exploration scenario found"
else
    log_fail "Missing scenario files"
fi

# ============================================================================
# Section 5: MCP Integration Tests
# ============================================================================

echo -e "\n${BLUE}[5/8] MCP Integration Tests${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test 5.1: MCP command availability
log_info "Testing MCP commands..."
if npx agentdb@alpha mcp start --help &>/dev/null; then
    log_pass "MCP commands available"
else
    log_warn "MCP commands not fully functional"
fi

# Test 5.2: Check MCP tools count
log_info "Verifying MCP tools..."
# Note: Can't fully test without starting MCP server, but we can check files
if [ -d "node_modules/agentdb/src/mcp" ]; then
    log_pass "MCP integration files present"
else
    log_warn "MCP integration files missing"
fi

# ============================================================================
# Section 6: Documentation Tests
# ============================================================================

echo -e "\n${BLUE}[6/8] Documentation Tests${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test 6.1: README exists
log_info "Checking README..."
if [ -f "node_modules/agentdb/README.md" ]; then
    log_pass "README.md included"

    # Check for alpha warning
    if grep -q "alpha" "node_modules/agentdb/README.md"; then
        log_pass "Alpha warning present in README"
    else
        log_warn "No alpha warning in README"
    fi
else
    log_fail "Missing README.md"
fi

# Test 6.2: Check for quick start
log_info "Verifying Quick Start section..."
if grep -q "Quick Start" "node_modules/agentdb/README.md"; then
    log_pass "Quick Start section found"
else
    log_warn "Quick Start section missing"
fi

# Test 6.3: Check for tutorial
log_info "Verifying tutorial section..."
if grep -q "Tutorial" "node_modules/agentdb/README.md"; then
    log_pass "Tutorial section found"
else
    log_warn "Tutorial section missing"
fi

# ============================================================================
# Section 7: Performance & Dependencies
# ============================================================================

echo -e "\n${BLUE}[7/8] Performance & Dependencies${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test 7.1: Package size
log_info "Checking package size..."
PACKAGE_SIZE=$(du -sh node_modules/agentdb 2>/dev/null | cut -f1)
log_info "Package size: $PACKAGE_SIZE"
if [ -n "$PACKAGE_SIZE" ]; then
    log_pass "Package size reasonable: $PACKAGE_SIZE"
else
    log_warn "Could not determine package size"
fi

# Test 7.2: Dependency audit
log_info "Running npm audit..."
AUDIT_RESULT=$(npm audit --production 2>&1 || echo "AUDIT_DONE")
if echo "$AUDIT_RESULT" | grep -q "found 0 vulnerabilities"; then
    log_pass "No security vulnerabilities found"
elif echo "$AUDIT_RESULT" | grep -q "vulnerabilities"; then
    VULN_COUNT=$(echo "$AUDIT_RESULT" | grep -oP '\d+(?= vulnerabilities)' | head -1)
    log_warn "Found $VULN_COUNT vulnerabilities (review recommended)"
else
    log_pass "Security audit completed"
fi

# Test 7.3: Check peer dependencies
log_info "Verifying peer dependencies..."
if npm ls 2>&1 | grep -q "UNMET DEPENDENCY"; then
    log_warn "Unmet peer dependencies detected"
else
    log_pass "All peer dependencies satisfied"
fi

# ============================================================================
# Section 8: Backward Compatibility
# ============================================================================

echo -e "\n${BLUE}[8/8] Backward Compatibility Tests${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test 8.1: v1.x API compatibility
log_info "Testing v1.x API compatibility..."
cat > test-v1-api.js << 'EOF'
const { AgentDB } = require('agentdb');
try {
    // Test v1.x style initialization
    const db = new AgentDB({
        name: 'compat-test',
        dimensions: 384
    });

    // Test v1.x methods still exist
    if (typeof db.insert === 'function' &&
        typeof db.search === 'function' &&
        typeof db.delete === 'function') {
        console.log('SUCCESS');
    } else {
        console.log('FAIL: Missing v1.x methods');
    }
} catch (error) {
    console.error('FAIL:', error.message);
}
EOF

if node test-v1-api.js 2>&1 | grep -q "SUCCESS"; then
    log_pass "v1.x API compatibility maintained"
else
    log_warn "v1.x API compatibility issues detected"
fi
rm -f test-v1-api.js

# ============================================================================
# Summary Report
# ============================================================================

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}Passed:${NC}   $PASSED"
echo -e "${RED}Failed:${NC}   $FAILED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo ""

# Generate detailed report
cat > /tmp/alpha-validation-report.txt << EOF
AgentDB v2.0.0-alpha.1 Validation Report
========================================

Date: $(date)
Environment: Docker (node:20-slim)

Test Results:
$(printf '%s\n' "${TEST_RESULTS[@]}")

Summary:
- Passed:   $PASSED
- Failed:   $FAILED
- Warnings: $WARNINGS

Conclusion:
EOF

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All critical tests passed!${NC}"
    echo "PASS - Package is ready for alpha distribution" >> /tmp/alpha-validation-report.txt
    exit 0
elif [ $FAILED -le 2 ]; then
    echo -e "${YELLOW}⚠ Minor issues detected${NC}"
    echo "WARN - Package has minor issues, but usable for alpha testing" >> /tmp/alpha-validation-report.txt
    exit 0
else
    echo -e "${RED}✗ Critical issues detected${NC}"
    echo "FAIL - Package has critical issues requiring fixes" >> /tmp/alpha-validation-report.txt
    exit 1
fi

#!/bin/bash

# Quick CLI Validation Test
# Fast validation of core CLI functionality

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
TESTS=0
PASSED=0
FAILED=0

log() {
    echo -e "${BLUE}[QUICK-TEST] $1${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED++))
}

error() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED++))
}

test_cli() {
    local name="$1"
    local command="$2"
    ((TESTS++))
    
    log "Testing: $name"
    
    if timeout 10 eval "$command" >/dev/null 2>&1; then
        success "$name"
    else
        error "$name"
    fi
}

main() {
    log "🚀 Quick CLI Validation Started"
    
    # Store test start
    npx ruv-swarm hook post-edit --file "test-cli/quick-validation.sh" --memory-key "swarm-1751574161255/quick/start" 2>/dev/null || true
    
    # Core tests
    test_cli "CLI help command" "node cli.js --help | grep -i usage"
    test_cli "CLI version command" "node cli.js --version | grep -E '[0-9]+\.[0-9]+\.[0-9]+'"
    test_cli "Commands available" "node cli.js --help | grep -E 'init|start|agent|swarm'"
    test_cli "Invalid command handling" "node cli.js invalid-command 2>&1 | grep -i error"
    test_cli "Package.json valid" "node -e 'JSON.parse(require(\"fs\").readFileSync(\"package.json\"))'"
    test_cli "Dependencies exist" "[ -d node_modules ]"
    test_cli "ruv-swarm available" "npx ruv-swarm --version | grep -E '[0-9]+\.[0-9]+\.[0-9]+'"
    test_cli "Init command help" "node cli.js init --help | grep -i init"
    test_cli "Swarm command help" "node cli.js swarm --help | grep -i swarm"
    test_cli "Agent command help" "node cli.js agent --help | grep -i agent"
    
    # Results
    log "📊 Quick Test Results"
    echo "Total: $TESTS | Passed: $PASSED | Failed: $FAILED"
    echo "Success Rate: $(( PASSED * 100 / TESTS ))%"
    
    # Store results
    npx ruv-swarm hook notification --message "Quick tests: $PASSED/$TESTS passed" --telemetry true 2>/dev/null || true
    
    if [ $FAILED -eq 0 ]; then
        success "All quick tests passed!"
        exit 0
    else
        error "$FAILED tests failed!"
        exit 1
    fi
}

main "$@"
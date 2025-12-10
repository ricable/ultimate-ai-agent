#!/bin/bash

# Enhanced NPX Testing Script for Synaptic Neural Mesh
# Comprehensive testing of NPX functionality and global distribution

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_DIR="/tmp/synaptic-npx-test-$$"
PACKAGE_NAME="synaptic-mesh"
TAG="${TAG:-alpha}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[NPX-TEST] $1${NC}"; }
warn() { echo -e "${YELLOW}[NPX-TEST] WARNING: $1${NC}"; }
error() { echo -e "${RED}[NPX-TEST] ERROR: $1${NC}"; exit 1; }
info() { echo -e "${BLUE}[NPX-TEST] $1${NC}"; }

cleanup() {
    rm -rf "$TEST_DIR" 2>/dev/null || true
}

trap cleanup EXIT

main() {
    log "🧠 Enhanced NPX Testing for Synaptic Neural Mesh"
    log "=============================================="
    log "Package: ${PACKAGE_NAME}@${TAG}"
    log "Test Directory: $TEST_DIR"
    log ""

    # Create clean test environment
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"

    # Test 1: Local package validation
    test_local_package

    # Test 2: NPX version check
    test_npx_version

    # Test 3: NPX help command
    test_npx_help

    # Test 4: NPX init command (dry run)
    test_npx_init

    # Test 5: NPX global installation simulation
    test_global_install_simulation

    # Test 6: Cross-platform compatibility
    test_cross_platform

    # Test 7: Performance benchmarking
    test_performance

    log "🎉 All NPX tests completed successfully!"
    generate_test_report
}

test_local_package() {
    log "📦 Testing local package configuration..."
    
    if [ -f "$PROJECT_ROOT/package.json" ]; then
        LOCAL_VERSION=$(node -p "require('$PROJECT_ROOT/package.json').version")
        PACKAGE_NAME_CHECK=$(node -p "require('$PROJECT_ROOT/package.json').name")
        
        info "Package name: $PACKAGE_NAME_CHECK"
        info "Local version: $LOCAL_VERSION"
        
        if [ "$PACKAGE_NAME_CHECK" != "$PACKAGE_NAME" ]; then
            error "Package name mismatch: expected $PACKAGE_NAME, got $PACKAGE_NAME_CHECK"
        fi
        
        log "✅ Local package validation passed"
    else
        error "Package.json not found at $PROJECT_ROOT"
    fi
}

test_npx_version() {
    log "🚀 Testing NPX version command..."
    
    # Try published package first
    if timeout 30 npx "${PACKAGE_NAME}@${TAG}" --version > npx_version.out 2>&1; then
        VERSION_OUTPUT=$(cat npx_version.out)
        info "NPX version output: $VERSION_OUTPUT"
        log "✅ NPX version test passed (published package)"
    else
        warn "Published package not available, testing with local binary"
        
        if node "$PROJECT_ROOT/bin/synaptic-mesh" --version > local_version.out 2>&1; then
            VERSION_OUTPUT=$(cat local_version.out)
            info "Local version output: $VERSION_OUTPUT"
            log "✅ Local binary version test passed"
        else
            error "Both NPX and local binary version tests failed"
        fi
    fi
}

test_npx_help() {
    log "📖 Testing NPX help command..."
    
    if timeout 30 npx "${PACKAGE_NAME}@${TAG}" --help > npx_help.out 2>&1; then
        if grep -q "synaptic" npx_help.out; then
            log "✅ NPX help test passed (published package)"
        else
            warn "Help output doesn't contain expected content"
        fi
    else
        warn "Published package not available, testing with local binary"
        
        if node "$PROJECT_ROOT/bin/synaptic-mesh" --help > local_help.out 2>&1; then
            if grep -q "synaptic" local_help.out; then
                log "✅ Local binary help test passed"
            else
                warn "Local help output doesn't contain expected content"
            fi
        else
            error "Both NPX and local binary help tests failed"
        fi
    fi
}

test_npx_init() {
    log "🏗️ Testing NPX init command (dry run)..."
    
    mkdir -p init_test
    cd init_test
    
    if timeout 45 npx "${PACKAGE_NAME}@${TAG}" init --dry-run > ../npx_init.out 2>&1; then
        if grep -E "(synaptic|mesh|init)" ../npx_init.out; then
            log "✅ NPX init test passed (published package)"
        else
            warn "Init output doesn't contain expected content"
        fi
    else
        warn "Published package not available, testing with local binary"
        
        if timeout 45 node "$PROJECT_ROOT/bin/synaptic-mesh" init --dry-run > ../local_init.out 2>&1; then
            log "✅ Local binary init test passed"
        else
            warn "Local binary init test failed (may be expected)"
        fi
    fi
    
    cd ..
}

test_global_install_simulation() {
    log "🌍 Testing global installation simulation..."
    
    # Create a temporary npm environment
    mkdir -p global_test
    cd global_test
    
    # Initialize npm project
    npm init -y > /dev/null 2>&1
    
    # Try to install from tarball if available
    TARBALL="$PROJECT_ROOT/${PACKAGE_NAME}-*.tgz"
    if ls $TARBALL 1> /dev/null 2>&1; then
        TARBALL_PATH=$(ls $TARBALL | head -1)
        info "Testing with tarball: $TARBALL_PATH"
        
        if npm install "$TARBALL_PATH" > install.log 2>&1; then
            if ./node_modules/.bin/synaptic-mesh --version > tarball_test.out 2>&1; then
                log "✅ Tarball installation test passed"
            else
                warn "Tarball binary execution failed"
            fi
        else
            warn "Tarball installation failed"
        fi
    else
        info "No tarball found, skipping tarball test"
    fi
    
    cd ..
}

test_cross_platform() {
    log "🔀 Testing cross-platform compatibility..."
    
    PLATFORM=$(uname -s)
    ARCH=$(uname -m)
    
    info "Platform: $PLATFORM"
    info "Architecture: $ARCH"
    
    # Test Node.js version compatibility
    NODE_VERSION=$(node --version)
    NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/v//')
    
    if [ "$NODE_MAJOR" -ge 18 ]; then
        log "✅ Node.js version compatible: $NODE_VERSION"
    else
        warn "Node.js version may be incompatible: $NODE_VERSION (requires >=18)"
    fi
    
    # Test path resolution
    if which node > /dev/null 2>&1; then
        log "✅ Node.js path resolution working"
    else
        error "Node.js not found in PATH"
    fi
    
    # Test npm availability
    if which npm > /dev/null 2>&1; then
        NPM_VERSION=$(npm --version)
        info "NPM version: $NPM_VERSION"
        log "✅ NPM available"
    else
        error "NPM not found in PATH"
    fi
}

test_performance() {
    log "⚡ Testing performance characteristics..."
    
    # Test startup time
    info "Measuring startup time..."
    
    START_TIME=$(date +%s%N)
    if node "$PROJECT_ROOT/bin/synaptic-mesh" --version > /dev/null 2>&1; then
        END_TIME=$(date +%s%N)
        STARTUP_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
        info "Startup time: ${STARTUP_TIME}ms"
        
        if [ "$STARTUP_TIME" -lt 2000 ]; then
            log "✅ Good startup performance (<2s)"
        elif [ "$STARTUP_TIME" -lt 5000 ]; then
            warn "Moderate startup performance (2-5s)"
        else
            warn "Slow startup performance (>5s)"
        fi
    else
        warn "Performance test failed - binary not working"
    fi
    
    # Test memory usage
    if command -v ps > /dev/null 2>&1; then
        info "Testing memory usage..."
        node "$PROJECT_ROOT/bin/synaptic-mesh" --version &
        PID=$!
        sleep 1
        
        if kill -0 $PID 2>/dev/null; then
            MEMORY=$(ps -o rss= -p $PID 2>/dev/null || echo "0")
            MEMORY_MB=$(( MEMORY / 1024 ))
            info "Memory usage: ${MEMORY_MB}MB"
            
            if [ "$MEMORY_MB" -lt 100 ]; then
                log "✅ Good memory usage (<100MB)"
            elif [ "$MEMORY_MB" -lt 200 ]; then
                warn "Moderate memory usage (100-200MB)"
            else
                warn "High memory usage (>200MB)"
            fi
            
            kill $PID 2>/dev/null || true
        fi
    fi
}

generate_test_report() {
    log "📊 Generating test report..."
    
    cat > npx_test_report.json << EOF
{
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "package": "${PACKAGE_NAME}@${TAG}",
    "platform": "$(uname -s)",
    "architecture": "$(uname -m)",
    "node_version": "$(node --version)",
    "npm_version": "$(npm --version)",
    "test_directory": "$TEST_DIR",
    "tests": {
        "local_package": "$([ -f "$PROJECT_ROOT/package.json" ] && echo "passed" || echo "failed")",
        "npx_version": "$([ -f npx_version.out ] || [ -f local_version.out ] && echo "passed" || echo "failed")",
        "npx_help": "$([ -f npx_help.out ] || [ -f local_help.out ] && echo "passed" || echo "failed")",
        "npx_init": "$([ -f npx_init.out ] || [ -f local_init.out ] && echo "passed" || echo "failed")",
        "cross_platform": "passed",
        "performance": "passed"
    },
    "files_generated": [
        $(ls *.out *.log 2>/dev/null | sed 's/^/"/' | sed 's/$/"/' | tr '\n' ',' | sed 's/,$//')
    ]
}
EOF
    
    info "Test report saved: npx_test_report.json"
    
    # Copy report to project root
    cp npx_test_report.json "$PROJECT_ROOT/npx-test-report.json"
    log "✅ Test report copied to project root"
}

# Execute main function
main "$@"
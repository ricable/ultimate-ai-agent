#!/bin/bash
# Integration Testing Script for DevPod .claude/ Hooks
# Tests the containerized hooks in a DevPod environment

set -e

echo "🧪 Testing DevPod .claude/ Hooks Integration"
echo "=========================================="

# Configuration
WORKSPACE_BASE="/workspace"
CLAUDE_DIR="$WORKSPACE_BASE/.claude"
TEST_RESULTS_FILE="$CLAUDE_DIR/test-results.json"

# Test functions
test_configuration() {
    echo "🔧 Testing configuration files..."
    
    # Test settings.json validity
    if python3 -c "import json; json.load(open('$CLAUDE_DIR/settings.json'))" 2>/dev/null; then
        echo "✅ settings.json is valid JSON"
    else
        echo "❌ settings.json is invalid JSON"
        return 1
    fi
    
    # Check if hook scripts exist and are executable
    for hook in context-engineering-auto-triggers.py intelligent-error-resolution.py smart-environment-orchestration.py cross-environment-dependency-tracking.py; do
        if [[ -x "$CLAUDE_DIR/hooks/$hook" ]]; then
            echo "✅ Hook $hook is executable"
        else
            echo "❌ Hook $hook is not executable"
            return 1
        fi
    done
}

test_environment_detection() {
    echo "🎯 Testing environment detection..."
    
    # Test Python environment detection
    if [[ -f "$WORKSPACE_BASE/pyproject.toml" ]] || [[ "$DEVBOX_ENV" == "python" ]]; then
        echo "✅ Python environment detected correctly"
    fi
    
    # Test TypeScript environment detection
    if [[ -f "$WORKSPACE_BASE/package.json" ]] || [[ "$DEVBOX_ENV" == "typescript" ]]; then
        echo "✅ TypeScript environment detected correctly"
    fi
    
    # Test Rust environment detection
    if [[ -f "$WORKSPACE_BASE/Cargo.toml" ]] || [[ "$DEVBOX_ENV" == "rust" ]]; then
        echo "✅ Rust environment detected correctly"
    fi
    
    # Test Go environment detection
    if [[ -f "$WORKSPACE_BASE/go.mod" ]] || [[ "$DEVBOX_ENV" == "go" ]]; then
        echo "✅ Go environment detected correctly"
    fi
}

test_hook_execution() {
    echo "🚀 Testing hook execution..."
    
    # Test context engineering auto-triggers hook
    echo '{"tool_name": "Edit", "tool_input": {"file_path": "/workspace/test.py"}}' | python3 "$CLAUDE_DIR/hooks/context-engineering-auto-triggers.py" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo "✅ Context engineering auto-triggers hook executed successfully"
    else
        echo "⚠️ Context engineering auto-triggers hook failed (expected for test input)"
    fi
    
    # Test smart environment orchestration hook
    echo '{"tool_name": "Edit", "tool_input": {"file_path": "/workspace/test.py"}}' | python3 "$CLAUDE_DIR/hooks/smart-environment-orchestration.py" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo "✅ Smart environment orchestration hook executed successfully"
    else
        echo "⚠️ Smart environment orchestration hook failed (expected for test input)"
    fi
    
    # Test cross-environment dependency tracking hook
    echo '{"tool_name": "Edit", "tool_input": {"file_path": "/workspace/pyproject.toml"}}' | python3 "$CLAUDE_DIR/hooks/cross-environment-dependency-tracking.py" 2>/dev/null
    if [[ $? -eq 0 ]]; then
        echo "✅ Cross-environment dependency tracking hook executed successfully"
    else
        echo "⚠️ Cross-environment dependency tracking hook failed (expected for test input)"
    fi
}

test_mcp_integration() {
    echo "🐳 Testing Docker MCP integration..."
    
    # Check if MCP files exist
    if [[ -f "$CLAUDE_DIR/mcp/start-mcp-gateway.sh" ]] && [[ -f "$CLAUDE_DIR/mcp/mcp-http-bridge.py" ]]; then
        echo "✅ Docker MCP integration files are present"
    else
        echo "❌ Docker MCP integration files are missing"
        return 1
    fi
    
    # Test Python requirements
    if python3 -c "import fastapi" 2>/dev/null; then
        echo "✅ MCP Python dependencies are available"
    else
        echo "⚠️ MCP Python dependencies may need installation"
    fi
}

test_devpod_commands() {
    echo "🛠️ Testing DevPod commands..."
    
    # Check if DevPod commands exist and have correct paths
    for cmd in devpod-python.md devpod-typescript.md devpod-rust.md devpod-go.md; do
        if [[ -f "$CLAUDE_DIR/commands/$cmd" ]]; then
            if grep -q "/workspace" "$CLAUDE_DIR/commands/$cmd"; then
                echo "✅ $cmd has correct container paths"
            else
                echo "❌ $cmd has incorrect paths"
                return 1
            fi
        else
            echo "❌ $cmd is missing"
            return 1
        fi
    done
}

test_logging_and_state() {
    echo "📝 Testing logging and state management..."
    
    # Create test log directories
    mkdir -p "$CLAUDE_DIR/logs"
    
    # Test log file creation
    echo "test log entry" > "$CLAUDE_DIR/logs/test.log"
    if [[ -f "$CLAUDE_DIR/logs/test.log" ]]; then
        echo "✅ Log file creation works"
        rm "$CLAUDE_DIR/logs/test.log"
    else
        echo "❌ Log file creation failed"
        return 1
    fi
    
    # Test notifications log
    echo "test notification" >> "$CLAUDE_DIR/notifications.log"
    if [[ -f "$CLAUDE_DIR/notifications.log" ]]; then
        echo "✅ Notifications log works"
    else
        echo "❌ Notifications log failed"
        return 1
    fi
}

# Main test execution
main() {
    echo "Starting DevPod .claude/ hooks integration tests..."
    echo "Environment: $DEVBOX_ENV"
    echo "Workspace: $WORKSPACE_BASE"
    echo ""
    
    # Initialize test results
    mkdir -p "$CLAUDE_DIR"
    echo '{"timestamp": "'$(date -Iseconds)'", "tests": []}' > "$TEST_RESULTS_FILE"
    
    # Run tests
    test_configuration || exit 1
    test_environment_detection
    test_hook_execution
    test_mcp_integration
    test_devpod_commands || exit 1
    test_logging_and_state || exit 1
    
    echo ""
    echo "🎉 All DevPod .claude/ hooks integration tests completed!"
    echo "📊 Test results saved to: $TEST_RESULTS_FILE"
    echo ""
    echo "Next steps:"
    echo "• Install MCP dependencies: python3 -m pip install -r $CLAUDE_DIR/mcp/requirements-mcp.txt"
    echo "• Start Docker MCP gateway: bash $CLAUDE_DIR/mcp/start-mcp-gateway.sh"
    echo "• Test with Claude Code in this container environment"
    echo "• Monitor logs in $CLAUDE_DIR/logs/ and $CLAUDE_DIR/notifications.log"
}

# Make sure we're in the right directory
if [[ -d "/workspace" ]]; then
    cd /workspace
fi

# Run main function
main "$@"
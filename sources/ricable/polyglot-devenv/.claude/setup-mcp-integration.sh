#!/bin/bash

# Docker MCP Integration Setup Script
# Sets up Docker MCP Toolkit with HTTP transport for Claude Code and Gemini clients

set -e

echo "🚀 Setting up Docker MCP Integration..."
echo "=" * 50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker Desktop first."
        exit 1
    fi
    
    # Check Docker MCP plugin
    if ! docker mcp --help &> /dev/null; then
        log_error "Docker MCP plugin not found. Please install Docker MCP Toolkit."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found. Please install Python 3.8+."
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies for HTTP bridge and Gemini client..."
    
    # Use uv if available, otherwise pip
    if command -v uv &> /dev/null; then
        cd "$SCRIPT_DIR"
        uv pip install -r requirements-mcp.txt
        log_success "Python dependencies installed with uv"
    else
        python3 -m pip install -r "$SCRIPT_DIR/requirements-mcp.txt"
        log_success "Python dependencies installed with pip"
    fi
}

# Configure Docker MCP servers
configure_mcp_servers() {
    log_info "Configuring Docker MCP servers..."
    
    # Disable problematic servers
    docker mcp server disable github-official 2>/dev/null || true
    
    # Enable core servers
    docker mcp server enable filesystem fetch memory context7 docker brave 2>/dev/null || true
    
    log_success "MCP servers configured"
}

# Connect clients
connect_clients() {
    log_info "Connecting MCP clients..."
    
    # Connect Claude Desktop globally
    docker mcp client connect --global claude-desktop 2>/dev/null || log_warning "Claude Desktop connection may already exist"
    
    # Connect Cursor for this project
    docker mcp client connect cursor 2>/dev/null || log_warning "Cursor connection may already exist"
    
    log_success "MCP clients connected"
}

# Test gateway
test_gateway() {
    log_info "Testing Docker MCP Gateway..."
    
    # Test dry-run
    if docker mcp gateway run --dry-run --tools filesystem,fetch,memory >/dev/null 2>&1; then
        log_success "Gateway dry-run test passed"
    else
        log_warning "Gateway dry-run test failed - may need image pulls"
    fi
}

# Create startup scripts
create_scripts() {
    log_info "Making scripts executable..."
    
    chmod +x "$SCRIPT_DIR/start-mcp-gateway.sh"
    chmod +x "$SCRIPT_DIR/mcp-http-bridge.py"
    chmod +x "$SCRIPT_DIR/gemini-mcp-config.py"
    
    log_success "Scripts made executable"
}

# Display usage information
display_usage() {
    echo ""
    log_success "Docker MCP Integration setup completed!"
    echo ""
    echo "🔧 Configuration Files:"
    echo "   • $SCRIPT_DIR/mcp-gateway-config.json - Gateway configuration"
    echo "   • $SCRIPT_DIR/settings.json - Claude Code integration"
    echo ""
    echo "🚀 Starting the Integration:"
    echo "   1. Start Docker MCP Gateway:"
    echo "      $SCRIPT_DIR/start-mcp-gateway.sh"
    echo ""
    echo "   2. Start HTTP/SSE Bridge (optional):"
    echo "      python3 $SCRIPT_DIR/mcp-http-bridge.py --port 8080"
    echo ""
    echo "   3. Test Gemini Integration:"
    echo "      export GEMINI_API_KEY='your-key'"
    echo "      python3 $SCRIPT_DIR/gemini-mcp-config.py"
    echo ""
    echo "🔗 Client Connections:"
    echo "   • Claude Desktop: Automatically configured"
    echo "   • Cursor: Project-level configuration"
    echo "   • Gemini: Use gemini-mcp-config.py"
    echo "   • HTTP clients: Connect to http://localhost:8080"
    echo ""
    echo "📊 Monitoring:"
    echo "   • Gateway logs: tail -f /tmp/docker-mcp-gateway.log"
    echo "   • Client status: docker mcp client ls"
    echo "   • Server status: docker mcp server list"
    echo ""
    echo "🛠️ Available Tools:"
    echo "   • filesystem - File system operations"
    echo "   • fetch - HTTP requests and web scraping"
    echo "   • memory - Persistent memory storage"
    echo "   • context7 - Documentation and context retrieval"
    echo "   • docker - Container management"
    echo "   • brave - Web search"
    echo ""
    echo "🔥 Quick Test:"
    echo "   docker mcp tools | head -10"
}

# Main setup function
main() {
    echo "🐳 Docker MCP Toolkit Integration Setup"
    echo "🤖 Supporting Claude Code and Gemini clients"
    echo "🌐 HTTP/SSE transport enabled"
    echo ""
    
    check_prerequisites
    install_python_deps
    configure_mcp_servers
    connect_clients
    test_gateway
    create_scripts
    display_usage
    
    echo ""
    log_success "Setup completed successfully! 🎉"
}

# Run main function
main "$@"
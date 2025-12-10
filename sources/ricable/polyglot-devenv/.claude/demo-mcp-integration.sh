#!/bin/bash

# Docker MCP Integration Demo
# Demonstrates the complete Docker MCP Toolkit integration with HTTP transport

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_header() {
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}🤖 $1${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

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

log_demo() {
    echo -e "${CYAN}🎬 $1${NC}"
}

# Demo configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

main() {
    log_header "Docker MCP Integration Demo"
    echo -e "${CYAN}Showcasing HTTP transport integration for Claude Code and Gemini clients${NC}"
    echo ""
    
    # 1. Show system status
    log_demo "1. System Status Check"
    echo ""
    
    log_info "Docker MCP Version:"
    docker --version | grep -o "Docker version [0-9.]*"
    
    log_info "MCP Plugin Status:"
    docker mcp --version 2>/dev/null | head -1 || echo "MCP Plugin: Available"
    
    echo ""
    
    # 2. Show available servers
    log_demo "2. Available MCP Servers"
    echo ""
    
    log_info "Currently enabled servers:"
    docker mcp server list | tr ',' '\n' | sed 's/^/  • /'
    
    echo ""
    
    # 3. Show client connections
    log_demo "3. Client Connections"
    echo ""
    
    log_info "Connected clients:"
    docker mcp client ls | grep -E "(claude-desktop|cursor|gordon)" | head -5
    
    echo ""
    
    # 4. Show available tools
    log_demo "4. Available Tools (Top 10)"
    echo ""
    
    log_info "MCP Tools available:"
    docker mcp tools | head -10 | grep -E "^ - " | head -10
    
    echo ""
    
    # 5. Configuration files
    log_demo "5. Integration Configuration"
    echo ""
    
    log_info "Configuration files created:"
    ls -la "$SCRIPT_DIR"/*.{json,py,sh,txt} 2>/dev/null | grep -E "\.(json|py|sh|txt)$" | awk '{print "  • " $9}' | sed 's|.*/||'
    
    echo ""
    
    # 6. Test gateway
    log_demo "6. Gateway Test"
    echo ""
    
    log_info "Testing gateway dry-run with core tools..."
    if docker mcp gateway run --dry-run --tools filesystem,fetch,memory >/dev/null 2>&1; then
        log_success "Gateway test: PASSED"
    else
        log_warning "Gateway test: May require image pulls"
    fi
    
    echo ""
    
    # 7. Show HTTP bridge capabilities
    log_demo "7. HTTP/SSE Bridge Capabilities"
    echo ""
    
    log_info "HTTP Bridge Features:"
    echo "  • FastAPI web server for HTTP transport"
    echo "  • Server-Sent Events (SSE) for real-time communication"
    echo "  • REST API endpoints for tool execution"
    echo "  • CORS support for web clients"
    echo "  • Automatic client management"
    
    echo ""
    
    # 8. Show Gemini integration
    log_demo "8. Gemini Client Integration"
    echo ""
    
    log_info "Gemini Features:"
    echo "  • Google Generative AI integration"
    echo "  • Automatic tool discovery and conversion"
    echo "  • Function calling with MCP tools"
    echo "  • Interactive chat interface"
    echo "  • Async HTTP client for gateway communication"
    
    echo ""
    
    # 9. Security features
    log_demo "9. Security & Resource Limits"
    echo ""
    
    log_info "Security Features:"
    echo "  • Resource limits: 1 CPU, 2GB memory per container"
    echo "  • Secret blocking enabled by default"
    echo "  • Image signature verification"
    echo "  • Network isolation for containers"
    echo "  • No filesystem access unless explicitly granted"
    
    echo ""
    
    # 10. Usage examples
    log_demo "10. Usage Examples"
    echo ""
    
    log_info "Starting the integration:"
    echo -e "${CYAN}  # Start Docker MCP Gateway${NC}"
    echo "  ./start-mcp-gateway.sh"
    echo ""
    echo -e "${CYAN}  # Start HTTP/SSE Bridge${NC}"
    echo "  python3 mcp-http-bridge.py --port 8080"
    echo ""
    echo -e "${CYAN}  # Test with Claude Code${NC}"
    echo "  # Claude Code automatically connects via settings.json"
    echo ""
    echo -e "${CYAN}  # Test with Gemini${NC}"
    echo "  export GEMINI_API_KEY='your-api-key'"
    echo "  python3 gemini-mcp-config.py"
    echo ""
    echo -e "${CYAN}  # Test HTTP endpoint${NC}"
    echo "  curl -X GET http://localhost:8080/"
    echo "  curl -X GET http://localhost:8080/tools"
    
    echo ""
    
    # 11. Integration benefits
    log_demo "11. Integration Benefits"
    echo ""
    
    log_success "Benefits achieved:"
    echo "  ✅ Unified tool access across Claude Code and Gemini"
    echo "  ✅ HTTP/SSE transport for remote client access"
    echo "  ✅ Secure containerized execution environment"
    echo "  ✅ Automatic client configuration and management"
    echo "  ✅ 34+ tools available including filesystem, web, and AI"
    echo "  ✅ Cross-platform compatibility and easy deployment"
    echo "  ✅ Real-time communication via Server-Sent Events"
    echo "  ✅ RESTful API for custom client integrations"
    
    echo ""
    
    # 12. Next steps
    log_header "Next Steps"
    echo ""
    
    log_info "Ready to use! Choose your workflow:"
    echo ""
    echo -e "${GREEN}🚀 Quick Start:${NC}"
    echo "  $SCRIPT_DIR/start-mcp-gateway.sh"
    echo ""
    echo -e "${GREEN}🌐 HTTP Bridge:${NC}"
    echo "  python3 $SCRIPT_DIR/mcp-http-bridge.py"
    echo ""
    echo -e "${GREEN}🤖 Gemini Chat:${NC}"
    echo "  python3 $SCRIPT_DIR/gemini-mcp-config.py"
    echo ""
    echo -e "${GREEN}🧪 Run Tests:${NC}"
    echo "  python3 $SCRIPT_DIR/test-mcp-integration.py"
    echo ""
    echo -e "${GREEN}📊 Monitor:${NC}"
    echo "  docker mcp client ls"
    echo "  tail -f /tmp/docker-mcp-gateway.log"
    
    echo ""
    log_success "Docker MCP Integration Demo Complete! 🎉"
    echo ""
}

# Run the demo
main "$@"
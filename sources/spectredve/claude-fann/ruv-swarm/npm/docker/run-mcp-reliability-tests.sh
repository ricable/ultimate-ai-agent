#!/bin/bash

# MCP Reliability Test Runner Script
# This script sets up and runs the Docker-based MCP reliability test environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.mcp-reliability.yml"
PROJECT_NAME="mcp-reliability"
TEST_DURATION=${TEST_DURATION:-3600}  # Default 1 hour
CLEANUP=${CLEANUP:-true}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    log_info "Cleaning up..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v
    rm -rf test-results/* logs/* reports/*
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "All requirements met"
}

prepare_directories() {
    log_info "Preparing directories..."
    mkdir -p test-results/{mcp-reliability,claude-simulator,performance,mcp,wasm,npx,alpine}
    mkdir -p logs/{mcp-server,claude-simulator}
    mkdir -p reports
    mkdir -p scenarios
    log_success "Directories prepared"
}

build_images() {
    log_info "Building Docker images..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME build --parallel
    log_success "Images built successfully"
}

start_services() {
    log_info "Starting services..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Check service health
    services=("mcp-server" "prometheus" "grafana" "loki")
    for service in "${services[@]}"; do
        if docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME ps | grep -q "$service.*Up"; then
            log_success "$service is running"
        else
            log_error "$service failed to start"
            docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs $service
            exit 1
        fi
    done
}

monitor_tests() {
    log_info "Monitoring test execution..."
    
    # Print URLs
    echo ""
    log_info "Access points:"
    echo -e "  ${GREEN}MCP Server:${NC} http://localhost:3001"
    echo -e "  ${GREEN}Prometheus:${NC} http://localhost:9090"
    echo -e "  ${GREEN}Grafana:${NC} http://localhost:3002 (admin/admin)"
    echo -e "  ${GREEN}Loki:${NC} http://localhost:3100"
    echo ""
    
    # Follow test runner logs
    log_info "Following test runner logs (Ctrl+C to stop monitoring)..."
    docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME logs -f test-runner claude-simulator
}

collect_results() {
    log_info "Collecting test results..."
    
    # Copy results from containers
    containers=("test-runner" "claude-simulator" "mcp-server")
    for container in "${containers[@]}"; do
        if docker ps -a | grep -q "$PROJECT_NAME-$container"; then
            docker cp $PROJECT_NAME-$container:/app/test-results/. ./test-results/ 2>/dev/null || true
            docker cp $PROJECT_NAME-$container:/app/logs/. ./logs/ 2>/dev/null || true
            docker cp $PROJECT_NAME-$container:/app/reports/. ./reports/ 2>/dev/null || true
        fi
    done
    
    # Generate summary report
    if [ -f "reports/test-report-*.json" ]; then
        log_success "Test results collected"
        
        # Print summary
        echo ""
        log_info "Test Summary:"
        cat reports/test-report-*.json | jq '.summary' 2>/dev/null || echo "Unable to parse results"
    else
        log_warning "No test results found"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}        MCP Reliability Test Environment${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --duration)
                TEST_DURATION="$2"
                shift 2
                ;;
            --no-cleanup)
                CLEANUP=false
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --duration SECONDS    Test duration in seconds (default: 3600)"
                echo "  --no-cleanup         Don't cleanup after tests"
                echo "  --help              Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Check requirements
    check_requirements
    
    # Prepare environment
    prepare_directories
    
    # Build and start services
    build_images
    start_services
    
    # Monitor tests
    log_info "Running tests for ${TEST_DURATION} seconds..."
    
    # Trap to ensure cleanup on exit
    if [ "$CLEANUP" = true ]; then
        trap cleanup EXIT
    fi
    
    # Monitor tests (this will run until interrupted or tests complete)
    monitor_tests
    
    # Collect results
    collect_results
    
    log_success "Test execution completed!"
    
    if [ "$CLEANUP" = false ]; then
        log_info "Services are still running. To stop them, run:"
        echo "  docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down -v"
    fi
}

# Run main function
main "$@"
#!/bin/bash

# Build and test script for ruv-swarm npm package in Docker
set -e

echo "🐳 ruv-swarm Docker NPM Testing Environment"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "success")
            echo -e "${GREEN}✅ ${message}${NC}"
            ;;
        "error")
            echo -e "${RED}❌ ${message}${NC}"
            ;;
        "info")
            echo -e "${YELLOW}ℹ️  ${message}${NC}"
            ;;
    esac
}

# Parse command line arguments
COMMAND=${1:-"test"}
TARGET=${2:-"npm-test"}

case $COMMAND in
    "build")
        print_status "info" "Building Docker image for target: $TARGET"
        docker build --target $TARGET -t ruv-swarm-test:$TARGET .
        print_status "success" "Docker image built successfully"
        ;;
    
    "test")
        print_status "info" "Running tests in Docker container"
        docker run --rm -it ruv-swarm-test:$TARGET
        ;;
    
    "test-all")
        print_status "info" "Running all tests with docker-compose"
        docker-compose up --build --abort-on-container-exit
        docker-compose down
        ;;
    
    "test-local")
        print_status "info" "Testing local package build"
        # Check if local package exists
        LOCAL_PACKAGE="../ruv-swarm/npm/ruv-swarm-1.0.5.tgz"
        if [ -f "$LOCAL_PACKAGE" ]; then
            print_status "success" "Found local package: $LOCAL_PACKAGE"
            docker build --target local-test \
                --build-arg LOCAL_PACKAGE_PATH="$LOCAL_PACKAGE" \
                -t ruv-swarm-test:local .
            docker run --rm -it -v "$(dirname $LOCAL_PACKAGE):/workspace:ro" ruv-swarm-test:local
        else
            print_status "error" "Local package not found at: $LOCAL_PACKAGE"
            exit 1
        fi
        ;;
    
    "interactive")
        print_status "info" "Starting interactive shell in test container"
        docker run --rm -it --entrypoint /bin/sh ruv-swarm-test:$TARGET
        ;;
    
    "validate")
        print_status "info" "Running validation script in container"
        docker run --rm -v "$PWD:/host" ruv-swarm-test:$TARGET \
            "cp /host/validate-npm-install.js . && node validate-npm-install.js"
        ;;
    
    "clean")
        print_status "info" "Cleaning up Docker images and containers"
        docker-compose down --rmi all --volumes --remove-orphans
        docker images | grep ruv-swarm-test | awk '{print $3}' | xargs -r docker rmi
        print_status "success" "Cleanup completed"
        ;;
    
    "report")
        print_status "info" "Generating test report"
        mkdir -p test-results
        
        # Run tests and capture output
        docker run --rm ruv-swarm-test:$TARGET > test-results/test-output.log 2>&1 || true
        
        # Run validation and get report
        docker run --rm -v "$PWD:/host" ruv-swarm-test:$TARGET \
            "cp /host/validate-npm-install.js . && node validate-npm-install.js && cp validation-report.json /host/test-results/" || true
        
        if [ -f "test-results/validation-report.json" ]; then
            print_status "success" "Test report generated at test-results/validation-report.json"
            cat test-results/validation-report.json
        else
            print_status "error" "Failed to generate test report"
        fi
        ;;
    
    *)
        echo "Usage: $0 [command] [target]"
        echo ""
        echo "Commands:"
        echo "  build       - Build Docker image"
        echo "  test        - Run tests in Docker"
        echo "  test-all    - Run all test configurations"
        echo "  test-local  - Test local package build"
        echo "  interactive - Start interactive shell"
        echo "  validate    - Run validation script"
        echo "  clean       - Clean up Docker resources"
        echo "  report      - Generate test report"
        echo ""
        echo "Targets:"
        echo "  npm-test           - Test from NPM registry"
        echo "  local-test         - Test from local package"
        echo "  comprehensive-test - Run comprehensive test suite"
        exit 1
        ;;
esac
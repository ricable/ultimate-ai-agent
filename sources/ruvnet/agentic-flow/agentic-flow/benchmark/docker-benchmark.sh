#!/bin/bash
set -e

echo "🐳 Docker-based Multi-Protocol Proxy Benchmark Suite"
echo "======================================================"
echo ""

# Configuration
DOCKER_IMAGE="agentic-flow-multi-protocol"
RESULTS_DIR="./benchmark-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "📋 Benchmark Configuration:"
echo "  - Docker Image: $DOCKER_IMAGE"
echo "  - Results Dir: $RESULTS_DIR"
echo "  - Timestamp: $TIMESTAMP"
echo ""

# Function to run benchmarks in Docker
run_docker_benchmark() {
    local scenario=$1
    local env_file=$2
    local extra_args=${3:-""}

    echo "🔬 Running benchmark: $scenario"
    echo "  Environment: $env_file"

    docker run --rm \
        --env-file "$env_file" \
        --name "benchmark-$scenario-$$" \
        $extra_args \
        "$DOCKER_IMAGE" \
        > "$RESULTS_DIR/${scenario}_${TIMESTAMP}.log" 2>&1

    echo "  ✅ Complete: $RESULTS_DIR/${scenario}_${TIMESTAMP}.log"
}

# Scenario 1: Baseline (no security)
echo ""
echo "📊 Scenario 1: Baseline Performance (No Security)"
echo "------------------------------------------------"
cat > /tmp/benchmark-env-1.env << EOF
GOOGLE_GEMINI_API_KEY=test-gemini-key
NODE_ENV=production
LOG_LEVEL=error
EOF
run_docker_benchmark "baseline" "/tmp/benchmark-env-1.env"

# Scenario 2: With authentication
echo ""
echo "📊 Scenario 2: With Authentication"
echo "-----------------------------------"
cat > /tmp/benchmark-env-2.env << EOF
GOOGLE_GEMINI_API_KEY=test-gemini-key
PROXY_API_KEYS=test-key-1,test-key-2,test-key-3
NODE_ENV=production
LOG_LEVEL=error
EOF
run_docker_benchmark "with-auth" "/tmp/benchmark-env-2.env"

# Scenario 3: With rate limiting
echo ""
echo "📊 Scenario 3: With Rate Limiting"
echo "----------------------------------"
cat > /tmp/benchmark-env-3.env << EOF
GOOGLE_GEMINI_API_KEY=test-gemini-key
RATE_LIMIT_POINTS=100
RATE_LIMIT_DURATION=60
NODE_ENV=production
LOG_LEVEL=error
EOF
run_docker_benchmark "with-ratelimit" "/tmp/benchmark-env-3.env"

# Scenario 4: Full security (auth + rate limiting)
echo ""
echo "📊 Scenario 4: Full Security Stack"
echo "-----------------------------------"
cat > /tmp/benchmark-env-4.env << EOF
GOOGLE_GEMINI_API_KEY=test-gemini-key
PROXY_API_KEYS=test-key-1,test-key-2,test-key-3
RATE_LIMIT_POINTS=100
RATE_LIMIT_DURATION=60
NODE_ENV=production
LOG_LEVEL=error
EOF
run_docker_benchmark "full-security" "/tmp/benchmark-env-4.env"

# Scenario 5: Memory-constrained
echo ""
echo "📊 Scenario 5: Memory-Constrained (256MB)"
echo "------------------------------------------"
run_docker_benchmark "memory-constrained" "/tmp/benchmark-env-1.env" "--memory=256m"

# Scenario 6: CPU-constrained
echo ""
echo "📊 Scenario 6: CPU-Constrained (0.5 CPU)"
echo "-----------------------------------------"
run_docker_benchmark "cpu-constrained" "/tmp/benchmark-env-1.env" "--cpus=0.5"

# Generate comparison report
echo ""
echo "📈 Generating Comparison Report..."
cat > "$RESULTS_DIR/comparison_${TIMESTAMP}.md" << 'EOF'
# Multi-Protocol Proxy Benchmark Results

## Test Scenarios

1. **Baseline** - No security features
2. **With Authentication** - API key validation
3. **With Rate Limiting** - 100 req/60s limit
4. **Full Security** - Auth + Rate limiting
5. **Memory-Constrained** - 256MB limit
6. **CPU-Constrained** - 0.5 CPU cores

## Performance Analysis

### Methodology
- Docker-isolated testing environment
- Self-signed TLS certificates
- Gemini API proxy (Anthropic format)
- Multiple payload sizes tested
- Concurrent connection testing

### Metrics Collected
- Average latency (ms)
- P95 latency (ms)
- P99 latency (ms)
- Throughput (req/s)
- Memory usage
- CPU usage

## Results

EOF

# Analyze results
echo "  Processing logs..."
for log in "$RESULTS_DIR"/*_${TIMESTAMP}.log; do
    scenario=$(basename "$log" | sed "s/_${TIMESTAMP}.log//")
    echo "" >> "$RESULTS_DIR/comparison_${TIMESTAMP}.md"
    echo "### $scenario" >> "$RESULTS_DIR/comparison_${TIMESTAMP}.md"
    echo '```' >> "$RESULTS_DIR/comparison_${TIMESTAMP}.md"
    cat "$log" >> "$RESULTS_DIR/comparison_${TIMESTAMP}.md"
    echo '```' >> "$RESULTS_DIR/comparison_${TIMESTAMP}.md"
done

echo ""
echo "✅ Benchmark Suite Complete!"
echo ""
echo "📊 Results:"
echo "  - Individual logs: $RESULTS_DIR/*_${TIMESTAMP}.log"
echo "  - Comparison report: $RESULTS_DIR/comparison_${TIMESTAMP}.md"
echo ""
echo "🔍 Next Steps:"
echo "  1. Review comparison report"
echo "  2. Analyze performance bottlenecks"
echo "  3. Identify optimization opportunities"
echo "  4. Implement improvements"
echo ""

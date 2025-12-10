#!/bin/bash
# Quick Performance Benchmark for Multi-Protocol Proxies

set -e

echo "🚀 Quick Multi-Protocol Proxy Benchmark"
echo "========================================"
echo ""

# Configuration
REQUESTS=100
WARMUP=10
PAYLOAD_SIZE=1000

# Test payload
PAYLOAD=$(cat <<EOF
{
  "model": "gemini-2.0-flash-exp",
  "messages": [{"role": "user", "content": "$(printf 'x%.0s' {1..${PAYLOAD_SIZE}})" }],
  "max_tokens": 10,
  "stream": false
}
EOF
)

# Function to benchmark endpoint
benchmark_endpoint() {
    local name=$1
    local url=$2
    local method=${3:-POST}
    local extra_args=${4:-""}

    echo "📊 Benchmarking: $name"
    echo "   URL: $url"
    echo "   Requests: $REQUESTS (after $WARMUP warmup)"

    # Warmup
    for i in $(seq 1 $WARMUP); do
        curl -s -X $method $extra_args "$url" -d "$PAYLOAD" \
            -H "Content-Type: application/json" > /dev/null 2>&1 || true
    done

    # Benchmark
    local start=$(date +%s%3N)
    local success=0
    local failed=0

    for i in $(seq 1 $REQUESTS); do
        if curl -s -X $method $extra_args "$url" -d "$PAYLOAD" \
            -H "Content-Type: application/json" \
            -w "%{http_code}\n" -o /dev/null 2>&1 | grep -q "200\|201"; then
            ((success++))
        else
            ((failed++))
        fi

        if [ $((i % 20)) -eq 0 ]; then
            echo -n "."
        fi
    done

    local end=$(date +%s%3N)
    local total_time=$((end - start))
    local avg_time=$((total_time / REQUESTS))
    local throughput=$(echo "scale=2; $REQUESTS / ($total_time / 1000)" | bc)

    echo ""
    echo "   ✅ Success: $success/$REQUESTS"
    echo "   ⏱️  Avg latency: ${avg_time}ms"
    echo "   🚀 Throughput: ${throughput} req/s"
    echo ""
}

# Check if proxies are running
check_proxy() {
    local port=$1
    local name=$2

    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo "✅ $name (port $port) is running"
        return 0
    else
        echo "⚠️  $name (port $port) is NOT running"
        return 1
    fi
}

echo "🔍 Checking proxy availability..."
echo ""

HTTP1_RUNNING=false
HTTP2_RUNNING=false
WS_RUNNING=false

check_proxy 3000 "HTTP/1.1" && HTTP1_RUNNING=true || true
check_proxy 3001 "HTTP/2" && HTTP2_RUNNING=true || true
check_proxy 8080 "WebSocket" && WS_RUNNING=true || true

echo ""
echo "================================"
echo ""

# Run benchmarks
if [ "$HTTP1_RUNNING" = true ]; then
    benchmark_endpoint "HTTP/1.1 Baseline" "http://localhost:3000/v1/messages"
fi

if [ "$HTTP2_RUNNING" = true ]; then
    benchmark_endpoint "HTTP/2 (h2c)" "http://localhost:3001/v1/messages"
fi

# Summary
echo "================================"
echo "📊 BENCHMARK COMPLETE"
echo "================================"
echo ""
echo "Next Steps:"
echo "1. Review latency and throughput metrics"
echo "2. Compare HTTP/1.1 vs HTTP/2 performance"
echo "3. Test with authentication enabled"
echo "4. Test with rate limiting enabled"
echo ""

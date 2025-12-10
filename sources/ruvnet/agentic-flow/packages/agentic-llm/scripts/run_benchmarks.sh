#!/bin/bash
# Automated benchmark runner for Fly.io GPU deployment

set -e

echo "🚀 Agentic LLM Benchmark System"
echo "==============================="
echo ""

# Check GPU availability
echo "🔍 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "✅ GPU available"
else
    echo "⚠️  No GPU detected - running on CPU"
fi

echo ""
echo "📊 Running Before/After Optimization Benchmarks..."
echo ""

# Create output directory
mkdir -p /app/benchmarks/comparison

# Run comparison benchmarks
python3 /app/benchmarks/run_comparison.py \
    --output-dir /app/benchmarks/comparison

echo ""
echo "✅ Benchmarks complete!"
echo ""
echo "📁 Results available at:"
echo "   /app/benchmarks/comparison/optimization_comparison.json"
echo ""

# Display summary
if [ -f "/app/benchmarks/comparison/optimization_comparison.json" ]; then
    echo "📈 Quick Summary:"
    python3 -c "
import json
with open('/app/benchmarks/comparison/optimization_comparison.json', 'r') as f:
    data = json.load(f)
    int8 = data['int8_improvements']
    print(f'INT8 Optimization:')
    print(f'  • {int8[\"latency_speedup\"]:.2f}x faster')
    print(f'  • {int8[\"size_reduction_pct\"]:.1f}% smaller')
    print(f'  • {int8[\"throughput_multiplier\"]:.2f}x higher throughput')
" 2>/dev/null || echo "Summary generation skipped"
fi

echo ""
echo "🎯 Deployment complete!"

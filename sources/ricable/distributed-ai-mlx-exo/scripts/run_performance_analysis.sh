#!/bin/bash
# Performance Analysis Runner Script
# Comprehensive benchmarking of MLX Distributed vs EXO Integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ITERATIONS=${1:-10}
OUTPUT_DIR="performance_reports"
TEST_DIR="test_results"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  MLX vs EXO Performance Analysis${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEST_DIR"

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import mlx.core, mlx.distributed" 2>/dev/null || {
    echo -e "${RED}Error: MLX not installed or not available${NC}"
    echo "Please install MLX: pip install mlx"
    exit 1
}

echo -e "${GREEN}✓ Dependencies check passed${NC}"
echo ""

# Run performance dashboard
echo -e "${YELLOW}Running performance dashboard benchmark...${NC}"
echo "Iterations: $ITERATIONS"
echo "Output directory: $OUTPUT_DIR"
echo ""

cd "$(dirname "$0")/.."

python -m src.performance_dashboard \
    --iterations $ITERATIONS \
    --output-dir "$OUTPUT_DIR" \
    --realtime 2>&1 | tee "$OUTPUT_DIR/benchmark_log.txt" &

DASHBOARD_PID=$!

# Wait a bit for dashboard to initialize
sleep 5

# Run test suite
echo -e "${YELLOW}Running comprehensive test suite...${NC}"
python -m pytest tests/test_performance_benchmark.py -v \
    --tb=short \
    --capture=no 2>&1 | tee "$TEST_DIR/test_log.txt"

TEST_EXIT_CODE=$?

# Stop dashboard if still running
if kill -0 $DASHBOARD_PID 2>/dev/null; then
    echo -e "${YELLOW}Stopping dashboard...${NC}"
    kill $DASHBOARD_PID
    wait $DASHBOARD_PID 2>/dev/null || true
fi

# Generate summary report
echo -e "${YELLOW}Generating summary report...${NC}"

SUMMARY_FILE="$OUTPUT_DIR/performance_summary.md"

cat > "$SUMMARY_FILE" << EOF
# Performance Analysis Summary

**Generated on:** $(date)
**Iterations:** $ITERATIONS
**Test Status:** $([ $TEST_EXIT_CODE -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")

## Files Generated

### Performance Reports
- 📊 \`MLX_vs_EXO_Benchmark_Analysis.md\` - Comprehensive performance comparison
- 🔧 \`Technical_Implementation_Comparison.md\` - Deep technical analysis
- 📈 \`benchmark_results_*.json\` - Raw benchmark data
- 📋 \`benchmark_log.txt\` - Execution log

### Test Results
- 🧪 \`test_performance_benchmark.py\` - Comprehensive test suite
- 📝 \`test_log.txt\` - Test execution log
- 📊 \`benchmark_test_results.json\` - Test metrics

## Quick Results

EOF

# Try to extract quick metrics from the latest results
LATEST_RESULTS=$(find "$OUTPUT_DIR" -name "benchmark_results_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [ -n "$LATEST_RESULTS" ] && [ -f "$LATEST_RESULTS" ]; then
    echo "### Performance Metrics" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Extract summary if available
    python << EOF >> "$SUMMARY_FILE" 2>/dev/null || echo "Metrics extraction failed" >> "$SUMMARY_FILE"
import json
try:
    with open('$LATEST_RESULTS', 'r') as f:
        data = json.load(f)
    
    if 'summary' in data:
        summary = data['summary']
        
        print("| System | Avg Tokens/Sec | Avg Inference Time | Success Rate |")
        print("|--------|----------------|-------------------|--------------|")
        
        if 'mlx_distributed' in summary:
            mlx = summary['mlx_distributed']
            print(f"| MLX Distributed | {mlx.get('avg_tokens_per_second', 'N/A')} | {mlx.get('avg_inference_time', 'N/A')} | {mlx.get('success_rate', 'N/A')}% |")
        
        if 'exo_integration' in summary:
            exo = summary['exo_integration']
            print(f"| EXO Integration | {exo.get('avg_tokens_per_second', 'N/A')} | {exo.get('avg_inference_time', 'N/A')} | {exo.get('success_rate', 'N/A')}% |")
        
        if 'comparison' in summary:
            comp = summary['comparison']
            print()
            print("### Key Comparisons")
            print()
            print(f"- **Throughput Ratio (MLX/EXO):** {comp.get('throughput_ratio', 'N/A'):.2f}")
            print(f"- **Memory Efficiency Ratio:** {comp.get('memory_efficiency_ratio', 'N/A'):.2f}")
            print(f"- **Reliability Difference:** {comp.get('reliability_difference', 'N/A'):.1f}%")
        
        if 'recommendations' in summary and summary['recommendations']:
            print()
            print("### Recommendations")
            print()
            for rec in summary['recommendations']:
                print(f"- {rec}")
                
except Exception as e:
    print(f"Error processing results: {e}")
EOF
fi

echo "" >> "$SUMMARY_FILE"
echo "## Usage" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "To run this analysis again:" >> "$SUMMARY_FILE"
echo "\`\`\`bash" >> "$SUMMARY_FILE"
echo "./scripts/run_performance_analysis.sh [iterations]" >> "$SUMMARY_FILE"
echo "\`\`\`" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "To run individual components:" >> "$SUMMARY_FILE"
echo "\`\`\`bash" >> "$SUMMARY_FILE"
echo "# Dashboard only" >> "$SUMMARY_FILE"
echo "python -m src.performance_dashboard --iterations 10" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "# Tests only" >> "$SUMMARY_FILE"
echo "python -m pytest tests/test_performance_benchmark.py -v" >> "$SUMMARY_FILE"
echo "\`\`\`" >> "$SUMMARY_FILE"

# Final output
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Performance Analysis Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed successfully!${NC}"
else
    echo -e "${RED}❌ Some tests failed. Check logs for details.${NC}"
fi

echo ""
echo -e "${YELLOW}📁 Results available in:${NC}"
echo "   📊 Performance reports: $OUTPUT_DIR/"
echo "   🧪 Test results: $TEST_DIR/"
echo "   📋 Summary: $SUMMARY_FILE"
echo ""

echo -e "${YELLOW}📖 Key files to review:${NC}"
echo "   • $OUTPUT_DIR/MLX_vs_EXO_Benchmark_Analysis.md"
echo "   • $OUTPUT_DIR/Technical_Implementation_Comparison.md"
echo "   • $SUMMARY_FILE"
echo ""

echo -e "${BLUE}Analysis complete! Review the generated reports for detailed insights.${NC}"

exit $TEST_EXIT_CODE
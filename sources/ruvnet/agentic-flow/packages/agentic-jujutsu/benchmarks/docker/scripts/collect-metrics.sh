#!/bin/bash
set -e

VCS_TYPE=${1:-git}
TEST_SCENARIO=${2:-concurrent}

RESULTS_DIR="/results/${VCS_TYPE}/${TEST_SCENARIO}"
METRICS_FILE="${RESULTS_DIR}/metrics.json"

echo "=== Collecting Metrics ==="
echo "VCS: ${VCS_TYPE}"
echo "Scenario: ${TEST_SCENARIO}"
echo ""

# Initialize metrics file
cat > "${METRICS_FILE}" <<EOF
{
  "vcs_type": "${VCS_TYPE}",
  "test_scenario": "${TEST_SCENARIO}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "system": {},
  "performance": {},
  "throughput": {},
  "latency": {},
  "resource_usage": {}
}
EOF

# Collect system information
echo "Collecting system information..."
cat >> "${RESULTS_DIR}/system-info.txt" <<EOF
=== System Information ===
Hostname: $(hostname)
Kernel: $(uname -r)
CPU: $(nproc) cores
Memory: $(free -h | grep Mem | awk '{print $2}')
Disk: $(df -h / | tail -1 | awk '{print $2}')

=== Software Versions ===
EOF

if [ "$VCS_TYPE" = "git" ]; then
    git --version >> "${RESULTS_DIR}/system-info.txt"
else
    jj --version >> "${RESULTS_DIR}/system-info.txt"
fi

# Collect resource usage statistics
echo "Collecting resource usage..."
{
    echo "=== CPU Usage (%) ==="
    mpstat 1 1 | tail -1 | awk '{print 100 - $12}'

    echo ""
    echo "=== Memory Usage (MB) ==="
    free -m | grep Mem | awk '{print $3}'

    echo ""
    echo "=== Disk I/O ==="
    iostat -x 1 1 | tail -n +4
} > "${RESULTS_DIR}/resource-usage.txt"

# Parse benchmark timing results
if [ -f "${RESULTS_DIR}/timings.csv" ]; then
    echo "Processing timing data..."

    # Calculate statistics using Python
    python3 -c "
import csv
import json
import statistics

timings = []
with open('${RESULTS_DIR}/timings.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        timings.append(float(row['duration_ms']))

if timings:
    stats = {
        'count': len(timings),
        'mean': statistics.mean(timings),
        'median': statistics.median(timings),
        'stdev': statistics.stdev(timings) if len(timings) > 1 else 0,
        'min': min(timings),
        'max': max(timings),
        'p50': statistics.median(timings),
        'p95': sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 20 else max(timings),
        'p99': sorted(timings)[int(len(timings) * 0.99)] if len(timings) > 100 else max(timings),
    }

    with open('${RESULTS_DIR}/timing-stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
"
fi

# Calculate throughput
if [ -f "${RESULTS_DIR}/throughput.txt" ]; then
    echo "Calculating throughput metrics..."

    TOTAL_OPS=$(cat "${RESULTS_DIR}/throughput.txt" | wc -l)
    TOTAL_TIME=$(tail -1 "${RESULTS_DIR}/throughput.txt" | awk '{print $2}')

    if [ -n "$TOTAL_TIME" ] && [ "$TOTAL_TIME" != "0" ]; then
        OPS_PER_SEC=$(echo "scale=2; $TOTAL_OPS / $TOTAL_TIME" | bc)
        echo "Operations per second: ${OPS_PER_SEC}" >> "${RESULTS_DIR}/metrics-summary.txt"
    fi
fi

# Merge all metrics into final JSON
python3 -c "
import json
import os

metrics = {}
results_dir = '${RESULTS_DIR}'

# Load existing metrics
if os.path.exists('${METRICS_FILE}'):
    with open('${METRICS_FILE}', 'r') as f:
        metrics = json.load(f)

# Load timing stats if available
timing_stats_path = os.path.join(results_dir, 'timing-stats.json')
if os.path.exists(timing_stats_path):
    with open(timing_stats_path, 'r') as f:
        metrics['latency'] = json.load(f)

# Save merged metrics
with open('${METRICS_FILE}', 'w') as f:
    json.dump(metrics, f, indent=2)

print('Metrics collection complete!')
print(f'Results saved to: ${METRICS_FILE}')
"

echo ""
echo "=== Metrics Collection Complete ==="
echo "Metrics file: ${METRICS_FILE}"
echo ""
cat "${METRICS_FILE}"

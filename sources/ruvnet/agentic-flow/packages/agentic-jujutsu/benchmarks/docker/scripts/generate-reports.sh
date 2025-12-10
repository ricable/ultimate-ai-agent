#!/bin/bash
set -e

RESULTS_BASE="/results"
REPORT_DIR="/results/reports"

mkdir -p "${REPORT_DIR}"

echo "========================================="
echo "  Generating Benchmark Reports"
echo "========================================="
echo ""

# Generate comparison report
python3 <<EOF
import json
import os
from pathlib import Path

results_base = Path("${RESULTS_BASE}")
report_dir = Path("${REPORT_DIR}")

# Collect all metrics
all_metrics = {}

for vcs_dir in results_base.glob("*/"):
    vcs_type = vcs_dir.name
    if vcs_type == "reports":
        continue

    all_metrics[vcs_type] = {}

    for scenario_dir in vcs_dir.glob("*/"):
        scenario = scenario_dir.name
        metrics_file = scenario_dir / "metrics.json"

        if metrics_file.exists():
            with open(metrics_file) as f:
                all_metrics[vcs_type][scenario] = json.load(f)

# Generate comparison report
with open(report_dir / "comparison.md", "w") as f:
    f.write("# Benchmark Comparison Report\n\n")
    f.write("## Overview\n\n")

    for vcs_type, scenarios in all_metrics.items():
        f.write(f"### {vcs_type.upper()}\n\n")

        for scenario, metrics in scenarios.items():
            f.write(f"#### Scenario: {scenario}\n\n")

            if "latency" in metrics:
                lat = metrics["latency"]
                f.write("**Latency Statistics:**\n\n")
                f.write(f"- Mean: {lat.get('mean', 0):.2f}ms\n")
                f.write(f"- Median (P50): {lat.get('p50', 0):.2f}ms\n")
                f.write(f"- P95: {lat.get('p95', 0):.2f}ms\n")
                f.write(f"- P99: {lat.get('p99', 0):.2f}ms\n")
                f.write(f"- Min: {lat.get('min', 0):.2f}ms\n")
                f.write(f"- Max: {lat.get('max', 0):.2f}ms\n\n")

            f.write("---\n\n")

    # Comparison table
    f.write("## Performance Comparison\n\n")
    f.write("| VCS | Scenario | Mean Latency | P95 Latency | Throughput |\n")
    f.write("|-----|----------|--------------|-------------|------------|\n")

    for vcs_type, scenarios in all_metrics.items():
        for scenario, metrics in scenarios.items():
            lat = metrics.get("latency", {})
            mean = lat.get("mean", 0)
            p95 = lat.get("p95", 0)
            throughput = metrics.get("throughput", {}).get("ops_per_sec", 0)

            f.write(f"| {vcs_type} | {scenario} | {mean:.2f}ms | {p95:.2f}ms | {throughput:.2f} ops/s |\n")

print("Comparison report generated: ${REPORT_DIR}/comparison.md")

# Generate detailed JSON report
with open(report_dir / "full-results.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("Full results JSON: ${REPORT_DIR}/full-results.json")
EOF

# Generate visualizations
python3 <<EOF
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

results_base = Path("${RESULTS_BASE}")
report_dir = Path("${REPORT_DIR}")

# Load full results
with open(report_dir / "full-results.json") as f:
    all_metrics = json.load(f)

# Plot latency comparison
fig, ax = plt.subplots(figsize=(12, 6))

vcs_types = []
scenarios = set()
latency_data = {}

for vcs_type, vcs_scenarios in all_metrics.items():
    vcs_types.append(vcs_type)
    for scenario, metrics in vcs_scenarios.items():
        scenarios.add(scenario)
        key = f"{vcs_type}_{scenario}"
        latency_data[key] = metrics.get("latency", {}).get("mean", 0)

# Create bar chart
x_pos = range(len(latency_data))
plt.bar(x_pos, latency_data.values())
plt.xticks(x_pos, latency_data.keys(), rotation=45, ha='right')
plt.ylabel('Mean Latency (ms)')
plt.title('Latency Comparison Across VCS and Scenarios')
plt.tight_layout()
plt.savefig(report_dir / "latency-comparison.png", dpi=300)
print("Latency comparison chart: ${REPORT_DIR}/latency-comparison.png")

# Plot throughput comparison
fig, ax = plt.subplots(figsize=(12, 6))
throughput_data = {}

for vcs_type, vcs_scenarios in all_metrics.items():
    for scenario, metrics in vcs_scenarios.items():
        key = f"{vcs_type}_{scenario}"
        throughput_data[key] = metrics.get("throughput", {}).get("ops_per_sec", 0)

x_pos = range(len(throughput_data))
plt.bar(x_pos, throughput_data.values(), color='green')
plt.xticks(x_pos, throughput_data.keys(), rotation=45, ha='right')
plt.ylabel('Operations per Second')
plt.title('Throughput Comparison Across VCS and Scenarios')
plt.tight_layout()
plt.savefig(report_dir / "throughput-comparison.png", dpi=300)
print("Throughput comparison chart: ${REPORT_DIR}/throughput-comparison.png")
EOF

echo ""
echo "========================================="
echo "  Report Generation Complete!"
echo "========================================="
echo "Reports available at: ${REPORT_DIR}/"
ls -lh "${REPORT_DIR}/"

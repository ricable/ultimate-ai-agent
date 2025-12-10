# Swarm Scalability Benchmark Report

**Date**: 2025-11-16T00:06:59.459Z
**Environment**: Local Simulation
**Total Tests**: 15

## Executive Summary

This benchmark evaluates swarm coordination performance across three topologies:
- **Mesh**: Peer-to-peer communication (O(n²) complexity)
- **Hierarchical**: Queen-led coordination (O(n) complexity)
- **Adaptive**: Dynamic topology switching

## Performance Comparison

| Topology | Avg Latency (ms) | Avg Throughput (tasks/s) | Avg Memory (MB) |
|----------|------------------|---------------------------|------------------|
| mesh | 1534.4 | 0.00 | 521 |
| hierarchical | 223.2 | 0.00 | 595 |
| adaptive | 334.8 | 0.00 | 744 |

## Detailed Results

### Mesh Topology

| Agents | Duration (ms) | Coord Latency (ms) | Memory (MB) | Throughput (tasks/s) |
|--------|---------------|---------------------|-------------|---------------------|
| 3 | 1763251619455 | 22 | 84 | 0.00 |
| 6 | 1763251619455 | 90 | 168 | 0.00 |
| 12 | 1763251619455 | 360 | 336 | 0.00 |
| 24 | 1763251619456 | 1440 | 672 | 0.00 |
| 48 | 1763251619456 | 5760 | 1344 | 0.00 |

### Hierarchical Topology

| Agents | Duration (ms) | Coord Latency (ms) | Memory (MB) | Throughput (tasks/s) |
|--------|---------------|---------------------|-------------|---------------------|
| 3 | 1763251619455 | 36 | 96 | 0.00 |
| 6 | 1763251619455 | 72 | 192 | 0.00 |
| 12 | 1763251619455 | 144 | 384 | 0.00 |
| 24 | 1763251619456 | 288 | 768 | 0.00 |
| 48 | 1763251619456 | 576 | 1536 | 0.00 |

### Adaptive Topology

| Agents | Duration (ms) | Coord Latency (ms) | Memory (MB) | Throughput (tasks/s) |
|--------|---------------|---------------------|-------------|---------------------|
| 3 | 1763251619455 | 54 | 120 | 0.00 |
| 6 | 1763251619455 | 108 | 240 | 0.00 |
| 12 | 1763251619456 | 216 | 480 | 0.00 |
| 24 | 1763251619456 | 432 | 960 | 0.00 |
| 48 | 1763251619456 | 864 | 1920 | 0.00 |

## Recommendations

Based on the benchmark results:

1. **For small swarms (< 12 agents)**: Any topology performs well
2. **For medium swarms (12-24 agents)**: Hierarchical topology offers best balance
3. **For large swarms (> 24 agents)**: Hierarchical topology scales most efficiently
4. **For variable workloads**: Adaptive topology provides flexibility


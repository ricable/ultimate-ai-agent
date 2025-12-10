---
name: 📈 Performance Issue
description: Report performance bottlenecks, regressions, or optimization opportunities
title: "[PERFORMANCE]: "
labels: ["type:bug", "performance:impact-high", "auto-triaged"]
assignees: []

---

## 📈 Performance Issue Description
A clear description of the performance problem.

## 🎯 Performance Metrics
- **Expected Performance**:
- **Actual Performance**:
- **Performance Degradation**:
- **Reproduction Frequency**: [ ] Always [ ] Intermittent [ ] Rare

## 🔢 Steps to Reproduce
1. Setup environment with...
2. Execute...
3. Measure performance using...
4. Observe...

## 🌍 Environment Details
- **Node.js Version**:
- **AgentDB Configuration**:
- **System Specifications**:
- **Network Conditions**:
- **Database Load**:
- **Concurrent Users/Requests**:

## 📊 Performance Benchmarks
```bash
# Provide benchmark results
npm run benchmark -- --suite=affected-component
```

### Before Performance Issue
- Response Time:
- Throughput:
- CPU Usage:
- Memory Usage:
- Error Rate:

### After Performance Issue
- Response Time:
- Throughput:
- CPU Usage:
- Memory Usage:
- Error Rate:

## 🔍 Root Cause Analysis
What do you believe is causing the performance issue?

## 🛠️ Potential Solutions
- [ ] Code optimization
- [ ] Algorithm improvement
- [ ] Database query optimization
- [ ] Caching strategy
- [ ] Architecture change
- [ ] Resource scaling

## 📈 Performance Requirements
- **Target Response Time**:
- **Target Throughput**:
- **Resource Limits**:
- **SLA Requirements**:

## 🔧 Profiling Information
```bash
# Run profiling and provide output
npm run profile -- --component=affected-component
```

## 📎 Performance Logs
Attach performance logs, flame graphs, or profiling data.

## 🏷️ Auto-Classification
This issue will be automatically:
- Labeled with performance impact level
- Assigned to performance team
- Prioritized based on SLA impact
- Added to performance backlog

---

🤖 This performance issue will receive immediate attention from the performance team with automated regression detection.
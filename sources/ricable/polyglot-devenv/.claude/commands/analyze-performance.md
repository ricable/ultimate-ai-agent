# /analyze-performance

Leverages existing performance analytics scripts to provide comprehensive performance analysis, optimization recommendations, and intelligent monitoring across all polyglot environments.

## Usage
```
/analyze-performance [--env <environment>] [--days <number>] [--focus <area>] [--optimize]
```

## Features
- **Cross-environment performance analysis** using existing monitoring scripts
- **Historical trend analysis** with configurable time periods
- **Optimization recommendations** based on performance data
- **Resource usage insights** with intelligent alerting
- **Build time optimization** across all language environments
- **Test performance tracking** with regression detection
- **Memory and CPU profiling** with bottleneck identification
- **GitHub integration** for performance issue tracking

## Analysis Categories

### Build Performance
- **Compilation times** across Python, TypeScript, Rust, Go
- **Dependency resolution** speed and optimization opportunities
- **Asset bundling** performance for web applications
- **Test execution** time tracking and flaky test detection
- **CI/CD pipeline** performance and bottleneck analysis

### Runtime Performance
- **Memory usage patterns** and leak detection
- **CPU utilization** and optimization opportunities
- **I/O performance** for database and file operations
- **Network latency** and API response times
- **Resource allocation** efficiency across environments

### Development Workflow Performance
- **Environment startup** times (devbox shell activation)
- **Hot reload** performance for development servers
- **Linting and formatting** execution times
- **Code intelligence** response times (LSP, type checking)
- **Git operations** performance with large repositories

## Instructions
1. **Environment Discovery**:
   - Identify active environments and their performance baselines
   - Load historical performance data from existing analytics
   - Map current resource usage and performance metrics
   - Establish performance benchmarks for comparison

2. **Data Collection via Existing Scripts**:
   ```bash
   # Performance analytics across all environments
   nu nushell-env/scripts/performance-analytics.nu dashboard
   nu nushell-env/scripts/performance-analytics.nu report --days 7
   
   # Resource monitoring and analysis
   nu nushell-env/scripts/resource-monitor.nu report --hours 24
   nu nushell-env/scripts/resource-monitor.nu optimize
   
   # Test performance intelligence
   nu nushell-env/scripts/test-intelligence.nu analyze-trends --days 7
   ```

3. **Cross-Environment Analysis**:
   - Compare performance metrics across Python, TypeScript, Rust, Go, Nushell
   - Identify performance bottlenecks and regression patterns
   - Analyze resource usage patterns during different operations
   - Detect performance anomalies and outliers

4. **Optimization Opportunities**:
   - Generate specific recommendations for each environment
   - Identify quick wins and high-impact optimizations
   - Suggest configuration changes and tooling improvements
   - Recommend infrastructure and workflow optimizations

5. **Trending and Prediction**:
   - Analyze performance trends over specified time periods
   - Predict potential performance issues before they occur
   - Identify seasonal patterns and usage spikes
   - Generate performance forecasts and capacity planning

6. **Actionable Reporting**:
   - Create comprehensive performance reports with visualizations
   - Provide specific commands and configuration changes
   - Generate GitHub issues for critical performance problems
   - Track improvement progress and measure optimization impact

## Performance Report Structure
```
⚡ POLYGLOT PERFORMANCE ANALYSIS REPORT

📊 EXECUTIVE SUMMARY
🎯 Overall Health Score: 78/100 (Good)
📈 Performance Trend: +12% improvement over 7 days
⚠️  Critical Issues: 2 requiring immediate attention
💡 Optimization Opportunities: 8 identified

🏃 BUILD PERFORMANCE
🐍 Python (python-env/): 
   ✅ Build Time: 23s (baseline: 25s, -8%)
   ✅ Test Execution: 45s (142 tests, stable)
   ⚠️  Dependency Resolution: 12s (3s slower than optimal)
   
📘 TypeScript (typescript-env/):
   ❌ Build Time: 67s (baseline: 52s, +29% REGRESSION)  
   ✅ Test Execution: 31s (89 tests, improved)
   ⚠️  Bundle Size: 2.4MB (target: <2MB)
   
🦀 Rust (rust-env/):
   ✅ Build Time: 89s (baseline: 91s, -2%)
   ✅ Test Execution: 12s (optimal performance)
   ✅ Memory Usage: 156MB (efficient)
   
🐹 Go (go-env/):
   ✅ Build Time: 8s (consistently fast)
   ✅ Test Execution: 6s (excellent)
   ✅ Binary Size: 12MB (optimized)
   
🐚 Nushell (nushell-env/):
   ✅ Script Execution: <1s (all automation scripts)
   ✅ Intelligence Analytics: 3s average
   ✅ Resource Monitoring: Real-time capable

💾 RESOURCE UTILIZATION
🧠 Memory Usage:
   • Peak: 3.2GB during TypeScript build (85% of available)
   • Average: 1.8GB (optimal range)
   • Python: 512MB average (efficient)
   • Rust: 256MB during compilation (good)
   
⚙️  CPU Utilization:
   • Peak: 92% during parallel builds
   • Average: 34% (good headroom)
   • TypeScript: Highest consumer (45% average)
   • Bottleneck: Single-threaded bundling process
   
💿 Disk I/O:
   • Read: 89MB/s average (SSD performing well)
   • Write: 45MB/s average (within expected range)
   • Cache Hit Rate: 78% (room for improvement)

🔥 CRITICAL PERFORMANCE ISSUES
1. 🚨 TypeScript Build Regression (+29% slower)
   Impact: High - affects development velocity
   Root Cause: webpack configuration change detected
   Fix: Revert to previous webpack-dev-server settings
   Command: `cd typescript-env && npm run build:analyze`
   
2. ⚠️  Python Dependency Resolution Slowdown
   Impact: Medium - slows environment startup
   Root Cause: uv cache invalidation detected
   Fix: Clear and rebuild uv cache
   Command: `cd python-env && uv cache clean && uv sync`

🎯 OPTIMIZATION OPPORTUNITIES
1. 📦 TypeScript Bundle Optimization
   Potential Gain: 30% build time reduction
   Actions: Enable tree-shaking, update dependencies
   Estimated Impact: Save 20s per build
   
2. 🧠 Memory Usage Optimization
   Potential Gain: 15% memory reduction
   Actions: Optimize Python object lifecycle
   Estimated Impact: Reduce peak memory by 480MB
   
3. 🚀 Parallel Test Execution
   Potential Gain: 40% test time reduction
   Actions: Enable pytest-xdist, jest --maxWorkers
   Estimated Impact: Save 18s in test execution

4. 📊 Dependency Caching Enhancement
   Potential Gain: 60% cold start improvement
   Actions: Implement layer caching in devbox
   Estimated Impact: Save 8s in environment activation

📈 PERFORMANCE TRENDS (7 days)
✅ Improvements:
   • Python build time: -8% (optimization work effective)
   • Rust memory usage: -12% (compiler upgrade benefit)
   • Test stability: +23% (fewer flaky tests)
   
⚠️  Regressions:
   • TypeScript build: +29% (requires immediate attention)
   • Overall memory peak: +18% (needs investigation)
   • Cache hit rate: -5% (cache strategy review needed)

🤖 INTELLIGENT RECOMMENDATIONS
🎯 Immediate Actions (Next 24 hours):
1. Fix TypeScript webpack configuration regression
2. Clear and rebuild Python uv cache
3. Enable parallel test execution in CI

📅 Short-term Improvements (Next week):
1. Implement bundle analysis and optimization
2. Set up automated performance regression detection
3. Optimize memory usage in Python services

🚀 Long-term Strategy (Next month):
1. Implement comprehensive caching strategy
2. Set up performance monitoring dashboards
3. Establish performance budgets and SLAs

💰 ESTIMATED IMPACT
⏱️  Time Savings: 47 seconds per development cycle
💾 Memory Reduction: 480MB peak usage
🔄 CI/CD Improvement: 23% faster pipeline execution
👥 Developer Productivity: 15% improvement estimated
```

## Focus Areas (--focus flag)
- `--focus build`: Deep analysis of compilation and build performance
- `--focus runtime`: Runtime performance and resource usage analysis
- `--focus memory`: Memory usage patterns and optimization
- `--focus tests`: Test execution performance and flaky test detection
- `--focus dependencies`: Dependency resolution and caching analysis
- `--focus ci`: CI/CD pipeline performance optimization

## Integration with Existing Scripts
- **performance-analytics.nu**: Historical data and trend analysis
- **resource-monitor.nu**: Real-time resource usage and optimization
- **test-intelligence.nu**: Test performance and regression detection
- **failure-pattern-learning.nu**: Performance failure correlation
- **github-integration.nu**: Automated issue creation for regressions

## Optimization Actions (--optimize flag)
When `--optimize` is specified, automatically:
1. Apply safe performance optimizations
2. Update configuration files for better performance  
3. Clear caches and rebuild environments
4. Generate optimized build configurations
5. Create performance improvement PR if applicable

## GitHub Integration
- **Automated issue creation** for performance regressions
- **Performance tracking** in PR descriptions
- **Benchmark reporting** on significant changes
- **Performance budget** enforcement in CI
- **Optimization progress** tracking and reporting

## Continuous Monitoring
- **Real-time performance alerts** for significant regressions
- **Daily performance reports** with trend analysis
- **Proactive optimization suggestions** based on usage patterns
- **Performance baseline** updates as improvements are implemented
- **Cross-environment performance** correlation and insights
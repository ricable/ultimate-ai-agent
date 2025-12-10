# Critic Mode Examples Documentation

## Overview

This document provides comprehensive practical examples of using the Critic mode within the SAFLA-aiGI integrated system. These examples demonstrate real-world usage patterns, CLI command sequences, MCP tool interactions, and complete workflow implementations for various analysis and evaluation scenarios.

## Example 1: Comprehensive System Health Analysis

### Scenario
Perform a complete system health analysis including performance, quality, and security assessment with actionable recommendations.

### Implementation
```bash
#!/bin/bash
# Example 1: Comprehensive System Health Analysis

echo "=== Example 1: Comprehensive System Health Analysis ==="

# Initialize analysis session
ANALYSIS_ID="health_analysis_$(date +%Y%m%d_%H%M%S)"
echo "Starting analysis: $ANALYSIS_ID"

# Step 1: System validation and readiness
echo "Step 1: Validating system readiness..."
validation_result=$(use_mcp_tool safla validate_installation '{}')
system_ready=$(echo "$validation_result" | jq -r '.validation_summary.overall_status')

if [ "$system_ready" != "passed" ]; then
  echo "ERROR: System validation failed"
  echo "$validation_result" | jq '.validation_issues[]'
  exit 1
fi

echo "✓ System validation passed"

# Step 2: Gather baseline metrics
echo "Step 2: Gathering baseline metrics..."
current_metrics=$(access_mcp_resource safla "safla://performance-metrics")
baseline_metrics=$(access_mcp_resource safla "safla://performance-baselines")

cpu_usage=$(echo "$current_metrics" | jq '.system_metrics.cpu_usage_percent')
memory_usage=$(echo "$current_metrics" | jq '.system_metrics.memory_usage_percent')
baseline_cpu=$(echo "$baseline_metrics" | jq '.baseline_metrics.cpu_usage_percent')
baseline_memory=$(echo "$baseline_metrics" | jq '.baseline_metrics.memory_usage_percent')

echo "Current Metrics: CPU ${cpu_usage}%, Memory ${memory_usage}%"
echo "Baseline Metrics: CPU ${baseline_cpu}%, Memory ${baseline_memory}%"

# Step 3: Performance bottleneck analysis
echo "Step 3: Analyzing performance bottlenecks..."
bottleneck_analysis=$(use_mcp_tool safla analyze_performance_bottlenecks '{
  "duration_seconds": 180,
  "include_memory_profile": true
}')

critical_bottlenecks=$(echo "$bottleneck_analysis" | jq '.analysis_summary.critical_issues')
performance_score=$(echo "$bottleneck_analysis" | jq '.analysis_summary.overall_performance_score')

echo "Performance Analysis: Score ${performance_score}, Critical Issues: ${critical_bottlenecks}"

# Step 4: Quality assessment
echo "Step 4: Running quality assessment..."
test_results=$(use_mcp_tool safla run_integration_tests '{
  "test_suite": "comprehensive",
  "parallel": true,
  "verbose": false
}')

total_tests=$(echo "$test_results" | jq '.test_execution_summary.total_tests')
passed_tests=$(echo "$test_results" | jq '.test_execution_summary.passed')
test_success_rate=$(echo "scale=1; $passed_tests * 100 / $total_tests" | bc)

echo "Quality Assessment: ${test_success_rate}% test success rate (${passed_tests}/${total_tests})"

# Step 5: Memory validation
echo "Step 5: Validating memory operations..."
memory_validation=$(use_mcp_tool safla validate_memory_operations '{
  "test_data_size": 25,
  "include_stress_test": true
}')

memory_status=$(echo "$memory_validation" | jq -r '.validation_summary.overall_status')
echo "Memory Validation: $memory_status"

# Step 6: Generate health score and recommendations
overall_health=$(echo "scale=1; ($performance_score + $test_success_rate) / 2" | bc)

echo ""
echo "=== SYSTEM HEALTH ANALYSIS RESULTS ==="
echo "Overall Health Score: $overall_health"
echo "Performance Score: $performance_score"
echo "Test Success Rate: ${test_success_rate}%"
echo "Memory Validation: $memory_status"
echo "Critical Issues: $critical_bottlenecks"

# Generate recommendations
if (( $(echo "$overall_health < 80" | bc -l) )); then
  echo ""
  echo "RECOMMENDATIONS:"
  
  if (( $(echo "$performance_score < 75" | bc -l) )); then
    echo "- PRIORITY HIGH: Performance optimization required"
    echo "  Action: $(echo "$bottleneck_analysis" | jq -r '.optimization_recommendations[0].description')"
  fi
  
  if (( $(echo "$test_success_rate < 95" | bc -l) )); then
    echo "- PRIORITY MEDIUM: Test reliability improvement needed"
    echo "  Action: Investigate and fix failing tests"
  fi
  
  if [ "$memory_status" != "passed" ]; then
    echo "- PRIORITY CRITICAL: Memory system requires immediate attention"
    echo "  Action: $(echo "$memory_validation" | jq -r '.recommendations[0]')"
  fi
else
  echo ""
  echo "✓ System health is within acceptable parameters"
fi

echo ""
echo "Analysis complete: $ANALYSIS_ID"
```

### Expected Output
```
=== Example 1: Comprehensive System Health Analysis ===
Starting analysis: health_analysis_20250602_153000
Step 1: Validating system readiness...
✓ System validation passed
Step 2: Gathering baseline metrics...
Current Metrics: CPU 45%, Memory 62%
Baseline Metrics: CPU 40%, Memory 55%
Step 3: Analyzing performance bottlenecks...
Performance Analysis: Score 78, Critical Issues: 1
Step 4: Running quality assessment...
Quality Assessment: 96.5% test success rate (193/200)
Step 5: Validating memory operations...
Memory Validation: passed

=== SYSTEM HEALTH ANALYSIS RESULTS ===
Overall Health Score: 87.2
Performance Score: 78
Test Success Rate: 96.5%
Memory Validation: passed
Critical Issues: 1

RECOMMENDATIONS:
- PRIORITY HIGH: Performance optimization required
  Action: Optimize memory allocation patterns in vector operations

✓ System health is within acceptable parameters

Analysis complete: health_analysis_20250602_153000
```

## Example 2: Code Quality Assessment with Agent Coordination

### Scenario
Perform detailed code quality assessment using multiple specialized agents for different aspects of analysis.

### Implementation
```bash
#!/bin/bash
# Example 2: Code Quality Assessment with Agent Coordination

echo "=== Example 2: Code Quality Assessment with Agent Coordination ==="

ANALYSIS_ID="code_quality_$(date +%Y%m%d_%H%M%S)"
declare -a agent_sessions=()

echo "Starting code quality analysis: $ANALYSIS_ID"

# Step 1: Create specialized analysis agents
echo "Step 1: Creating specialized analysis agents..."

# Code complexity analysis agent
complexity_agent=$(use_mcp_tool safla create_agent_session '{
  "agent_type": "cognitive",
  "session_config": {
    "focus": "code_complexity_analysis",
    "optimization_level": "comprehensive",
    "learning_enabled": true
  },
  "timeout_seconds": 3600
}' | jq -r '.session_id')

agent_sessions+=("$complexity_agent")
echo "✓ Created complexity analysis agent: $complexity_agent"

# Security analysis agent
security_agent=$(use_mcp_tool safla create_agent_session '{
  "agent_type": "cognitive",
  "session_config": {
    "focus": "security_analysis",
    "optimization_level": "comprehensive",
    "learning_enabled": true
  },
  "timeout_seconds": 3600
}' | jq -r '.session_id')

agent_sessions+=("$security_agent")
echo "✓ Created security analysis agent: $security_agent"

# Maintainability analysis agent
maintainability_agent=$(use_mcp_tool safla create_agent_session '{
  "agent_type": "cognitive",
  "session_config": {
    "focus": "maintainability_analysis",
    "optimization_level": "comprehensive",
    "learning_enabled": true
  },
  "timeout_seconds": 3600
}' | jq -r '.session_id')

agent_sessions+=("$maintainability_agent")
echo "✓ Created maintainability analysis agent: $maintainability_agent"

# Step 2: Execute parallel analysis
echo "Step 2: Executing parallel code analysis..."

# Complexity analysis
echo "  Analyzing code complexity..."
complexity_analysis=$(use_mcp_tool safla interact_with_agent '{
  "session_id": "'$complexity_agent'",
  "command": "analyze_complexity",
  "parameters": {
    "target_directory": "src/",
    "analysis_depth": "comprehensive",
    "include_metrics": ["cyclomatic", "cognitive", "halstead"],
    "threshold_warnings": true
  }
}')

complexity_score=$(echo "$complexity_analysis" | jq '.analysis_results.overall_complexity_score')
high_complexity_files=$(echo "$complexity_analysis" | jq '.analysis_results.high_complexity_files | length')

echo "    Complexity Score: $complexity_score"
echo "    High Complexity Files: $high_complexity_files"

# Security analysis
echo "  Analyzing security vulnerabilities..."
security_analysis=$(use_mcp_tool safla interact_with_agent '{
  "session_id": "'$security_agent'",
  "command": "security_scan",
  "parameters": {
    "scope": "full_codebase",
    "vulnerability_types": ["injection", "authentication", "authorization", "data_exposure"],
    "severity_threshold": "medium",
    "include_recommendations": true
  }
}')

security_score=$(echo "$security_analysis" | jq '.security_assessment.overall_score')
vulnerabilities_found=$(echo "$security_analysis" | jq '.security_assessment.vulnerabilities_found')
critical_vulns=$(echo "$security_analysis" | jq '.security_assessment.critical_vulnerabilities')

echo "    Security Score: $security_score"
echo "    Vulnerabilities Found: $vulnerabilities_found"
echo "    Critical Vulnerabilities: $critical_vulns"

# Maintainability analysis
echo "  Analyzing code maintainability..."
maintainability_analysis=$(use_mcp_tool safla interact_with_agent '{
  "session_id": "'$maintainability_agent'",
  "command": "maintainability_assessment",
  "parameters": {
    "target_scope": "full_project",
    "metrics": ["readability", "modularity", "documentation", "test_coverage"],
    "include_refactoring_suggestions": true
  }
}')

maintainability_score=$(echo "$maintainability_analysis" | jq '.maintainability_metrics.overall_score')
documentation_coverage=$(echo "$maintainability_analysis" | jq '.maintainability_metrics.documentation_coverage')
test_coverage=$(echo "$maintainability_analysis" | jq '.maintainability_metrics.test_coverage')

echo "    Maintainability Score: $maintainability_score"
echo "    Documentation Coverage: ${documentation_coverage}%"
echo "    Test Coverage: ${test_coverage}%"

# Step 3: Aggregate and analyze results
echo "Step 3: Aggregating analysis results..."

overall_quality_score=$(echo "scale=1; ($complexity_score + $security_score + $maintainability_score) / 3" | bc)

echo ""
echo "=== CODE QUALITY ANALYSIS RESULTS ==="
echo "Overall Quality Score: $overall_quality_score"
echo ""
echo "Detailed Scores:"
echo "  Complexity: $complexity_score (High complexity files: $high_complexity_files)"
echo "  Security: $security_score (Vulnerabilities: $vulnerabilities_found, Critical: $critical_vulns)"
echo "  Maintainability: $maintainability_score (Test coverage: ${test_coverage}%)"

# Step 4: Generate prioritized recommendations
echo ""
echo "=== PRIORITIZED RECOMMENDATIONS ==="

if [ "$critical_vulns" -gt 0 ]; then
  echo "CRITICAL PRIORITY:"
  echo "- Address $critical_vulns critical security vulnerabilities immediately"
  echo "$security_analysis" | jq -r '.security_assessment.critical_issues[].recommendation'
fi

if (( $(echo "$complexity_score < 70" | bc -l) )); then
  echo "HIGH PRIORITY:"
  echo "- Refactor high complexity code modules ($high_complexity_files files)"
  echo "$complexity_analysis" | jq -r '.refactoring_recommendations[].description'
fi

if (( $(echo "$test_coverage < 80" | bc -l) )); then
  echo "MEDIUM PRIORITY:"
  echo "- Improve test coverage from ${test_coverage}% to 85%+"
  echo "$maintainability_analysis" | jq -r '.test_improvement_suggestions[].action'
fi

if (( $(echo "$documentation_coverage < 75" | bc -l) )); then
  echo "LOW PRIORITY:"
  echo "- Enhance documentation coverage from ${documentation_coverage}% to 80%+"
fi

# Step 5: Cleanup agents
echo ""
echo "Step 5: Cleaning up analysis agents..."
for session_id in "${agent_sessions[@]}"; do
  use_mcp_tool safla terminate_agent_session '{"session_id": "'$session_id'"}' > /dev/null
  echo "✓ Terminated agent: $session_id"
done

echo ""
echo "Code quality analysis complete: $ANALYSIS_ID"
echo "Overall Quality Score: $overall_quality_score"
```

### Expected Output
```
=== Example 2: Code Quality Assessment with Agent Coordination ===
Starting code quality analysis: code_quality_20250602_153500
Step 1: Creating specialized analysis agents...
✓ Created complexity analysis agent: agent_001_complexity
✓ Created security analysis agent: agent_002_security
✓ Created maintainability analysis agent: agent_003_maintainability
Step 2: Executing parallel code analysis...
  Analyzing code complexity...
    Complexity Score: 72
    High Complexity Files: 3
  Analyzing security vulnerabilities...
    Security Score: 85
    Vulnerabilities Found: 5
    Critical Vulnerabilities: 1
  Analyzing code maintainability...
    Maintainability Score: 78
    Documentation Coverage: 68%
    Test Coverage: 82%

=== CODE QUALITY ANALYSIS RESULTS ===
Overall Quality Score: 78.3

Detailed Scores:
  Complexity: 72 (High complexity files: 3)
  Security: 85 (Vulnerabilities: 5, Critical: 1)
  Maintainability: 78 (Test coverage: 82%)

=== PRIORITIZED RECOMMENDATIONS ===
CRITICAL PRIORITY:
- Address 1 critical security vulnerabilities immediately
- Fix SQL injection vulnerability in user authentication module

HIGH PRIORITY:
- Refactor high complexity code modules (3 files)
- Simplify nested conditional logic in data processing pipeline

MEDIUM PRIORITY:
- Improve test coverage from 82% to 85%+
- Add integration tests for API endpoints

LOW PRIORITY:
- Enhance documentation coverage from 68% to 80%+

Step 5: Cleaning up analysis agents...
✓ Terminated agent: agent_001_complexity
✓ Terminated agent: agent_002_security
✓ Terminated agent: agent_003_maintainability

Code quality analysis complete: code_quality_20250602_153500
Overall Quality Score: 78.3
```

## Example 3: Performance Optimization Analysis

### Scenario
Conduct comprehensive performance analysis with benchmarking and optimization recommendations.

### Implementation
```bash
#!/bin/bash
# Example 3: Performance Optimization Analysis

echo "=== Example 3: Performance Optimization Analysis ==="

ANALYSIS_ID="performance_opt_$(date +%Y%m%d_%H%M%S)"
PERFORMANCE_REPORT="performance_analysis_${ANALYSIS_ID}.json"

echo "Starting performance optimization analysis: $ANALYSIS_ID"

# Step 1: Baseline performance measurement
echo "Step 1: Measuring baseline performance..."
baseline_metrics=$(access_mcp_resource safla "safla://performance-metrics")
current_cpu=$(echo "$baseline_metrics" | jq '.system_metrics.cpu_usage_percent')
current_memory=$(echo "$baseline_metrics" | jq '.system_metrics.memory_usage_percent')
current_io=$(echo "$baseline_metrics" | jq '.system_metrics.io_wait_percent')

echo "Baseline Metrics:"
echo "  CPU Usage: ${current_cpu}%"
echo "  Memory Usage: ${current_memory}%"
echo "  I/O Wait: ${current_io}%"

# Step 2: Performance bottleneck analysis
echo "Step 2: Analyzing performance bottlenecks..."
bottleneck_analysis=$(use_mcp_tool safla analyze_performance_bottlenecks '{
  "duration_seconds": 300,
  "include_memory_profile": true
}')

critical_bottlenecks=$(echo "$bottleneck_analysis" | jq '.analysis_summary.critical_issues')
performance_score=$(echo "$bottleneck_analysis" | jq '.analysis_summary.overall_performance_score')
memory_hotspots=$(echo "$bottleneck_analysis" | jq '.memory_analysis.hotspots | length')

echo "Bottleneck Analysis:"
echo "  Performance Score: $performance_score"
echo "  Critical Bottlenecks: $critical_bottlenecks"
echo "  Memory Hotspots: $memory_hotspots"

# Step 3: Memory performance benchmarking
echo "Step 3: Benchmarking memory performance..."
memory_benchmark=$(use_mcp_tool safla benchmark_memory_performance '{
  "test_duration": 120,
  "memory_patterns": ["sequential_access", "random_access", "bulk_operations", "concurrent_access"]
}')

memory_efficiency=$(echo "$memory_benchmark" | jq '.benchmark_results.memory_efficiency')
throughput_score=$(echo "$memory_benchmark" | jq '.benchmark_results.throughput_score')
latency_score=$(echo "$memory_benchmark" | jq '.benchmark_results.latency_score')

echo "Memory Benchmark Results:"
echo "  Memory Efficiency: ${memory_efficiency}%"
echo "  Throughput Score: $throughput_score"
echo "  Latency Score: $latency_score"

# Step 4: Vector operations benchmarking
echo "Step 4: Benchmarking vector operations..."
vector_benchmark=$(use_mcp_tool safla benchmark_vector_operations '{
  "vector_count": 10000,
  "vector_dimensions": 512,
  "operations": ["similarity", "clustering", "indexing", "search"]
}')

vector_performance=$(echo "$vector_benchmark" | jq '.benchmark_summary.overall_performance')
similarity_ops_sec=$(echo "$vector_benchmark" | jq '.operation_benchmarks.similarity.operations_per_second')
search_latency=$(echo "$vector_benchmark" | jq '.operation_benchmarks.search.average_latency_ms')

echo "Vector Operations Benchmark:"
echo "  Overall Performance: $vector_performance"
echo "  Similarity Ops/sec: $similarity_ops_sec"
echo "  Search Latency: ${search_latency}ms"

# Step 5: MCP throughput analysis
echo "Step 5: Analyzing MCP throughput..."
mcp_benchmark=$(use_mcp_tool safla benchmark_mcp_throughput '{
  "request_count": 1000,
  "concurrent_connections": 10,
  "payload_size": "medium"
}')

mcp_throughput=$(echo "$mcp_benchmark" | jq '.throughput_metrics.requests_per_second')
mcp_latency=$(echo "$mcp_benchmark" | jq '.latency_metrics.average_response_time_ms')
error_rate=$(echo "$mcp_benchmark" | jq '.error_metrics.error_rate_percent')

echo "MCP Throughput Analysis:"
echo "  Requests/sec: $mcp_throughput"
echo "  Average Latency: ${mcp_latency}ms"
echo "  Error Rate: ${error_rate}%"

# Step 6: Calculate composite performance score
echo "Step 6: Calculating composite performance score..."
composite_score=$(echo "scale=1; ($performance_score + $memory_efficiency + $vector_performance) / 3" | bc)

# Step 7: Generate optimization recommendations
echo ""
echo "=== PERFORMANCE OPTIMIZATION ANALYSIS RESULTS ==="
echo "Composite Performance Score: $composite_score"
echo ""
echo "Component Scores:"
echo "  System Performance: $performance_score"
echo "  Memory Efficiency: ${memory_efficiency}%"
echo "  Vector Operations: $vector_performance"
echo "  MCP Throughput: ${mcp_throughput} req/sec"

# Generate detailed recommendations
echo ""
echo "=== OPTIMIZATION RECOMMENDATIONS ==="

if (( $(echo "$composite_score < 75" | bc -l) )); then
  echo "CRITICAL: Overall performance below acceptable threshold"
fi

if (( $(echo "$performance_score < 70" | bc -l) )); then
  echo "HIGH PRIORITY - System Performance:"
  echo "- $(echo "$bottleneck_analysis" | jq -r '.optimization_recommendations[0].description')"
  echo "- Estimated Impact: $(echo "$bottleneck_analysis" | jq -r '.optimization_recommendations[0].estimated_improvement')"
fi

if (( $(echo "$memory_efficiency < 80" | bc -l) )); then
  echo "HIGH PRIORITY - Memory Optimization:"
  echo "- $(echo "$memory_benchmark" | jq -r '.optimization_suggestions[0].recommendation')"
  echo "- Focus Area: $(echo "$memory_benchmark" | jq -r '.optimization_suggestions[0].focus_area')"
fi

if (( $(echo "$vector_performance < 75" | bc -l) )); then
  echo "MEDIUM PRIORITY - Vector Operations:"
  echo "- $(echo "$vector_benchmark" | jq -r '.performance_recommendations[0].action')"
  echo "- Expected Improvement: $(echo "$vector_benchmark" | jq -r '.performance_recommendations[0].expected_improvement')"
fi

if (( $(echo "$mcp_latency > 100" | bc -l) )); then
  echo "MEDIUM PRIORITY - MCP Performance:"
  echo "- Optimize MCP request handling (current latency: ${mcp_latency}ms)"
  echo "- Target: Reduce latency below 50ms"
fi

# Step 8: Generate performance report
cat > "$PERFORMANCE_REPORT" << EOF
{
  "analysis_id": "$ANALYSIS_ID",
  "timestamp": "$(date -Iseconds)",
  "performance_summary": {
    "composite_score": $composite_score,
    "system_performance": $performance_score,
    "memory_efficiency": $memory_efficiency,
    "vector_performance": $vector_performance,
    "mcp_throughput": $mcp_throughput
  },
  "detailed_results": {
    "bottleneck_analysis": $bottleneck_analysis,
    "memory_benchmark": $memory_benchmark,
    "vector_benchmark": $vector_benchmark,
    "mcp_benchmark": $mcp_benchmark
  }
}
EOF

echo ""
echo "Performance optimization analysis complete: $ANALYSIS_ID"
echo "Report saved: $PERFORMANCE_REPORT"
echo "Composite Performance Score: $composite_score"
```

### Expected Output
```
=== Example 3: Performance Optimization Analysis ===
Starting performance optimization analysis: performance_opt_20250602_154000
Step 1: Measuring baseline performance...
Baseline Metrics:
  CPU Usage: 52%
  Memory Usage: 68%
  I/O Wait: 8%
Step 2: Analyzing performance bottlenecks...
Bottleneck Analysis:
  Performance Score: 73
  Critical Bottlenecks: 2
  Memory Hotspots: 4
Step 3: Benchmarking memory performance...
Memory Benchmark Results:
  Memory Efficiency: 76%
  Throughput Score: 82
  Latency Score: 78
Step 4: Benchmarking vector operations...
Vector Operations Benchmark:
  Overall Performance: 71
  Similarity Ops/sec: 1250
  Search Latency: 45ms
Step 5: Analyzing MCP throughput...
MCP Throughput Analysis:
  Requests/sec: 850
  Average Latency: 65ms
  Error Rate: 0.2%

=== PERFORMANCE OPTIMIZATION ANALYSIS RESULTS ===
Composite Performance Score: 73.3

Component Scores:
  System Performance: 73
  Memory Efficiency: 76%
  Vector Operations: 71
  MCP Throughput: 850 req/sec

=== OPTIMIZATION RECOMMENDATIONS ===
HIGH PRIORITY - System Performance:
- Optimize memory allocation patterns in data processing pipeline
- Estimated Impact: 15-20% performance improvement

HIGH PRIORITY - Memory Optimization:
- Implement memory pooling for frequent allocations
- Focus Area: Vector operations and data caching

MEDIUM PRIORITY - Vector Operations:
- Optimize similarity computation algorithms
- Expected Improvement: 25% reduction in computation time

Performance optimization analysis complete: performance_opt_20250602_154000
Report saved: performance_analysis_performance_opt_20250602_154000.json
Composite Performance Score: 73.3
```

## Example 4: Meta-Cognitive Learning Analysis

### Scenario
Analyze system learning patterns, adaptation effectiveness, and meta-cognitive capabilities.

### Implementation
```bash
#!/bin/bash
# Example 4: Meta-Cognitive Learning Analysis

echo "=== Example 4: Meta-Cognitive Learning Analysis ==="

ANALYSIS_ID="metacognitive_$(date +%Y%m%d_%H%M%S)"

echo "Starting meta-cognitive learning analysis: $ANALYSIS_ID"

# Step 1: Assess current awareness state
echo "Step 1: Assessing meta-cognitive awareness state..."
awareness_state=$(use_mcp_tool safla get_system_awareness '{}')
awareness_level=$(echo "$awareness_state" | jq '.awareness_state.attention_level')
introspection_depth=$(echo "$awareness_state" | jq -r '.introspection.depth')
focus_areas=$(echo "$awareness_state" | jq -r '.focus_configuration.active_areas[]' | tr '\n' ', ' | sed 's/,$//')

echo "Current Awareness State:"
echo "  Awareness Level: $awareness_level"
echo "  Introspection Depth: $introspection_depth"
echo "  Focus Areas: $focus_areas"

# Step 2: Analyze learning metrics
echo "Step 2: Analyzing learning performance metrics..."
learning_metrics=$(use_mcp_tool safla get_learning_metrics '{
  "metric_type": "all",
  "time_range_hours": 168
}')

learning_velocity=$(echo "$learning_metrics" | jq '.learning_performance.velocity')
knowledge_retention=$(echo "$learning_metrics" | jq '.learning_performance.retention_rate')
adaptation_success=$(echo "$learning_metrics" | jq '.adaptation_metrics.success_rate')
learning_efficiency=$(echo "$learning_metrics" | jq '.learning_performance.efficiency_score')

echo "Learning Performance Metrics:"
echo "  Learning Velocity: $learning_velocity"
echo "  Knowledge Retention: ${knowledge_retention}%"
echo "  Adaptation Success: ${adaptation_success}%"
echo "  Learning Efficiency: $learning_efficiency"

# Step 3: Analyze adaptation patterns
echo "Step 3: Analyzing adaptation patterns..."
adaptation_patterns=$(use_mcp_tool safla analyze_adaptation_patterns '{
  "pattern_type": "all",
  "analysis_depth": "comprehensive",
  "time_window_days": 30
}')

behavioral_evolution=$(echo "$adaptation_patterns" | jq '.evolution_metrics.behavioral_change_rate')
pattern_stability=$(echo "$adaptation_patterns" | jq '.pattern_analysis.stability_score')
adaptation_frequency=$(echo "$adaptation_patterns" | jq '.adaptation_frequency.daily_average')

echo "Adaptation Pattern Analysis:"
echo "  Behavioral Evolution Rate: $behavioral_evolution"
echo "  Pattern Stability: $pattern_stability"
echo "  Daily Adaptation Frequency: $adaptation_frequency"

# Step 4: Evaluate goal progress and strategy effectiveness
echo "Step 4: Evaluating goals and strategies..."
goal_progress=$(use_mcp_tool safla evaluate_goal_progress '{
  "include_recommendations": true
}')

active_goals=$(echo "$goal_progress" | jq '.goal_summary.active_goals')
completion_rate=$(echo "$goal_progress" | jq '.goal_summary.completion_rate')
average_progress=$(echo "$goal_progress" | jq '.goal_summary.average_progress')

echo "Goal Management:"
echo "  Active Goals: $active_goals"
echo "  Completion Rate: ${completion_rate}%"
echo "  Average Progress: ${average_progress}%"

# Strategy effectiveness analysis
strategies_data=$(access_mcp_resource safla "safla://strategies")
total_strategies=$(echo "$strategies_data" | jq '.strategy_summary.total_strategies')
effective_strategies=$(echo "$strategies_data" | jq '.strategy_summary.high_performing_count')
strategy_effectiveness=$(echo "scale=1; $effective_strategies * 100 / $total_strategies" | bc)

echo "Strategy Effectiveness:"
echo "  Total Strategies: $total_strategies"
echo "  High Performing: $effective_strategies"
echo "  Effectiveness Rate: ${strategy_effectiveness}%"

# Step 5: Calculate meta-cognitive effectiveness score
echo "Step 5: Calculating meta-cognitive effectiveness..."
metacognitive_score=$(echo "scale=1; ($awareness_level * 100 + $knowledge_retention + $adaptation_success + $completion_rate + $strategy_effectiveness) / 5" | bc)

echo ""
echo "=== META-COGNITIVE LEARNING ANALYSIS RESULTS ==="
echo "Meta-Cognitive Effectiveness Score: $metacognitive_score"
echo ""
echo "Component Analysis:"
echo "  Awareness Level: $awareness_level (${awareness_level}0%)"
echo "  Knowledge Retention: ${knowledge_retention}%"
echo "  Adaptation Success: ${adaptation_success}%"
echo "  Goal Completion: ${completion_rate}%"
echo "  Strategy Effectiveness: ${strategy_effectiveness}%"

# Step 6: Generate learning optimization recommendations
echo ""
echo "=== LEARNING OPTIMIZATION RECOMMENDATIONS ==="

if (( $(echo "$metacognitive_score < 80" | bc -l) )); then
  echo "OVERALL: Meta-cognitive effectiveness below optimal threshold"
fi

if (( $(echo "$awareness_level < 0.8" | bc -l) )); then
  echo "HIGH PRIORITY - Awareness Enhancement:"
  echo "- Increase awareness level from $awareness_level to 0.85+"
  echo "- Expand focus areas to include learning optimization"
  
  # Apply awareness optimization
  use_mcp_tool safla update_awareness_state '{
    "awareness_level": 0.85,
    "focus_areas": ["learning_optimization", "adaptation_effectiveness", "goal_achievement"],
    "introspection_depth": "deep"
  }' > /dev/null
  echo "✓ Applied awareness optimization"
fi

if (( $(echo "$knowledge_retention < 85" | bc -l) )); then
  echo "HIGH PRIORITY - Knowledge Retention:"
  echo "- Improve retention from ${knowledge_retention}% to 90%+"
  echo "- Implement enhanced memory consolidation"
fi

if (( $(echo "$adaptation_success < 80" | bc -l) )); then
  echo "MEDIUM PRIORITY - Adaptation Improvement:"
  echo "- Enhance adaptation mechanisms"
  echo "- Target: Increase success rate to 85%+"
  
  # Trigger learning cycle for adaptation improvement
  use_mcp_tool safla trigger_learning_cycle '{
    "learning_type": "reinforcement",
    "data_sources": ["adaptation_outcomes", "strategy_effectiveness"],
    "focus_areas": ["adaptation_optimization", "strategy_refinement"]
  }' > /dev/null
  echo "✓ Triggered adaptation learning cycle"
fi

if (( $(echo "$completion_rate < 80" | bc -l) )); then
  echo "MEDIUM PRIORITY - Goal Management:"
  echo "- Improve goal completion from ${completion_rate}% to 85%+"
  echo "- Review and optimize goal setting strategies"
fi

if (( $(echo "$strategy_effectiveness < 75" | bc -l) )); then
  echo "LOW PRIORITY - Strategy Optimization:"
  echo "- Enhance strategy selection and performance"
  echo "- Current effectiveness: ${strategy_effectiveness}%"
fi

# Step 7: Update learning parameters if needed
if (( $(echo "$learning_velocity < 10" | bc -l) )); then
  echo ""
  echo "Optimizing learning parameters..."
  use_mcp_tool safla update_learning_parameters '{
    "learning_rate": 0.15,
    "adaptation_threshold": 0.8,
    "memory_retention": 0.92,
    "exploration_factor": 0.2
  }' > /dev/null
  echo "✓ Updated learning parameters for improved velocity"
fi

echo ""
echo "Meta-cognitive learning analysis complete: $ANALYSIS_ID"
echo "Meta-Cognitive Effectiveness Score: $metacognitive_score"
```

### Expected Output
```
=== Example 4: Meta-Cognitive Learning Analysis ===
Starting meta-cognitive learning analysis: metacognitive_20250602_154500
Step 1: Assessing meta-cognitive awareness state...
Current Awareness State:
  Awareness Level: 0.75
  Introspection Depth: moderate
  Focus Areas: system_optimization, performance_analysis
Step 2: Analyzing learning performance metrics...
Learning Performance Metrics:
  Learning Velocity: 8.5
  Knowledge Retention: 82%
  Adaptation Success: 76%
  Learning Efficiency: 78
Step 3: Analyzing adaptation patterns...
Adaptation Pattern Analysis:
  Behavioral Evolution Rate: 0.12
  Pattern Stability: 0.85
  Daily Adaptation Frequency: 3.2
Step 4: Evaluating goals and strategies...
Goal Management:
  Active Goals: 12
  Completion Rate: 78%
  Average Progress: 65%
Strategy Effectiveness:
  Total Strategies: 24
  High Performing: 17
  Effectiveness Rate: 70.8%

=== META-COGNITIVE LEARNING ANALYSIS RESULTS ===
Meta-Cognitive Effectiveness Score: 76.4

Component Analysis:
  Awareness Level: 0.75 (75%)
  Knowledge Retention: 82%
  Adaptation Success: 76%
  Goal Completion: 78%
  Strategy Effectiveness: 70.8%

=== LEARNING OPTIMIZATION RECOMMENDATIONS ===
OVERALL: Meta-cognitive effectiveness below optimal threshold
HIGH PRIORITY - Awareness Enhancement:
- Increase awareness level from 0.75 to 0.85+
- Expand focus areas to include learning optimization
✓ Applied awareness optimization
HIGH PRIORITY - Knowledge Retention:
- Improve retention from 82% to 90%+
- Implement enhanced memory consolidation
MEDIUM PRIORITY - Adaptation Improvement:
- Enhance adaptation mechanisms
- Target: Increase success rate to 85%+
✓ Triggered adaptation learning cycle
MEDIUM PRIORITY - Goal Management:
- Improve goal completion from 78% to 85%+
- Review and optimize goal setting strategies
LOW PRIORITY - Strategy Optimization:
- Enhance strategy selection and performance
- Current effectiveness: 70.8%

Optimizing learning parameters...
✓ Updated learning parameters for improved velocity

Meta-cognitive learning analysis complete: metacognitive_20250602_154500
Meta-Cognitive Effectiveness Score: 76.4
```

These comprehensive examples demonstrate practical usage patterns for the Critic mode, showcasing real-world scenarios with complete CLI command sequences, MCP tool interactions, and expected outputs for various analysis and evaluation workflows within the SAFLA-aiGI integrated system.
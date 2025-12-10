# Critic Mode Resources Documentation

## Overview

This document provides comprehensive guidance on accessing and utilizing SAFLA resources within the Critic mode. SAFLA's resource system provides real-time access to system data, configuration, performance metrics, and analysis results through the `safla://` URI scheme, enabling dynamic analysis and evaluation workflows.

## Core Resource Categories

### 1. System Configuration and Status Resources

#### Primary System Resources
```bash
# Access current SAFLA configuration for analysis baseline
$ access_mcp_resource safla "safla://config"
# Returns: Complete system configuration including analysis parameters, thresholds, and optimization settings

# Monitor real-time system status during analysis
$ access_mcp_resource safla "safla://status"
# Returns: Current system health, resource utilization, active processes, and operational status

# Review deployment information for context analysis
$ access_mcp_resource safla "safla://deployments"
# Returns: Active deployments, configuration details, resource allocation, and deployment health

# Access deployment templates for configuration analysis
$ access_mcp_resource safla "safla://deployment-templates"
# Returns: Available deployment configurations, best practices, and template specifications
```

#### Configuration Analysis Workflow
```bash
#!/bin/bash
# Comprehensive configuration analysis using SAFLA resources

echo "=== Configuration Analysis Workflow ==="

# 1. Retrieve and analyze system configuration
echo "Analyzing system configuration..."
config_data=$(access_mcp_resource safla "safla://config")
echo "$config_data" | jq '.configuration_analysis' > config_analysis.json

# Extract critical configuration parameters
analysis_enabled=$(echo "$config_data" | jq -r '.analysis_configuration.enabled')
performance_monitoring=$(echo "$config_data" | jq -r '.performance_monitoring.active')
security_level=$(echo "$config_data" | jq -r '.security_configuration.level')

echo "Analysis enabled: $analysis_enabled"
echo "Performance monitoring: $performance_monitoring"
echo "Security level: $security_level"

# 2. Validate configuration against best practices
if [ "$analysis_enabled" != "true" ]; then
  echo "WARNING: Analysis configuration not fully enabled"
fi

if [ "$performance_monitoring" != "true" ]; then
  echo "WARNING: Performance monitoring not active"
fi

if [ "$security_level" != "high" ]; then
  echo "RECOMMENDATION: Consider increasing security level for production"
fi

# 3. Check system status for analysis readiness
echo "Checking system status..."
status_data=$(access_mcp_resource safla "safla://status")
system_health=$(echo "$status_data" | jq -r '.system_health.overall_status')
resource_availability=$(echo "$status_data" | jq -r '.resource_availability.analysis_ready')

echo "System health: $system_health"
echo "Analysis ready: $resource_availability"

# 4. Generate configuration recommendations
cat > config_recommendations.json << EOF
{
  "analysis_timestamp": "$(date -Iseconds)",
  "configuration_assessment": {
    "analysis_enabled": "$analysis_enabled",
    "performance_monitoring": "$performance_monitoring",
    "security_level": "$security_level",
    "system_health": "$system_health",
    "analysis_ready": "$resource_availability"
  },
  "recommendations": []
}
EOF

# Add specific recommendations based on analysis
if [ "$analysis_enabled" != "true" ] || [ "$performance_monitoring" != "true" ]; then
  echo '{"priority": "high", "category": "configuration", "description": "Enable comprehensive analysis and monitoring"}' | \
    jq '. as $rec | $rec' >> temp_rec.json
fi

if [ "$security_level" != "high" ]; then
  echo '{"priority": "medium", "category": "security", "description": "Upgrade security configuration to high level"}' | \
    jq '. as $rec | $rec' >> temp_rec.json
fi

# Merge recommendations if any exist
if [ -f temp_rec.json ]; then
  jq '.recommendations += [inputs]' config_recommendations.json temp_rec.json > temp_config.json
  mv temp_config.json config_recommendations.json
  rm temp_rec.json
fi

echo "Configuration analysis complete. Report: config_recommendations.json"
```

### 2. Performance and Optimization Resources

#### Performance Metrics and Analysis
```bash
# Access real-time performance metrics
$ access_mcp_resource safla "safla://performance-metrics"
# Returns: CPU usage, memory utilization, I/O statistics, network performance, and system load

# Get AI-generated optimization recommendations
$ access_mcp_resource safla "safla://optimization-recommendations"
# Returns: Performance optimization suggestions, resource allocation recommendations, and efficiency improvements

# Access performance baselines for comparison
$ access_mcp_resource safla "safla://performance-baselines"
# Returns: Established performance benchmarks, historical trends, and comparison metrics

# Review benchmark results and trends
$ access_mcp_resource safla "safla://benchmark-results"
# Returns: Latest benchmark execution results, performance trends, and comparative analysis
```

#### Performance Analysis and Optimization Workflow
```bash
#!/bin/bash
# Comprehensive performance analysis using SAFLA resources

PERFORMANCE_REPORT="performance_analysis_$(date +%Y%m%d_%H%M%S).json"

echo "=== Performance Analysis and Optimization ==="

# 1. Gather current performance metrics
echo "Gathering performance metrics..."
current_metrics=$(access_mcp_resource safla "safla://performance-metrics")
echo "$current_metrics" > current_performance.json

# Extract key performance indicators
cpu_usage=$(echo "$current_metrics" | jq '.system_metrics.cpu_usage_percent')
memory_usage=$(echo "$current_metrics" | jq '.system_metrics.memory_usage_percent')
io_wait=$(echo "$current_metrics" | jq '.system_metrics.io_wait_percent')
network_throughput=$(echo "$current_metrics" | jq '.network_metrics.throughput_mbps')

echo "Current Performance Metrics:"
echo "  CPU Usage: ${cpu_usage}%"
echo "  Memory Usage: ${memory_usage}%"
echo "  I/O Wait: ${io_wait}%"
echo "  Network Throughput: ${network_throughput} Mbps"

# 2. Get performance baselines for comparison
echo "Retrieving performance baselines..."
baselines=$(access_mcp_resource safla "safla://performance-baselines")
baseline_cpu=$(echo "$baselines" | jq '.baseline_metrics.cpu_usage_percent')
baseline_memory=$(echo "$baselines" | jq '.baseline_metrics.memory_usage_percent')
baseline_io=$(echo "$baselines" | jq '.baseline_metrics.io_wait_percent')

# 3. Calculate performance deviations
cpu_deviation=$(echo "scale=2; $cpu_usage - $baseline_cpu" | bc)
memory_deviation=$(echo "scale=2; $memory_usage - $baseline_memory" | bc)
io_deviation=$(echo "scale=2; $io_wait - $baseline_io" | bc)

echo "Performance Deviations from Baseline:"
echo "  CPU: ${cpu_deviation}%"
echo "  Memory: ${memory_deviation}%"
echo "  I/O Wait: ${io_deviation}%"

# 4. Get optimization recommendations
echo "Retrieving optimization recommendations..."
optimizations=$(access_mcp_resource safla "safla://optimization-recommendations")

# 5. Analyze benchmark results
echo "Analyzing benchmark results..."
benchmarks=$(access_mcp_resource safla "safla://benchmark-results")
latest_benchmark=$(echo "$benchmarks" | jq '.latest_benchmark')
benchmark_trend=$(echo "$benchmarks" | jq '.performance_trend')

# 6. Generate comprehensive performance report
cat > "$PERFORMANCE_REPORT" << EOF
{
  "analysis_timestamp": "$(date -Iseconds)",
  "performance_summary": {
    "current_metrics": {
      "cpu_usage": $cpu_usage,
      "memory_usage": $memory_usage,
      "io_wait": $io_wait,
      "network_throughput": $network_throughput
    },
    "baseline_comparison": {
      "cpu_deviation": $cpu_deviation,
      "memory_deviation": $memory_deviation,
      "io_deviation": $io_deviation
    },
    "performance_score": $(echo "scale=1; (100 - $cpu_usage + 100 - $memory_usage + 100 - $io_wait) / 3" | bc)
  },
  "detailed_analysis": {
    "current_metrics": $current_metrics,
    "baselines": $baselines,
    "optimizations": $optimizations,
    "benchmarks": $benchmarks
  }
}
EOF

# 7. Generate performance alerts and recommendations
performance_score=$(echo "scale=1; (100 - $cpu_usage + 100 - $memory_usage + 100 - $io_wait) / 3" | bc)

echo "Overall Performance Score: $performance_score"

if (( $(echo "$performance_score < 70" | bc -l) )); then
  echo "ALERT: Performance score below acceptable threshold"
  echo "Critical recommendations:"
  echo "$optimizations" | jq '.critical_optimizations[].description'
fi

if (( $(echo "$cpu_usage > 80" | bc -l) )); then
  echo "WARNING: High CPU usage detected"
fi

if (( $(echo "$memory_usage > 85" | bc -l) )); then
  echo "WARNING: High memory usage detected"
fi

if (( $(echo "$io_wait > 20" | bc -l) )); then
  echo "WARNING: High I/O wait time detected"
fi

echo "Performance analysis complete. Report: $PERFORMANCE_REPORT"
```

### 3. Testing and Quality Assurance Resources

#### Test Results and Coverage Analysis
```bash
# Access latest test execution results
$ access_mcp_resource safla "safla://test-results"
# Returns: Test execution summaries, pass/fail rates, error details, and execution times

# Review test coverage metrics
$ access_mcp_resource safla "safla://test-coverage"
# Returns: Code coverage percentages, uncovered areas, and coverage trends

# Access system logs for analysis
$ access_mcp_resource safla "safla://system-logs"
# Returns: System events, error logs, performance logs, and audit trails

# Review backup and restore status
$ access_mcp_resource safla "safla://backup-status"
# Returns: Backup schedules, completion status, restore points, and data integrity
```

#### Quality Assurance Analysis Workflow
```bash
#!/bin/bash
# Comprehensive quality assurance analysis using SAFLA resources

QA_REPORT="qa_analysis_$(date +%Y%m%d_%H%M%S).json"

echo "=== Quality Assurance Analysis ==="

# 1. Analyze test results
echo "Analyzing test results..."
test_results=$(access_mcp_resource safla "safla://test-results")
total_tests=$(echo "$test_results" | jq '.test_summary.total_tests')
passed_tests=$(echo "$test_results" | jq '.test_summary.passed_tests')
failed_tests=$(echo "$test_results" | jq '.test_summary.failed_tests')
test_success_rate=$(echo "scale=2; $passed_tests * 100 / $total_tests" | bc)

echo "Test Results Summary:"
echo "  Total Tests: $total_tests"
echo "  Passed: $passed_tests"
echo "  Failed: $failed_tests"
echo "  Success Rate: ${test_success_rate}%"

# 2. Analyze test coverage
echo "Analyzing test coverage..."
coverage_data=$(access_mcp_resource safla "safla://test-coverage")
line_coverage=$(echo "$coverage_data" | jq '.coverage_metrics.line_coverage_percent')
branch_coverage=$(echo "$coverage_data" | jq '.coverage_metrics.branch_coverage_percent')
function_coverage=$(echo "$coverage_data" | jq '.coverage_metrics.function_coverage_percent')

echo "Coverage Metrics:"
echo "  Line Coverage: ${line_coverage}%"
echo "  Branch Coverage: ${branch_coverage}%"
echo "  Function Coverage: ${function_coverage}%"

# 3. Analyze system logs for issues
echo "Analyzing system logs..."
system_logs=$(access_mcp_resource safla "safla://system-logs")
error_count=$(echo "$system_logs" | jq '.log_analysis.error_count')
warning_count=$(echo "$system_logs" | jq '.log_analysis.warning_count')
critical_issues=$(echo "$system_logs" | jq '.log_analysis.critical_issues')

echo "Log Analysis:"
echo "  Errors: $error_count"
echo "  Warnings: $warning_count"
echo "  Critical Issues: $critical_issues"

# 4. Check backup status
echo "Checking backup status..."
backup_status=$(access_mcp_resource safla "safla://backup-status")
last_backup=$(echo "$backup_status" | jq -r '.backup_summary.last_successful_backup')
backup_health=$(echo "$backup_status" | jq -r '.backup_summary.overall_health')

echo "Backup Status:"
echo "  Last Backup: $last_backup"
echo "  Health: $backup_health"

# 5. Calculate overall quality score
overall_coverage=$(echo "scale=1; ($line_coverage + $branch_coverage + $function_coverage) / 3" | bc)
quality_score=$(echo "scale=1; ($test_success_rate + $overall_coverage) / 2" | bc)

# Adjust quality score based on critical issues
if [ "$critical_issues" -gt 0 ]; then
  quality_score=$(echo "scale=1; $quality_score - ($critical_issues * 5)" | bc)
fi

echo "Overall Quality Score: $quality_score"

# 6. Generate comprehensive QA report
cat > "$QA_REPORT" << EOF
{
  "analysis_timestamp": "$(date -Iseconds)",
  "quality_summary": {
    "test_success_rate": $test_success_rate,
    "overall_coverage": $overall_coverage,
    "quality_score": $quality_score,
    "critical_issues": $critical_issues
  },
  "detailed_metrics": {
    "test_results": $test_results,
    "coverage_data": $coverage_data,
    "system_logs": $system_logs,
    "backup_status": $backup_status
  }
}
EOF

# 7. Generate quality recommendations
echo "Generating quality recommendations..."

if (( $(echo "$test_success_rate < 95" | bc -l) )); then
  echo "RECOMMENDATION: Improve test success rate (current: ${test_success_rate}%)"
  echo "$test_results" | jq '.failed_tests[].recommendation'
fi

if (( $(echo "$overall_coverage < 80" | bc -l) )); then
  echo "RECOMMENDATION: Increase test coverage (current: ${overall_coverage}%)"
  echo "$coverage_data" | jq '.uncovered_areas[].suggestion'
fi

if [ "$critical_issues" -gt 0 ]; then
  echo "CRITICAL: $critical_issues critical issues require immediate attention"
  echo "$system_logs" | jq '.critical_issues[].description'
fi

if [ "$backup_health" != "healthy" ]; then
  echo "WARNING: Backup system requires attention"
  echo "$backup_status" | jq '.recommendations[]'
fi

echo "Quality assurance analysis complete. Report: $QA_REPORT"
```

### 4. Agent and Session Management Resources

#### Agent Coordination and Analysis
```bash
# Access active agent sessions
$ access_mcp_resource safla "safla://agent-sessions"
# Returns: Active agent sessions, session details, performance metrics, and coordination status

# Review agent capabilities
$ access_mcp_resource safla "safla://agent-capabilities"
# Returns: Available agent types, capabilities, performance characteristics, and usage patterns

# Access user session information
$ access_mcp_resource safla "safla://user-sessions"
# Returns: Active user sessions, access patterns, resource usage, and session analytics
```

#### Agent-Based Analysis Coordination Workflow
```bash
#!/bin/bash
# Agent-based analysis coordination using SAFLA resources

AGENT_ANALYSIS_REPORT="agent_analysis_$(date +%Y%m%d_%H%M%S).json"

echo "=== Agent-Based Analysis Coordination ==="

# 1. Review agent capabilities for analysis planning
echo "Reviewing agent capabilities..."
agent_capabilities=$(access_mcp_resource safla "safla://agent-capabilities")
available_agents=$(echo "$agent_capabilities" | jq '.available_agent_types[]')

echo "Available Agent Types:"
echo "$agent_capabilities" | jq -r '.available_agent_types[].agent_type'

# 2. Check current agent sessions
echo "Checking current agent sessions..."
agent_sessions=$(access_mcp_resource safla "safla://agent-sessions")
active_sessions=$(echo "$agent_sessions" | jq '.active_sessions | length')
session_performance=$(echo "$agent_sessions" | jq '.performance_metrics')

echo "Active Agent Sessions: $active_sessions"

# 3. Analyze agent performance metrics
if [ "$active_sessions" -gt 0 ]; then
  echo "Analyzing agent performance..."
  avg_response_time=$(echo "$agent_sessions" | jq '.performance_metrics.average_response_time_ms')
  success_rate=$(echo "$agent_sessions" | jq '.performance_metrics.success_rate_percent')
  resource_efficiency=$(echo "$agent_sessions" | jq '.performance_metrics.resource_efficiency_percent')
  
  echo "Agent Performance Metrics:"
  echo "  Average Response Time: ${avg_response_time}ms"
  echo "  Success Rate: ${success_rate}%"
  echo "  Resource Efficiency: ${resource_efficiency}%"
else
  echo "No active agent sessions for performance analysis"
fi

# 4. Review user session patterns
echo "Analyzing user session patterns..."
user_sessions=$(access_mcp_resource safla "safla://user-sessions")
active_users=$(echo "$user_sessions" | jq '.session_summary.active_users')
session_duration=$(echo "$user_sessions" | jq '.session_analytics.average_duration_minutes')
resource_usage=$(echo "$user_sessions" | jq '.resource_usage.average_per_session')

echo "User Session Analytics:"
echo "  Active Users: $active_users"
echo "  Average Session Duration: ${session_duration} minutes"
echo "  Average Resource Usage: $resource_usage"

# 5. Generate agent coordination recommendations
echo "Generating agent coordination recommendations..."

# Calculate agent efficiency score
if [ "$active_sessions" -gt 0 ]; then
  agent_efficiency=$(echo "scale=1; ($success_rate + $resource_efficiency) / 2" | bc)
else
  agent_efficiency=0
fi

# 6. Create comprehensive agent analysis report
cat > "$AGENT_ANALYSIS_REPORT" << EOF
{
  "analysis_timestamp": "$(date -Iseconds)",
  "agent_coordination_summary": {
    "active_sessions": $active_sessions,
    "agent_efficiency_score": $agent_efficiency,
    "active_users": $active_users,
    "session_performance": {
      "avg_response_time": ${avg_response_time:-0},
      "success_rate": ${success_rate:-0},
      "resource_efficiency": ${resource_efficiency:-0}
    }
  },
  "detailed_analysis": {
    "agent_capabilities": $agent_capabilities,
    "agent_sessions": $agent_sessions,
    "user_sessions": $user_sessions
  }
}
EOF

# 7. Generate optimization recommendations
if [ "$active_sessions" -gt 0 ]; then
  if (( $(echo "${avg_response_time:-0} > 1000" | bc -l) )); then
    echo "RECOMMENDATION: Agent response times are high - consider optimization"
  fi
  
  if (( $(echo "${success_rate:-0} < 95" | bc -l) )); then
    echo "RECOMMENDATION: Agent success rate below threshold - investigate failures"
  fi
  
  if (( $(echo "${resource_efficiency:-0} < 80" | bc -l) )); then
    echo "RECOMMENDATION: Agent resource efficiency could be improved"
  fi
else
  echo "RECOMMENDATION: Consider deploying agents for enhanced analysis capabilities"
fi

if [ "$active_users" -gt 10 ] && [ "$active_sessions" -lt 3 ]; then
  echo "RECOMMENDATION: High user load with few agents - consider scaling agent deployment"
fi

echo "Agent coordination analysis complete. Report: $AGENT_ANALYSIS_REPORT"
```

### 5. Meta-Cognitive and Learning Resources

#### Meta-Cognitive State and Learning Analysis
```bash
# Access meta-cognitive awareness state
$ access_mcp_resource safla "safla://meta-cognitive-state"
# Returns: Current awareness levels, introspection depth, focus areas, and cognitive metrics

# Review system goals and progress
$ access_mcp_resource safla "safla://goals"
# Returns: Active goals, progress tracking, completion metrics, and goal dependencies

# Access available strategies
$ access_mcp_resource safla "safla://strategies"
# Returns: Strategy library, performance metrics, context mapping, and effectiveness scores

# Review learning metrics
$ access_mcp_resource safla "safla://learning-metrics"
# Returns: Learning performance, knowledge retention, adaptation rates, and improvement trends

# Access adaptation patterns
$ access_mcp_resource safla "safla://adaptation-patterns"
# Returns: Behavioral evolution, adaptation trends, learning patterns, and system evolution
```

#### Meta-Cognitive Analysis and Learning Optimization Workflow
```bash
#!/bin/bash
# Meta-cognitive analysis and learning optimization using SAFLA resources

METACOGNITIVE_REPORT="metacognitive_analysis_$(date +%Y%m%d_%H%M%S).json"

echo "=== Meta-Cognitive Analysis and Learning Optimization ==="

# 1. Analyze meta-cognitive state
echo "Analyzing meta-cognitive state..."
metacognitive_state=$(access_mcp_resource safla "safla://meta-cognitive-state")
awareness_level=$(echo "$metacognitive_state" | jq '.awareness_metrics.current_level')
introspection_depth=$(echo "$metacognitive_state" | jq -r '.introspection.depth')
focus_areas=$(echo "$metacognitive_state" | jq '.focus_configuration.active_areas')

echo "Meta-Cognitive State:"
echo "  Awareness Level: $awareness_level"
echo "  Introspection Depth: $introspection_depth"
echo "  Active Focus Areas: $(echo "$focus_areas" | jq -r '.[]' | tr '\n' ', ' | sed 's/,$//')"

# 2. Review goal progress and effectiveness
echo "Reviewing goal progress..."
goals_data=$(access_mcp_resource safla "safla://goals")
total_goals=$(echo "$goals_data" | jq '.goal_summary.total_goals')
completed_goals=$(echo "$goals_data" | jq '.goal_summary.completed_goals')
active_goals=$(echo "$goals_data" | jq '.goal_summary.active_goals')
goal_completion_rate=$(echo "scale=2; $completed_goals * 100 / $total_goals" | bc)

echo "Goal Progress:"
echo "  Total Goals: $total_goals"
echo "  Completed: $completed_goals"
echo "  Active: $active_goals"
echo "  Completion Rate: ${goal_completion_rate}%"

# 3. Analyze strategy effectiveness
echo "Analyzing strategy effectiveness..."
strategies_data=$(access_mcp_resource safla "safla://strategies")
total_strategies=$(echo "$strategies_data" | jq '.strategy_summary.total_strategies')
high_performing=$(echo "$strategies_data" | jq '.strategy_summary.high_performing_count')
strategy_effectiveness=$(echo "scale=2; $high_performing * 100 / $total_strategies" | bc)

echo "Strategy Analysis:"
echo "  Total Strategies: $total_strategies"
echo "  High Performing: $high_performing"
echo "  Effectiveness Rate: ${strategy_effectiveness}%"

# 4. Review learning metrics
echo "Reviewing learning metrics..."
learning_data=$(access_mcp_resource safla "safla://learning-metrics")
learning_velocity=$(echo "$learning_data" | jq '.learning_performance.velocity')
knowledge_retention=$(echo "$learning_data" | jq '.learning_performance.retention_rate')
adaptation_success=$(echo "$learning_data" | jq '.adaptation_metrics.success_rate')

echo "Learning Metrics:"
echo "  Learning Velocity: $learning_velocity"
echo "  Knowledge Retention: ${knowledge_retention}%"
echo "  Adaptation Success: ${adaptation_success}%"

# 5. Analyze adaptation patterns
echo "Analyzing adaptation patterns..."
adaptation_data=$(access_mcp_resource safla "safla://adaptation-patterns")
behavioral_evolution=$(echo "$adaptation_data" | jq '.evolution_metrics.behavioral_change_rate')
learning_patterns=$(echo "$adaptation_data" | jq '.pattern_analysis.dominant_patterns')
system_evolution=$(echo "$adaptation_data" | jq '.evolution_metrics.system_evolution_score')

echo "Adaptation Analysis:"
echo "  Behavioral Evolution Rate: $behavioral_evolution"
echo "  System Evolution Score: $system_evolution"

# 6. Calculate overall meta-cognitive effectiveness
metacognitive_score=$(echo "scale=1; ($awareness_level * 100 + $goal_completion_rate + $strategy_effectiveness + $knowledge_retention + $adaptation_success) / 5" | bc)

echo "Overall Meta-Cognitive Effectiveness: $metacognitive_score"

# 7. Generate comprehensive meta-cognitive report
cat > "$METACOGNITIVE_REPORT" << EOF
{
  "analysis_timestamp": "$(date -Iseconds)",
  "metacognitive_summary": {
    "awareness_level": $awareness_level,
    "goal_completion_rate": $goal_completion_rate,
    "strategy_effectiveness": $strategy_effectiveness,
    "learning_velocity": $learning_velocity,
    "adaptation_success": $adaptation_success,
    "overall_effectiveness": $metacognitive_score
  },
  "detailed_analysis": {
    "metacognitive_state": $metacognitive_state,
    "goals_data": $goals_data,
    "strategies_data": $strategies_data,
    "learning_data": $learning_data,
    "adaptation_data": $adaptation_data
  }
}
EOF

# 8. Generate optimization recommendations
echo "Generating meta-cognitive optimization recommendations..."

if (( $(echo "$awareness_level < 0.8" | bc -l) )); then
  echo "RECOMMENDATION: Increase meta-cognitive awareness level"
  echo "  Current: $awareness_level, Target: 0.85+"
fi

if (( $(echo "$goal_completion_rate < 80" | bc -l) )); then
  echo "RECOMMENDATION: Improve goal completion effectiveness"
  echo "  Current: ${goal_completion_rate}%, Target: 85%+"
fi

if (( $(echo "$strategy_effectiveness < 75" | bc -l) )); then
  echo "RECOMMENDATION: Optimize strategy selection and performance"
  echo "  Current: ${strategy_effectiveness}%, Target: 80%+"
fi

if (( $(echo "$knowledge_retention < 85" | bc -l) )); then
  echo "RECOMMENDATION: Enhance knowledge retention mechanisms"
  echo "  Current: ${knowledge_retention}%, Target: 90%+"
fi

if (( $(echo "$adaptation_success < 80" | bc -l) )); then
  echo "RECOMMENDATION: Improve adaptation success rate"
  echo "  Current: ${adaptation_success}%, Target: 85%+"
fi

if (( $(echo "$metacognitive_score < 80" | bc -l) )); then
  echo "CRITICAL: Overall meta-cognitive effectiveness below optimal threshold"
  echo "  Comprehensive optimization required across multiple dimensions"
fi

echo "Meta-cognitive analysis complete. Report: $METACOGNITIVE_REPORT"
```

This comprehensive resources documentation provides detailed guidance for accessing and utilizing SAFLA's extensive resource system within the Critic mode, enabling thorough analysis and evaluation through real-time data access and systematic resource utilization workflows.